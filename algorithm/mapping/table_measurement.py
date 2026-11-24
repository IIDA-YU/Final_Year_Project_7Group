"""基準辺の3点と残りの3辺を、決まった順番で測る。"""

from dataclasses import replace
from math import dist, radians
from threading import Lock

from algorithm.mapping.measurement_geometry import fit_baseline, measured_bounds, position_on_axis
from algorithm.mapping.measurement_motion import SENSOR_FORWARD_MM, MeasurementMotion
from algorithm.mapping.table_bounds import TableBounds
from algorithm.table_roaming import drive_in_background
from data.roaming_map_files import MAP_DIRECTORY, TRAIL_INTERVAL_MM, save_roaming_map


BACKOFF_MM = 150.0
SIDE_STEP_MM = 90.0
MAX_BACKOFF_EXTRA_MM = 35.0
MAX_OUTWARD_SHIFT_MM = 10.0
MAX_INWARD_SHIFT_MM = 45.0
LATERAL_TOLERANCE_MM = 6.0


class MeasurementMap:
    """既存のマップ画面へ、計測中の軌跡・検知点・推定した辺を渡す。"""

    def __init__(self):
        self._lock = Lock()
        self._origin = None
        self._start_heading = None
        self._pose = None
        self._trail = []
        self._edges = []
        self._bounds = TableBounds()
        self._paused = False

    def record_pose(self, state):
        with self._lock:
            if self._origin is None:
                self._origin = state.position
                self._start_heading = state.heading_deg
            x = state.position[0] - self._origin[0]
            y = state.position[1] - self._origin[1]
            point = (x, y)
            self._pose = (x, y, radians(state.heading_deg))
            if not self._trail or dist(self._trail[-1], point) >= TRAIL_INTERVAL_MM:
                self._trail.append(point)
        return replace(state, position=point)

    def record_edge(self, point):
        with self._lock:
            self._edges.append(point)

    def set_bounds(self, bounds):
        with self._lock:
            self._bounds = bounds

    def table_bounds(self):
        with self._lock:
            return self._bounds

    def snapshot(self):
        with self._lock:
            trail = self._trail.copy()
            if self._pose is not None and trail[-1] != self._pose[:2]:
                trail.append(self._pose[:2])
            return trail, self._edges.copy(), self._pose, self._paused

    def pause(self):
        with self._lock:
            self._paused = True

    def finish(self, bounds):
        self.set_bounds(bounds)
        trail, edges, _, _ = self.snapshot()
        save_roaming_map(
            MAP_DIRECTORY, trail, edges, bounds,
            start_heading=self._start_heading, sensor_forward=SENSOR_FORWARD_MM,
        )


def measure_table(robot, map_data):
    def drive(robot, read):
        def read_and_record():
            return map_data.record_pose(read())

        try:
            motion = MeasurementMotion(robot, read_and_record)
            bounds = _measure(motion, map_data)
            map_data.finish(bounds)
            print(f"6点の計測が完了しました。推定寸法 {bounds.axis_u_mm / 10:.1f} × {bounds.axis_v_mm / 10:.1f} cm")
        finally:
            map_data.pause()

    return drive_in_background(robot, drive)


def _measure(motion, map_data):
    start = motion.read()
    if start.cliff_detected:
        raise RuntimeError("開始位置で段差を検知しました。机の中央付近に置いてください。")
    initial_heading = start.heading_deg

    def probe(heading, label):
        print(f"{label}を探索します。")
        point = motion.probe(heading)
        map_data.record_edge(point)
        return point

    # ②の後退位置を、両側90 mmにある①③へ移動する基準にする。
    middle = probe(initial_heading, "② 基準辺・中央")
    reference = motion.move(initial_heading, -BACKOFF_MM).position

    _move_lateral(motion, reference, initial_heading, SIDE_STEP_MM)
    motion.rotate_to(initial_heading)
    first = probe(initial_heading, "① 基準辺・片側")
    edge_depth, _ = position_on_axis(motion.read().position, reference, initial_heading)
    if edge_depth > BACKOFF_MM + MAX_BACKOFF_EXTRA_MM:
        raise RuntimeError("①から②の後退位置へ戻るための距離が長すぎます。")
    retreat = max(BACKOFF_MM, edge_depth - MAX_OUTWARD_SHIFT_MM)
    motion.move(initial_heading, -retreat)

    _move_lateral(motion, reference, initial_heading, -SIDE_STEP_MM)
    motion.rotate_to(initial_heading)
    last = probe(initial_heading, "③ 基準辺・反対側")
    motion.move(initial_heading, -BACKOFF_MM)

    baseline = fit_baseline(first, middle, last, initial_heading)
    map_data.set_bounds(baseline.visible_bounds())

    # 基準辺の向きで中央へ戻り、法線で④、接線の両方向で⑤⑥を測る。
    toward_middle = baseline.tangent_deg + 180.0
    motion.rotate_to(toward_middle)
    motion.move(toward_middle, SIDE_STEP_MM)
    inward = baseline.outward_deg + 180.0
    motion.rotate_to(inward)
    opposite = probe(inward, "④ 基準辺の向かい")
    motion.move(inward, -BACKOFF_MM)

    tangent = baseline.tangent_deg
    motion.rotate_to(tangent)
    first_side = probe(tangent, "⑤ 隣接辺・片側")
    motion.move(tangent, -BACKOFF_MM)
    motion.rotate_to(toward_middle)
    second_side = probe(toward_middle, "⑥ 隣接辺・反対側")
    motion.move(toward_middle, -BACKOFF_MM)

    return measured_bounds(baseline, opposite, first_side, second_side)


def _move_lateral(motion, reference, initial_heading, offset):
    direction = 1 if offset > 0 else -1
    travel_heading = initial_heading + direction * 90.0
    motion.rotate_to(travel_heading)
    start = motion.read().position
    depth, lateral = position_on_axis(start, reference, initial_heading)
    _check_lateral_depth(depth)

    # 旋回後の実際の座標から、②を基準とした目標横座標までの距離を求める。
    distance = direction * (offset - lateral)
    max_travel = (SIDE_STEP_MM if direction > 0 else 2 * SIDE_STEP_MM) + 30.0
    if not -LATERAL_TOLERANCE_MM <= distance <= max_travel:
        raise RuntimeError("横移動の目標が予定の範囲を外れています。")
    if distance <= LATERAL_TOLERANCE_MM:
        return

    end = motion.move(travel_heading, distance, position_tolerance=LATERAL_TOLERANCE_MM).position
    end_depth, end_lateral = position_on_axis(end, reference, initial_heading)
    _check_lateral_depth(end_depth)
    if abs(end_depth - depth) > 12.0 or abs(end_lateral - offset) > 12.0:
        raise RuntimeError("横移動後の位置が目標から大きくずれています。")


def _check_lateral_depth(depth):
    if not -MAX_INWARD_SHIFT_MM <= depth <= MAX_OUTWARD_SHIFT_MM:
        raise RuntimeError("横移動の位置が②の後退ラインから大きくずれています。")

"""学習マップを使ってStartキューブに接近し、検知位置を反映する。"""

from dataclasses import replace
from math import atan2, degrees, dist
from time import monotonic

from algorithm.mapping.drive_limits import heading_vector, rotate_point
from algorithm.table_roaming import (
    angle_difference,
    drive_in_background,
    turn_in_place,
    wait_for_braking,
)


FORWARD_SPEED_MMPS = 20.0
MIN_SENSOR_CLEARANCE_MM = 30.0
STOP_MARGIN_MM = 10.0
TURN_MARGIN_MM = 15.0
FORWARD_LOOKAHEAD_MM = FORWARD_SPEED_MMPS * 1.0
START_SEARCH_RADIUS_MM = 130.0
START_STANDOFF_MM = 75.0
CUBE_STANDOFF_MM = 95.0
MAX_STEP_MM = 25.0
MIN_STEP_MM = 5.0
OBSERVATION_TIMEOUT_SECONDS = 5.0
RUN_TIMEOUT_SECONDS = 180.0

# カメラを水平にしたときの暫定値。焦点距離は320×240画像を基準とする。
FOCAL_LENGTH_PX = 235.0
CUBE_HEIGHT_MM = 44.0
CAMERA_FORWARD_MM = 20.0


def drive_on_map(robot, drive_map):
    def drive(robot, read):
        try:
            _navigate(robot, read, drive_map)
        except RuntimeError:
            drive_map.set_status("Stopped. See terminal for details")
            raise
        finally:
            drive_map.pause()

    return drive_in_background(robot, drive)


def _navigate(robot, read, drive_map):
    table = drive_map.table
    start_x, start_y = drive_map.start_position
    origin = None
    rotation = 0.0
    deadline = None
    # 旋回中のセンサー軌道全体が内側に収まるよう、機体原点の制限を決める。
    inset = MIN_SENSOR_CLEARANCE_MM + drive_map.sensor_forward_mm + TURN_MARGIN_MM

    def read_aligned():
        nonlocal origin, rotation
        state = read()
        if origin is None:
            origin = state.position
            rotation = drive_map.start_heading - state.heading_deg

        x, y = state.position
        dx, dy = rotate_point((x - origin[0], y - origin[1]), rotation)
        position = (start_x + dx, start_y + dy)
        heading = state.heading_deg + rotation
        drive_map.record_pose(position, heading)

        if state.cliff_detected:
            drive_map.record_cliff(position, heading)
            raise RuntimeError("走行中に段差を検知しました。マップとCozmoの位置を確認してください。")

        if deadline is None:
            return replace(state, position=position, heading_deg=heading)
        if monotonic() >= deadline:
            raise RuntimeError("キューブへの接近が時間内に完了しませんでした。")
        if not drive_map.cube_frame()[0]:
            raise RuntimeError("走行中にカメラがOFFになりました。")
        sensor = drive_map.sensor_position(position, heading)
        if min(table.clearances(sensor)) <= MIN_SENSOR_CLEARANCE_MM + STOP_MARGIN_MM:
            raise RuntimeError("センサーが机の端に近づきすぎました。マップとCozmoの位置を確認してください。")
        return replace(state, position=position, heading_deg=heading)

    state = read_aligned()
    while (target := drive_map.claim_start()) is None:
        state = read_aligned()

    deadline = monotonic() + RUN_TIMEOUT_SECONDS
    if min(table.clearances(target)) <= inset + FORWARD_LOOKAHEAD_MM:
        print("Startは机の縁に近いため、内側の走行可能な範囲から探します。")

    drive_map.set_status("Heading toward Start")
    state = _turn_toward(robot, read_aligned, state, table, inset, _bearing(state.position, target))
    scan_heading = state.heading_deg
    scan_step = 0
    previous_cube = None
    while True:
        state, cube = _observe_cube(read_aligned, drive_map, target, previous_cube)
        if cube is not None:
            if previous_cube is None:
                drive_map.set_status("Approaching Start cube")
            previous_cube = cube
            gap = dist(state.position, cube) - CUBE_STANDOFF_MM
            if gap <= MIN_STEP_MM:
                drive_map.correct_start(cube)
                print(f"Startキューブの位置を更新しました。X={cube[0]:.1f} mm, Y={cube[1]:.1f} mm")
                return
            heading = _bearing(state.position, cube)
            if abs(angle_difference(heading, state.heading_deg)) > 7.0:
                state = _turn_toward(robot, read_aligned, state, table, inset, heading)
                continue
        elif previous_cube is not None:
            raise RuntimeError("接近中にStartキューブを見失いました。")
        else:
            gap = dist(state.position, target) - START_STANDOFF_MM if not scan_step else 0.0

        clearance = table.approaching_edge(state.position, state.heading_deg, inset_mm=inset)
        distance = min(gap, MAX_STEP_MM, clearance.travel_mm - FORWARD_LOOKAHEAD_MM)
        if distance >= MIN_STEP_MM:
            state = _advance_step(robot, read_aligned, state, distance)
        elif cube is not None:
            drive_map.correct_start(cube)
            print("机の縁の手前で停止し、Startキューブの位置を更新しました。")
            return
        elif scan_step < 7:
            drive_map.set_status("Searching near Start")
            scan_step += 1
            state = _turn_toward(
                robot, read_aligned, state, table, inset, scan_heading + 45.0 * scan_step,
            )
        else:
            raise RuntimeError("Start付近にキューブを検知できませんでした。")


def _require_turn_room(table, state, inset):
    if min(table.clearances(state.position)) <= inset:
        raise RuntimeError("机の端から十分に離れて旋回できません。Cozmoの位置を確認してください。")


def _bearing(position, target):
    return degrees(atan2(target[1] - position[1], target[0] - position[0]))


def _turn_toward(robot, read, state, table, inset, heading):
    change = angle_difference(heading, state.heading_deg)
    if abs(change) <= 3.0:
        return state
    _require_turn_room(table, state, inset)
    return turn_in_place(robot, read, state, 1 if change > 0 else -1, abs(change))


def _advance_step(robot, read, state, distance):
    start = state.position
    deadline = monotonic() + distance / FORWARD_SPEED_MMPS + 1.0
    robot.drive_wheels(FORWARD_SPEED_MMPS, FORWARD_SPEED_MMPS)
    try:
        while dist(state.position, start) < distance:
            state = read()
            if monotonic() >= deadline:
                raise RuntimeError("予定した距離まで前進できませんでした。")
    finally:
        robot.stop_all_motors()
    return wait_for_braking(read)


def _observe_cube(read, drive_map, target, previous_cube):
    # 制動後の映像だけ使い、移動中の古い検知枠を現在位置と組み合わせない。
    stopped_at = monotonic()
    while True:
        state = read()
        _, frame = drive_map.cube_frame()
        now = monotonic()
        if frame is not None and frame[0] >= max(stopped_at, now - 2.0):
            _, (width, height), boxes = frame
            candidates = (
                point for point in _cube_positions(state, width, height, boxes)
                if dist(point, target) <= START_SEARCH_RADIUS_MM
                and min(drive_map.table.clearances(point)) > 0
                and (previous_cube is None or dist(point, previous_cube) <= 50.0)
            )
            reference = previous_cube if previous_cube is not None else target
            return state, min(candidates, key=lambda point: dist(point, reference), default=None)
        if now - stopped_at >= OBSERVATION_TIMEOUT_SECONDS:
            raise RuntimeError("キューブ検知用の新しい映像を受信できません。")


def _cube_positions(state, width, height, boxes):
    focal_x = FOCAL_LENGTH_PX * width / 320.0
    focal_y = FOCAL_LENGTH_PX * height / 240.0
    for x1, y1, x2, y2 in boxes:
        if not (1 < x1 < x2 < width - 1 and 1 < y1 < y2 < height - 1 and y2 - y1 >= height / 60):
            continue
        depth = CUBE_HEIGHT_MM * focal_y / (y2 - y1)
        lateral = (width / 2 - (x1 + x2) / 2) * depth / focal_x
        dx, dy = rotate_point((CAMERA_FORWARD_MM + depth + CUBE_HEIGHT_MM / 2, lateral), state.heading_deg)
        yield state.position[0] + dx, state.position[1] + dy

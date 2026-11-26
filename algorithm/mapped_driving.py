"""学習マップの4辺から距離を取りながら机上を走る。"""

from dataclasses import replace
from math import atan, degrees, dist, radians
from time import monotonic

from pycozmo.robot import TRACK_WIDTH

from algorithm.mapping.drive_limits import rotate_point
from algorithm.pawn_avoidance import (
    BYPASS_EXIT_MM,
    FRONT_MM,
    HALF_WIDTH_MM,
    PAWN_CLEARANCE_MM,
    blocking_pawn,
    pawn_clearance,
    plan_bypass,
    turn_clears_pawns,
)
from algorithm.pawn_confirmation import PAWN_MATCH_DISTANCE_MM, PawnConfirmation
from algorithm.table_roaming import (
    TURN_ANGLES,
    angle_difference,
    drive_in_background,
    turn_in_place,
    wait_for_braking,
)
from camera.pawn_detection import MAX_FRAME_AGE_SECONDS, PAWN_RADIUS_MM


FORWARD_SPEED_MMPS = 40.0
PROGRESS_CHECK_SECONDS = 1.0
MIN_PROGRESS_MM = 3.0
MIN_SENSOR_CLEARANCE_MM = 30.0
STOP_MARGIN_MM = 10.0
TURN_MARGIN_MM = 15.0
# 状態の受信と停止にかかる時間を見込んで1秒分を先読みする。
FORWARD_LOOKAHEAD_MM = FORWARD_SPEED_MMPS * 1.0
TURN_RUNWAY_MM = FORWARD_LOOKAHEAD_MM + 20.0
BYPASS_SPEED_MMPS = 28.0
MAX_STEERING_CURVATURE = 1 / 30.0


class _VisionPause(Exception):
    def __init__(self, state):
        self.state = state


def drive_on_map(robot, drive_map, *, pawn_camera=None):
    def drive(robot, read):
        try:
            _navigate(robot, read, drive_map, pawn_camera=pawn_camera)
        finally:
            drive_map.pause()

    return drive_in_background(robot, drive)


def _navigate(robot, read, drive_map, *, pawn_camera=None):
    table = drive_map.table
    start_x, start_y = drive_map.start_position
    origin = None
    rotation = 0.0
    last_frame = None
    vision_enabled = False
    pose_frame = None
    pawns = ()
    confirmation = PawnConfirmation()
    # 旋回中のセンサー軌道全体が内側に収まるよう、機体原点の制限を決める。
    inset = MIN_SENSOR_CLEARANCE_MM + drive_map.sensor_forward_mm + TURN_MARGIN_MM

    def align_position(position):
        x, y = position
        dx, dy = rotate_point((x - origin[0], y - origin[1]), rotation)
        return start_x + dx, start_y + dy

    def vision_ready():
        return not vision_enabled or (
            last_frame is not None
            and monotonic() - last_frame.received_at <= MAX_FRAME_AGE_SECONDS
            and not confirmation.waiting
        )

    def on_vision_stop():
        confirmation.stopped(monotonic())

    def read_aligned():
        nonlocal origin, rotation, pose_frame, vision_enabled, last_frame, pawns
        state = read()
        if origin is None:
            origin = state.position
            rotation = drive_map.start_heading - state.heading_deg
            pose_frame = state.pose_frame

        position = align_position(state.position)
        heading = state.heading_deg + rotation
        drive_map.record_pose(position, heading)

        if state.cliff_detected:
            drive_map.record_cliff(position, heading)
            raise RuntimeError("走行中に段差を検知しました。マップとCozmoの位置を確認してください。")

        sensor = drive_map.sensor_position(position, heading)
        if min(table.clearances(sensor)) <= MIN_SENSOR_CLEARANCE_MM + STOP_MARGIN_MM:
            raise RuntimeError("センサーが机の端に近づきすぎました。マップとCozmoの位置を確認してください。")

        if pawn_camera is not None:
            vision_enabled, frames = pawn_camera.read_frames()
            now = monotonic()
            confirmation.expire(now)
            if not vision_enabled:
                last_frame = None
                confirmation.cancel()
            for frame in frames:
                if frame.pose_frame != pose_frame:
                    continue
                last_frame = frame
                if not confirmation.waiting and now - frame.received_at > MAX_FRAME_AGE_SECONDS:
                    continue
                points = []
                for raw_point in frame.positions:
                    point = align_position(raw_point)
                    if min(table.clearances(point)) >= -PAWN_RADIUS_MM and all(
                        dist(point, known) > PAWN_MATCH_DISTANCE_MM for known in pawns
                    ):
                        points.append(point)
                points.sort(key=lambda point: dist(point, position))
                confirmed = confirmation.observe(points, frame.received_at, now)
                if confirmed:
                    drive_map.record_pawns(confirmed)
                    pawns = drive_map.pawn_positions()
            if any(pawn_clearance(position, heading, pawn) <= PAWN_CLEARANCE_MM for pawn in pawns):
                raise RuntimeError("ポーンが機体の安全距離内にあるため停止しました。")
        return replace(state, position=position, heading_deg=heading)

    def read_turn():
        state = read_aligned()
        if not vision_ready():
            raise _VisionPause(state)
        return state

    state = read_aligned()
    preferred_direction = 1
    while True:
        _require_turn_room(table, state, inset)
        state = _advance_to_turn(
            robot, read_aligned, state, table, inset,
            get_pawns=lambda: pawns, vision_ready=vision_ready, on_vision_stop=on_vision_stop,
        )
        state = _wait_for_vision(robot, read_aligned, state, vision_ready, on_vision_stop)
        _require_turn_room(table, state, inset)
        direction, angle = _choose_turn(table, state, inset, preferred_direction, pawns)
        try:
            state = turn_in_place(robot, read_turn, state, direction, angle)
        except _VisionPause as pause:
            state = _wait_for_vision(robot, read_aligned, pause.state, vision_ready, on_vision_stop)
            continue
        preferred_direction = -direction


def _require_turn_room(table, state, inset):
    if min(table.clearances(state.position)) <= inset:
        raise RuntimeError("机の端から十分に離れて旋回できません。Cozmoの位置を確認してください。")


def _advance_to_turn(
    robot, read, state, table, inset, *, get_pawns=lambda: (), vision_ready=lambda: True,
    on_vision_stop=None,
):
    checkpoint = state.position
    next_check = 0.0
    moving = False

    while True:
        if not vision_ready():
            state = _wait_for_vision(robot, read, state, vision_ready, on_vision_stop)
            checkpoint = state.position
            moving = False

        edge = table.approaching_edge(state.position, state.heading_deg, inset_mm=inset)
        if edge.travel_mm <= FORWARD_LOOKAHEAD_MM:
            break

        pawns = get_pawns()
        pawn = blocking_pawn(state.position, state.heading_deg, pawns)
        if pawn is not None:
            path = plan_bypass(state.position, state.heading_deg, pawn, pawns, table, inset)
            if path is not None:
                state = _follow_bypass(
                    robot, read, state, table, inset, path, get_pawns, vision_ready, on_vision_stop,
                )
                checkpoint = state.position
                moving = False
                continue
            space_after_pawn = table.approaching_edge(pawn, state.heading_deg, inset_mm=inset).travel_mm
            if (
                min(table.clearances(pawn)) <= inset + HALF_WIDTH_MM + PAWN_CLEARANCE_MM
                or space_after_pawn <= BYPASS_EXIT_MM + FORWARD_LOOKAHEAD_MM
            ):
                break
            raise RuntimeError("進行方向にポーンがあり、安全な迂回経路を確保できませんでした。")

        now = monotonic()
        if not moving:
            robot.drive_wheels(FORWARD_SPEED_MMPS, FORWARD_SPEED_MMPS)
            next_check = now + PROGRESS_CHECK_SECONDS
            moving = True
        elif now >= next_check:
            if dist(state.position, checkpoint) < MIN_PROGRESS_MM:
                raise RuntimeError("前進中にCozmoの位置が変わりません。")
            checkpoint = state.position
            next_check = now + PROGRESS_CHECK_SECONDS
        state = read()

    robot.stop_all_motors()
    return wait_for_braking(read)


def _wait_for_vision(robot, read, state, vision_ready, on_vision_stop=None):
    if not vision_ready():
        robot.stop_all_motors()
        if on_vision_stop is not None:
            on_vision_stop()
        while not vision_ready():
            state = read()
            if on_vision_stop is not None:
                on_vision_stop()
    return state


def _follow_bypass(robot, read, state, table, inset, path, get_pawns, vision_ready, on_vision_stop=None):
    checkpoint = state.position
    next_check = monotonic() + PROGRESS_CHECK_SECONDS
    while True:
        if not vision_ready():
            state = _wait_for_vision(robot, read, state, vision_ready, on_vision_stop)
            checkpoint = state.position
            next_check = monotonic() + PROGRESS_CHECK_SECONDS

        x, y = path.local_position(state.position, state.heading_deg)
        target_y, slope, change = path.profile(x)
        heading_error = angle_difference(path.heading + degrees(atan(slope)), state.heading_deg)
        cross_track = y - target_y
        if x >= path.end_x and abs(cross_track) <= 5 and abs(heading_error) <= 6:
            return state

        if (
            abs(cross_track) > 14
            or x > path.end_x + 25
            or min(table.clearances(state.position)) <= inset
        ):
            raise RuntimeError("ポーンの迂回経路から外れたため停止しました。")

        # 曲線の少し先も確認し、迂回中に別の駒が確定した場合は接近前に停止する。
        ahead, ahead_heading = path.pose_at(min(x + 18, max(x, path.end_x)))
        if min(table.clearances(ahead)) <= inset or any(
            pawn_clearance(ahead, ahead_heading, pawn) <= PAWN_CLEARANCE_MM + 4
            for pawn in get_pawns()
        ):
            raise RuntimeError("ポーンの迂回を続けられる空間がなくなったため停止しました。")

        now = monotonic()
        if now >= next_check:
            if dist(state.position, checkpoint) < MIN_PROGRESS_MM:
                raise RuntimeError("迂回中にCozmoの位置が変わりません。")
            checkpoint = state.position
            next_check = now + PROGRESS_CHECK_SECONDS

        curvature = change / (1 + slope * slope) ** 1.5
        curvature += radians(heading_error) / 20 - cross_track / 480
        curvature = max(-MAX_STEERING_CURVATURE, min(MAX_STEERING_CURVATURE, curvature))
        difference = BYPASS_SPEED_MMPS * curvature * TRACK_WIDTH.mm / 2
        robot.drive_wheels(BYPASS_SPEED_MMPS - difference, BYPASS_SPEED_MMPS + difference)
        state = read()


def _choose_turn(table, state, inset, preferred_direction, pawns=()):
    for angle in TURN_ANGLES:
        for direction in (preferred_direction, -preferred_direction):
            heading = state.heading_deg + direction * angle
            edge = table.approaching_edge(state.position, heading, inset_mm=inset)
            if (
                edge.travel_mm > TURN_RUNWAY_MM
                and blocking_pawn(state.position, heading, pawns, TURN_RUNWAY_MM + FRONT_MM) is None
                and turn_clears_pawns(state.position, state.heading_deg, direction, angle, pawns)
            ):
                return direction, angle

    raise RuntimeError("机の端やポーンから距離を保って前進できる向きが見つかりません。")

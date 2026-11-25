"""学習マップの4辺から距離を取りながら机上を走る。"""

from dataclasses import replace
from math import dist
from time import monotonic

from algorithm.mapping.drive_limits import rotate_point
from algorithm.table_roaming import (
    TURN_ANGLES,
    drive_in_background,
    turn_in_place,
    wait_for_braking,
)


FORWARD_SPEED_MMPS = 40.0
PROGRESS_CHECK_SECONDS = 1.0
MIN_PROGRESS_MM = 3.0
MIN_SENSOR_CLEARANCE_MM = 30.0
STOP_MARGIN_MM = 10.0
TURN_MARGIN_MM = 15.0
# 状態の受信と停止にかかる時間を見込んで1秒分を先読みする。
FORWARD_LOOKAHEAD_MM = FORWARD_SPEED_MMPS * 1.0
TURN_RUNWAY_MM = FORWARD_LOOKAHEAD_MM + 20.0


def drive_on_map(robot, drive_map):
    def drive(robot, read):
        try:
            _navigate(robot, read, drive_map)
        finally:
            drive_map.pause()

    return drive_in_background(robot, drive)


def _navigate(robot, read, drive_map):
    table = drive_map.table
    start_x, start_y = drive_map.start_position
    origin = None
    rotation = 0.0
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

        sensor = drive_map.sensor_position(position, heading)
        if min(table.clearances(sensor)) <= MIN_SENSOR_CLEARANCE_MM + STOP_MARGIN_MM:
            raise RuntimeError("センサーが机の端に近づきすぎました。マップとCozmoの位置を確認してください。")
        return replace(state, position=position, heading_deg=heading)

    state = read_aligned()
    preferred_direction = 1
    while True:
        _require_turn_room(table, state, inset)
        state = _advance_to_turn(robot, read_aligned, state, table, inset)
        _require_turn_room(table, state, inset)
        direction, angle = _choose_turn(table, state, inset, preferred_direction)
        state = turn_in_place(robot, read_aligned, state, direction, angle)
        preferred_direction = -direction


def _require_turn_room(table, state, inset):
    if min(table.clearances(state.position)) <= inset:
        raise RuntimeError("机の端から十分に離れて旋回できません。Cozmoの位置を確認してください。")


def _advance_to_turn(robot, read, state, table, inset):
    checkpoint = state.position
    next_check = 0.0
    moving = False

    while table.approaching_edge(state.position, state.heading_deg, inset_mm=inset).travel_mm > FORWARD_LOOKAHEAD_MM:
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


def _choose_turn(table, state, inset, preferred_direction):
    for angle in TURN_ANGLES:
        for direction in (preferred_direction, -preferred_direction):
            heading = state.heading_deg + direction * angle
            edge = table.approaching_edge(state.position, heading, inset_mm=inset)
            if edge.travel_mm > TURN_RUNWAY_MM:
                return direction, angle

    raise RuntimeError("机の端から距離を保って前進できる向きが見つかりません。")

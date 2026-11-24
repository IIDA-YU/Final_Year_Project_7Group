"""学習した机の縁の直前まで進み、机の内側へ旋回する。"""

from dataclasses import replace
from math import atan2, degrees, dist
from time import monotonic

from algorithm.mapping.drive_limits import STOP_AHEAD_MM, heading_vector, rotate_point
from algorithm.table_roaming import drive_in_background, turn_in_place, wait_for_braking


FORWARD_SPEED_MMPS = 40.0
MAX_OVERRUN_MM = 8.0
PROGRESS_CHECK_SECONDS = 1.0
MIN_PROGRESS_MM = 3.0
TURN_ANGLES = (80, -110, 135, -80, 110, -135)
TURN_HEADING_MARGIN_DEG = 6.0


def drive_on_map(robot, drive_map):
    def drive(robot, read):
        try:
            _navigate(robot, read, drive_map)
        finally:
            drive_map.pause()

    return drive_in_background(robot, drive)


def _navigate(robot, read, drive_map):
    first = read()
    start_x, start_y = drive_map.start_position
    origin_x, origin_y = first.position
    rotation = drive_map.start_heading - first.heading_deg
    area = drive_map.area

    def align(state):
        x, y = state.position
        dx, dy = rotate_point((x - origin_x, y - origin_y), rotation)
        position = (start_x + dx, start_y + dy)
        heading = state.heading_deg + rotation
        drive_map.record_pose(position, heading)

        if state.cliff_detected:
            robot.stop_all_motors()
            print("段差を検知しました。旋回位置の補正を保存します。")
            drive_map.learn_cliff(position, heading)
            raise RuntimeError(f"旋回位置を補正しました。保存先 {drive_map.path}")

        if min(area.clearances(position)) < -MAX_OVERRUN_MM:
            raise RuntimeError("旋回範囲を外れました。開始位置とマップを確認してください。")
        return replace(state, position=position, heading_deg=heading)

    def read_aligned():
        return align(read())

    state = align(first)
    turn_count = 0
    while True:
        if area.forward_room(state.position, state.heading_deg) > STOP_AHEAD_MM:
            state = _move_forward(robot, read_aligned, area, state.position)

        angle = _choose_turn(area, state.position, state.heading_deg, turn_count)
        direction = 1 if angle > 0 else -1
        direction_name = "左" if direction == 1 else "右"
        print(f"学習マップの縁に近づきました。{direction_name}へ{abs(angle):.0f}度旋回します。")
        state = turn_in_place(robot, read_aligned, state, direction, abs(angle))
        turn_count += 1
        if _room_after_turn(area, state.position, state.heading_deg) <= 0:
            raise RuntimeError("旋回後に机の内側へ進める方位を確認できません。")


def _move_forward(robot, read, area, start_position):
    robot.drive_wheels(FORWARD_SPEED_MMPS, FORWARD_SPEED_MMPS)
    checkpoint = start_position
    next_check = monotonic() + PROGRESS_CHECK_SECONDS
    while True:
        state = read()
        if area.forward_room(state.position, state.heading_deg) <= STOP_AHEAD_MM:
            robot.stop_all_motors()
            return wait_for_braking(read)

        now = monotonic()
        if now >= next_check:
            if dist(state.position, checkpoint) < MIN_PROGRESS_MM:
                raise RuntimeError("前進中にCozmoの位置が変わりません。")
            checkpoint = state.position
            next_check = now + PROGRESS_CHECK_SECONDS


def _choose_turn(area, position, heading, turn_count):
    # 旋回角の候補を順にずらして、同じ往復経路に偏りにくくする。
    index = turn_count % len(TURN_ANGLES)
    angles = TURN_ANGLES[index:] + TURN_ANGLES[:index]
    center_x, center_y = area.center()
    toward_center = degrees(atan2(center_y - position[1], center_x - position[0]))
    center_turn = (toward_center - heading + 180.0) % 360.0 - 180.0
    desired_room = min(160.0, max(area.width, area.depth) / 3)
    best_angle, best_room = 0.0, 0.0

    for angle in angles + (center_turn,):
        # 停止時に方位が少しずれても、机の内側へ進める角度を選ぶ。
        room = min(
            _room_after_turn(area, position, heading + angle + error)
            for error in (-TURN_HEADING_MARGIN_DEG, 0.0, TURN_HEADING_MARGIN_DEG)
        )
        if room >= desired_room:
            return angle
        if room > best_room:
            best_angle, best_room = angle, room

    if best_room <= 0:
        raise RuntimeError("旋回後に机の内側へ向かう経路が見つかりません。")
    return best_angle


def _room_after_turn(area, position, heading):
    room = area.forward_room(position, heading)
    if room <= 2 * STOP_AHEAD_MM:
        return 0.0

    # 角では、旋回後の進行方向が両方の辺から離れることを確かめる。
    distance = min(35.0, room / 2)
    dx, dy = heading_vector(heading)
    inside = (position[0] + distance * dx, position[1] + distance * dy)
    if min(area.clearances(inside)) <= STOP_AHEAD_MM:
        return 0.0
    return room

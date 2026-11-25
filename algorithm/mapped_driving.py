"""学習マップで浅い進入角を補正しながら、段差センサーを使って走る。"""

from dataclasses import replace
from functools import partial
from math import copysign, dist
from time import monotonic

from algorithm.mapping.drive_limits import rotate_point
from algorithm.table_roaming import (
    angle_difference,
    drive_in_background,
    explore_table,
    turn_in_place,
    wait_for_braking,
)


FORWARD_SPEED_MMPS = 40.0
PROGRESS_CHECK_SECONDS = 1.0
MIN_PROGRESS_MM = 3.0
APPROACH_BAND_MM = 90.0  # 辺との垂直距離。この範囲で進入角を確認する。
SHALLOW_ANGLE_DEG = 25.0
TARGET_APPROACH_ANGLE_DEG = 35.0
ADJUSTMENT_SPEED_MMPS = 16.0
ADJUSTMENT_TOLERANCE_DEG = 1.5


def drive_on_map(robot, drive_map):
    def drive(robot, read):
        try:
            _navigate(robot, read, drive_map)
        finally:
            drive_map.pause()

    return drive_in_background(robot, drive)


def _navigate(robot, read, drive_map):
    start_x, start_y = drive_map.start_position
    origin = None
    rotation = 0.0
    cliff_active = False

    def read_aligned():
        nonlocal origin, rotation, cliff_active
        state = read()
        if origin is None:
            origin = state.position
            rotation = drive_map.start_heading - state.heading_deg

        x, y = state.position
        dx, dy = rotate_point((x - origin[0], y - origin[1]), rotation)
        position = (start_x + dx, start_y + dy)
        heading = state.heading_deg + rotation
        drive_map.record_pose(position, heading)

        if state.cliff_detected and not cliff_active:
            drive_map.record_cliff(position, heading)
        cliff_active = state.cliff_detected
        return replace(state, position=position, heading_deg=heading)

    advance = partial(_advance_until_cliff, table=drive_map.table)
    explore_table(robot, read_aligned, advance=advance)


def _advance_until_cliff(robot, read, state, *, table):
    forward_start = state.position
    checkpoint = state.position
    next_check = 0.0
    moving = False

    while not state.cliff_detected:
        adjustment = _shallow_approach_adjustment(table, state)
        if adjustment and moving:
            robot.stop_all_motors()
            state = wait_for_braking(read, return_on_cliff=True)
            if state.cliff_detected:
                break
            moving = False
            adjustment = _shallow_approach_adjustment(table, state)

        if adjustment:
            direction = 1 if adjustment > 0 else -1
            print(f"机の辺への進入角が浅いため、方位を {adjustment:+.0f}° 補正します。")
            state = turn_in_place(
                robot, read, state, direction, abs(adjustment),
                speed=ADJUSTMENT_SPEED_MMPS, tolerance=ADJUSTMENT_TOLERANCE_DEG,
            )
            # 後退するときは、補正後にまっすぐ前進した区間を使う。
            forward_start = state.position

        now = monotonic()
        if not moving:
            robot.drive_wheels(FORWARD_SPEED_MMPS, FORWARD_SPEED_MMPS)
            checkpoint = state.position
            next_check = now + PROGRESS_CHECK_SECONDS
            moving = True
        elif now >= next_check:
            if dist(state.position, checkpoint) < MIN_PROGRESS_MM:
                raise RuntimeError("前進中にCozmoの位置が変わりません。")
            checkpoint = state.position
            next_check = now + PROGRESS_CHECK_SECONDS
        state = read()

    return state, forward_start


def _shallow_approach_adjustment(table, state):
    edge = table.approaching_edge(state.position, state.heading_deg)
    if abs(edge.clearance_mm) > APPROACH_BAND_MM:
        return 0.0

    toward_normal = angle_difference(edge.outward_heading_deg, state.heading_deg)
    approach_angle = 90.0 - abs(toward_normal)
    if approach_angle >= SHALLOW_ANGLE_DEG:
        return 0.0

    return copysign(TARGET_APPROACH_ANGLE_DEG - approach_angle, toward_normal)

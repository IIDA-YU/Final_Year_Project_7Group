"""机の角で段差検知を繰り返したとき、別の向きに逃げる。"""

from contextlib import contextmanager
from math import hypot
import sys
from threading import Event, Thread
from time import monotonic

from hardware.step_detection import StepDetector


FORWARD_SPEED = 60.0  # mm/s
BACKUP_SPEED = 30.0
TURN_SPEED = 30.0
BRAKE_SECONDS = 0.15
CORNER_SECONDS = 3.0
CORNER_DISTANCE_MM = 85.0
BACKUP_DISTANCE_MM = 42.0
MIN_BACKUP_MM = 5.0
BACKUP_TIMEOUT_SECONDS = 4.0
TURN_TIMEOUT_SECONDS = 6.0
MAX_TURNS_PER_CORNER = 5
HEADING_MARGIN_DEG = 28.0
TURN_TOLERANCE_DEG = 3.0
TURN_ANGLES = (125, 90, 180, 135, 150, 60, 75, 45)


class _StopRequested(Exception):
    """カメラ画面を閉じたときに走行ループを抜ける。"""


def roam_table(robot):
    """段差センサーを使った探索走行を開始する。"""
    return drive_in_background(robot, explore_table)


@contextmanager
def drive_in_background(robot, drive):
    """カメラ表示中は別スレッドで走行し、画面を閉じると終了する。"""
    stop_requested = Event()
    worker = Thread(target=_drive, args=(robot, stop_requested, drive))
    worker.start()
    try:
        yield
    finally:
        stop_requested.set()
        worker.join()


def _drive(robot, stop_requested, drive):
    try:
        robot.stop_all_motors()
        with StepDetector(robot) as detector:
            pose_frame = None

            def read():
                nonlocal pose_frame
                if stop_requested.is_set():
                    raise _StopRequested
                state = detector.read()
                if stop_requested.is_set():
                    raise _StopRequested
                if state.unsafe_to_drive:
                    raise RuntimeError("Cozmoを机の中央に置いてから再実行してください。")
                if pose_frame is None:
                    pose_frame = state.pose_frame
                elif state.pose_frame != pose_frame:
                    raise RuntimeError("Cozmoの座標の基準が変わりました。")
                return state

            try:
                drive(robot, read)
            finally:
                robot.stop_all_motors()
    except _StopRequested:
        pass
    except Exception as exc:
        if not stop_requested.is_set():
            print(f"机上走行を終了しました。{exc}", file=sys.stderr)


def _drive_straight_until_cliff(robot, read, state):
    forward_start = state.position
    robot.drive_wheels(FORWARD_SPEED, FORWARD_SPEED)
    while not state.cliff_detected:
        state = read()
    return state, forward_start


def explore_table(robot, read, *, advance=_drive_straight_until_cliff):
    """前進して段差を検知するたび、後退して別の向きに旋回する。"""
    state = read()
    if state.cliff_detected:
        raise RuntimeError("開始位置で段差を検知しました。")

    turns = []  # (旋回方向、旋回角度、旋回後の方位)
    cliff_headings = []
    last_cliff_position = None
    initial_direction = -1

    while True:
        started_at = monotonic()
        state, forward_start = advance(robot, read, state)
        robot.stop_all_motors()

        quick_detection = monotonic() - started_at <= CORNER_SECONDS
        near_last_cliff = (
            last_cliff_position is not None
            and _distance(state.position, last_cliff_position) <= CORNER_DISTANCE_MM
        )
        if not turns or not (quick_detection or near_last_cliff):
            turns.clear()
            cliff_headings.clear()
            initial_direction *= -1

        if len(turns) >= MAX_TURNS_PER_CORNER:
            raise RuntimeError("同じ付近で段差検知が続いたため停止しました。")

        last_cliff_position = state.position
        cliff_headings.append(state.heading_deg)
        backup_distance = BACKUP_DISTANCE_MM + 20.0 * min(len(turns), 2)
        state = wait_for_braking(read)
        state = _back_up(robot, read, state, forward_start, backup_distance)

        direction, angle = _choose_turn(
            state.heading_deg, turns, cliff_headings, initial_direction
        )
        direction_name = "左" if direction == 1 else "右"
        print(f"段差を検知。後退後、{direction_name}へ{angle}度旋回します。")
        state = turn_in_place(robot, read, state, direction, angle)
        turns.append((direction, angle, state.heading_deg))


def _back_up(robot, read, state, forward_start, desired_distance):
    # 直前に前進した範囲を目安に、後退しすぎないよう距離を制限する。
    start = state.position
    distance = min(desired_distance, _distance(start, forward_start))
    if distance < MIN_BACKUP_MM:
        raise RuntimeError("旋回前に後退できる距離が足りません。")

    deadline = monotonic() + BACKUP_TIMEOUT_SECONDS
    robot.drive_wheels(-BACKUP_SPEED, -BACKUP_SPEED)
    while True:
        state = read()
        if _distance(state.position, start) >= distance:
            break
        if monotonic() >= deadline:
            raise RuntimeError("後退に必要な距離へ到達できませんでした。")

    robot.stop_all_motors()
    state = wait_for_braking(read)
    if state.cliff_detected:
        raise RuntimeError("後退後も段差を検知しています。")
    return state


def _choose_turn(heading, turns, cliff_headings, initial_direction):
    """未使用の角度から、これまで危険を検知した向きを避けて選ぶ。"""
    preferred_angle = TURN_ANGLES[min(len(turns), 2)]
    preferred_direction = turns[-1][0] if turns else initial_direction
    used_angles = {turn[1] for turn in turns}
    old_headings = cliff_headings + [turn[2] for turn in turns]

    for angle in (preferred_angle,) + TURN_ANGLES:
        if angle in used_angles:
            continue
        for direction in (preferred_direction, -preferred_direction):
            target = heading + direction * angle
            if all(
                abs(angle_difference(target, old)) >= HEADING_MARGIN_DEG
                for old in old_headings
            ):
                return direction, angle

    raise RuntimeError("同じ角で試せる旋回先が見つかりませんでした。")


def turn_in_place(
    robot, read, state, direction, angle, *, speed=TURN_SPEED, tolerance=TURN_TOLERANCE_DEG,
):
    previous_heading = state.heading_deg
    turned = 0.0
    deadline = monotonic() + TURN_TIMEOUT_SECONDS
    speed *= direction
    robot.drive_wheels(-speed, speed)

    while True:
        state = read()
        if state.cliff_detected:
            raise RuntimeError("旋回中に段差を検知しました。")

        change = angle_difference(state.heading_deg, previous_heading)
        turned += direction * change
        previous_heading = state.heading_deg
        if turned >= angle - tolerance:
            break
        if turned < -15.0 or monotonic() >= deadline:
            raise RuntimeError("指定した方向・角度へ旋回できませんでした。")

    robot.stop_all_motors()
    return wait_for_braking(read, require_clear=True)


def wait_for_braking(read, require_clear=False, *, return_on_cliff=False):
    deadline = monotonic() + BRAKE_SECONDS
    while True:
        state = read()
        if state.cliff_detected:
            if require_clear:
                raise RuntimeError("旋回直後に段差を検知しました。")
            if return_on_cliff:
                return state
        if monotonic() >= deadline:
            return state


def angle_difference(target, reference):
    # 例: +179度から-179度に移ったときの変化は+2度。
    return (target - reference + 180.0) % 360.0 - 180.0


def _distance(point_a, point_b):
    return hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

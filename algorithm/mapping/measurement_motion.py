"""6点計測中の端探索、走行方向の補正、定角度旋回を行う。"""

from math import atan2, degrees, dist
from time import monotonic

from algorithm.mapping.drive_limits import heading_vector
from algorithm.mapping.measurement_geometry import angle_error, position_on_axis


# 添付の計測プログラムと同じ暫定値。実機の校正値が分かったら変更する。
SENSOR_FORWARD_MM = 17.5
APPROACH_SPEED = 40.0
TRAVEL_SPEED = 65.0
COMMAND_INTERVAL_SECONDS = 0.06
HEADING_GAIN = 1.2
MAX_HEADING_ERROR_DEG = 18.0
MAX_SCAN_MM = 950.0
MAX_SCAN_SECONDS = 55.0
MAX_SCAN_CROSS_MM = 22.0
MAX_DETECTION_HEADING_ERROR_DEG = 6.0
MAX_BRACKET_MM = 15.0
MAX_BRACKET_SECONDS = 0.35

TURN_SPEED = 30.0
TURN_SLOW_SPEED = 10.0
TURN_LEAD_SECONDS = 0.04
TURN_FINAL_TOLERANCE_DEG = 0.5
TURN_RETRY_BACKOFF_DEG = 5.0
MAX_RETRY_BRAKING_DRIFT_DEG = 2.0
TURN_TIMEOUT_SECONDS = 60.0


class MeasurementMotion:
    def __init__(self, robot, read):
        self.robot = robot
        self.read = read
        self._next_command = 0.0
        self._reverse_track_first = True

    def probe(self, heading):
        """端の検知直前と直後の中間姿勢から、センサーの位置を求める。"""
        state = start = self.read()
        self._require_clear(start)
        self._require_heading(start, heading)
        deadline = monotonic() + MAX_SCAN_SECONDS
        try:
            while True:
                travel, cross = position_on_axis(state.position, start.position, heading)
                if not -10.0 <= travel <= MAX_SCAN_MM or monotonic() >= deadline:
                    raise RuntimeError("段差探索の距離または時間の上限に達しました。")
                if abs(cross) > MAX_SCAN_CROSS_MM:
                    raise RuntimeError("段差探索中に進行ラインから外れました。")
                self._steer(state, heading, APPROACH_SPEED)
                previous = state
                state = self.read()
                if state.cliff_detected:
                    break
        finally:
            self._stop()

        self._require_heading(state, heading, MAX_DETECTION_HEADING_ERROR_DEG)
        travel, cross = position_on_axis(state.position, start.position, heading)
        if not 0.0 <= travel <= MAX_SCAN_MM or abs(cross) > MAX_SCAN_CROSS_MM:
            raise RuntimeError("段差検知時の位置が探索範囲の外にあります。")
        return _interpolate_edge(previous, state)

    def move(self, heading, distance, *, position_tolerance=0.0):
        """指定の方位を保って進む。負の距離は検知した辺からの後退。"""
        reverse = distance < 0
        direction = -1 if reverse else 1
        start = state = self._settle(0.18, allow_cliff=reverse)
        self._require_heading(start, heading, 6.0 if reverse else 3.0)
        cliff_cleared = not start.cliff_detected
        distance = abs(distance)
        max_cross = 25.0 if reverse else 32.0
        overrun = 9.0 if reverse else 25.0
        max_bias = 3.0 if reverse else 9.0
        lookahead = 130.0 if reverse else TRAVEL_SPEED / 0.22
        deadline = monotonic() + max(7.0, 4.0 + 2.5 * distance / TRAVEL_SPEED)

        try:
            while True:
                along, cross = position_on_axis(state.position, start.position, heading)
                progress = direction * along
                if abs(cross) > max_cross or not -10.0 <= progress <= distance + overrun:
                    raise RuntimeError("移動中の横ずれ、または移動距離が許容範囲を超えました。")
                if not state.cliff_detected:
                    cliff_cleared = True
                elif cliff_cleared or progress > 60.0:
                    raise RuntimeError("移動中に段差を検知しました。")
                if distance - progress <= position_tolerance:
                    break
                if monotonic() >= deadline:
                    raise RuntimeError("指定した距離の移動が時間切れになりました。")
                bias = _clamp(degrees(atan2(-direction * cross, lookahead)), max_bias)
                self._steer(state, heading + bias, direction * TRAVEL_SPEED)
                state = self.read()
        finally:
            self._stop()

        state = self._settle(0.18)
        along, cross = position_on_axis(state.position, start.position, heading)
        if direction * along > distance + overrun or abs(cross) > max_cross:
            raise RuntimeError("停止後の位置が移動の許容範囲を超えています。")
        return state

    def rotate_to(self, target):
        """停止後の方位を確認し、必要なら5度離れて低速で合わせ直す。"""
        state = self.read()
        self._require_clear(state)
        error = angle_error(target, state.heading_deg)
        if abs(error) <= TURN_FINAL_TOLERANCE_DEG:
            return

        deadline = monotonic() + TURN_TIMEOUT_SECONDS
        initial_direction = 1 if error > 0 else -1
        self._turn_phase(target, min(deadline, monotonic() + 35.0), TURN_SPEED,
                         TURN_SLOW_SPEED, 18.0, 1.2, initial_direction)
        retry_stop_heading = None
        while monotonic() < deadline:
            state = self._settle(0.50)
            error = angle_error(target, state.heading_deg)
            if abs(error) <= TURN_FINAL_TOLERANCE_DEG:
                state = self._settle(0.35)
                error = angle_error(target, state.heading_deg)
                if abs(error) <= TURN_FINAL_TOLERANCE_DEG:
                    return

            side = -1 if error > 0 else 1
            away = state.heading_deg + side * TURN_RETRY_BACKOFF_DEG
            aim = target
            if retry_stop_heading is not None:
                # 低速で停止を指示した後に回った分を、次の再接近で見込む。
                drift = abs(angle_error(state.heading_deg, retry_stop_heading))
                aim += side * min(drift, MAX_RETRY_BRAKING_DRIFT_DEG)
            self._turn_phase(away, min(deadline, monotonic() + 9.0), 13.0, 13.0, 0.0, 1.0)
            self._settle(0.20)
            retry_stop_heading = self._turn_phase(
                aim, min(deadline, monotonic() + 9.0), 10.0, 8.0, 4.0, TURN_FINAL_TOLERANCE_DEG,
            )

        raise RuntimeError("再調整しても旋回角度が許容範囲に入りませんでした。")

    def _turn_phase(self, target, deadline, speed, slow_speed, slow_zone, tolerance, initial_direction=None):
        started = False
        try:
            while True:
                state = self.read()
                self._require_clear(state)
                error = angle_error(target, state.heading_deg)
                if abs(error) <= tolerance:
                    return state.heading_deg
                if monotonic() >= deadline:
                    raise RuntimeError(f"旋回が時間切れになりました。残り角度 {error:+.1f}°")

                direction = 1 if error > 0 else -1
                if initial_direction is not None and abs(error) >= 150.0:
                    direction = initial_direction
                turn_speed = direction * (speed if abs(error) > slow_zone else slow_speed)
                if not started:
                    self._begin_turn(-turn_speed, turn_speed)
                    started = True
                else:
                    self._command(-turn_speed, turn_speed)
        finally:
            self._stop()

    def _begin_turn(self, left, right):
        # 各旋回で後退側と前進側の先行を交互に切り替える。
        left_first = (left < 0) == self._reverse_track_first
        self._reverse_track_first = not self._reverse_track_first
        self.robot.drive_wheels(left if left_first else 0.0, 0.0 if left_first else right)
        deadline = monotonic() + TURN_LEAD_SECONDS
        while monotonic() < deadline:
            self._require_clear(self.read())
        self._command(left, right)

    def _steer(self, state, heading, speed):
        error = self._require_heading(state, heading, MAX_HEADING_ERROR_DEG)
        correction = _clamp(HEADING_GAIN * error, 12.0)
        self._command(speed - correction, speed + correction)

    def _require_heading(self, state, heading, tolerance=3.0):
        error = angle_error(heading, state.heading_deg)
        if abs(error) > tolerance:
            raise RuntimeError(f"計測中の進行方向が目標から {error:+.1f}° ずれています。")
        return error

    def _require_clear(self, state):
        if state.cliff_detected:
            raise RuntimeError("端探索以外の動作中に段差を検知しました。")

    def _settle(self, seconds, *, allow_cliff=False):
        deadline = monotonic() + seconds
        while True:
            state = self.read()
            if not allow_cliff:
                self._require_clear(state)
            if monotonic() >= deadline:
                return state

    def _command(self, left, right):
        now = monotonic()
        if now >= self._next_command:
            self.robot.drive_wheels(left, right)
            self._next_command = now + COMMAND_INTERVAL_SECONDS

    def _stop(self):
        self.robot.stop_all_motors()
        self._next_command = 0.0


def _interpolate_edge(clear, detected):
    elapsed = detected.received_at - clear.received_at
    gap = dist(_sensor_position(clear.position, clear.heading_deg),
               _sensor_position(detected.position, detected.heading_deg))
    if not 0 <= elapsed <= MAX_BRACKET_SECONDS or gap > MAX_BRACKET_MM:
        raise RuntimeError("段差検知前後の状態の間隔が広く、端の位置を確定できません。")
    if elapsed == 0 and not 0 < gap <= 2.5:
        raise RuntimeError("同じ受信時刻の状態から端の位置を確定できません。")

    x = (clear.position[0] + detected.position[0]) / 2
    y = (clear.position[1] + detected.position[1]) / 2
    heading = clear.heading_deg + angle_error(detected.heading_deg, clear.heading_deg) / 2
    return _sensor_position((x, y), heading)


def _sensor_position(position, heading):
    dx, dy = heading_vector(heading)
    return (position[0] + SENSOR_FORWARD_MM * dx, position[1] + SENSOR_FORWARD_MM * dy)


def _clamp(value, limit):
    return max(-limit, min(limit, value))

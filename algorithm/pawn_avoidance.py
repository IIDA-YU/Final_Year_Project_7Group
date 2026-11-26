"""前進のままポーンを迂回し、元の走行線へ戻る経路を求める。"""

from dataclasses import dataclass
from math import atan, ceil, cos, degrees, dist, hypot, pi, sin

from algorithm.mapping.drive_limits import heading_vector, rotate_point
from camera.pawn_detection import PAWN_RADIUS_MM


# 機体座標の原点から、フォークを含む機体外周までの概算値。
FRONT_MM = 42.0
REAR_MM = 68.0
HALF_WIDTH_MM = 32.0
TURN_CENTER_BACK_MM = 19.0
PAWN_CLEARANCE_MM = PAWN_RADIUS_MM + 12.0
PLAN_CLEARANCE_MM = PAWN_CLEARANCE_MM + 8.0
DETOUR_LOOKAHEAD_MM = 180.0
MAX_PATH_CURVATURE = 1 / 42.0
TABLE_PATH_MARGIN_MM = 5.0
BYPASS_EXIT_MM = REAR_MM + PLAN_CLEARANCE_MM + 3.0


@dataclass(frozen=True)
class Bypass:
    start: tuple[float, float]
    heading: float
    peak_x: float
    end_x: float
    offset: float

    def local_position(self, position, heading):
        return _relative_to(_turn_center(position, heading), self.start, self.heading)

    def profile(self, x):
        """横位置と、前方距離に対する1階・2階微分を返す。"""
        if x < 0 or x >= self.end_x:
            return 0.0, 0.0, 0.0
        if x < self.peak_x:
            span = self.peak_x
            phase = pi * x / span
        else:
            span = self.end_x - self.peak_x
            phase = pi + pi * (x - self.peak_x) / span
        rate = pi / span
        half_offset = self.offset / 2
        return (
            half_offset * (1 - cos(phase)),
            half_offset * rate * sin(phase),
            half_offset * rate * rate * cos(phase),
        )

    def pose_at(self, x):
        y, slope, _ = self.profile(x)
        dx, dy = rotate_point((x, y), self.heading)
        heading = self.heading + degrees(atan(slope))
        position = _shift_forward((self.start[0] + dx, self.start[1] + dy), heading, TURN_CENTER_BACK_MM)
        return position, heading


def blocking_pawn(position, heading, pawns, lookahead=DETOUR_LOOKAHEAD_MM):
    """直進した場合、進路と安全幅が重なる最も手前のポーンを返す。"""
    candidates = []
    for pawn in pawns:
        x, y = _relative_to(pawn, position, heading)
        if 0 < x <= lookahead and abs(y) <= HALF_WIDTH_MM + PAWN_CLEARANCE_MM:
            candidates.append((x, pawn))
    return min(candidates)[1] if candidates else None


def pawn_clearance(position, heading, pawn):
    x, y = _relative_to(pawn, position, heading)
    return hypot(max(x - FRONT_MM, -REAR_MM - x, 0.0), max(abs(y) - HALF_WIDTH_MM, 0.0))


def plan_bypass(position, heading, pawn, pawns, table, inset):
    center = _turn_center(position, heading)
    peak_x, pawn_y = _relative_to(pawn, center, heading)
    max_end = table.approaching_edge(
        position, heading, inset_mm=inset + TABLE_PATH_MARGIN_MM,
    ).travel_mm
    exit_span = BYPASS_EXIT_MM - TURN_CENTER_BACK_MM
    if peak_x <= FRONT_MM + TURN_CENTER_BACK_MM + PLAN_CLEARANCE_MM or max_end < peak_x + exit_span:
        return None

    best_path = None
    best_length = float("inf")
    for side in (1, -1):
        for extra in range(0, 46, 5):
            offset = pawn_y + side * (HALF_WIDTH_MM + PLAN_CLEARANCE_MM + extra)
            min_span = pi * (abs(offset) / (2 * MAX_PATH_CURVATURE)) ** 0.5
            if min_span > peak_x:
                break
            min_end = peak_x + max(min_span, exit_span)
            last_end = min(max_end, min_end + 60)
            for end_x in _end_distances(min_end, last_end):
                if end_x >= best_length:
                    break
                path = Bypass(center, heading, peak_x, end_x, offset)
                length = _safe_length(path, pawns, table, inset)
                if length < best_length:
                    best_path, best_length = path, length
    return best_path


def _end_distances(first, last):
    end = first
    while end < last:
        yield end
        end += 12.0
    if last >= first:
        yield last


def _safe_length(path, pawns, table, inset):
    steps = ceil(path.end_x / 5.0)
    previous = path.pose_at(0)[0]
    length = 0.0
    for step in range(steps + 1):
        position, heading = path.pose_at(step * path.end_x / steps)
        if min(table.clearances(position)) < inset + TABLE_PATH_MARGIN_MM or any(
            pawn_clearance(position, heading, pawn) < PLAN_CLEARANCE_MM for pawn in pawns
        ):
            return float("inf")
        length += dist(previous, position)
        previous = position
    return length


def turn_clears_pawns(position, heading, direction, angle, pawns):
    if not pawns:
        return True
    # 原点は前輪付近にあるため、超信地旋回でも原点自体は円弧を描く。
    center = _turn_center(position, heading)
    steps = ceil(angle / 5)
    for step in range(steps + 1):
        yaw = heading + direction * angle * step / steps
        pose = _shift_forward(center, yaw, TURN_CENTER_BACK_MM)
        if any(pawn_clearance(pose, yaw, pawn) <= PAWN_CLEARANCE_MM + 4 for pawn in pawns):
            return False
    return True


def _turn_center(position, heading):
    return _shift_forward(position, heading, -TURN_CENTER_BACK_MM)


def _shift_forward(position, heading, distance):
    dx, dy = heading_vector(heading)
    return (position[0] + distance * dx, position[1] + distance * dy)


def _relative_to(point, origin, heading):
    return rotate_point((point[0] - origin[0], point[1] - origin[1]), -heading)

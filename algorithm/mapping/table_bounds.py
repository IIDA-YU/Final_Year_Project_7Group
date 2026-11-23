"""段差を検知した点から、長方形の机の辺と寸法を推定する。"""

from dataclasses import dataclass
from math import cos, radians, sin
from statistics import median
from typing import Optional


LINE_TOLERANCE_MM = 15.0
MIN_POINT_SPAN_MM = 25.0
MIN_OUTWARD_MM = 7.0
SIDES = ((0, -1), (0, 1), (1, -1), (1, 1))

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class EdgeHit:
    point: Point
    robot_position: Point
    sample_count: int = 1


@dataclass(frozen=True)
class TableBounds:
    segments_mm: tuple[Segment, ...] = ()
    axis_u_mm: Optional[float] = None
    axis_v_mm: Optional[float] = None
    angle_deg: Optional[float] = None


@dataclass(frozen=True)
class _Side:
    offset: float
    points: tuple
    low: float
    high: float
    error: float


def estimate_table(hits):
    """90度違う方位は同じ長方形なので、0〜90度を調べる。"""
    if len(hits) < 2:
        return TableBounds()

    best = None
    for angle in range(0, 90, 2):
        fitted = _fit_angle(hits, angle)
        if best is None or fitted[0] > best[0]:
            best = fitted

    # 粗い探索の前後を0.25度刻みで調べ直す。
    coarse_angle = best[1]
    for step in range(-8, 9):
        angle = (coarse_angle + step * 0.25) % 90
        fitted = _fit_angle(hits, angle)
        if fitted[0] > best[0]:
            best = fitted

    _, angle, sides = best
    return _make_bounds(angle, sides)


def _fit_angle(hits, angle):
    c, s = cos(radians(angle)), sin(radians(angle))
    projected = [
        (*_project(hit.point, c, s), *_project(hit.robot_position, c, s))
        for hit in hits
    ]
    sides = [_find_side(projected, axis, sign) for axis, sign in SIDES]

    # 角付近の1点を、同時に2本の辺の根拠にしない。
    owners = {}
    for side_id, side in enumerate(sides):
        if side is None:
            continue
        for index, normal, _, outward in side.points:
            rank = (abs(normal - side.offset), -outward, -len(side.points))
            if index not in owners or rank < owners[index][0]:
                owners[index] = (rank, side_id)

    distinct_sides = []
    for side_id, side in enumerate(sides):
        points = () if side is None else tuple(
            point for point in side.points if owners[point[0]][1] == side_id
        )
        distinct_sides.append(_fit_side(points))

    confirmed = [side for side in distinct_sides if side is not None]
    score = (
        len(confirmed),
        sum(len(side.points) for side in confirmed),
        -sum(side.error for side in confirmed),
    )
    return score, angle, distinct_sides


def _find_side(projected, axis, sign):
    # 法線方向の値は符号を統一し、常に机の外側ほど大きくする。
    inside_limit = max(sign * point[axis + 2] for point in projected)
    eligible = []
    for index, point in enumerate(projected):
        normal = sign * point[axis]
        outward = normal - sign * point[axis + 2]
        if outward >= MIN_OUTWARD_MM and normal >= inside_limit - LINE_TOLERANCE_MM:
            eligible.append((index, normal, point[1 - axis], outward))

    best = None
    best_rank = ()
    for _, anchor, _, _ in eligible:
        nearby = [
            point[1] for point in eligible
            if abs(point[1] - anchor) <= LINE_TOLERANCE_MM
        ]
        if len(nearby) < 2:
            continue
        center = median(nearby)
        points = tuple(
            point for point in eligible if abs(point[1] - center) <= LINE_TOLERANCE_MM
        )
        side = _fit_side(points)
        if side is not None:
            rank = (len(points), -side.error, side.high - side.low)
            if rank > best_rank:
                best, best_rank = side, rank
    return best


def _fit_side(points):
    if len(points) < 2:
        return None
    low = min(point[2] for point in points)
    high = max(point[2] for point in points)
    if high - low < MIN_POINT_SPAN_MM:
        return None
    offset = median(point[1] for point in points)
    error = sum(abs(point[1] - offset) for point in points)
    return _Side(offset, points, low, high, error)


def _make_bounds(angle, sides):
    c, s = cos(radians(angle)), sin(radians(angle))

    def world(u, v):
        return (u * c - v * s, u * s + v * c)

    u_low, u_high, v_low, v_high = sides
    u_size = _opposite_distance(u_low, u_high)
    v_size = _opposite_distance(v_low, v_high)
    if u_size is not None and v_size is not None:
        corners = (
            world(-u_low.offset, -v_low.offset),
            world(u_high.offset, -v_low.offset),
            world(u_high.offset, v_high.offset),
            world(-u_low.offset, v_high.offset),
        )
        segments = tuple((corners[i], corners[(i + 1) % 4]) for i in range(4))
    else:
        segments = []
        for (axis, sign), side in zip(SIDES, sides):
            if side is None:
                continue
            offset = sign * side.offset
            if axis == 0:
                segments.append((world(offset, side.low), world(offset, side.high)))
            else:
                segments.append((world(side.low, offset), world(side.high, offset)))
        segments = tuple(segments)

    return TableBounds(segments, u_size, v_size, angle if segments else None)


def _opposite_distance(low, high):
    if low is None or high is None:
        return None
    distance = low.offset + high.offset
    return distance if distance > 0 else None


def _project(point, c, s):
    x, y = point
    return (x * c + y * s, -x * s + y * c)
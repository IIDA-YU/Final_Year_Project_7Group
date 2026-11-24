"""3点から基準辺を求め、残りの3点で長方形を決める。"""

from dataclasses import dataclass
from math import atan2, degrees, dist
from statistics import mean

from algorithm.mapping.drive_limits import rotate_point
from algorithm.mapping.table_bounds import TableBounds, rectangle_bounds


MIN_BASELINE_SPAN_MM = 125.0
MIN_SEGMENT_MM = 60.0
MAX_RESIDUAL_MM = 12.0
MAX_SEGMENT_ANGLE_DEG = 6.0
MAX_BASELINE_UNCERTAINTY_DEG = 3.0
MAX_SKEW_DEG = 35.0
MIN_TABLE_SIDE_MM = 160.0


def angle_error(target, heading):
    return (target - heading + 180.0) % 360.0 - 180.0


def position_on_axis(position, origin, heading):
    x, y = position
    return rotate_point((x - origin[0], y - origin[1]), -heading)


@dataclass(frozen=True)
class Baseline:
    outward_deg: float
    tangent_deg: float
    offset_mm: float
    sample_min_mm: float
    sample_max_mm: float

    def visible_bounds(self):
        start = rotate_point((self.offset_mm, self.sample_min_mm), self.outward_deg)
        end = rotate_point((self.offset_mm, self.sample_max_mm), self.outward_deg)
        return TableBounds(segments_mm=((start, end),), angle_deg=self.outward_deg)


def fit_baseline(first, middle, last, initial_heading):
    points = (first, middle, last)
    center = (mean(point[0] for point in points), mean(point[1] for point in points))
    deltas = [(x - center[0], y - center[1]) for x, y in points]
    xx = sum(x * x for x, _ in deltas)
    yy = sum(y * y for _, y in deltas)
    xy = sum(x * y for x, y in deltas)
    tangent = degrees(atan2(2 * xy, xx - yy)) / 2
    if position_on_axis(last, first, tangent)[0] < 0:
        tangent += 180.0

    normal = tangent + 90.0
    if abs(angle_error(normal, initial_heading)) > MAX_SKEW_DEG:
        raise RuntimeError("基準辺の向きが初期の進行方向から大きくずれています。")

    projected = [rotate_point(point, -normal) for point in points]
    offsets, along = zip(*projected)
    offset = mean(offsets)
    if not along[0] > along[1] > along[2] or min(dist(first, middle), dist(middle, last)) < MIN_SEGMENT_MM:
        raise RuntimeError("基準辺の①②③が順に並んでいないか、点間隔が足りません。")
    if along[0] - along[2] < MIN_BASELINE_SPAN_MM:
        raise RuntimeError("基準辺の①③間隔が足りません。")
    if max(abs(value - offset) for value in offsets) > MAX_RESIDUAL_MM:
        raise RuntimeError("基準辺の3点が同じ直線上にありません。")

    angles = [degrees(atan2(b[1] - a[1], b[0] - a[0])) for a, b in ((first, middle), (middle, last))]
    deviation = max(abs(offsets[0] - offsets[1]), abs(offsets[1] - offsets[2]))
    uncertainty = degrees(atan2(deviation, dist(first, last) / 2))
    if abs(angle_error(*angles)) > MAX_SEGMENT_ANGLE_DEG or uncertainty > MAX_BASELINE_UNCERTAINTY_DEG:
        raise RuntimeError("基準辺の2区間の向きが一致せず、角度を確定できません。")

    return Baseline(normal, tangent, offset, along[2], along[0])


def measured_bounds(baseline, opposite, first_side, second_side):
    angle = baseline.outward_deg
    low_u, opposite_v = rotate_point(opposite, -angle)
    first_u, low_v = rotate_point(first_side, -angle)
    second_u, high_v = rotate_point(second_side, -angle)
    high_u = baseline.offset_mm
    if min(high_u - low_u, high_v - low_v) < MIN_TABLE_SIDE_MM:
        raise RuntimeError("机の推定寸法が小さすぎるか、別の辺を検知しています。")

    if not (low_u < 0 < high_u and low_v < 0 < high_v):
        raise RuntimeError("推定した机の範囲に開始位置が入っていません。")
    if not (
        low_v - MAX_RESIDUAL_MM <= min(opposite_v, baseline.sample_min_mm)
        and max(opposite_v, baseline.sample_max_mm) <= high_v + MAX_RESIDUAL_MM
        and low_u - MAX_RESIDUAL_MM <= min(first_u, second_u)
        and max(first_u, second_u) <= high_u + MAX_RESIDUAL_MM
    ):
        raise RuntimeError("一部の検知点が推定した机の四辺と一致しません。")

    return rectangle_bounds(angle, low_u, high_u, low_v, high_v)

"""推定した長方形を基に、机の縁まで走れる距離を計算する。"""

from dataclasses import dataclass
from math import cos, radians, sin


SIDE_NAMES = ("u_min", "u_max", "v_min", "v_max")
FIT_TOLERANCE_MM = 1.0
STOP_AHEAD_MM = 6.0  # 状態の受信間隔と、停止するまでの移動分を見込む。


@dataclass(frozen=True)
class TableArea:
    # u、v はそれぞれ机の辺に平行な軸。単位はmm。
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    angle_deg: float

    @classmethod
    def from_map(cls, saved_map, angle):
        incomplete = "机の4辺がそろったマップが必要です。1. マッピングで記録してください。"
        if len(saved_map.segments) != 4 or saved_map.axis_u_mm is None or saved_map.axis_v_mm is None:
            raise ValueError(incomplete)

        projected = [
            (rotate_point(start, -angle), rotate_point(end, -angle))
            for start, end in saved_map.segments
        ]
        us, vs = zip(*(point for segment in projected for point in segment))
        area = cls(min(us), max(us), min(vs), max(vs), angle)
        if (
            abs(area.width - saved_map.axis_u_mm) > FIT_TOLERANCE_MM
            or abs(area.depth - saved_map.axis_v_mm) > FIT_TOLERANCE_MM
            or not area._matches_edges(projected)
        ):
            raise ValueError("マップの4辺と寸法から長方形の机を確認できません。")
        return area

    def _matches_edges(self, segments):
        for axis, offset, length in (
            (0, self.u_min, self.depth),
            (0, self.u_max, self.depth),
            (1, self.v_min, self.width),
            (1, self.v_max, self.width),
        ):
            if not any(
                abs(start[axis] - offset) <= FIT_TOLERANCE_MM
                and abs(end[axis] - offset) <= FIT_TOLERANCE_MM
                and abs(abs(end[1 - axis] - start[1 - axis]) - length) <= FIT_TOLERANCE_MM
                for start, end in segments
            ):
                return False
        return True

    @property
    def width(self):
        return self.u_max - self.u_min

    @property
    def depth(self):
        return self.v_max - self.v_min

    def inset(self, distances):
        return TableArea(
            self.u_min + distances["u_min"],
            self.u_max - distances["u_max"],
            self.v_min + distances["v_min"],
            self.v_max - distances["v_max"],
            self.angle_deg,
        )

    def center(self):
        uv = (self.u_min + self.width / 2, self.v_min + self.depth / 2)
        return rotate_point(uv, self.angle_deg)

    def clearances(self, position):
        """各辺の内側なら正、外側なら負の距離を返す。"""
        u, v = rotate_point(position, -self.angle_deg)
        return (u - self.u_min, self.u_max - u, v - self.v_min, self.v_max - v)

    def forward_room(self, position, heading):
        """今の向きに直進するとき、最初に達する辺までの距離を返す。"""
        du, dv = heading_vector(heading - self.angle_deg)
        outward_speeds = (-du, du, -dv, dv)
        distances = self.clearances(position)
        return min(
            distance / speed
            for distance, speed in zip(distances, outward_speeds)
            if speed > 1e-6
        )


def heading_vector(heading_deg):
    angle = radians(heading_deg)
    return cos(angle), sin(angle)


def rotate_point(point, angle_deg):
    x, y = point
    c, s = heading_vector(angle_deg)
    return (x * c - y * s, x * s + y * c)

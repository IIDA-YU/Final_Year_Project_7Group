"""実走行用の地図と、段差を検知した辺ごとの旋回位置を管理する。"""

from math import atan2, cos, degrees, dist, radians, sin
from threading import Lock

from algorithm.mapping.drive_limits import SIDE_NAMES, STOP_AHEAD_MM, TableArea
from algorithm.mapping.table_bounds import TableBounds
from data.roaming_map_files import (
    SENSOR_FORWARD_MM,
    TRAIL_INTERVAL_MM,
    number,
    parse_map,
    read_map_json,
    write_map_json,
)


TURN_MARGIN_MM = 15.0
CORRECTION_MM = 12.0
CORNER_BAND_MM = 12.0
INSET_FIELD = "drive_test_insets_mm"


class DriveMap:
    def __init__(self, path):
        self.path = path
        data = read_map_json(path)
        saved = parse_map(data)
        self.start_position = saved.trail[0]
        self.start_heading = _start_heading(data, saved.trail)

        if len(saved.segments) != 4:
            raise ValueError("机の4辺がそろったマップが必要です。1. マッピングで記録してください。")
        angle = number(data["table_bounds"].get("angle_deg"), "机の角度")
        self.table = TableArea.from_map(saved, angle)
        self._bounds = TableBounds(tuple(saved.segments), saved.axis_u_mm, saved.axis_v_mm, angle)

        self._sensor_forward = number(data.get("sensor_forward_mm", SENSOR_FORWARD_MM), "センサーの位置")
        if self._sensor_forward <= 0:
            raise ValueError("センサーの位置は正の数で指定してください。")
        self._margin = self._sensor_forward + TURN_MARGIN_MM
        self._original_data = data.copy()
        self._insets = _read_insets(self._original_data.pop(INSET_FIELD, {}))
        self.area = self._drive_area(self._insets)
        if min(self.area.width, self.area.depth) <= 4 * STOP_AHEAD_MM:
            raise ValueError("安全に走行・旋回できる範囲が足りません。")
        if min(self.area.clearances(self.start_position)) <= STOP_AHEAD_MM:
            raise ValueError("開始位置が旋回範囲の外側か縁付近です。机の内側からマッピングしてください。")

        self._lock = Lock()
        self._edges = saved.edges
        self._trail = []
        self._pose = None
        self._paused = False

    def _drive_area(self, insets):
        return self.table.inset({side: self._margin + insets[side] for side in SIDE_NAMES})

    def learn_cliff(self, position, heading):
        """検知した辺と、角なら隣の辺も内側へ寄せて同じJSONへ保存する。"""
        latest = read_map_json(self.path)
        insets = _read_insets(latest.pop(INSET_FIELD, {}))
        if latest != self._original_data:
            raise RuntimeError("選んだマップが走行中に変更されたため、補正を保存できません。")

        distances = self.table.clearances(position)
        nearest = min(distances)
        for side, distance in zip(SIDE_NAMES, distances):
            insets[side] = max(insets[side], self._insets[side])
            if distance <= nearest + CORNER_BAND_MM:
                insets[side] = round(max(
                    insets[side] + CORRECTION_MM,
                    distance + CORRECTION_MM - self._margin,
                ), 3)

        latest[INSET_FIELD] = insets
        write_map_json(self.path, latest)
        self._insets = insets
        self.area = self._drive_area(insets)

        angle = radians(heading)
        x, y = position
        edge = (x + self._sensor_forward * cos(angle), y + self._sensor_forward * sin(angle))
        with self._lock:
            self._edges.append(edge)

    def record_pose(self, position, heading):
        with self._lock:
            self._pose = (*position, radians(heading))
            if not self._trail or dist(position, self._trail[-1]) >= TRAIL_INTERVAL_MM:
                self._trail.append(position)

    def pause(self):
        with self._lock:
            self._paused = True

    def snapshot(self):
        with self._lock:
            trail = self._trail.copy()
            if self._pose is not None and trail[-1] != self._pose[:2]:
                trail.append(self._pose[:2])
            return trail, self._edges.copy(), self._pose, self._paused

    def table_bounds(self):
        return self._bounds


def _start_heading(data, trail):
    if "start_heading_deg" in data:
        return number(data["start_heading_deg"], "開始時の方位")

    # 以前の記録では、探索の最初の直進から方位を求める。
    start_x, start_y = trail[0]
    for point in trail[1:]:
        if dist(trail[0], point) >= TRAIL_INTERVAL_MM:
            return degrees(atan2(point[1] - start_y, point[0] - start_x))
    raise ValueError("開始時の方位が分かりません。1. マッピングで記録してください。")


def _read_insets(raw):
    if not isinstance(raw, dict):
        raise ValueError("旋回位置の補正は辺ごとに指定してください。")
    insets = {side: number(raw.get(side, 0.0), "旋回位置の補正") for side in SIDE_NAMES}
    if min(insets.values()) < 0:
        raise ValueError("旋回位置の補正に負の値があります。")
    return insets

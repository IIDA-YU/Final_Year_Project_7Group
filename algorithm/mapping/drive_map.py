"""実走行用に学習マップを読み込み、画面に表示する走行軌跡を記録する。"""

from math import atan2, degrees, dist, radians
from threading import Lock

from algorithm.mapping.drive_limits import TableArea, heading_vector
from algorithm.mapping.table_bounds import TableBounds
from data.roaming_map_files import (
    SENSOR_FORWARD_MM,
    TRAIL_INTERVAL_MM,
    number,
    parse_map,
    read_map_json,
)


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

        self.sensor_forward_mm = number(data.get("sensor_forward_mm", SENSOR_FORWARD_MM), "センサーの位置")
        if self.sensor_forward_mm <= 0:
            raise ValueError("センサーの位置は正の数で指定してください。")

        self._lock = Lock()
        self._edges = saved.edges
        self._trail = []
        self._pose = None
        self._paused = False

    def sensor_position(self, position, heading):
        x, y = position
        dx, dy = heading_vector(heading)
        return (x + self.sensor_forward_mm * dx, y + self.sensor_forward_mm * dy)

    def record_cliff(self, position, heading):
        """実走行中の検知点をマップ画面に表示する。"""
        edge = self.sensor_position(position, heading)
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

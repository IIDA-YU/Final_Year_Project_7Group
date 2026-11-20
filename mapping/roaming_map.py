"""Cozmoの走行軌跡と、段差を検知した位置を記録する。"""

from dataclasses import asdict
from datetime import datetime
import json
from math import cos, hypot, isfinite, sin
from pathlib import Path
from threading import Lock

from pycozmo.event import EvtRobotStateUpdated
from pycozmo.robot import RobotStatusFlag

from hardware.step_detection import UNSAFE_STATUS
from mapping.table_bounds import EdgeHit, TableBounds, estimate_table


TRAIL_INTERVAL_MM = 8.0
EDGE_MERGE_MM = 25.0
SENSOR_FORWARD_MM = 35.0  # 機体座標の原点から前方のセンサーまでの概算値。
MAP_DIRECTORY = Path("data") / "roaming_maps"


class RoamingMap:
    def __init__(self, robot):
        self.robot = robot
        self._lock = Lock()
        self._origin = None
        self._frame = None
        self._trail = []
        self._edges = []
        self._edge_hits = []
        self._bounds = TableBounds()
        self._edge_revision = 0
        self._fitted_revision = 0
        self._pose = None
        self._paused = False
        self._cliff_detected = False
        self._active = False

    def __enter__(self):
        self._active = True
        self._handler = self.robot.add_handler(EvtRobotStateUpdated, self._record)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self._lock:
            self._active = False
        self.robot.del_handler(EvtRobotStateUpdated, self._handler)
        self.save_json()

    def _record(self, client):
        pose = client.pose
        status = client.robot_status
        x = pose.position.x
        y = pose.position.y
        heading = pose.rotation.angle_z.radians
        frame = (pose.origin_id, client.pose_frame_id)

        with self._lock:
            if not self._active or self._paused:
                return
            if status & UNSAFE_STATUS or not pose.is_valid:
                self._paused = self._origin is not None
                return
            if not all(isfinite(value) for value in (x, y, heading)):
                return

            if self._origin is None:
                self._origin = (x, y)
                self._frame = frame
            elif frame != self._frame:
                # 座標の基準が変わったら、最後の有効な地図を保持する。
                self._paused = True
                return

            x -= self._origin[0]
            y -= self._origin[1]
            self._pose = (x, y, heading)
            point = (x, y)
            if not self._trail or _distance(point, self._trail[-1]) >= TRAIL_INTERVAL_MM:
                self._trail.append(point)

            cliff_detected = bool(status & RobotStatusFlag.CLIFF_DETECTED)
            if cliff_detected and not self._cliff_detected:
                edge = (
                    x + SENSOR_FORWARD_MM * cos(heading),
                    y + SENSOR_FORWARD_MM * sin(heading),
                )
                self._record_edge(edge, point)
            self._cliff_detected = cliff_detected

    def _record_edge(self, edge, position):
        for index, known in enumerate(self._edges):
            if _distance(edge, known) >= EDGE_MERGE_MM:
                continue

            # 近い場所を同じ方向から再測定した場合は、点の数を増やさず平均する。
            hit = self._edge_hits[index]
            dot = sum(
                (edge[axis] - position[axis])
                * (hit.point[axis] - hit.robot_position[axis])
                for axis in (0, 1)
            )
            if dot >= 0.5 * SENSOR_FORWARD_MM ** 2:
                count = min(hit.sample_count + 1, 8)
                self._edge_hits[index] = EdgeHit(
                    _average_point(hit.point, edge, count),
                    _average_point(hit.robot_position, position, count),
                    count,
                )
                self._edge_revision += 1
            return

        self._edges.append(edge)
        self._edge_hits.append(EdgeHit(edge, position))
        self._edge_revision += 1

    def snapshot(self):
        """描画中に更新されないよう、記録のコピーを返す。"""
        with self._lock:
            trail = list(self._trail)
            if self._pose is not None and trail[-1] != self._pose[:2]:
                trail.append(self._pose[:2])
            return trail, list(self._edges), self._pose, self._paused

    def table_bounds(self):
        # 描画側で計算し、通信スレッドの受信処理を待たせない。
        with self._lock:
            revision = self._edge_revision
            if revision == self._fitted_revision:
                return self._bounds
            hits = tuple(self._edge_hits)

        bounds = estimate_table(hits)
        with self._lock:
            if revision > self._fitted_revision:
                self._bounds = bounds
                self._fitted_revision = revision
            return self._bounds

    def save_json(self):
        trail, edges, _, _ = self.snapshot()
        if not trail:
            return

        MAP_DIRECTORY.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = MAP_DIRECTORY / f"roaming_map_{timestamp}.json"
        temporary = path.with_suffix(".tmp")
        data = {
            "sensor_forward_mm": SENSOR_FORWARD_MM,
            "trail_mm": trail,
            "detected_edges_mm": edges,
            "table_bounds": asdict(self.table_bounds()),
        }
        try:
            temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"走行マップを保存しました。{path}")


def _distance(point_a, point_b):
    return hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _average_point(previous, current, count):
    return (
        previous[0] + (current[0] - previous[0]) / count,
        previous[1] + (current[1] - previous[1]) / count,
    )
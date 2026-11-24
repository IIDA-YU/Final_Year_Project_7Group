"""走行マップのファイル一覧・読み込み・保存を扱う。"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional


MAP_DIRECTORY = Path(__file__).resolve().parent / "roaming_maps"
MAP_FILE_PREFIX = "roaming_map_"
SENSOR_FORWARD_MM = 35.0  # 機体座標の原点から前方のセンサーまでの概算値。
TRAIL_INTERVAL_MM = 8.0

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class MapRecord:
    path: Path
    recorded_at: datetime
    uses_modified_time: bool = False

    @property
    def date_label(self):
        label = self.recorded_at.strftime("%Y/%m/%d %H:%M:%S")
        if self.uses_modified_time:
            label += "（更新日時）"
        return label


@dataclass(frozen=True)
class SavedMap:
    trail: list[Point]
    edges: list[Point]
    segments: list[Segment]
    axis_u_mm: Optional[float]
    axis_v_mm: Optional[float]


def list_records(directory):
    """記録日時が新しい順にJSONファイルを並べる。"""
    records = []
    for path in directory.glob("*.json"):
        if not path.is_file():
            continue
        timestamp = path.stem.removeprefix(MAP_FILE_PREFIX)
        date_format = "%Y%m%d_%H%M%S"
        if timestamp.count("_") == 2:
            date_format += "_%f"
        try:
            record = MapRecord(path, datetime.strptime(timestamp, date_format))
        except ValueError:
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime)
            except FileNotFoundError:
                continue
            record = MapRecord(path, modified_at, uses_modified_time=True)
        records.append(record)
    return sorted(records, key=lambda record: (record.recorded_at, record.path.name), reverse=True)


def read_map_json(path):
    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("走行マップのJSON形式ではありません。")
    return data


def write_map_json(path, data):
    """同じフォルダーに書き終えてから、元のファイルを置き換える。"""
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_roaming_map(directory, trail, edges, bounds, *, start_heading, sensor_forward):
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"{MAP_FILE_PREFIX}{timestamp}.json"
    write_map_json(path, {
        "sensor_forward_mm": sensor_forward,
        "start_heading_deg": start_heading,
        "trail_mm": trail,
        "detected_edges_mm": edges,
        "table_bounds": asdict(bounds),
    })
    print(f"走行マップを保存しました。{path}")
    return path


def number(value, field):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{field} に無効な数値があります。")
    return float(value)


def _points(values, field):
    if not isinstance(values, list):
        raise ValueError(f"{field} は座標の一覧である必要があります。")
    points = []
    for point in values:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{field} に X・Y 座標の組が必要です。")
        x, y = point
        points.append((number(x, field), number(y, field)))
    return points


def parse_map(data):
    trail = _points(data.get("trail_mm"), "走行軌跡")
    if not trail:
        raise ValueError("走行軌跡が記録されていません。")
    edges = _points(data.get("detected_edges_mm", []), "段差検知点")
    bounds = data.get("table_bounds")
    if bounds is None:
        bounds = {}
    elif not isinstance(bounds, dict):
        raise ValueError("机の推定結果の形式が正しくありません。")

    stored_segments = bounds.get("segments_mm", [])
    if not isinstance(stored_segments, list):
        raise ValueError("机の辺は線分の一覧である必要があります。")
    segments = []
    for segment in stored_segments:
        points = _points(segment, "机の辺")
        if len(points) != 2:
            raise ValueError("机の各辺に両端の座標が必要です。")
        segments.append((points[0], points[1]))

    axis_u = bounds.get("axis_u_mm")
    axis_v = bounds.get("axis_v_mm")
    if axis_u is not None:
        axis_u = number(axis_u, "机の寸法")
    if axis_v is not None:
        axis_v = number(axis_v, "机の寸法")
    return SavedMap(trail, edges, segments, axis_u, axis_v)


def load_map(path):
    return parse_map(read_map_json(path))

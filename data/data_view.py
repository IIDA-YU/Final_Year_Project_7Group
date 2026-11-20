"""Cozmoの保存済み走行マップを、記録日時を選んで表示する。"""

from dataclasses import dataclass
from datetime import datetime
import json
from math import isfinite
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional


MAP_DIRECTORY = Path(__file__).resolve().parent / "roaming_maps"
MAP_FILE_PREFIX = "roaming_map_"
GRID_INTERVAL_MM = 100

BACKGROUND = "#121719"
GRID_COLOR = "#293237"
TRAIL_COLOR = "#45d7e1"
EDGE_COLOR = "#ff7562"
BOUNDARY_COLOR = "#f5d241"
START_COLOR = "#e0e0e0"
END_COLOR = "#9bf682"

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
    """ファイル名の保存日時で並べる。作成後にコピーしても順番は変わらない。"""
    records = []
    for path in directory.glob(f"{MAP_FILE_PREFIX}*.json"):
        if not path.is_file():
            continue
        timestamp = path.stem[len(MAP_FILE_PREFIX):]
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


def _number(value, field):
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
        points.append((_number(x, field), _number(y, field)))
    return points


def load_map(path):
    """保存された座標と推定結果を読み込む。"""
    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("走行マップのJSON形式ではありません。")

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
        axis_u = _number(axis_u, "机の寸法")
    if axis_v is not None:
        axis_v = _number(axis_v, "机の寸法")
    return SavedMap(trail, edges, segments, axis_u, axis_v)


def _size_label(saved_map):
    edge_count = f"推定した辺 {len(saved_map.segments)}/4 本"
    if saved_map.axis_u_mm is not None and saved_map.axis_v_mm is not None:
        long_side, short_side = sorted((saved_map.axis_u_mm, saved_map.axis_v_mm), reverse=True)
        return f"推定寸法 {long_side / 10:.1f} × {short_side / 10:.1f} cm　｜　{edge_count}"
    for size in (saved_map.axis_u_mm, saved_map.axis_v_mm):
        if size is not None:
            return f"推定幅 {size / 10:.1f} cm　｜　{edge_count}"
    return edge_count


def draw_map(canvas, saved_map, width, height):
    """X・Yの縮尺をそろえ、ウィンドウのサイズに合わせて描く。"""
    canvas.delete("all")
    points = saved_map.trail + saved_map.edges
    for start, end in saved_map.segments:
        points.extend((start, end))
    x_values, y_values = zip(*points)
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    span_x = max(300.0, max_x - min_x + 80.0)
    span_y = max(300.0, max_y - min_y + 80.0)
    scale = min(max(width - 32, 1) / span_x, max(height - 32, 1) / span_y)

    def to_pixel(point):
        x, y = point
        # 地図は上が＋Y、キャンバスは下が＋Y。
        return (width / 2 + (x - center_x) * scale, height / 2 - (y - center_y) * scale)

    spacing = round(GRID_INTERVAL_MM * scale)
    if spacing >= 20:
        origin_x, origin_y = to_pixel((0, 0))
        for x in range(round(origin_x) % spacing, width, spacing):
            canvas.create_line(x, 0, x, height, fill=GRID_COLOR)
        for y in range(round(origin_y) % spacing, height, spacing):
            canvas.create_line(0, y, width, y, fill=GRID_COLOR)

    trail_pixels = [to_pixel(point) for point in saved_map.trail]
    if len(trail_pixels) >= 2:
        coordinates = [coordinate for point in trail_pixels for coordinate in point]
        canvas.create_line(*coordinates, fill=TRAIL_COLOR, width=2)
    for start, end in saved_map.segments:
        canvas.create_line(*to_pixel(start), *to_pixel(end), fill=BOUNDARY_COLOR, width=2)
    for edge in saved_map.edges:
        x, y = to_pixel(edge)
        canvas.create_line(x - 5, y - 5, x + 5, y + 5, fill=EDGE_COLOR, width=2)
        canvas.create_line(x - 5, y + 5, x + 5, y - 5, fill=EDGE_COLOR, width=2)

    start_x, start_y = trail_pixels[0]
    canvas.create_oval(start_x - 7, start_y - 7, start_x + 7, start_y + 7, outline=START_COLOR, width=2)
    end_x, end_y = trail_pixels[-1]
    canvas.create_oval(end_x - 4, end_y - 4, end_x + 4, end_y + 4, fill=END_COLOR, outline="")


def show_map(parent, record, saved_map):
    window = tk.Toplevel(parent)
    window.title(f"Cozmo 保存マップ — {record.date_label}")
    window.geometry("640x560")
    window.minsize(480, 360)

    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=record.date_label).pack(anchor="w")
    ttk.Label(frame, text=record.path.name).pack(anchor="w", pady=(2, 10))

    canvas = tk.Canvas(frame, background=BACKGROUND, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    def redraw(event):
        draw_map(canvas, saved_map, event.width, event.height)

    canvas.bind("<Configure>", redraw)
    ttk.Label(frame, text=_size_label(saved_map)).pack(anchor="w", pady=(10, 4))
    legend = "水色 走行軌跡　赤× 段差検知　黄線 推定した辺　白○ 開始　緑● 最終記録"
    ttk.Label(frame, text=legend).pack(anchor="w")
    window.bind("<Escape>", lambda event: window.destroy())
    return window


def main():
    root = tk.Tk()
    root.title("Cozmo 走行マップの記録")
    root.geometry("680x320")
    root.minsize(520, 240)
    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)

    try:
        records = list_records(MAP_DIRECTORY)
    except OSError as error:
        ttk.Label(frame, text=f"記録を一覧にできませんでした。\n{error}").pack(anchor="w")
        root.mainloop()
        return
    if not records:
        ttk.Label(frame, text="保存済みの走行マップがありません。").pack(anchor="w")
        ttk.Label(frame, text="保存先  data/roaming_maps").pack(anchor="w", pady=(6, 0))
        root.mainloop()
        return

    ttk.Label(frame, text="表示する記録を選んでください。").pack(anchor="w", pady=(0, 8))
    list_frame = ttk.Frame(frame)
    list_frame.pack(fill="both", expand=True)
    table = ttk.Treeview(list_frame, columns=("date", "file"), show="headings", selectmode="browse")
    table.heading("date", text="記録日時")
    table.heading("file", text="ファイル名")
    table.column("date", width=200, stretch=False)
    table.column("file", width=395)
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=table.yview)
    table.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    table.pack(side="left", fill="both", expand=True)

    for index, record in enumerate(records):
        table.insert("", "end", iid=str(index), values=(record.date_label, record.path.name))

    def open_selected(event=None):
        selected = table.selection()
        if event is not None and event.type == tk.EventType.ButtonPress:
            row = table.identify_row(event.y)
            selected = (row,) if row else ()
        if not selected:
            return
        record = records[int(selected[0])]
        try:
            saved_map = load_map(record.path)
        except (OSError, ValueError, OverflowError) as error:
            messagebox.showerror("マップの読み込みエラー", f"{record.path.name}\n\n{error}", parent=root)
            return
        show_map(root, record, saved_map)

    ttk.Button(frame, text="マップを開く", command=open_selected).pack(anchor="e", pady=(10, 0))
    table.bind("<Return>", open_selected)
    table.bind("<Double-1>", open_selected)
    table.selection_set("0")
    table.focus("0")
    table.focus_set()
    if len(records) == 1:
        root.after_idle(open_selected)
    root.mainloop()


if __name__ == "__main__":
    main()

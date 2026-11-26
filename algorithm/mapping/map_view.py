"""走行マップをOpenCVのウィンドウに描画する。"""

from math import cos, sin
from time import monotonic

import cv2
import numpy as np


WINDOW_NAME = "Cozmo Map"
WIDTH, HEIGHT = 420, 480
PLOT_TOP, PLOT_BOTTOM = 42, 402
REFRESH_SECONDS = 0.1
RECENT_POINTS = 35

BACKGROUND = (25, 23, 18)
GRID = (47, 42, 33)
TRAIL = (120, 116, 52)
RECENT_TRAIL = (225, 215, 60)
EDGE = (75, 100, 255)
BOUNDARY = (65, 210, 245)
ROBOT = (130, 246, 155)
PAWN = (75, 175, 255)
TEXT = (222, 224, 224)


class MapWindow:
    def __init__(self, map_data, camera_window_name, camera_width):
        self.map_data = map_data
        self.pawn_positions = getattr(map_data, "pawn_positions", None)
        self.last_refresh = None
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(camera_window_name, 24, 60)
        cv2.moveWindow(WINDOW_NAME, 24 + camera_width + 28, 60)
        self.refresh()

    def is_open(self):
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def refresh(self):
        now = monotonic()
        if self.last_refresh is not None and now - self.last_refresh < REFRESH_SECONDS:
            return
        trail, edges, pose, paused = self.map_data.snapshot()
        bounds = self.map_data.table_bounds()
        pawns = None if self.pawn_positions is None else self.pawn_positions()
        cv2.imshow(WINDOW_NAME, draw_map(trail, edges, pose, paused, bounds, pawns=pawns))
        self.last_refresh = now


def draw_map(trail, edges, pose, paused, bounds=None, *, pawns=None):
    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    segments = () if bounds is None else bounds.segments_mm
    endpoints = [point for segment in segments for point in segment]
    points = trail + edges + endpoints or [(0.0, 0.0)]
    xs, ys = zip(*points)
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    span_x = max(300.0, max(xs) - min(xs) + 80.0)
    span_y = max(300.0, max(ys) - min(ys) + 80.0)
    scale = min((WIDTH - 36) / span_x, (PLOT_BOTTOM - PLOT_TOP - 24) / span_y)

    def to_pixel(point):
        x, y = point
        # 地図のY軸は上向き、画像のY軸は下向き。
        return (
            round(WIDTH / 2 + (x - center_x) * scale),
            round((PLOT_TOP + PLOT_BOTTOM) / 2 - (y - center_y) * scale),
        )

    _draw_grid(image, to_pixel((0.0, 0.0)), round(100 * scale))

    pixels = [to_pixel(point) for point in trail]
    _draw_trail(image, pixels, TRAIL, 2)
    _draw_trail(image, pixels[-RECENT_POINTS:], RECENT_TRAIL, 3)
    if pixels:
        cv2.circle(image, pixels[0], 5, TEXT, 1, cv2.LINE_AA)

    for start, end in segments:
        cv2.line(image, to_pixel(start), to_pixel(end), BOUNDARY, 2, cv2.LINE_AA)

    for edge in edges:
        cv2.drawMarker(image, to_pixel(edge), EDGE, cv2.MARKER_TILTED_CROSS, 14, 2)

    if pose is not None:
        x, y, heading = pose
        start_x, start_y = to_pixel((x, y))
        arrow_end = (start_x + round(17 * cos(heading)), start_y - round(17 * sin(heading)))
        cv2.circle(image, (start_x, start_y), 5, ROBOT, -1, cv2.LINE_AA)
        cv2.arrowedLine(image, (start_x, start_y), arrow_end, ROBOT, 2, cv2.LINE_AA)

    for pawn in pawns or ():
        _draw_pawn(image, to_pixel(pawn), pawn)

    title = "Map paused" if paused else "Cozmo Map"
    _text(image, title, (16, 28), TEXT, 0.65)
    if pose is None:
        _text(image, "Waiting for robot position...", (80, 234), TEXT)
    _text(image, _size_label(bounds), (16, 425), TEXT, 0.52)
    _text(image, "Trail", (16, 449), RECENT_TRAIL)
    _text(image, "X  Edge hit", (80, 449), EDGE)
    _text(image, "Fitted edge", (181, 449), BOUNDARY)
    _text(image, "Cozmo", (303, 449), ROBOT)
    _text(image, "Circle: start point", (16, 469), TEXT)
    if pawns is not None:
        _text(image, "Pawn (x, y) cm", (181, 469), PAWN)
    return image


def _draw_pawn(image, pixel, position):
    x, y = pixel
    # 駒の台座の下にある点が、検知したマップ座標。
    body = np.array(
        [(x - 8, y - 5), (x - 8, y - 9), (x - 4, y - 10), (x - 2, y - 16),
         (x + 2, y - 16), (x + 4, y - 10), (x + 8, y - 9), (x + 8, y - 5)],
        dtype=np.int32,
    )
    cv2.polylines(image, [body], True, BACKGROUND, 3, cv2.LINE_AA)
    cv2.fillPoly(image, [body], PAWN, cv2.LINE_AA)
    cv2.circle(image, (x, y - 19), 6, BACKGROUND, -1, cv2.LINE_AA)
    cv2.circle(image, (x, y - 19), 4, PAWN, -1, cv2.LINE_AA)
    cv2.circle(image, pixel, 4, BACKGROUND, -1, cv2.LINE_AA)
    cv2.circle(image, pixel, 2, PAWN, -1, cv2.LINE_AA)
    label = f"({position[0] / 10:.1f}, {position[1] / 10:.1f})"
    (label_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    label_x = x + 11 if x + 11 + label_width < WIDTH - 5 else x - 11 - label_width
    cv2.putText(image, label, (label_x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                BACKGROUND, 3, cv2.LINE_AA)
    _text(image, label, (label_x, y - 6), PAWN, 0.38)


def _size_label(bounds):
    if bounds is None:
        return "Fitted edges: 0/4"
    sizes = [size for size in (bounds.axis_u_mm, bounds.axis_v_mm) if size is not None]
    if len(sizes) == 2:
        long_side, short_side = sorted(sizes, reverse=True)
        return f"Table estimate: {long_side / 10:.1f} x {short_side / 10:.1f} cm"
    count = len(bounds.segments_mm)
    if sizes:
        return f"Span estimate: {sizes[0] / 10:.1f} cm  ({count}/4 edges)"
    return f"Fitted edges: {count}/4"


def _draw_grid(image, origin, spacing):
    # 10 cmごとのマス。縮小して密集する場合は省く。
    if spacing < 20:
        return
    origin_x, origin_y = origin
    for x in range(origin_x % spacing, WIDTH, spacing):
        cv2.line(image, (x, PLOT_TOP), (x, PLOT_BOTTOM), GRID)
    first_y = PLOT_TOP + (origin_y - PLOT_TOP) % spacing
    for y in range(first_y, PLOT_BOTTOM + 1, spacing):
        cv2.line(image, (0, y), (WIDTH - 1, y), GRID)


def _draw_trail(image, pixels, color, thickness):
    if len(pixels) >= 2:
        line = np.array(pixels, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [line], False, color, thickness, cv2.LINE_AA)


def _text(image, text, position, color, size=0.47):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

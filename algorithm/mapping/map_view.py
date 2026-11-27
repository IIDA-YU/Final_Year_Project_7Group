"""走行マップをOpenCVのウィンドウに描画する。"""

from math import cos, sin
from time import monotonic

import cv2
import numpy as np


WINDOW_NAME = "Cozmo Map"
WIDTH, HEIGHT = 420, 550
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
TEXT = (222, 224, 224)
CUBE_BUTTONS = {
    "Start": ((16, 488, 196, 521), (255, 190, 110)),
    "Goal": ((212, 488, 392, 521), (200, 130, 255)),
}


class MapWindow:
    def __init__(self, map_data, camera_window_name, camera_width):
        self.map_data = map_data
        self.last_refresh = None
        self.cube_points = {}
        self.pending_cube = None
        self.projection = None
        self.click_error = None
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._click)
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
        cube_points, status = (
            self.map_data.cube_display() if hasattr(self.map_data, "cube_display")
            else (self.cube_points, None)
        )
        image, projection = _render_map(
            trail, edges, pose, paused, bounds, cube_points, self.pending_cube,
            self.click_error or status,
        )
        cv2.imshow(WINDOW_NAME, image)
        self.projection = projection
        self.last_refresh = now

    def _click(self, event, x, y, flags, user_data):
        if event != cv2.EVENT_LBUTTONUP:
            return
        for name, ((left, top, right, bottom), _) in CUBE_BUTTONS.items():
            if left <= x <= right and top <= y <= bottom:
                self.click_error = None
                self.pending_cube = None if self.pending_cube == name else name
                self.last_refresh = None
                return
        if self.pending_cube and self.projection and 0 <= x < WIDTH and PLOT_TOP <= y <= PLOT_BOTTOM:
            origin_x, origin_y, scale = self.projection
            point = ((x - origin_x) / scale, (origin_y - y) / scale)
            if hasattr(self.map_data, "set_cube_point"):
                self.click_error = self.map_data.set_cube_point(self.pending_cube, point)
            else:
                self.cube_points[self.pending_cube] = point
            self.pending_cube = None
            self.last_refresh = None


def draw_map(trail, edges, pose, paused, bounds=None):
    image, _ = _render_map(trail, edges, pose, paused, bounds, {}, None)
    return image


def _render_map(trail, edges, pose, paused, bounds, cube_points, pending_cube, status=None):
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
    origin_x = WIDTH / 2 - center_x * scale
    origin_y = (PLOT_TOP + PLOT_BOTTOM) / 2 + center_y * scale

    def to_pixel(point):
        x, y = point
        # 地図のY軸は上向き、画像のY軸は下向き。
        return round(origin_x + x * scale), round(origin_y - y * scale)

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

    title = "Map paused" if paused else "Cozmo Map"
    _text(image, title, (16, 28), TEXT, 0.65)
    if pose is None:
        _text(image, "Waiting for robot position...", (80, 234), TEXT)
    _text(image, _size_label(bounds), (16, 425), TEXT, 0.52)
    _text(image, "Trail", (16, 449), RECENT_TRAIL)
    _text(image, "X  Edge hit", (80, 449), EDGE)
    _text(image, "Fitted edge", (181, 449), BOUNDARY)
    _text(image, "Cozmo", (303, 449), ROBOT)
    _text(image, "Circle: start of trail", (16, 469), TEXT)

    plot = image[PLOT_TOP:PLOT_BOTTOM + 1]
    for name, ((left, top, right, bottom), color) in CUBE_BUTTONS.items():
        if name in cube_points:
            x, y = to_pixel(cube_points[name])
            if 0 <= x < WIDTH and PLOT_TOP <= y <= PLOT_BOTTOM:
                marker_y = y - PLOT_TOP
                cv2.circle(plot, (x, marker_y), 10, color, cv2.FILLED, cv2.LINE_AA)
                _text(plot, name[0], (x - 4, marker_y + 5), BACKGROUND)

        active = name == pending_cube
        cv2.rectangle(image, (left, top), (right, bottom), color, cv2.FILLED if active else 1)
        label = f"{name} cube" + ("  [set]" if name in cube_points else "")
        _text(image, label, (left + 10, top + 22), BACKGROUND if active else color)

    hint = f"Click the map to place the {pending_cube.lower()} marker" if pending_cube else (
        status or "Select a cube, then click the map"
    )
    _text(image, hint, (16, 541), TEXT)
    return image, (origin_x, origin_y, scale)


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

"""ポーンの検知枠を、画像受信時のCozmoの座標へ投影する。"""

from collections import deque
from dataclasses import dataclass
from math import cos, isfinite, radians, sin
from threading import Lock

from algorithm.mapping.drive_limits import rotate_point


PAWN_HEIGHT_MM = 28.0
PAWN_DIAMETER_MM = 26.0
PAWN_RADIUS_MM = PAWN_DIAMETER_MM / 2
PAWN_VIEW_HEAD_ANGLE_RAD = radians(-12)
MAX_FRAME_AGE_SECONDS = 1.0

# QVGAでの焦点距離と光学中心。機体の校正値を取得できる場合はそちらを使う。
DEFAULT_INTRINSICS = (288.0, 288.0, 160.0, 120.0)
# 機体原点からヘッド軸、ヘッド軸からカメラへの前方・上方オフセット。
NECK_FORWARD_MM = -13.0
CAMERA_FORWARD_MM = 17.52
CAMERA_UP_MM = -8.0
CAMERA_TILT_RAD = radians(-4)


@dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float]
    heading_deg: float
    head_angle_rad: float
    pose_frame: tuple[int, int]


@dataclass(frozen=True)
class PawnFrame:
    received_at: float
    pose_frame: tuple[int, int]
    positions: tuple[tuple[float, float], ...]


class PawnCamera:
    """検知がなかったフレームも、到着順に走行スレッドへ渡す。"""

    def __init__(self):
        self._lock = Lock()
        self._enabled = False
        self._frames = deque(maxlen=64)

    def set_enabled(self, enabled):
        with self._lock:
            self._enabled = enabled
            self._frames.clear()

    def read_frames(self):
        with self._lock:
            frames = tuple(self._frames)
            self._frames.clear()
            return self._enabled, frames

    def publish(self, received_at, pose, boxes, image_size, intrinsics):
        if pose is None:
            return
        positions = []
        for box in boxes:
            offset = _pawn_offset(box, image_size, intrinsics, pose.head_angle_rad)
            if offset is not None:
                x, y = rotate_point(offset, pose.heading_deg)
                positions.append((pose.position[0] + x, pose.position[1] + y))

        with self._lock:
            if self._enabled:
                self._frames.append(PawnFrame(received_at, pose.pose_frame, tuple(positions)))


def capture_camera_pose(robot):
    pose = robot.pose
    x, y = pose.position.x, pose.position.y
    heading = pose.rotation.angle_z.degrees
    head_angle = robot.head_angle.radians
    if not pose.is_valid or not all(isfinite(value) for value in (x, y, heading, head_angle)):
        return None
    return CameraPose((x, y), heading, head_angle, (pose.origin_id, robot.pose_frame_id))


def camera_intrinsics(robot):
    config = getattr(robot, "camera_config", None)
    if config is None:
        return DEFAULT_INTRINSICS
    values = (config.focal_length.x, config.focal_length.y, config.center.x, config.center.y)
    if all(isfinite(value) and value > 0 for value in values):
        return values
    return DEFAULT_INTRINSICS


def _pawn_offset(box, image_size, intrinsics, head_angle):
    width, height = image_size
    x_scale, y_scale = width / 320, height / 240
    fx, fy, cx, cy = intrinsics
    fx, fy, cx, cy = fx * x_scale, fy * y_scale, cx * x_scale, cy * y_scale
    x1, y1, x2, y2 = box
    if not all(isfinite(value) for value in box) or x2 - x1 < 1 or y2 - y1 < 1:
        return None

    pitch = head_angle + CAMERA_TILT_RAD
    cos_pitch, sin_pitch = cos(pitch), sin(pitch)
    # 左右や上下に見切れている場合は、画像に収まった方の寸法を使う。
    full_width = x1 > 2 * x_scale and x2 < width - 2 * x_scale
    full_height = y1 > 2 * y_scale and y2 < height - 2 * y_scale
    depths = []
    if full_width:
        depths.append(fx * PAWN_DIAMETER_MM / (x2 - x1))
    if full_height:
        middle_y = ((y1 + y2) / 2 - cy) / fy
        depths.append(fy * PAWN_HEIGHT_MM * (cos_pitch + middle_y * sin_pitch) / (y2 - y1))
    if not depths:
        return None
    depth = sum(depths) / len(depths)
    if depth <= 0:
        return None

    # 見切れている場合は既知の幅から駒の中心を補う。
    center_x = (x1 + x2) / 2
    half_width = fx * PAWN_RADIUS_MM / depth
    if x1 <= 2 * x_scale and x2 < width - 2 * x_scale:
        center_x = x2 - half_width
    elif x2 >= width - 2 * x_scale and x1 > 2 * x_scale:
        center_x = x1 + half_width

    camera_forward = NECK_FORWARD_MM + CAMERA_FORWARD_MM * cos(head_angle) - CAMERA_UP_MM * sin(head_angle)
    # 光軸に沿った距離を机と平行な距離へ変換する。
    middle_y = ((y1 + y2) / 2 - cy) / fy
    forward = camera_forward + depth * (cos_pitch + middle_y * sin_pitch)
    left = (cx - center_x) * depth / fx
    return forward, left

"""ArUcoマーカーを検知し、すれ違い距離内で双方向通信を行う。"""

import threading
import time

import cv2
import numpy as np
import pycozmo
from PIL import ImageDraw

from telecommunication.pc_communication import CoordCommunication


def cozmo_image_to_bgr(image):
    """PIL / NumPy 形式のCozmo画像をOpenCVのBGR画像へ変換する。"""
    image_np = np.asarray(image)
    if image_np.ndim == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    if image_np.ndim != 3:
        raise ValueError(f"未対応のカメラ画像形式です: shape={image_np.shape}")

    channels = image_np.shape[2]
    if channels == 1:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if channels == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"未対応のカメラ画像形式です: shape={image_np.shape}")


class ArucoCoordinateTracker:
    """カメラ映像からArUcoマーカーを監視し、すれ違い時に相互通信を発動する。"""

    def __init__(
        self,
        cli,
        target_ip_dict,
        comm=None,
        robot_id="Cozmo-A",
        cooldown_sec=4.0,
        get_pose_cb=None,
        marker_real_size_mm=30.0,
        focal_length_px=235.0,
        encounter_dist_max_mm=450.0,
    ):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco が利用できません。opencv-contrib-python をインストールしてください。"
            )

        self.cli = cli
        self.target_ip_dict = dict(target_ip_dict or {})
        self.robot_id = robot_id
        self.get_pose_cb = get_pose_cb
        self.comm = comm if comm is not None else CoordCommunication(robot_id=robot_id)

        self.marker_real_size_mm = float(marker_real_size_mm)
        self.focal_length_px = float(focal_length_px)
        self.encounter_dist_max_mm = float(encounter_dist_max_mm)

        CoordCommunication.register_robot_context(
            cli=self.cli,
            robot_id=self.robot_id,
            get_local_info_cb=self._get_live_pose,
        )
        if self.comm.get_local_info_cb is None:
            self.comm.get_local_info_cb = self._get_live_pose

        # ArUco検知の初期設定（OpenCV新旧両対応）
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        else:
            self.detector = None

        self.cooldown_sec = float(cooldown_sec)
        self.last_exchange_times = {}
        self.exchanging_ids = set()

        self.running = False
        self._camera_handler = None
        self._worker = None
        self._latest_raw_image = None
        self._latest_boxes = []
        self._latest_source_size = (320, 240)
        self._lock = threading.Lock()

        print(
            f"📡 [ArucoTracker] 初期化完了: 監視対象ID群 {list(self.target_ip_dict.keys())}"
        )

    def _get_live_pose(self):
        if callable(self.get_pose_cb):
            try:
                data = self.get_pose_cb()
                if isinstance(data, dict) and "x" in data and "y" in data:
                    return {
                        "id": data.get("id", self.robot_id),
                        "x": float(data["x"]),
                        "y": float(data["y"]),
                    }
            except Exception:
                pass

        if self.cli and getattr(self.cli, "pose", None) is not None:
            return {
                "id": self.robot_id,
                "x": float(self.cli.pose.position.x),
                "y": float(self.cli.pose.position.y),
            }
        return {"id": self.robot_id, "x": 0.0, "y": 0.0}

    def _estimate_marker_distance(self, pts):
        w_px = float(np.linalg.norm(pts[0] - pts[1]))
        h_px = float(np.linalg.norm(pts[1] - pts[2]))
        size_px = max(w_px, h_px, 1.0)
        return (self.marker_real_size_mm * self.focal_length_px) / size_px

    def _can_exchange_with(self, marker_id):
        now = time.time()
        if marker_id in self.exchanging_ids:
            return False
        last_t = self.last_exchange_times.get(marker_id, 0.0)
        return (now - last_t) >= self.cooldown_sec

    def _detect_markers(self, frame):
        if self.detector is not None:
            return self.detector.detectMarkers(frame)
        return cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.parameters
        )

    def _tracker_loop(self):
        while self.running:
            with self._lock:
                raw_image = self._latest_raw_image

            if raw_image is not None:
                try:
                    frame = cozmo_image_to_bgr(raw_image)
                    src_h, src_w = frame.shape[:2]
                    corners, ids, _ = self._detect_markers(frame)
                    if not self.running:
                        break

                    boxes = []
                    if ids is not None and len(ids) > 0:
                        flat_ids = np.asarray(ids).reshape(-1)

                        for i, raw_id in enumerate(flat_ids):
                            marker_id = int(raw_id)
                            pts = np.asarray(corners[i], dtype=np.float32).reshape(-1, 2)
                            if pts.shape[0] < 4:
                                continue

                            x1 = int(np.min(pts[:, 0]))
                            y1 = int(np.min(pts[:, 1]))
                            x2 = int(np.max(pts[:, 0]))
                            y2 = int(np.max(pts[:, 1]))
                            dist_mm = self._estimate_marker_distance(pts)

                            recently_synced = (
                                time.time() - self.last_exchange_times.get(marker_id, 0.0)
                                < 3.0
                            )
                            status_tag = " [SYNCED]" if recently_synced else ""
                            boxes.append(
                                {
                                    "box": (x1, y1, x2, y2),
                                    "label": f"ID:{marker_id} ({int(dist_mm)}mm){status_tag}",
                                    "synced": recently_synced,
                                }
                            )

                            if (
                                marker_id in self.target_ip_dict
                                and dist_mm <= self.encounter_dist_max_mm
                                and self._can_exchange_with(marker_id)
                            ):
                                target_addr = self.target_ip_dict[marker_id]
                                self.exchanging_ids.add(marker_id)

                                pose_info = self._get_live_pose()
                                my_x = float(pose_info["x"])
                                my_y = float(pose_info["y"])

                                print(
                                    f"🎯 すれ違い検知！Cozmo (ID: {marker_id}, 距離: 約{dist_mm:.0f}mm) "
                                    f"とリアルタイム相互通信を開始 -> {target_addr}"
                                )

                                def exchange_task(m_id, addr, mx, my):
                                    try:
                                        peer_reply = self.comm.exchange_data(
                                            target_ip=addr,
                                            x=mx,
                                            y=my,
                                            robot_id=self.robot_id,
                                        )
                                        if peer_reply:
                                            self.last_exchange_times[m_id] = time.time()
                                            print(
                                                f" ┗ ✅ [ID:{m_id}] 双方向交換完了！"
                                                f" (自機送信: X={mx:.1f}, Y={my:.1f})"
                                            )
                                        else:
                                            self.last_exchange_times[m_id] = (
                                                time.time() - self.cooldown_sec + 1.0
                                            )
                                            print(
                                                f" ┗ ❌ [ID:{m_id}] 相手({addr})からの応答なし"
                                            )
                                    finally:
                                        self.exchanging_ids.discard(m_id)

                                threading.Thread(
                                    target=exchange_task,
                                    args=(marker_id, target_addr, my_x, my_y),
                                    daemon=True,
                                ).start()

                    with self._lock:
                        self._latest_boxes = boxes
                        self._latest_source_size = (src_w, src_h)

                except Exception as e:
                    print(f"⚠️ ArUco検知エラー: {e}")

            time.sleep(0.05)

    def draw_on_image(self, image):
        """Pillow画像上に最新のArUco検知枠をスケーリングして重ねる。"""
        with self._lock:
            boxes = list(self._latest_boxes)
            src_w, src_h = self._latest_source_size

        if not boxes or src_w <= 0 or src_h <= 0:
            return image

        scale_x = image.width / src_w
        scale_y = image.height / src_h
        draw = ImageDraw.Draw(image)

        for det in boxes:
            x1, y1, x2, y2 = det["box"]
            sx1 = max(0, min(image.width - 1, int(round(x1 * scale_x))))
            sy1 = max(0, min(image.height - 1, int(round(y1 * scale_y))))
            sx2 = max(0, min(image.width - 1, int(round(x2 * scale_x))))
            sy2 = max(0, min(image.height - 1, int(round(y2 * scale_y))))
            color = (0, 230, 255) if det.get("synced") else (255, 0, 255)

            draw.rectangle((sx1, sy1, sx2, sy2), outline=color, width=3)

            label = det.get("label", "")
            text_box = draw.textbbox((0, 0), label)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            label_x = max(0, min(sx1, image.width - text_w - 8))
            label_y = max(0, sy1 - text_h - 8)

            draw.rectangle(
                (label_x, label_y, label_x + text_w + 8, label_y + text_h + 8),
                fill=color,
            )
            draw.text(
                (label_x + 4 - text_box[0], label_y + 4 - text_box[1]),
                label,
                fill="black",
            )

        return image

    def start(self):
        if self.running:
            return

        worker = threading.Thread(target=self._tracker_loop, daemon=True)

        def receive_image(cli, image):
            with self._lock:
                if self.running and self._worker is worker:
                    self._latest_raw_image = image.copy()

        self._camera_handler = self.cli.add_handler(
            pycozmo.event.EvtNewRawCameraImage, receive_image
        )
        with self._lock:
            self._worker = worker
            self.running = True
        try:
            worker.start()
        except RuntimeError:
            self._worker = None
            self.stop()
            raise

    def stop(self):
        with self._lock:
            self.running = False
        try:
            if self._camera_handler is not None and hasattr(self.cli, "del_handler"):
                self.cli.del_handler(
                    pycozmo.event.EvtNewRawCameraImage, self._camera_handler
                )
                self._camera_handler = None
        finally:
            if self._worker is not None:
                self._worker.join()
                self._worker = None
            with self._lock:
                self._latest_raw_image = None
                self._latest_boxes = []

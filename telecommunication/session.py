"""相互通信サーバー・ArUco検知・定期座標送信のライフサイクルをまとめて管理する。"""

import threading
import time

from camera.aruco_tracker import ArucoCoordinateTracker
from telecommunication.config import TELECOM_CONFIG
from telecommunication.coordinate_sender import CoordinateSender
from telecommunication.pc_communication import CoordCommunication


class EncounterCommSession:
    """with ブロックの間だけ相互通信関連モジュールを起動し、終了時にすべて停止する。"""

    def __init__(self, robot, config=None, map_data=None, *, track_camera=True):
        self.robot = robot
        self.config = config if config is not None else TELECOM_CONFIG
        self.map_data = map_data
        self.track_camera = track_camera
        self.comm = None
        self.aruco_tracker = None
        self.coord_sender = None

    def get_my_pose(self):
        """RoamingMap が有効ならマップ相対座標、無ければ機体の生座標を返す。"""
        if self.map_data is not None:
            try:
                _, _, pose, _ = self.map_data.snapshot()
                if pose is not None:
                    return {
                        "id": self.config.robot_id,
                        "x": float(pose[0]),
                        "y": float(pose[1]),
                    }
            except Exception:
                pass

        if getattr(self.robot, "pose", None) is not None:
            return {
                "id": self.config.robot_id,
                "x": float(self.robot.pose.position.x),
                "y": float(self.robot.pose.position.y),
            }
        return {"id": self.config.robot_id, "x": 0.0, "y": 0.0}

    def __enter__(self):
        # 1. すれ違い双方向通信サーバー（Bluetooth & TCP）を起動
        self.comm = CoordCommunication(
            port=self.config.listen_port,
            bt_channel=self.config.bt_channel,
            robot_id=self.config.robot_id,
            get_local_info_cb=self.get_my_pose,
        )
        CoordCommunication.register_robot_context(cli=self.robot)
        threading.Thread(
            target=self.comm.start_server,
            name="CoordinateReceiver",
            daemon=True,
        ).start()
        time.sleep(0.3)

        # 2. ArUcoマーカーによるすれ違い自動通信を起動
        if self.track_camera:
            self.start_tracking()

        # 3. 定期送信先が指定されている場合のみ CoordinateSender を起動
        if self.config.periodic_target_addr:
            self.coord_sender = CoordinateSender(
                cli=self.robot,
                target_ip=self.config.periodic_target_addr,
                robot_id=self.config.robot_id,
                interval=self.config.periodic_interval_sec,
                get_pose_cb=self.get_my_pose,
                comm=self.comm,
            )
            self.coord_sender.start()

        return self

    def start_tracking(self):
        if self.aruco_tracker is None:
            self.aruco_tracker = ArucoCoordinateTracker(
                cli=self.robot,
                target_ip_dict=self.config.target_addr_dict,
                comm=self.comm,
                robot_id=self.config.robot_id,
                cooldown_sec=self.config.cooldown_sec,
                get_pose_cb=self.get_my_pose,
                marker_real_size_mm=self.config.marker_real_size_mm,
                focal_length_px=self.config.focal_length_px,
                encounter_dist_max_mm=self.config.encounter_dist_max_mm,
            )
        self.aruco_tracker.start()

    def stop_tracking(self):
        if self.aruco_tracker is not None:
            self.aruco_tracker.stop()

    def draw_on_image(self, frame):
        if self.aruco_tracker is not None:
            return self.aruco_tracker.draw_on_image(frame)
        return frame

    def __exit__(self, exc_type, exc_value, traceback):
        if self.coord_sender is not None:
            try:
                self.coord_sender.stop()
            except Exception:
                pass
        try:
            self.stop_tracking()
        except Exception:
            pass
        if self.comm is not None:
            try:
                self.comm.stop()
            except Exception:
                pass

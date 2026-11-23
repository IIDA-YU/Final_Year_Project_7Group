"""Cozmoの座標を非同期で定期送信するモジュール。"""

import queue
import threading
import time

from telecommunication.pc_communication import CoordCommunication


class CoordinateSender:
    """Cozmoの座標を裏側で取得し、指定した宛先へ非同期で送信し続ける。"""

    def __init__(
        self,
        cli,
        target_ip,
        robot_id="Cozmo-01",
        interval=0.2,
        get_pose_cb=None,
        comm=None,
    ):
        self.cli = cli
        self.target_ip = target_ip
        self.robot_id = robot_id
        self.interval = interval
        self.get_pose_cb = get_pose_cb
        self.env_client = comm if comm is not None else CoordCommunication(robot_id=robot_id)
        self.running = False
        self.send_queue = queue.Queue(maxsize=1)

    def _read_xy(self):
        if callable(self.get_pose_cb):
            data = self.get_pose_cb()
            if isinstance(data, dict) and "x" in data and "y" in data:
                return float(data["x"]), float(data["y"])
        if self.cli and getattr(self.cli, "pose", None) is not None:
            return float(self.cli.pose.position.x), float(self.cli.pose.position.y)
        return None

    def _collect_loop(self):
        while self.running:
            xy = self._read_xy()
            if xy is not None:
                if self.send_queue.full():
                    try:
                        self.send_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.send_queue.put(xy)
            time.sleep(self.interval)

    def _send_loop(self):
        while self.running:
            try:
                x, y = self.send_queue.get(timeout=0.5)
                self.env_client.send_data(
                    self.target_ip, x, y, robot_id=self.robot_id
                )
            except queue.Empty:
                continue

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self._collect_loop, daemon=True).start()
            threading.Thread(target=self._send_loop, daemon=True).start()
            print(
                f"[{self.robot_id}] 非同期の座標送信スレッドを起動しました (宛先: {self.target_ip})"
            )

    def stop(self):
        self.running = False
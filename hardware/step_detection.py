"""Cozmoの機体状態から段差と位置・向きを読み取る。"""

from dataclasses import dataclass
from math import isfinite
from queue import Empty, Queue
from time import monotonic

from pycozmo.event import EvtRobotStateUpdated
from pycozmo.protocol_encoder import EnableStopOnCliff
from pycozmo.robot import RobotStatusFlag


STATE_TIMEOUT_SECONDS = 0.35
UNSAFE_STATUS = (
    RobotStatusFlag.IS_PICKED_UP
    | RobotStatusFlag.IS_FALLING
    | RobotStatusFlag.IS_ON_CHARGER
)


@dataclass(frozen=True)
class StepReading:
    cliff_detected: bool
    unsafe_to_drive: bool
    position: tuple[float, float]
    heading_deg: float
    pose_frame: tuple[int, int]


class StepDetector:
    """withブロックの間だけ状態の受信ハンドラーを登録する。"""

    def __init__(self, robot):
        self.robot = robot
        self.states = Queue()
        self.handler = None

    def __enter__(self):
        # 本体側でも段差を検知したら自動停止するように設定する。
        self.robot.conn.send(EnableStopOnCliff(enable=True))
        self.handler = self.robot.add_handler(
            EvtRobotStateUpdated, self._receive_state
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.robot.del_handler(EvtRobotStateUpdated, self.handler)

    def _receive_state(self, client):
        # 段差フラグと、そのフラグを受信したときの座標をセットで渡す。
        self.states.put(
            (monotonic(), client.robot_status, client.pose, client.pose_frame_id)
        )

    def read(self):
        """次の状態を待ち、受信が途絶えたり遅れたりした場合は例外を出す。"""
        try:
            received_at, status, pose, frame_id = self.states.get(
                timeout=STATE_TIMEOUT_SECONDS
            )
        except Empty:
            raise RuntimeError("Cozmoの状態を受信できません。") from None

        if monotonic() - received_at > STATE_TIMEOUT_SECONDS:
            raise RuntimeError("Cozmoの状態更新が遅れています。")

        x, y = pose.position.x, pose.position.y
        heading = pose.rotation.angle_z.degrees
        if not pose.is_valid or not all(isfinite(value) for value in (x, y, heading)):
            raise RuntimeError("Cozmoの位置や向きを取得できません。")

        return StepReading(
            cliff_detected=bool(status & RobotStatusFlag.CLIFF_DETECTED),
            unsafe_to_drive=bool(status & UNSAFE_STATUS),
            position=(x, y),
            heading_deg=heading,
            pose_frame=(pose.origin_id, frame_id),
        )
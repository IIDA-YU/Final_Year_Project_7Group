"""すれ違い相互通信（Bluetooth / TCP）とArUco検知の設定。"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TelecomConfig:
    robot_id: str = "Cozmo-A"
    listen_port: int = 50007
    bt_channel: int = 22

    # ArUcoマーカーID → 相手のBluetooth MACアドレスまたはIPアドレス
    target_addr_dict: dict[int, str] = field(default_factory=dict)

    # 定期座標送信（CoordinateSender）の宛先。使わない場合は None
    periodic_target_addr: Optional[str] = None
    periodic_interval_sec: float = 0.2

    # ArUco距離推定・すれ違い判定パラメータ
    marker_real_size_mm: float = 30.0
    focal_length_px: float = 235.0
    encounter_dist_max_mm: float = 450.0
    cooldown_sec: float = 4.0


# ---------------------------------------------------------
# 機体・PCごとに変更する設定
# ---------------------------------------------------------
TELECOM_CONFIG = TelecomConfig(
    robot_id="Cozmo-A",
    listen_port=50007,
    bt_channel=22,
    target_addr_dict={
        1: "98:59:7A:E2:CA:79",
        2: "98:59:7A:E3:45:85",
        3: "192.168.1.12",
        4: "192.168.1.13",
        5: "192.168.1.14",
    },
    # 他PCへ0.2秒ごとに定期送信したい場合のみアドレスを設定（不要なら None）
    periodic_target_addr=None,
)
"""Cozmoに接続し、カメラ・マップ・すれ違い相互通信を行いながら机上を走らせる。"""

import sys

from algorithm.table_roaming import roam_table
from camera.camera_movie import show_camera
from connect.network_check import run
from telecommunication.config import TELECOM_CONFIG

# Cozmo 本体の Wi-Fi 名に合わせる。
COZMO_SSID = "Cozmo_613F59"


def start(robot):
    show_camera(
        robot,
        drive=roam_table,
        show_map=True,
        telecom_config=TELECOM_CONFIG,
    )


if __name__ == "__main__":
    sys.exit(run(COZMO_SSID, start))
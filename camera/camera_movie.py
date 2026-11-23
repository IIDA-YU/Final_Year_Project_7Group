"""Cozmoのカラー映像に物体の検知枠を重ね、露光を調整して表示する。"""

from collections import deque
from contextlib import ExitStack
import sys
import time

import cv2
import numpy as np
import pycozmo
from PIL import Image, ImageDraw

from algorithm.mapping.map_view import MapWindow
from algorithm.mapping.roaming_map import RoamingMap
from camera.camera_adjustment import CameraAdjustment, KEY_HELP, prepare_color_frame
from camera.learning_data_check import LearningDataDetector
from hardware.battery import draw_battery
from telecommunication.session import EncounterCommSession


WINDOW_NAME = "Cozmo Camera"
DISPLAY_SIZE = (640, 480)  # 元の表示サイズを維持する。
IMAGE_TIMEOUT = 15


def show_camera(robot, drive=None, show_map=False, telecom_config=None):
    """Cozmoの映像を表示し、指定に応じて走行・マップ表示・相互通信も行う。"""
    print("学習済みモデルを読み込んでいます。")
    detector = LearningDataDetector()
    adjustment = CameraAdjustment(robot)

    # 最新の1枚だけを保持する。
    images = deque(maxlen=1)

    def receive_image(client, image):
        # 通信スレッドでは画像の受け渡しだけを行う。
        images.append((time.monotonic(), image.copy()))

    event = pycozmo.event.EvtNewRawCameraImage
    handler = robot.add_handler(event, receive_image)

    waiting_frame = Image.new("RGB", DISPLAY_SIZE, (20, 25, 36))
    ImageDraw.Draw(waiting_frame).text(
        (20, 220),
        "Waiting for Cozmo camera...",
        fill="white",
    )
    detected_frame = waiting_frame
    processed_at = None
    last_status = None
    resources = ExitStack()
    map_data = None
    map_window = None
    comm_session = None

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if show_map:
            map_data = resources.enter_context(RoamingMap(robot))
            map_window = MapWindow(map_data, WINDOW_NAME, DISPLAY_SIZE[0])
        if telecom_config is not None:
            comm_session = resources.enter_context(
                EncounterCommSession(robot, config=telecom_config, map_data=map_data)
            )
        robot.enable_camera(enable=True, color=True)
        adjustment.start()
        started_at = time.monotonic()

        print(
            "検知枠付きのカメラを表示します。"
            "Q・Esc・閉じるボタンで終了します。"
        )
        print("カメラ画面を選択してキー操作してください。" + KEY_HELP)

        while True:
            latest_image = images[-1] if images else None

            if latest_image is not None:
                received_at, image = latest_image
            else:
                received_at = started_at

            if time.monotonic() - received_at > IMAGE_TIMEOUT:
                raise RuntimeError(
                    "カメラ画像を受信できません。"
                    "Cozmoとの接続を確認してください。"
                )

            # 新しい画像でのみ、測光・補正・検知する。
            if latest_image is not None and received_at != processed_at:
                adjustment.update(image, received_at)
                frame = prepare_color_frame(image).resize(
                    DISPLAY_SIZE,
                    Image.Resampling.BILINEAR,
                )
                detected_frame = detector.detect_and_draw(frame)
                processed_at = received_at

            # バッテリー表示・ArUco枠が積み重ならないようコピーする。
            frame = detected_frame.copy()
            if comm_session is not None:
                comm_session.draw_on_image(frame)

            frame = draw_battery(
                frame,
                robot.battery_voltage,
            )

            # RGB→BGR変換は表示直前に1回だけ行う。
            screen = cv2.cvtColor(
                np.array(frame.convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )
            cv2.imshow(WINDOW_NAME, screen)

            # 映像の寸法や既存描画を変えず、設定はタイトルに表示する。
            status = adjustment.status_text
            if status != last_status:
                cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} | {status}")
                last_status = status

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            if adjustment.handle_key(key):
                print(adjustment.status_text)

            if cv2.getWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

            if map_window is not None:
                if not map_window.is_open():
                    break
                map_window.refresh()

            # 最初の映像を表示できてから、一度だけ走行を開始する。
            if drive is not None and latest_image is not None:
                resources.enter_context(drive(robot))
                drive = None

    finally:
        # 一部の後始末に失敗しても、残りを必ず試みる。
        cleanup_steps = (
            ("走行・マップ・相互通信の停止", resources.close),
            ("自動露光の復帰", adjustment.close),
            ("カメラの停止", lambda: robot.enable_camera(enable=False)),
            ("画像ハンドラーの解除", lambda: robot.del_handler(event, handler)),
            ("ウィンドウの終了", cv2.destroyAllWindows),
        )

        for label, cleanup in cleanup_steps:
            try:
                cleanup()
            except Exception as exc:
                print(f"{label}に失敗しました: {exc}", file=sys.stderr)
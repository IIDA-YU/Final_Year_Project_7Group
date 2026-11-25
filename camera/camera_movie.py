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
from hardware.battery import draw_battery
from telecommunication.session import EncounterCommSession


WINDOW_NAME = "Cozmo Camera"
DISPLAY_SIZE = (640, 480)  # 元の表示サイズを維持する。
IMAGE_TIMEOUT = 15
OFF_REFRESH_SECONDS = 0.5
CONTROLS_HEIGHT = 58
BUTTON_BOUNDS = {
    True: (96, DISPLAY_SIZE[1] + 10, 172, DISPLAY_SIZE[1] + 48),
    False: (180, DISPLAY_SIZE[1] + 10, 256, DISPLAY_SIZE[1] + 48),
}


def show_camera(
    robot, drive=None, show_map=False, telecom_config=None, *, map_data=None,
    camera_enabled=True,
):
    """映像・走行・マップ表示を管理する。camera_enabled=None なら画面上の選択を待つ。"""
    detector = None
    adjustment = None
    resources = ExitStack()
    camera_resources = ExitStack()
    map_window = None
    comm_session = None
    requested_camera = None

    def start_camera():
        nonlocal detector, adjustment
        if detector is None:
            from camera.learning_data_check import LearningDataDetector

            print("学習済みモデルを読み込んでいます。")
            detector = LearningDataDetector()
            adjustment = CameraAdjustment(robot)

        # ONにするたび別の受信バッファを使い、切替前の画像を破棄する。
        images = deque(maxlen=1)

        def receive_image(client, image):
            images.append((time.monotonic(), image.copy()))

        event = pycozmo.event.EvtNewRawCameraImage
        handler = robot.add_handler(event, receive_image)
        camera_resources.callback(robot.del_handler, event, handler)
        camera_resources.callback(robot.enable_camera, enable=False)
        camera_resources.callback(adjustment.close)
        if comm_session is not None:
            camera_resources.callback(comm_session.stop_tracking)
            comm_session.start_tracking()
        robot.enable_camera(enable=True, color=True)
        adjustment.start()
        return images, time.monotonic()

    def click_button(event, x, y, flags, user_data):
        nonlocal requested_camera
        if event == cv2.EVENT_LBUTTONUP:
            for enabled, (left, top, right, bottom) in BUTTON_BOUNDS.items():
                if left <= x <= right and top <= y <= bottom:
                    requested_camera = enabled
                    break

    off_frame = Image.new("RGB", DISPLAY_SIZE, (20, 25, 36))
    waiting_frame = off_frame.copy()
    ImageDraw.Draw(waiting_frame).text(
        (20, 220),
        "Waiting for Cozmo camera...",
        fill="white",
    )
    images = None
    camera_on = camera_enabled
    detected_frame = waiting_frame if camera_on else off_frame
    processed_at = None
    last_status = None
    next_off_refresh = 0.0

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, click_button)
        if show_map and map_data is None:
            map_data = resources.enter_context(RoamingMap(robot))
        if show_map:
            map_window = MapWindow(map_data, WINDOW_NAME, DISPLAY_SIZE[0])
        if telecom_config is not None:
            comm_session = resources.enter_context(
                EncounterCommSession(
                    robot, config=telecom_config, map_data=map_data, track_camera=False,
                )
            )
        if camera_on:
            images, started_at = start_camera()
        else:
            robot.enable_camera(enable=False)

        print(
            "Cozmo Cameraの映像下にあるON・OFFボタンでカメラを切り替えます。"
            "Q・Esc・閉じるボタンで終了します。"
        )
        print("カメラ画面を選択してキー操作してください。" + KEY_HELP)

        while True:
            now = time.monotonic()
            if camera_on:
                received_at, image = images[-1] if images else (started_at, None)
                if now - received_at > IMAGE_TIMEOUT:
                    raise RuntimeError(
                        "カメラ画像を受信できません。"
                        "Cozmoとの接続を確認してください。"
                    )

                # 新しい画像でのみ、測光・補正・検知する。
                if image is not None and received_at != processed_at:
                    adjustment.update(image, received_at)
                    frame = prepare_color_frame(image).resize(
                        DISPLAY_SIZE,
                        Image.Resampling.BILINEAR,
                    )
                    detected_frame = detector.detect_and_draw(frame)
                    processed_at = received_at

            if camera_on or now >= next_off_refresh:
                frame = detected_frame.copy()
                if camera_on and comm_session is not None:
                    comm_session.draw_on_image(frame)
                frame = draw_battery(frame, robot.battery_voltage)
                cv2.imshow(WINDOW_NAME, _render_camera(frame, camera_on))
                next_off_refresh = now + OFF_REFRESH_SECONDS

            if camera_on is None:
                status = "Select ON or OFF"
            else:
                status = f"ON | {adjustment.status_text}" if camera_on else "OFF"
            if status != last_status:
                cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} | {status}")
                last_status = status

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if map_window is not None:
                if not map_window.is_open():
                    break
                map_window.refresh()

            if requested_camera is None and key in (ord("c"), ord("C")):
                requested_camera = not camera_on

            if requested_camera is not None:
                if requested_camera != camera_on:
                    if requested_camera:
                        images, started_at = start_camera()
                        detected_frame = waiting_frame
                    else:
                        camera_resources.close()
                        images = None
                        detected_frame = off_frame
                    next_off_refresh = 0.0
                    processed_at = None
                    camera_on = requested_camera
                requested_camera = None
            elif camera_on and adjustment.handle_key(key):
                print(adjustment.status_text)

            # OFFの選択後、またはONで最初の映像を確認してから走行を開始する。
            if drive is not None and (camera_on is False or processed_at is not None):
                resources.enter_context(drive(robot))
                drive = None

    finally:
        # 一部の後始末に失敗しても、残りを必ず試みる。
        cleanup_steps = (
            ("走行・マップ・相互通信の停止", resources.close),
            ("カメラの停止", camera_resources.close),
            ("ウィンドウの終了", cv2.destroyAllWindows),
        )

        for label, cleanup in cleanup_steps:
            try:
                cleanup()
            except Exception as exc:
                print(f"{label}に失敗しました: {exc}", file=sys.stderr)


def _render_camera(frame, camera_on):
    width, height = frame.size
    screen = np.empty((height + CONTROLS_HEIGHT, width, 3), dtype=np.uint8)
    screen[:height] = np.asarray(frame)[:, :, ::-1]
    screen[height:] = (29, 25, 22)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (240, 240, 238)
    cv2.putText(screen, "Camera", (14, height + 35), font, 0.55, text_color, 1, cv2.LINE_AA)
    for enabled, (left, top, right, bottom) in BUTTON_BOUNDS.items():
        color = (78, 120, 32) if camera_on is enabled else (69, 61, 53)
        cv2.rectangle(screen, (left, top), (right, bottom), color, cv2.FILLED)
        label = "ON" if enabled else "OFF"
        cv2.putText(screen, label, (left + 22, top + 25), font, 0.55, text_color, 1, cv2.LINE_AA)

    hint = "Select ON or OFF to start" if camera_on is None else "C  Toggle camera"
    cv2.putText(
        screen, hint, (278, height + 35), font, 0.48, (185, 185, 182), 1, cv2.LINE_AA,
    )
    return screen

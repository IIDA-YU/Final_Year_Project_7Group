"""Windows用。Wi-Fiの切り替えとCozmoの接続・終了処理を行う。"""

import subprocess
import sys
import time

import pycozmo


def _run_netsh(arguments):
    """WindowsのWi-Fi操作コマンドを実行し、結果を返す。"""
    result = subprocess.run(
        ["netsh", "wlan"] + arguments,
        capture_output=True,
        text=True,
        encoding="oem",  # Windowsのコマンド出力の文字コード
        errors="replace",
        timeout=10,
    )

    if result.returncode != 0:
        message = result.stdout.strip() or result.stderr.strip()
        raise RuntimeError(f"Wi-Fi操作に失敗しました。{message}")

    return result.stdout


def get_current_ssid():
    """接続中のWi-Fi名を返す。未接続ならNoneを返す。"""
    output = _run_netsh(["show", "interfaces"])

    for line in output.splitlines():
        name, separator, value = line.partition(":")

        if separator and name.strip() == "SSID":
            return value.strip() or None

    return None


def switch_wifi(ssid, timeout=20):
    """保存済みのWi-Fiへ切り替える。失敗したら例外を発生させる。"""
    if get_current_ssid() == ssid:
        return

    print(f"Wi-Fi接続中：{ssid}")

    # 保存済みプロファイル名がSSIDと同じであることを前提とする。
    _run_netsh(["connect", f"name={ssid}", f"ssid={ssid}"])

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if get_current_ssid() == ssid:
            # 切り替え直後の通信が落ち着くまで少し待つ。
            time.sleep(2)
            return

        time.sleep(1)

    raise RuntimeError(
        f"{ssid} に接続できませんでした。"
        "保存済みのWi-Fi設定を確認してください。"
    )


def _disconnect_robot(robot):
    """切断に失敗した場合も、通信の停止を試みる。"""
    try:
        robot.disconnect()
    except Exception as error:
        print(f"警告：Cozmoの切断に失敗しました。{error}")

    try:
        robot.stop()
    except Exception as error:
        print(f"警告：通信の停止に失敗しました。{error}")


def run(cozmo_ssid, on_connected=None):
    """接続後に指定の関数を実行する。正常終了は0、接続・復帰の失敗は1。"""
    if sys.platform != "win32":
        print("このプログラムのWi-Fi切り替え機能はWindows専用です。")
        return 1

    previous_ssid = None
    robot = None
    exit_code = 0

    try:
        # 開始前のWi-Fi名を保存してから、CozmoのWi-Fiへ切り替える。
        previous_ssid = get_current_ssid()
        switch_wifi(cozmo_ssid)

        print("Cozmo本体に接続しています...")

        robot = pycozmo.Client()
        robot.start()
        robot.connect()
        robot.wait_for_robot(timeout=10)

        print("Cozmoに接続しました。")

        if on_connected is None:
            input("Enterキーで切断して終了します。")
        else:
            # 接続済みのCozmoを、カメラ表示などの関数へ渡す。
            on_connected(robot)

    except KeyboardInterrupt:
        print("\nプログラムを終了します。")

    except Exception as error:
        print(f"エラー：{error}")
        exit_code = 1

    finally:
        # カメラ表示の終了後や、エラー発生時に切断する。
        if robot is not None:
            _disconnect_robot(robot)

        # 接続に失敗した場合も、開始前のWi-Fiへの復帰を試みる。
        if previous_ssid and previous_ssid != cozmo_ssid:
            try:
                print(f"元のWi-Fiに戻します：{previous_ssid}")
                switch_wifi(previous_ssid)

            except Exception as error:
                print(
                    "警告：元のWi-Fiに戻せませんでした。"
                    f"手動で接続してください。{error}"
                )
                exit_code = 1

    return exit_code
"""実行モードと学習マップを選び、Cozmoに接続する。"""

from functools import partial

from algorithm.mapping.drive_map import DriveMap
from data.roaming_map_files import MAP_DIRECTORY, list_records


COZMO_SSID = "Cozmo_42A832"  # Cozmo本体のWi-Fi名に合わせる。


def main():
    try:
        while True:
            print("\n1. マッピング\n2. 実走行試験\n0. 終了")
            mode = _choose_number("番号を入力してください > ", 2)
            if mode == 0:
                return 0

            drive_map = None
            if mode == 1:
                print("長方形の机の中央付近で、Cozmoを最初に測る辺へほぼ直角に向けてください。")
                print("開始位置と向きに印を付け、段差を検知できる落下防止ガードを設置してください。")
            else:
                drive_map = _select_drive_map()
                if drive_map is None:
                    continue
                print("選んだマップを記録したときと同じ位置・向きにCozmoを置いてください。")

            input("準備できたらEnterキーで接続します > ")
            from connect.network_check import run

            return run(COZMO_SSID, partial(_start_mode, drive_map=drive_map))
    except (EOFError, KeyboardInterrupt):
        print("\nプログラムを終了します。")
        return 0
    except (OSError, ImportError) as error:
        print(f"起動できませんでした。{error}")
        return 1


def _select_drive_map():
    records = list_records(MAP_DIRECTORY)
    if not records:
        print("学習マップがありません。1. マッピングで記録してください。")
        return None

    while True:
        if len(records) == 1:
            record = records[0]
        else:
            print("\n使用する学習マップを選んでください。")
            for index, record in enumerate(records, start=1):
                print(f"{index}. {record.date_label}  {record.path.name}")
            print("0. モード選択に戻る")
            choice = _choose_number("マップの番号を入力してください > ", len(records))
            if choice == 0:
                return None
            record = records[choice - 1]

        try:
            drive_map = DriveMap(record.path)
        except (OSError, ValueError, OverflowError) as error:
            print(f"{record.path.name} は実走行に使えません。{error}")
            if len(records) == 1:
                return None
            continue

        print(f"使用するマップ {record.path.name}")
        return drive_map


def _choose_number(prompt, largest):
    while True:
        choice = input(prompt).strip()
        try:
            number = int(choice)
        except ValueError:
            number = -1
        if 0 <= number <= largest:
            return number
        print(f"0〜{largest}の番号を入力してください。")


def _start_mode(robot, drive_map):
    from algorithm.mapped_driving import drive_on_map
    from algorithm.mapping.table_measurement import MeasurementMap, measure_table
    from camera.camera_movie import show_camera
    from camera.pawn_detection import PawnCamera
    from telecommunication.config import TELECOM_CONFIG

    pawn_camera = None
    if drive_map is None:
        map_data = MeasurementMap()
        drive = partial(measure_table, map_data=map_data)
    else:
        map_data = drive_map
        pawn_camera = PawnCamera()
        drive = partial(drive_on_map, drive_map=drive_map, pawn_camera=pawn_camera)
    show_camera(
        robot,
        drive=drive,
        show_map=True,
        telecom_config=TELECOM_CONFIG,
        map_data=map_data,
        camera_enabled=None,
        pawn_camera=pawn_camera,
    )

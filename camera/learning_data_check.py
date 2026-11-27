"""学習済みのYOLOモデルを読み込み、検知した物体に枠を描く。"""

from pathlib import Path

from PIL import ImageDraw
from ultralytics import YOLO


# このファイルの場所を基準にするので、起動するディレクトリに依存しない。
LEARNING_DATA_DIR = Path(__file__).resolve().parent.parent / "learningData"

# 信頼度がこの値以上の物体を表示する。大きくすると判定が厳しくなる。
CONFIDENCE_THRESHOLD = 0.5

# ファイル名、画面に表示する名前、枠の色（RGB）の順。
MODEL_SETTINGS = (
    ("best_powercube.pt", "Power Cube", (0, 255, 0)),
    ("best_chess_4.0.pt", "Chess", (255, 180, 0)),
)


class LearningDataDetector:
    """2つのモデルを起動時に一度だけ読み込んで使い続ける。"""

    def __init__(self):
        # YOLOに渡す前に存在を確認し、ファイル不足をわかりやすく知らせる。
        for filename, model_name, color in MODEL_SETTINGS:
            model_path = LEARNING_DATA_DIR / filename
            if not model_path.is_file():
                raise FileNotFoundError(
                    "学習済みモデルが見つかりません: {}".format(model_path)
                )

        self.models = []

        for filename, model_name, color in MODEL_SETTINGS:
            model_path = LEARNING_DATA_DIR / filename
            model = YOLO(str(model_path))

            if model.task != "detect":
                raise ValueError(
                    "物体検知用のYOLOモデルを指定してください: {}".format(
                        model_path
                    )
                )

            self.models.append((model, model_name, color))

    def detect_and_draw(self, image, *, cube_boxes=None):
        """Pillow画像を受け取り、検知枠付きの新しい画像を返す。"""
        # 受信した画像は変更せず、描画用の画像を別に作る。
        source = image.convert("RGB")
        frame = source.copy()
        draw = ImageDraw.Draw(frame)

        for model, model_name, color in self.models:
            # どちらのモデルにも、枠やバッテリー表示のない画像を渡す。
            results = model.predict(
                source=source,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )
            result = results[0]

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()

            for box, score, class_id in zip(boxes, scores, class_ids):
                if cube_boxes is not None and model_name == "Power Cube":
                    cube_boxes.append(tuple(box))
                x1, y1, x2, y2 = [int(value) for value in box]

                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline=color,
                    width=3,
                )

                class_name = result.names[int(class_id)]
                label = "{} / {} {:.0%}".format(
                    model_name, class_name, score
                )

                # 文字が映像に埋もれないよう、背景を付けて表示する。
                text_box = draw.textbbox((0, 0), label)
                text_width = text_box[2] - text_box[0]
                text_height = text_box[3] - text_box[1]

                label_x = max(0, min(x1, frame.width - text_width - 8))
                label_y = max(0, y1 - text_height - 8)

                draw.rectangle(
                    (
                        label_x,
                        label_y,
                        label_x + text_width + 8,
                        label_y + text_height + 8,
                    ),
                    fill=color,
                )
                draw.text(
                    (
                        label_x + 4 - text_box[0],
                        label_y + 4 - text_box[1],
                    ),
                    label,
                    fill="black",
                )

        return frame

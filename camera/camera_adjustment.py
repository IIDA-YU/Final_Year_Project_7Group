"""pycozmoの露光制御と、サイズ・色情報を保つ軽い暗部補正。"""

import math
import time

import numpy as np
from PIL import Image
from pycozmo.protocol_encoder import SetCameraParams


# 毎回手動モードで開始する場合はFalseに変更する。
DEFAULT_EXPOSURE_MS = 8
DEFAULT_GAIN = 1.0
DEFAULT_AUTO_ADJUST = True

ADJUST_INTERVAL = 0.5
SETTLE_TIME = 0.25

KEY_HELP = (
    "[ / ]: 露光を1ms下げる / 上げる、"
    "- / +（=も可）: ゲインを0.1下げる / 上げる、"
    "M: 現在の設定で手動固定、A: 自動調整、R: 初期設定に戻す"
)


def prepare_color_frame(image):
    """RGB比率を保って暗部を少し持ち上げ、同じサイズで返す。

    センサーで既に白飛びした細部は復元できない。
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    peak = rgb.max(axis=2, keepdims=True) / 255.0
    scale = 1.0 + 0.20 * (1.0 - peak) ** 2
    corrected = np.clip(np.rint(rgb * scale), 0, 255).astype(np.uint8)
    return Image.fromarray(corrected)


class CameraAdjustment:
    """白飛びを優先する自動露光と、キー操作による手動露光。

    自動モードも本クラスが制御する。本体の自動露光は無効にし、
    競合を防ぐ。表示する数値は本体へ送信した要求値。
    """

    def __init__(
        self,
        robot,
        exposure_ms=DEFAULT_EXPOSURE_MS,
        gain=DEFAULT_GAIN,
        auto_adjust=DEFAULT_AUTO_ADJUST,
    ):
        self.robot = robot
        config = getattr(robot, "camera_config", None)

        # 設定情報がない場合はpycozmo 0.9の既定範囲を使用する。
        self.min_exposure_ms = int(getattr(config, "min_exposure_time_ms", 1))
        self.max_exposure_ms = int(getattr(config, "max_exposure_time_ms", 67))
        self.min_gain = float(getattr(config, "min_gain", 0.1))
        self.max_gain = float(getattr(config, "max_gain", 3.984375))

        if not (
            1 <= self.min_exposure_ms <= self.max_exposure_ms <= 65535
            and math.isfinite(self.min_gain)
            and math.isfinite(self.max_gain)
            and 0 < self.min_gain <= self.max_gain
        ):
            raise ValueError("Cozmoのカメラ設定範囲が不正です。")

        self._initial_exposure, self._initial_gain = self._clamp(exposure_ms, gain)
        self._initial_auto = bool(auto_adjust)

        self.exposure_ms = self._initial_exposure
        self.gain = self._initial_gain
        self.auto_adjust = self._initial_auto

        self._started = False
        self._last_sent_at = float("-inf")
        self._last_metered_at = float("-inf")

    def _clamp(self, exposure_ms, gain):
        if not (math.isfinite(exposure_ms) and math.isfinite(gain)):
            raise ValueError("露光時間とゲインには有限の数値を指定してください。")

        exposure_ms = max(
            self.min_exposure_ms,
            min(self.max_exposure_ms, int(round(exposure_ms))),
        )
        gain = max(self.min_gain, min(self.max_gain, float(gain)))
        return exposure_ms, gain

    def _send(self):
        self.robot.conn.send(
            SetCameraParams(
                exposure_ms=self.exposure_ms,
                gain=self.gain,
                auto_exposure_enabled=False,
            )
        )
        self._last_sent_at = time.monotonic()

    def start(self):
        """カラー配信の有効化後に初期露光を送信する。"""
        self._send()
        self._started = True

    def close(self):
        """終了時は本体の自動露光を有効に戻す。"""
        if self._started:
            self.robot.conn.send(
                SetCameraParams(
                    exposure_ms=self.exposure_ms,
                    gain=self.gain,
                    auto_exposure_enabled=True,
                )
            )
            self._started = False

    def set_manual(self, exposure_ms=None, gain=None):
        """手動設定を送信する。以後は自動変更しない。"""
        exposure_ms, gain = self._clamp(
            self.exposure_ms if exposure_ms is None else exposure_ms,
            self.gain if gain is None else gain,
        )
        self.exposure_ms, self.gain = exposure_ms, gain
        self.auto_adjust = False
        self.start()

    def update(self, image, received_at):
        """加工・拡大・検知枠の描画前の受信画像で測光する。"""
        now = time.monotonic()
        if (
            not self._started
            or not self.auto_adjust
            or now - self._last_metered_at < ADJUST_INTERVAL
            or received_at - self._last_sent_at < SETTLE_TIME
        ):
            return

        self._last_metered_at = now
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

        # 平均輝度だけでなく、いずれかの色チャンネルの飽和も検出する。
        peak = rgb.max(axis=2)
        highlight = float(np.percentile(peak, 98))
        clipped_fraction = float(np.mean(peak >= 250))

        if clipped_fraction > 0.02:
            factor = 0.80
        elif highlight > 235:
            factor = 225.0 / highlight
        elif highlight < 210:
            # 暗い場合はゆっくり明るくし、ちらつきを抑える。
            factor = min(1.15, 225.0 / max(highlight, 1.0))
        else:
            return

        # 露光時間×ゲインを調整し、整数msの差分をゲインで補う。
        level = self.exposure_ms * self.gain * factor
        level = max(
            self.min_exposure_ms * self.min_gain,
            min(self.max_exposure_ms * self.max_gain, level),
        )

        preferred_gain = max(self.min_gain, min(self.max_gain, 1.0))
        exposure_ms = max(
            self.min_exposure_ms,
            min(self.max_exposure_ms, int(level / preferred_gain)),
        )
        exposure_ms, gain = self._clamp(exposure_ms, level / exposure_ms)

        if exposure_ms != self.exposure_ms or not math.isclose(
            gain, self.gain, rel_tol=0.001, abs_tol=0.0001
        ):
            self.exposure_ms, self.gain = exposure_ms, gain
            self._send()

    def handle_key(self, key):
        """調整キーを処理した場合はTrueを返す。"""
        if key == ord("["):
            self.set_manual(exposure_ms=self.exposure_ms - 1)
        elif key == ord("]"):
            self.set_manual(exposure_ms=self.exposure_ms + 1)
        elif key == ord("-"):
            self.set_manual(gain=self.gain - 0.1)
        elif key in (ord("+"), ord("=")):
            self.set_manual(gain=self.gain + 0.1)
        elif key in (ord("m"), ord("M")):
            self.set_manual()
        elif key in (ord("a"), ord("A")):
            self.auto_adjust = True
            self._last_metered_at = time.monotonic()
        elif key in (ord("r"), ord("R")):
            self.exposure_ms = self._initial_exposure
            self.gain = self._initial_gain
            self.auto_adjust = self._initial_auto
            self.start()
        else:
            return False

        return True

    @property
    def status_text(self):
        mode = "AUTO" if self.auto_adjust else "MANUAL"
        return f"{mode} | exposure={self.exposure_ms}ms | gain={self.gain:.3f}"
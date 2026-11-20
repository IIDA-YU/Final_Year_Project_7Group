"""電圧を平滑化し、コンパクトなバッテリー表示を描く。"""

import math
import time
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


# 実測に合わせて調整する仮の基準値。
EMPTY_VOLTAGE = 3.5
FULL_VOLTAGE = 4.2

AVERAGE_SECONDS = 10.0
SAMPLE_INTERVAL = 0.5
DISPLAY_INTERVAL = 3.0
PERCENT_DEADBAND = 2
STALE_SECONDS = 30.0

# 電圧推移による充電推定の基準。
CHARGE_TREND_SECONDS = 10.0
CHARGE_RISE_VOLTAGE = 0.02
CHARGE_FALL_VOLTAGE = 0.02
CHARGE_PEAK_DROP = 0.04
CHARGE_CONFIRM_SECONDS = 3.0


def _valid_voltage(voltage):
    try:
        voltage = float(voltage)
    except (TypeError, ValueError, OverflowError):
        return None

    return voltage if math.isfinite(voltage) and voltage > 0 else None


def estimate_percent(voltage):
    """残量を0～100で返す。不正値・未取得ならNone。"""
    voltage = _valid_voltage(voltage)
    if voltage is None:
        return None

    percent = (
        (voltage - EMPTY_VOLTAGE)
        / (FULL_VOLTAGE - EMPTY_VOLTAGE)
        * 100
    )
    return max(0, min(100, round(percent)))


@dataclass(frozen=True)
class BatteryState:
    voltage: Optional[float]
    percent: Optional[int]
    is_charging: bool


class BatteryMonitor:
    """1台分の電圧履歴と表示状態を保持する。"""

    def __init__(self):
        self.reset()

    def reset(self):
        """別の本体へ接続するときなどに履歴を消去する。"""
        self._samples = deque()
        self._last_valid = None
        self._display_percent = None
        self._last_display = None
        self._charging = False
        self._peak_voltage = None
        self._candidate = None
        self._candidate_since = None

    def _detect_charging(self, average, now):
        # 前半・後半の中央値を比較し、単発のスパイクを除外する。
        recent = [
            (t, v) for t, v in self._samples
            if t >= now - CHARGE_TREND_SECONDS
        ]
        span = recent[-1][0] - recent[0][0]
        trend = None

        if span >= CHARGE_TREND_SECONDS - SAMPLE_INTERVAL:
            midpoint = now - CHARGE_TREND_SECONDS / 2
            older = [v for t, v in recent if t < midpoint]
            newer = [v for t, v in recent if t >= midpoint]

            if len(older) >= 3 and len(newer) >= 3:
                trend = median(newer) - median(older)

        desired = self._charging

        if self._charging:
            self._peak_voltage = max(
                self._peak_voltage or average, average
            )
            if (
                self._peak_voltage - average >= CHARGE_PEAK_DROP
                or (
                    trend is not None
                    and trend <= -CHARGE_FALL_VOLTAGE
                )
            ):
                desired = False
        elif trend is not None and trend >= CHARGE_RISE_VOLTAGE:
            desired = True

        if desired == self._charging:
            self._candidate = None
            self._candidate_since = None
        elif self._candidate != desired:
            self._candidate = desired
            self._candidate_since = now
        elif now - self._candidate_since >= CHARGE_CONFIRM_SECONDS:
            self._charging = desired
            self._peak_voltage = average if desired else None
            self._candidate = None
            self._candidate_since = None

    def update(self, voltage, *, is_charging=None, now=None):
        """充電フラグ指定時は優先。省略時は電圧から推定する。"""
        now = time.monotonic() if now is None else now

        if (
            self._last_valid is not None
            and (
                now < self._last_valid
                or now - self._last_valid > STALE_SECONDS
            )
        ):
            self.reset()

        if is_charging is not None:
            self._charging = bool(is_charging)
            self._peak_voltage = None
            self._candidate = None
            self._candidate_since = None

        voltage = _valid_voltage(voltage)
        if voltage is None:
            return BatteryState(None, None, self._charging)

        self._last_valid = now
        accepted = (
            not self._samples
            or now - self._samples[-1][0] >= SAMPLE_INTERVAL
        )
        if accepted:
            self._samples.append((now, voltage))

        history_seconds = max(
            AVERAGE_SECONDS, CHARGE_TREND_SECONDS
        )
        while (
            self._samples
            and self._samples[0][0] < now - history_seconds
        ):
            self._samples.popleft()

        values = [
            v for t, v in self._samples
            if t >= now - AVERAGE_SECONDS
        ]
        average = sum(values) / len(values)

        was_charging = self._charging
        if accepted and is_charging is None:
            self._detect_charging(average, now)

        if self._charging:
            self._display_percent = None
            self._last_display = None
            return BatteryState(average, None, True)

        percent = estimate_percent(average)

        if self._display_percent is None or was_charging:
            self._display_percent = percent
            self._last_display = now
        elif now - self._last_display >= DISPLAY_INTERVAL:
            difference = abs(percent - self._display_percent)
            if (
                difference >= PERCENT_DEADBAND
                or percent in (0, 100)
            ):
                self._display_percent = percent
                self._last_display = now

        return BatteryState(
            average, self._display_percent, False
        )


def _load_font(size):
    for name in ("segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue

    return ImageFont.load_default()


def _load_charging_font():
    for name in (
        "YuGothM.ttc",
        "meiryo.ttc",
        "msgothic.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    ):
        try:
            return ImageFont.truetype(name, 14), "充電中"
        except OSError:
            continue

    # 日本語フォントがない環境での文字化けを防ぐ。
    return _load_font(12), "CHARGING"


LABEL_FONT = _load_font(9)
VALUE_FONT = _load_font(18)
DETAIL_FONT = _load_font(10)
CHARGING_FONT, CHARGING_TEXT = _load_charging_font()

_DEFAULT_MONITOR = BatteryMonitor()


def draw_battery(frame, voltage, is_charging=None, *, monitor=None):
    """右上にコンパクトな表示を重ね、新しいRGB画像を返す。"""
    monitor = _DEFAULT_MONITOR if monitor is None else monitor
    state = monitor.update(voltage, is_charging=is_charging)
    percent = state.percent

    if state.is_charging:
        color = (56, 189, 248)
    elif percent is None:
        color = (148, 163, 184)
    elif percent <= 20:
        color = (248, 113, 113)
    elif percent <= 50:
        color = (251, 191, 36)
    else:
        color = (45, 212, 191)

    # カード本体44px + 影3px。
    card = Image.new("RGBA", (136, 47), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)

    draw.rounded_rectangle(
        (0, 3, 135, 46),
        radius=12,
        fill=(0, 0, 0, 55),
    )
    draw.rounded_rectangle(
        (0, 0, 135, 43),
        radius=12,
        fill=(15, 23, 42, 218),
        outline=(255, 255, 255, 32),
    )

    # バッテリーの外枠と端子。
    draw.rounded_rectangle(
        (11, 14, 35, 29),
        radius=3,
        outline=color,
        width=2,
    )
    draw.rounded_rectangle(
        (36, 18, 38, 25),
        radius=1,
        fill=color,
    )

    if state.is_charging:
        # 充電中は残量バーを描かず、稲妻を表示する。
        draw.polygon(
            (
                (24, 16), (19, 23), (23, 23), (21, 28),
                (28, 20), (24, 20), (26, 16),
            ),
            fill=color,
        )
        draw.text(
            (47, 4),
            CHARGING_TEXT,
            font=CHARGING_FONT,
            fill=color,
        )
        detail = (
            "POWER INPUT"
            if is_charging is not None
            else "EST. CHARGING"
        )
    else:
        if percent is not None and percent > 0:
            width = max(1, round(18 * percent / 100))
            draw.rectangle(
                (14, 17, 13 + width, 26),
                fill=color,
            )

        draw.text(
            (47, 2),
            "--%" if percent is None else f"{percent}%",
            font=VALUE_FONT,
            fill=(248, 250, 252),
        )
        detail = (
            "WAITING"
            if state.voltage is None
            else f"{state.voltage:.2f} V  EST."
        )

    draw.text(
        (48, 27),
        detail,
        font=LABEL_FONT if state.is_charging else DETAIL_FONT,
        fill=(148, 163, 184),
    )

    # 小さな画像ではカード全体を縮小する。
    margin = min(8, max(0, (min(frame.size) - 1) // 2))
    scale = min(
        1.0,
        (frame.width - margin * 2) / card.width,
        (frame.height - margin * 2) / card.height,
    )
    if scale < 1:
        card = card.resize(
            (
                max(1, int(card.width * scale)),
                max(1, int(card.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    overlay.paste(
        card,
        (frame.width - card.width - margin, margin),
    )

    return Image.alpha_composite(
        frame.convert("RGBA"), overlay
    ).convert("RGB")
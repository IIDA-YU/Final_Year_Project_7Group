"""車輪を停止した状態で、同じポーンを連続して検知できるか確認する。"""

from dataclasses import dataclass, field
from math import dist
from statistics import median

from camera.pawn_detection import MAX_FRAME_AGE_SECONDS


PAWN_MATCH_DISTANCE_MM = 32.0
SETTLE_SECONDS = 0.25
CONFIRM_SECONDS = 4.0
CONFIRM_TIMEOUT_SECONDS = 6.0
MIN_RESUME_SECONDS = 1.0
RETRY_EVIDENCE_SECONDS = 0.75
RETRY_WINDOW_SECONDS = 8.0


@dataclass
class _Candidate:
    position: tuple[float, float]
    first_seen: float
    last_seen: float
    samples: list[tuple[float, float]] = field(default_factory=list)


class PawnConfirmation:
    def __init__(self):
        self._stopped_at = None
        self._candidates = []
        self._retry_candidate = None
        self._retry_at = float("-inf")
        self._retry_until = float("-inf")

    @property
    def waiting(self):
        return bool(self._candidates)

    def stopped(self, now):
        if self.waiting and self._stopped_at is None:
            self._stopped_at = now

    def cancel(self):
        self._candidates.clear()
        self._stopped_at = None
        self._retry_candidate = None

    def expire(self, now):
        if self._stopped_at is not None and now - self._stopped_at >= CONFIRM_TIMEOUT_SECONDS:
            self._reject(now)

    def observe(self, points, received_at, now):
        if self._stopped_at is not None:
            return self._observe_while_stopped(points, received_at, now)
        if not self.waiting and self._ready_to_check(points, received_at, now):
            self._candidates = [_Candidate(point, received_at, received_at) for point in points]
        return ()

    def _ready_to_check(self, points, received_at, now):
        if not points:
            self._retry_candidate = None
            return False
        if now >= self._retry_until:
            return True

        # 誤検知の直後は少し前進し、再確認には走行中の連続検知を求める。
        candidate = self._retry_candidate
        if (
            candidate is None
            or received_at - candidate.last_seen > MAX_FRAME_AGE_SECONDS
            or _matching_index(points, candidate.position) is None
        ):
            self._retry_candidate = _Candidate(points[0], received_at, received_at)
            return False
        candidate.last_seen = received_at
        return now >= self._retry_at and received_at - candidate.first_seen >= RETRY_EVIDENCE_SECONDS

    def _observe_while_stopped(self, points, received_at, now):
        if received_at < self._stopped_at + SETTLE_SECONDS:
            return ()

        remaining = list(points)
        visible = []
        confirmed = []
        for candidate in self._candidates:
            index = _matching_index(remaining, candidate.position)
            if index is None:
                continue
            point = remaining.pop(index)
            if not candidate.samples:
                candidate.position = point
                candidate.first_seen = received_at
            elif received_at - candidate.last_seen > MAX_FRAME_AGE_SECONDS:
                continue
            candidate.last_seen = received_at
            candidate.samples.append(point)
            if received_at - candidate.first_seen >= CONFIRM_SECONDS:
                xs, ys = zip(*candidate.samples)
                confirmed.append((median(xs), median(ys)))
            else:
                visible.append(candidate)

        if confirmed:
            self.cancel()
        elif visible:
            self._candidates = visible
        else:
            self._reject(now)
        return tuple(confirmed)

    def _reject(self, now):
        self.cancel()
        self._retry_at = now + MIN_RESUME_SECONDS
        self._retry_until = now + RETRY_WINDOW_SECONDS


def _matching_index(points, position):
    nearest = min(range(len(points)), key=lambda index: dist(points[index], position), default=None)
    if nearest is not None and dist(points[nearest], position) <= PAWN_MATCH_DISTANCE_MM:
        return nearest
    return None

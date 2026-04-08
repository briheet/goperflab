from __future__ import annotations

import math

SCORE_MIN = 0.0
SCORE_MAX = 1.0


def normalize_score(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return SCORE_MIN
    if value <= 0.0:
        return SCORE_MIN
    if value >= 1.0:
        return SCORE_MAX
    return value

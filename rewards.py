from __future__ import annotations

from models import GoPerfObservation, GoPerfReward, GoPerfState
from tasks import TaskConfig
from graders import _weighted_speedup

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def compute_reward(
    task: TaskConfig, state: GoPerfState, observation: GoPerfObservation
) -> GoPerfReward:
    components: dict[str, float] = {}

    baseline = state.baseline_metrics
    current = state.current_metrics
    prev_best = state.prev_best_metrics

    speedup_base = _weighted_speedup(baseline, current, task.metrics_weights)
    speedup_prev = _weighted_speedup(prev_best, current, task.metrics_weights)

    delta_base = speedup_base - 1.0
    delta_prev = speedup_prev - 1.0
    components["delta_vs_baseline"] = delta_base
    components["delta_vs_prev_best"] = delta_prev

    base_reward = 0.7 * delta_base + 0.3 * delta_prev
    if base_reward > 1.0:
        base_reward = 1.0
    if base_reward < 0.0:
        base_reward = 0.0
    components["speedup_reward"] = base_reward

    penalty = 0.0
    if observation.exit_code != 0:
        penalty -= 0.5
        components["exec_penalty"] = -0.5
    total = base_reward + penalty
    components["total"] = total

    score = total
    # Keep reward score strictly within (0, 1) for validator compatibility.
    if score <= 0.0:
        score = _SCORE_MIN
    elif score >= 1.0:
        score = _SCORE_MAX

    return GoPerfReward(score=score, components=components)

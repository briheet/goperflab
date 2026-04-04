from __future__ import annotations

from models import GoPerfObservation, GoPerfReward, GoPerfState
from tasks import TaskConfig
from graders import _weighted_speedup


def compute_reward(
    task: TaskConfig, state: GoPerfState, observation: GoPerfObservation
) -> GoPerfReward:
    components: dict[str, float] = {}

    speedup = _weighted_speedup(
        state.baseline_metrics, state.current_metrics, task.metrics_weights
    )
    if speedup <= 1.0:
        base_reward = (speedup - 1.0) * 0.5
    else:
        base_reward = min(speedup - 1.0, 2.0)
    components["speedup_reward"] = base_reward

    penalty = 0.0
    if observation.exit_code != 0:
        penalty -= 0.5
        components["exec_penalty"] = -0.5
    if observation.test_passed is False:
        penalty -= 0.25
        components["test_penalty"] = -0.25

    total = base_reward + penalty
    components["total"] = total
    return GoPerfReward(score=total, components=components)

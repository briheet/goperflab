from __future__ import annotations

from models import GoPerfState
from score_utils import normalize_score
from tasks import TaskConfig


def _weighted_speedup(
    baseline: dict | None, current: dict | None, weights: dict
) -> float:
    if not baseline or not current:
        return 1.0
    total = 0.0
    weight_sum = 0.0
    for metric, weight in weights.items():
        base_val = baseline.get(metric)
        cur_val = current.get(metric)
        if base_val is None or cur_val in (None, 0):
            continue
        total += (base_val / cur_val) * weight
        weight_sum += weight
    if weight_sum == 0:
        return 1.0
    return total / weight_sum


def grade_task(task: TaskConfig, state: GoPerfState) -> dict:
    speedup = _weighted_speedup(
        state.baseline_metrics, state.current_metrics, task.metrics_weights
    )
    target = task.target_speedup
    max_speedup = task.max_speedup

    if speedup <= 1.0:
        raw = 0.0
    else:
        raw = (speedup - 1.0) / max(target - 1.0, 1e-6)
    score = min(max(raw, 0.0), 1.0)

    penalties = 0.0
    if state.current_metrics is None:
        penalties += 0.1

    improved_count = 0
    regressed_hard = False
    if state.current_metrics and state.baseline_metrics:
        for metric in task.metrics_weights:
            base_val = state.baseline_metrics.get(metric)
            cur_val = state.current_metrics.get(metric)
            if base_val is None or cur_val is None:
                continue
            if cur_val < base_val:
                improved_count += 1
            if cur_val > base_val * 1.10:
                regressed_hard = True

    # Require improvement in at least 2 metrics for full credit.
    if improved_count < 2:
        score *= 0.5

    # Stronger penalty for >10% regression in any metric.
    if regressed_hard:
        penalties += 0.2

    final_score = normalize_score(score - penalties)
    return {
        "task_id": task.task_id,
        "speedup": speedup,
        "score": final_score,
        "penalties": penalties,
        "improved_metrics": improved_count,
        "hard_regression": regressed_hard,
        "target_speedup": target,
    }

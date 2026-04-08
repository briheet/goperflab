import math
import unittest

from graders import grade_task
from models import GoPerfObservation, GoPerfReward, GoPerfState
from rewards import compute_reward
from score_utils import SCORE_MAX, SCORE_MIN, normalize_score
from tasks import get_task


class ScoreNormalizationTests(unittest.TestCase):
    def test_normalize_score_clamps_to_closed_unit_interval(self) -> None:
        self.assertEqual(normalize_score(None), SCORE_MIN)
        self.assertEqual(normalize_score(float("nan")), SCORE_MIN)
        self.assertEqual(normalize_score(float("inf")), SCORE_MIN)
        self.assertEqual(normalize_score(-1.0), SCORE_MIN)
        self.assertEqual(normalize_score(0.0), SCORE_MIN)
        self.assertEqual(normalize_score(1.0), SCORE_MAX)
        self.assertEqual(normalize_score(2.0), SCORE_MAX)
        self.assertEqual(normalize_score(0.5), 0.5)

    def test_reward_default_score_is_in_range(self) -> None:
        reward = GoPerfReward()
        self.assertEqual(reward.score, 0.0)

    def test_grade_task_returns_closed_interval_score(self) -> None:
        task = get_task("easy")
        state = GoPerfState(
            baseline_metrics={"ns/op": 100.0, "B/op": 100.0, "allocs/op": 10.0},
            current_metrics={"ns/op": 50.0, "B/op": 50.0, "allocs/op": 1.0},
        )

        grade = grade_task(task, state)

        self.assertGreaterEqual(grade["score"], 0.0)
        self.assertLessEqual(grade["score"], 1.0)
        self.assertEqual(grade["score"], 1.0)

    def test_compute_reward_returns_closed_interval_score(self) -> None:
        task = get_task("easy")
        state = GoPerfState(
            baseline_metrics={"ns/op": 100.0, "B/op": 100.0, "allocs/op": 10.0},
            current_metrics={"ns/op": 50.0, "B/op": 50.0, "allocs/op": 1.0},
            prev_best_metrics={"ns/op": 100.0, "B/op": 100.0, "allocs/op": 10.0},
        )
        observation = GoPerfObservation(exit_code=0)

        reward = compute_reward(task, state, observation)

        self.assertTrue(math.isfinite(reward.score))
        self.assertGreaterEqual(reward.score, 0.0)
        self.assertLessEqual(reward.score, 1.0)

    def test_compute_reward_defaults_to_zero_on_regression(self) -> None:
        task = get_task("easy")
        state = GoPerfState(
            baseline_metrics={"ns/op": 100.0, "B/op": 100.0, "allocs/op": 10.0},
            current_metrics={"ns/op": 120.0, "B/op": 130.0, "allocs/op": 12.0},
            prev_best_metrics={"ns/op": 100.0, "B/op": 100.0, "allocs/op": 10.0},
        )
        observation = GoPerfObservation(exit_code=0)

        reward = compute_reward(task, state, observation)

        self.assertEqual(reward.score, 0.0)


if __name__ == "__main__":
    unittest.main()

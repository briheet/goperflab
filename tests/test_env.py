import shutil
import tempfile
import unittest
from pathlib import Path

from models import GoPerfAction
from server.env import GoPerfEnvironment


class GoPerfEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.env = GoPerfEnvironment(workspace_root=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _repo_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "go-bench-repo"

    def test_init_repo_copies_and_inits_git(self) -> None:
        repo_path = self._repo_path()
        self.assertTrue(repo_path.exists())

        self.env.reset(task_id="easy")
        action = GoPerfAction(
            action_type="init_repo",
            repo_path=str(repo_path),
            repo_copy=True,
            repo_init_if_missing=True,
        )
        obs = self.env.step(action)

        self.assertEqual(obs.exit_code, 0)
        workspace = Path(self.env.state.workspace_path or "")
        self.assertTrue((workspace / ".git").exists())
        self.assertEqual(self.env.state.repo_name, repo_path.name)

    @unittest.skipUnless(shutil.which("go"), "go toolchain required")
    def test_benchmarks_run(self) -> None:
        repo_path = self._repo_path()

        self.env.reset(task_id="easy")
        init_action = GoPerfAction(
            action_type="init_repo",
            repo_path=str(repo_path),
            repo_copy=True,
            repo_init_if_missing=True,
        )
        self.env.step(init_action)

        bench_action = GoPerfAction(
            action_type="benchmarks",
            bench_suite=".",
            bench_mem_required=True,
            bench_count=1,
            bench_timeout=10,
        )
        obs = self.env.step(bench_action)

        self.assertEqual(
            obs.exit_code,
            0,
            msg=f"stderr={obs.stderr[:400]} stdout={obs.stdout[:400]}",
        )
        self.assertIsInstance(obs.bench_summary, list)
        self.assertTrue(
            len(obs.bench_summary) > 0,
            msg=f"stderr={obs.stderr[:400]} stdout={obs.stdout[:400]}",
        )

    @unittest.skipUnless(shutil.which("go"), "go toolchain required")
    def test_baseline_metrics_set_on_init(self) -> None:
        repo_path = self._repo_path()

        self.env.reset(task_id="easy")
        init_action = GoPerfAction(
            action_type="init_repo",
            repo_path=str(repo_path),
            repo_copy=True,
            repo_init_if_missing=True,
        )
        self.env.step(init_action)

        self.assertIsNotNone(self.env.state.baseline_metrics)
        self.assertIsNotNone(self.env.state.prev_best_metrics)


if __name__ == "__main__":
    unittest.main()

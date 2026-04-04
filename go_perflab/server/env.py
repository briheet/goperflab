import subprocess
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional
import re

from openenv.core.env_server import Environment

from models import GoPerfAction, GoPerfObservation, GoPerfState
from tasks import get_task
from graders import grade_task
from rewards import compute_reward


class GoPerfEnvironment(Environment[GoPerfAction, GoPerfObservation, GoPerfState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, workspace_root: Optional[str] = None):
        super().__init__()
        self._workspace_root = (
            Path(workspace_root).expanduser()
            if workspace_root
            else Path(tempfile.gettempdir()) / "go_perflab"
        )
        self._state = GoPerfState()

    @property
    def state(self) -> GoPerfState:
        return self._state

    def reset(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs
    ):
        self._reset_rubric()
        self._state = GoPerfState()
        self._state.episode_id = episode_id or f"episode-{uuid.uuid4().hex[:8]}"
        self._state.step_count = 0
        self._state.action_history = []
        self._state.reward_trace = []
        self._state.build_flags = []
        self._state.perf_artifacts = []
        self._state.escape_artifacts = []
        task_id = kwargs.get("task_id")
        task = get_task(task_id)
        self._state.task_config = task.__dict__
        self._state.budget_remaining = kwargs.get("budget", task.budget)

        self._state.workspace_path = str(self._make_workspace(self._state.episode_id))

        obs = self._base_observation(stdout="reset")
        obs.task_id = task.task_id
        obs.task_goal = task.goal
        return self._apply_transform(obs)

    def step(self, action: GoPerfAction, timeout_s: Optional[float] = None, **kwargs):
        self._state.step_count += 1
        if self._state.action_history is None:
            self._state.action_history = []
        self._state.last_action = action.model_dump()
        self._state.action_history.append(self._state.last_action)

        observation = GoPerfObservation()
        if action.action_type == "init_repo":
            observation = self._handle_init_repo(action)
        elif action.action_type == "git":
            observation = self._handle_git(action, timeout_s=timeout_s)
        elif action.action_type == "patch":
            observation = self._handle_patch(action)
        elif action.action_type == "build_flags":
            observation = self._handle_build_flags(action)
        elif action.action_type == "benchmarks":
            observation = self._handle_benchmarks(action, timeout_s=timeout_s)
        elif action.action_type == "tests":
            observation = self._handle_tests(action, timeout_s=timeout_s)
        elif action.action_type == "escape_analysis":
            observation = self._handle_escape_analysis(action, timeout_s=timeout_s)
        elif action.action_type == "perf":
            observation = self._handle_perf(action, timeout_s=timeout_s)
        else:
            observation = self._base_observation(
                stderr=f"unknown action_type: {action.action_type}", exit_code=2
            )

        observation = self._with_context(observation)
        self._update_metrics_from_observation(observation)

        task = get_task(
            self._state.task_config.get("task_id") if self._state.task_config else None
        )
        reward = compute_reward(task, self._state, observation)
        observation.reward = reward.score
        observation.metadata["reward_components"] = reward.components

        grade = grade_task(task, self._state)
        observation.metadata["grade"] = grade

        if self._state.budget_remaining is not None:
            self._state.budget_remaining -= 1
        if (
            self._state.budget_remaining is not None
            and self._state.budget_remaining <= 0
        ):
            observation.done = True
        if grade.get("score", 0.0) >= 1.0:
            observation.done = True

        return self._apply_transform(observation)

    def _make_workspace(self, episode_id: str) -> Path:
        path = self._workspace_root / episode_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _handle_init_repo(self, action: GoPerfAction) -> GoPerfObservation:
        if not action.repo_path:
            return self._base_observation(stderr="repo_path is required", exit_code=2)

        src = Path(action.repo_path).expanduser()
        if not src.exists():
            return self._base_observation(
                stderr=f"repo_path not found: {src}", exit_code=2
            )

        workspace = self._workspace_path()
        use_copy = bool(action.repo_copy)
        if use_copy:
            shutil.copytree(src, workspace, dirs_exist_ok=True)
            target = workspace
        else:
            target = src
            self._state.workspace_path = str(target.resolve())

        if action.repo_init_if_missing:
            if not (target / ".git").exists():
                self._run_command(["git", "init"], target, None)
                self._run_command(
                    ["git", "config", "user.email", "seed@example.com"], target, None
                )
                self._run_command(
                    ["git", "config", "user.name", "GoPerfLab"], target, None
                )
                self._run_command(["git", "add", "-A"], target, None)
                self._run_command(["git", "commit", "-m", "init repo"], target, None)

        self._state.repo_name = target.name
        return self._base_observation(stdout=f"repo ready at {target}")

    def _workspace_path(self) -> Path:
        if not self._state.workspace_path:
            self._state.workspace_path = str(self._make_workspace("default"))
        return Path(self._state.workspace_path)

    def _base_observation(
        self, stdout: str = "", stderr: str = "", exit_code: int = 0
    ) -> GoPerfObservation:
        return GoPerfObservation(stdout=stdout, stderr=stderr, exit_code=exit_code)

    def _with_context(self, observation: GoPerfObservation) -> GoPerfObservation:
        observation.repo_name = self._state.repo_name
        observation.repo_worktree_path = self._state.workspace_path
        observation.task_step_count = self._state.step_count
        observation.task_remaining_budget = self._state.budget_remaining
        observation.task_build_flags = self._state.build_flags
        if self._state.workspace_path:
            observation.repo_git_status = self._git_status()
        return observation

    def _update_metrics_from_observation(self, observation: GoPerfObservation) -> None:
        if observation.bench_summary:
            if self._state.baseline_metrics is None:
                self._state.baseline_metrics = observation.bench_summary[0].get(
                    "metrics"
                )
            self._state.current_metrics = observation.bench_summary[0].get("metrics")

    def _run_command(
        self, cmd: list[str], cwd: Optional[Path], timeout_s: Optional[float]
    ):
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                timeout=timeout_s,
                capture_output=True,
                text=True,
            )
            return completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            return "", f"timeout after {timeout_s}s: {exc}", 124
        except FileNotFoundError as exc:
            return "", str(exc), 127

    def _git_status(self) -> dict | None:
        workspace = self._workspace_path()
        if not (workspace / ".git").exists():
            return None
        stdout, stderr, code = self._run_command(
            ["git", "status", "--porcelain", "-b"], workspace, None
        )
        return {"exit_code": code, "stdout": stdout, "stderr": stderr}

    def _handle_git(self, action: GoPerfAction, timeout_s: Optional[float]):
        workspace = self._workspace_path()
        cmd: list[str] = ["git"]
        if action.git_op:
            cmd.append(action.git_op)
        if action.git_args:
            cmd.extend(action.git_args)
        stdout, stderr, code = self._run_command(cmd, workspace, timeout_s)

        if action.git_op == "clone" and action.git_target_dir:
            self._state.repo_name = action.git_repo_name
            self._state.workspace_path = str(
                (workspace / action.git_target_dir).resolve()
            )

        return self._base_observation(stdout=stdout, stderr=stderr, exit_code=code)

    def _handle_patch(self, action: GoPerfAction):
        workspace = self._workspace_path()
        if not action.patch_diff:
            return self._base_observation(stderr="patch_diff is required", exit_code=2)

        patch_path = workspace / "action.patch"
        patch_path.write_text(action.patch_diff, encoding="utf-8")

        if (workspace / ".git").exists():
            cmd = ["git", "apply", str(patch_path)]
        else:
            cmd = ["patch", "-p0", "-i", str(patch_path)]

        stdout, stderr, code = self._run_command(cmd, workspace, None)
        return self._base_observation(stdout=stdout, stderr=stderr, exit_code=code)

    def _handle_build_flags(self, action: GoPerfAction):
        self._state.build_flags = action.build_flags or []
        return self._base_observation(stdout="build flags set")

    def _handle_benchmarks(self, action: GoPerfAction, timeout_s: Optional[float]):
        workspace = self._workspace_path()
        bench = action.bench_suite or "."
        cmd = ["go", "test", "-run=^$", f"-bench={bench}"]
        if action.bench_mem_required:
            cmd.append("-benchmem")
        if action.bench_time:
            cmd.append(f"-benchtime={action.bench_time}s")
        if action.bench_count:
            cmd.append(f"-count={action.bench_count}")
        if action.bench_timeout:
            cmd.append(f"-timeout={action.bench_timeout}s")
        if self._state.build_flags:
            cmd.append(f"-gcflags={' '.join(self._state.build_flags)}")

        stdout, stderr, code = self._run_command(cmd, workspace, timeout_s)
        if action.bench_file_name:
            (workspace / action.bench_file_name).write_text(stdout, encoding="utf-8")
        obs = self._base_observation(stdout=stdout, stderr=stderr, exit_code=code)
        obs.bench_summary = self._parse_bench_output(stdout)
        return obs

    def _handle_tests(self, action: GoPerfAction, timeout_s: Optional[float]):
        workspace = self._workspace_path()
        suite = action.test_suite or "./..."
        cmd = ["go", "test", suite]
        if action.test_verbose:
            cmd.append("-v")
        if action.test_timeout:
            cmd.append(f"-timeout={action.test_timeout}s")
        stdout, stderr, code = self._run_command(cmd, workspace, timeout_s)
        if action.test_output_save_file:
            (workspace / action.test_output_save_file).write_text(
                stdout + stderr, encoding="utf-8"
            )
        obs = self._base_observation(stdout=stdout, stderr=stderr, exit_code=code)
        obs.test_passed = code == 0
        return obs

    def _handle_escape_analysis(self, action: GoPerfAction, timeout_s: Optional[float]):
        workspace = self._workspace_path()
        target = action.escape_target or "./..."
        flags = action.escape_flags or ["-m"]
        cmd = ["go", "test", "-run=^$", f"-gcflags=all={' '.join(flags)}", target]
        stdout, stderr, code = self._run_command(cmd, workspace, timeout_s)
        if action.escape_output_file:
            (workspace / action.escape_output_file).write_text(
                stdout + stderr, encoding="utf-8"
            )
        if code == 0 and action.escape_output_file:
            if self._state.escape_artifacts is None:
                self._state.escape_artifacts = []
            self._state.escape_artifacts.append(
                str(workspace / action.escape_output_file)
            )
        obs = self._base_observation(stdout=stdout, stderr=stderr, exit_code=code)
        obs.escape_summary, obs.escape_count_total = self._parse_escape_output(
            stdout + stderr
        )
        return obs

    def _handle_perf(self, action: GoPerfAction, timeout_s: Optional[float]):
        workspace = self._workspace_path()
        if not action.perf_mode:
            return self._base_observation(stderr="perf_mode is required", exit_code=2)
        bench = action.perf_bench or "."
        go_cmd = ["go", "test", "-run=^$", f"-bench={bench}"]
        perf_args = action.perf_args or []
        cmd = ["perf", action.perf_mode, *perf_args, "--", *go_cmd]
        stdout, stderr, code = self._run_command(cmd, workspace, timeout_s)
        if action.perf_output_file:
            (workspace / action.perf_output_file).write_text(
                stdout + stderr, encoding="utf-8"
            )
        if code == 0 and action.perf_output_file:
            if self._state.perf_artifacts is None:
                self._state.perf_artifacts = []
            self._state.perf_artifacts.append(str(workspace / action.perf_output_file))
        obs = self._base_observation(stdout=stdout, stderr=stderr, exit_code=code)
        obs.perf_summary = self._parse_perf_output(stdout + stderr)
        return obs

    def _parse_bench_output(self, text: str) -> list[dict]:
        results: list[dict] = []
        line_re = re.compile(
            r"^(Benchmark\\S+)\\s+\\d+\\s+([\\d\\.]+)\\s+ns/op"
            r"(?:\\s+([\\d\\.]+)\\s+B/op)?"
            r"(?:\\s+([\\d\\.]+)\\s+allocs/op)?"
        )
        for line in text.splitlines():
            match = line_re.match(line.strip())
            if not match:
                continue
            name, ns_op, b_op, allocs_op = match.groups()
            metrics = {"ns/op": float(ns_op)}
            if b_op is not None:
                metrics["B/op"] = float(b_op)
            if allocs_op is not None:
                metrics["allocs/op"] = float(allocs_op)
            results.append({"name": name, "metrics": metrics})
        return results

    def _parse_escape_output(self, text: str) -> tuple[list[dict], int]:
        escapes = 0
        no_escapes = 0
        for line in text.splitlines():
            if "escapes to heap" in line:
                escapes += 1
            if "does not escape" in line:
                no_escapes += 1
        summary = [{"escapes_to_heap": escapes, "does_not_escape": no_escapes}]
        return summary, escapes

    def _parse_perf_output(self, text: str) -> dict:
        summary: dict[str, float | str] = {"raw": text[:2000]}
        elapsed_re = re.compile(r"([\\d\\.]+)\\s+seconds time elapsed")
        match = elapsed_re.search(text)
        if match:
            summary["seconds_elapsed"] = float(match.group(1))
        return summary

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
        self._state.baseline_metrics = None
        self._state.current_metrics = None
        self._state.prev_best_metrics = None
        task_id = kwargs.get("task_id")
        task = get_task(task_id)
        self._state.task_config = task.__dict__
        self._state.budget_remaining = kwargs.get("budget", task.budget)

        self._state.workspace_path = str(self._make_workspace(self._state.episode_id))

        obs = self._base_observation(stdout="reset")
        obs.task_id = task.task_id
        obs.task_goal = task.goal

        # auto-init: clone/init repo + run baseline benchmark.
        # At the start of inference we need to clone and run baseline to compare and go.
        if kwargs.get("auto_init"):
            # Build action
            init_action = GoPerfAction(
                action_type="init_repo",
                repo_path=kwargs.get("repo_path"),
                repo_copy=kwargs.get("repo_copy", True),
                repo_init_if_missing=kwargs.get("repo_init_if_missing", True),
            )

            init_obs = self._handle_init_repo(init_action)
            if init_obs.exit_code != 0:
                obs.stderr = init_obs.stderr
                obs.exit_code = init_obs.exit_code
                obs.metadata["auto_init_error"] = init_obs.stderr
            else:
                obs.stdout = init_obs.stdout
                obs.metadata["auto_init"] = True
        obs = self._with_context(obs)
        return self._apply_transform(obs)

    def step(self, action: GoPerfAction, timeout_s: Optional[float] = None, **kwargs):
        self._state.step_count += 1
        if self._state.action_history is None:
            self._state.action_history = []
        self._state.last_action = action.model_dump()
        self._state.action_history.append(self._state.last_action)

        observation = GoPerfObservation()
        if action.action_type == "init_repo":
            # Internal-only: use reset(auto_init=True) instead.
            observation = self._base_observation(
                stderr="init_repo is internal-only; use reset(auto_init=True)",
                exit_code=2,
            )
        elif action.action_type == "git":
            observation = self._handle_git(action, timeout_s=timeout_s)
        elif action.action_type == "patch":
            patch_obs = self._handle_patch(action)
            if patch_obs.exit_code != 0:
                observation = patch_obs
            else:
                bench_action = GoPerfAction(
                    action_type="benchmarks",
                    bench_suite=".",
                    bench_mem_required=True,
                    bench_count=1,
                )
                bench_obs = self._handle_benchmarks(bench_action, timeout_s=timeout_s)
                bench_obs.metadata["patch_stdout"] = patch_obs.stdout
                bench_obs.metadata["patch_stderr"] = patch_obs.stderr
                bench_obs.metadata["patch_exit_code"] = patch_obs.exit_code
                observation = bench_obs
        elif action.action_type == "build_flags":
            observation = self._handle_build_flags(action)
        elif action.action_type == "benchmarks":
            observation = self._handle_benchmarks(action, timeout_s=timeout_s)
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
        observation.delta_baseline = reward.components.get("delta_vs_baseline")
        observation.delta_prev_best = reward.components.get("delta_vs_prev_best")

        # Auto-revert on failure, negative reward, or regression vs prev-best if a patch was applied
        if getattr(self._state, "last_patch_files", None):
            failed = observation.exit_code != 0
            negative = observation.reward is not None and observation.reward < 0
            regress_prev = False
            if observation.metadata.get("reward_components"):
                delta_prev = observation.metadata["reward_components"].get(
                    "delta_vs_prev_best"
                )
                if delta_prev is not None and delta_prev < 0:
                    regress_prev = True
            if failed or negative or regress_prev:
                files = self._state.last_patch_files
                for rel in files:
                    self._run_command(
                        ["git", "checkout", "--", rel], self._workspace_path(), None
                    )
                observation.stderr += "\n[auto-revert] reverted patch files"
                self._state.last_patch_files = None
            else:
                # update prev-best on successful improvement
                if self._state.current_metrics:
                    self._state.prev_best_metrics = self._state.current_metrics

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

        if action.action_type == "patch" and observation.exit_code == 0:
            self._state.last_action_type = "benchmarks"
        else:
            self._state.last_action_type = action.action_type
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

        # Lets say somehow is it not a repo with a existing git stuff, have some so called configs
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

        self._state.repo_name = src.name

        # Capture baseline metrics once per task
        bench_action = GoPerfAction(
            action_type="benchmarks",
            bench_suite=".",
            bench_mem_required=True,
            bench_count=1,
        )
        bench_obs = self._handle_benchmarks(bench_action, timeout_s=None)
        if bench_obs.bench_summary:
            self._state.baseline_metrics = bench_obs.bench_summary[0].get("metrics")
            self._state.current_metrics = self._state.baseline_metrics
            self._state.prev_best_metrics = self._state.baseline_metrics

        return self._base_observation(stdout=f"repo ready at {target}")

    def _workspace_path(self) -> Path:
        if not self._state.workspace_path:
            self._state.workspace_path = str(self._make_workspace("default"))
        return Path(self._state.workspace_path)

    def _base_observation(
        self, stdout: str = "", stderr: str = "", exit_code: int = 0
    ) -> GoPerfObservation:
        return GoPerfObservation(
            stdout=stdout, stderr=stderr, exit_code=exit_code, metadata={}
        )

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

        def _sanitize_apply_patch(diff: str) -> str:
            diff = diff.replace("\r\n", "\n").replace("\r", "\n")
            lines = diff.splitlines()
            out: list[str] = []
            current_go_file = False
            in_hunk = False
            saw_unified = False
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith("*** Begin Patch"):
                    out.append("*** Begin Patch")
                    continue
                if stripped.startswith("*** End Patch"):
                    out.append("*** End Patch")
                    continue
                if stripped.startswith("*** Update File: "):
                    path = stripped.split(": ", 1)[1].strip()
                    out.append(f"*** Update File: {path}")
                    current_go_file = path.endswith(".go")
                    in_hunk = False
                    continue
                if stripped.startswith("--- a/") or stripped.startswith("+++ b/"):
                    path = stripped.split("/", 1)[1].strip()
                    current_go_file = path.endswith(".go")
                    saw_unified = True
                    out.append(stripped.rstrip())
                    continue
                if stripped == "@@":
                    in_hunk = True
                    out.append("__HUNK__")
                    continue
                if stripped.startswith("@@"):
                    in_hunk = True
                    out.append(stripped.rstrip())
                    continue
                if in_hunk and not line.startswith((" ", "+", "-")) and line != "":
                    line = " " + line
                if saw_unified and in_hunk:
                    out.append(line.rstrip())
                    continue
                if (
                    line.startswith(" ")
                    or line.startswith("+")
                    or line.startswith("-")
                    or not line
                ):
                    if current_go_file and in_hunk:
                        prefix = line[0] if line and line[0] in {" ", "+", "-"} else ""
                        content = line[1:] if prefix else line
                        content_stripped = content.lstrip()
                        if content.startswith(" \t"):
                            content = content[1:]
                            content_stripped = content.lstrip()
                        if (
                            (not saw_unified)
                            and prefix == " "
                            and content_stripped.startswith(
                                ("func ", "package ", "import ")
                            )
                        ):
                            prefix = ""
                            content = content_stripped
                            content_stripped = content.lstrip()
                        if content_stripped == "}" and content.startswith("\t"):
                            fixed = prefix + "\t}"
                            out.append(fixed)
                            continue
                        if (not saw_unified) and content and not content[0].isspace():
                            if not content_stripped.startswith(
                                ("func ", "package ", "import ", "}", ")")
                            ):
                                fixed = prefix + "\t" + content
                                out.append(fixed)
                                continue
                        fixed = (prefix + content).rstrip()
                        out.append(fixed)
                        continue
                    fixed = line.rstrip()
                    out.append(fixed)
                    continue
                fixed = line.rstrip()
                if in_hunk and not fixed.startswith(("+", "-")):
                    if not fixed.startswith(" "):
                        fixed = " " + fixed
                out.append(fixed)
            if saw_unified:
                norm: list[str] = []
                in_h = False
                for line in out:
                    if line.startswith("diff --git"):
                        in_h = False
                        norm.append(line)
                        continue
                    if line.startswith("@@"):
                        in_h = True
                        norm.append(line)
                        continue
                    if line.startswith("--- ") or line.startswith("+++ "):
                        norm.append(line)
                        continue
                    if in_h:
                        if line != "" and not line.startswith((" ", "+", "-")):
                            line = " " + line
                        if line.startswith("+ \t"):
                            line = "+" + line[2:]
                        elif line.startswith("- \t"):
                            line = "-" + line[2:]
                        elif line.startswith("  \t"):
                            line = " \t" + line[3:]
                    norm.append(line)
                out = norm
            cleaned = "\n".join(out)
            if not cleaned.endswith("\n"):
                cleaned += "\n"
            return cleaned

        def _fill_empty_hunks(diff: str, workspace: Path) -> str:
            lines = diff.splitlines()
            out: list[str] = []
            current_file: Path | None = None
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.startswith("+++ b/"):
                    current_file = workspace / line.split("+++ b/", 1)[1].strip()
                    out.append(line)
                    i += 1
                    continue
                if line == "__HUNK__":
                    # collect hunk lines until next hunk or file header
                    hunk_lines: list[str] = []
                    j = i + 1
                    while j < len(lines):
                        nxt = lines[j]
                        if (
                            nxt.startswith("@@")
                            or nxt.startswith("diff --git")
                            or nxt.startswith("--- a/")
                            or nxt.startswith("+++ b/")
                        ):
                            break
                        if nxt and not nxt.startswith((" ", "+", "-")):
                            nxt = " " + nxt
                        hunk_lines.append(nxt)
                        j += 1
                    old_count = sum(
                        1 for l in hunk_lines if l.startswith(" ") or l.startswith("-")
                    )
                    new_count = sum(
                        1 for l in hunk_lines if l.startswith(" ") or l.startswith("+")
                    )
                    start_line = 1
                    if current_file and current_file.exists():
                        file_lines = current_file.read_text().splitlines()
                        context = next(
                            (l for l in hunk_lines if l.startswith(" ")), None
                        )
                        if context:
                            ctx = context[1:]
                            for idx, fl in enumerate(file_lines, 1):
                                if fl == ctx:
                                    start_line = idx
                                    break
                    out.append(
                        f"@@ -{start_line},{max(old_count, 1)} +{start_line},{max(new_count, 1)} @@"
                    )
                    out.extend(hunk_lines)
                    i = j
                    continue
                out.append(line)
                i += 1
            return "\n".join(out) + ("\n" if diff.endswith("\n") else "")

        workspace = self._workspace_path()
        if not action.patch_diff:
            return self._base_observation(stderr="patch_diff is required", exit_code=2)
        base_patch = _sanitize_apply_patch(action.patch_diff)
        if "__HUNK__" in base_patch:
            base_patch = _fill_empty_hunks(base_patch, workspace)
        base_strip = base_patch.lstrip()
        if not (base_strip.startswith("diff --git") or base_strip.startswith("--- ")):
            return self._base_observation(
                stderr="patch_diff must be unified diff format",
                exit_code=2,
            )
        try:
            import os
        except Exception as exc:
            return self._base_observation(stderr=str(exc), exit_code=2)

        cwd = Path.cwd()
        try:
            os.chdir(workspace)
            unified = base_patch
            tmp_patch = workspace / ".openenv_patch_check.patch"
            tmp_patch.write_text(unified, encoding="utf-8")
            _, stderr, code = self._run_command(
                [
                    "git",
                    "apply",
                    "--check",
                    "--recount",
                    "--unidiff-zero",
                    "--whitespace=nowarn",
                    str(tmp_patch),
                ],
                workspace,
                None,
            )
            if code != 0:
                return self._base_observation(
                    stderr=stderr or "patch check failed",
                    exit_code=2,
                )
            _, stderr, code = self._run_command(
                [
                    "git",
                    "apply",
                    "--recount",
                    "--unidiff-zero",
                    "--whitespace=nowarn",
                    str(tmp_patch),
                ],
                workspace,
                None,
            )
            if code != 0:
                _, stderr3, code3 = self._run_command(
                    [
                        "git",
                        "apply",
                        "--3way",
                        "--recount",
                        "--unidiff-zero",
                        "--whitespace=nowarn",
                        str(tmp_patch),
                    ],
                    workspace,
                    None,
                )
                if code3 != 0:
                    return self._base_observation(
                        stderr=(stderr3 or stderr or "patch apply failed"),
                        exit_code=2,
                    )
        except Exception as exc:
            return self._base_observation(stderr=str(exc), exit_code=2)
        finally:
            try:
                try:
                    tmp_patch.unlink(missing_ok=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
                os.chdir(cwd)
            except Exception:
                pass

        # Extract file paths from unified diff
        patch_files = []
        for line in unified.splitlines():
            if line.startswith("+++ b/"):
                patch_files.append(line.split("+++ b/", 1)[1].strip())

        success = True
        files = patch_files
        obs = self._base_observation(
            stdout="applied patch diff" if success else "",
            stderr="" if success else "patch failed",
            exit_code=0 if success else 1,
        )
        obs.metadata["patch_files"] = files
        if success:
            self._state.last_patch_files = files
        return obs

    def _handle_build_flags(self, action: GoPerfAction):
        self._state.build_flags = action.build_flags or []
        return self._base_observation(stdout="build flags set")

    def _handle_benchmarks(self, action: GoPerfAction, timeout_s: Optional[float]):
        workspace = self._workspace_path()
        bench = action.bench_suite or "."
        cmd = ["go", "test", "-run=^$", f"-bench={bench}", "./..."]
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
            r"^(Benchmark\S+)\s+\d+\s+([\d\.]+)\s+ns/op"
            r"(?:\s+([\d\.]+)\s+B/op)?"
            r"(?:\s+([\d\.]+)\s+allocs/op)?"
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

from typing import List, Literal
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


class GoPerfAction(Action):
    # Common stuff
    action_type: Literal[
        "git",
        "benchmarks",
        "build_flags",
        "escape_analysis",
        "perf",
        "patch",
        "init_repo",
    ]

    # Git specific stuff
    git_op: str | None = None
    git_repo_name: str | None = None
    git_target_dir: str | None = None
    git_workspace: str | None = None
    git_args: List[str] | None = None

    # Benchmarking specific stuff
    bench_suite: str | None = None
    bench_mem_required: bool | None = None
    bench_time: int | None = None
    bench_count: int | None = None
    bench_stat: bool | None = None
    bench_file_name: str | None = None
    bench_timeout: int | None = None  # time in seconds
    # TODO: Check if we can somehow automate the profiling data and compare


    # Build flags
    build_flags: List[str] | None = None

    # Escape analysis
    escape_target: str | None = None
    escape_flags: List[str] | None = None
    escape_output_file: str | None = None

    # Perf analysis
    perf_mode: Literal["stat", "mem", "c2c"] | None = None
    perf_bench: str | None = None
    perf_args: List[str] | None = None
    perf_output_file: str | None = None

    # Patch
    patch_file: str | None = None
    patch_diff: str | None = None

    # Init repo
    repo_path: str | None = None
    repo_copy: bool | None = None
    repo_init_if_missing: bool | None = None


class GoPerfObservation(Observation):
    # Common results
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    metadata: dict = Field(default_factory=dict)

    # Workspace context
    repo_name: str | None = None
    repo_worktree_path: str | None = None
    repo_git_status: dict | None = None

    # Task metadata
    task_id: str | None = None
    task_goal: str | None = None
    task_step_count: int | None = None
    task_remaining_budget: int | None = None
    task_build_flags: List[str] | None = None
    delta_baseline: float | None = None
    delta_prev_best: float | None = None

    # Benchmark results
    bench_summary: List[dict] | None = None
    benchstat_summary: dict | None = None


    # Escape analysis
    escape_summary: List[dict] | None = None
    escape_count_total: int | None = None

    # Perf
    perf_summary: dict | None = None


class GoPerfState(State):
    workspace_path: str | None = None
    repo_name: str | None = None
    repo_revision: str | None = None
    worktree_id: str | None = None

    baseline_metrics: dict | None = None
    current_metrics: dict | None = None
    prev_best_metrics: dict | None = None
    patch_cycles: int | None = None

    last_action: dict | None = None
    last_action_type: str | None = None
    action_history: List[dict] | None = None

    task_config: dict | None = None
    reward_trace: List[dict] | None = None
    budget_remaining: int | None = None

    build_flags: List[str] | None = None
    perf_artifacts: List[str] | None = None
    escape_artifacts: List[str] | None = None


class GoPerfReward(BaseModel):
    score: float = 0.0
    components: dict = Field(default_factory=dict)

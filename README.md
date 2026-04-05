# GoPerfLab (OpenEnv)

GoPerfLab is a real-world environment for Go performance tuning with typed
OpenEnv models, deterministic graders, and three task tiers (easy/medium/hard).
One of the important reasons to build this was to add this in my developer workflow.
As a engineer working on trading/ticketing bots in golang, i tends to be hard and
time consuming to do perfing, benching, profiling. Hence ended up writing this to asist me.

## Environment overview

GoPerfLab simulates the developer workflow of diagnosing and improving Golang
performance. Agents interact by running benchmarks, performing escape/perf
analysis, and applying patches or build flags. Rewards are based on performance
improvement and correctness.

Tasks:
- Easy: baseline cleanup (reduce allocations, simple refactors)
- Medium: common performance patterns across two benchmarks
- Hard: escape analysis + perf tooling with stricter targets

The server seeds each episode workspace with the bundled Go repo at
`go-bench-repo/` and initializes a local git history, so agents can run git
patches and benchmarks immediately.

## Episode structure

Each task is an episode:

1) `reset(auto_init=True)` creates a workspace, initializes git, and runs a baseline benchmark.
2) `step(patch)` applies a patch and **auto-runs benchmarks** in the environment.
3) The environment ends the episode when grade ≥ threshold **and** the minimum
   patch cycles are met, or when budget is exhausted.

## Reward & grading

Rewards combine weighted improvements across `ns/op`, `B/op`, and `allocs/op`.
We use a multi-metric gate and a regression penalty:

- **Multi-metric gate**: at least 2 metrics must improve for full score.
- **Regression penalty**: >10% regression in any metric incurs a strong penalty.
- **Task-specific weights**:
  - Easy: allocations-heavy
  - Medium: ns/op + allocs
  - Hard: ns/op dominant

## Determinism & reproducibility

- Benchmarks run with `-count=1` by default (configurable via action).
- Performance can vary by CPU load; run on an idle machine for best stability.
- Docker image pins dependencies to keep results consistent across runs.

## Action space (GoPerfAction)

- `action_type`: `git` | `benchmarks` | `build_flags` | `escape_analysis` | `perf` | `patch`
- Git: `git_op`, `git_repo_name`, `git_target_dir`, `git_workspace`, `git_args`
- Benchmarks: `bench_suite`, `bench_mem_required`, `bench_time`, `bench_count`, `bench_stat`,
  `bench_file_name`, `bench_timeout`
- Build flags: `build_flags`
- Escape analysis: `escape_target`, `escape_flags`, `escape_output_file`
- Perf: `perf_mode`, `perf_bench`, `perf_args`, `perf_output_file`
- Patch: `patch_file`, `patch_diff`

## Observation space (GoPerfObservation)

- Execution: `stdout`, `stderr`, `exit_code`
- Repo context: `repo_name`, `repo_worktree_path`, `repo_git_status`
- Task context: `task_id`, `task_goal`, `task_step_count`, `task_remaining_budget`, `task_build_flags`
- Benchmarks: `bench_summary`, `benchstat_summary`
- Escape analysis: `escape_summary`, `escape_count_total`
- Perf: `perf_summary`
- Reward/grade: `reward` and `metadata.reward_components`, `metadata.grade`

## Quickstart

Install deps:
```
uv pip install -r requirements.txt
```

Validate OpenEnv:
```
./scripts/validate_openenv.sh
```

Run server:
```
uv run server.app
```

Health check:
```
curl -s http://localhost:8000/health
```

Smoke test (client):
```
uv run python - <<'PY'
from client import GoPerfEnv
from models import GoPerfAction

env = GoPerfEnv(base_url="http://localhost:8000").sync()
with env:
    result = env.reset(task_id="easy")
    result = env.step(action)
    print(result.observation.metadata.get("grade"))
PY
```

## Docker

```
docker build -f server/Dockerfile .
docker run -p 8000:8000 <image>
```

## Baseline inference (required format)

```
HF_TOKEN=... API_BASE_URL=... MODEL_NAME=... OPENENV_BASE_URL=http://localhost:8000 \
  uv run python inference.py
```

## Pre-submission validator

```
./scripts/validate-submission.sh https://your-space.hf.space .
```

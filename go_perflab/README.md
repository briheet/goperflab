# GoPerfLab (OpenEnv)

GoPerfLab is a real-world environment for Go performance tuning with typed
OpenEnv models, deterministic graders, and three task tiers (easy/medium/hard).

## Environment overview

GoPerfLab simulates the developer workflow of diagnosing and improving Go
performance. Agents interact by running tests/benchmarks, performing escape/perf
analysis, and applying patches or build flags. Rewards are based on performance
improvement and correctness.

Tasks:
- Easy: baseline cleanup (reduce allocations, simple refactors)
- Medium: common performance patterns across two benchmarks
- Hard: escape analysis + perf tooling with stricter targets

The server seeds each episode workspace with the bundled Go repo at
`go-bench-repo/` and initializes a local git history, so agents can run git,
tests, and benchmarks immediately.

## Action space (GoPerfAction)

- `action_type`: `git` | `benchmarks` | `tests` | `build_flags` | `escape_analysis` | `perf` | `patch`
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

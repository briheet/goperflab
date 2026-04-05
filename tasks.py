from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    title: str
    goal: str
    target_speedup: float
    max_speedup: float
    metrics_weights: dict
    budget: int
    min_patches: int


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        title="Baseline cleanup",
        goal="Reduce allocations with simple refactors while keeping output stable.",
        target_speedup=1.15,
        max_speedup=2.0,
        metrics_weights={"ns/op": 0.2, "B/op": 0.2, "allocs/op": 0.6},
        budget=25,
        min_patches=2,
    ),
    "medium": TaskConfig(
        task_id="medium",
        title="Common patterns",
        goal="Apply multiple goperf.dev patterns to improve two benchmarks.",
        target_speedup=1.25,
        max_speedup=2.5,
        metrics_weights={"ns/op": 0.5, "B/op": 0.1, "allocs/op": 0.4},
        budget=35,
        min_patches=2,
    ),
    "hard": TaskConfig(
        task_id="hard",
        title="Deep tuning",
        goal="Combine escape analysis and perf tooling to improve hot paths.",
        target_speedup=1.35,
        max_speedup=3.0,
        metrics_weights={"ns/op": 0.8, "B/op": 0.1, "allocs/op": 0.1},
        budget=45,
        min_patches=2,
    ),
}


def get_task(task_id: str | None) -> TaskConfig:
    if task_id and task_id in TASKS:
        return TASKS[task_id]
    return TASKS["easy"]


def list_tasks() -> list[TaskConfig]:
    return list(TASKS.values())

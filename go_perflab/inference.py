import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import GoPerfEnv
from models import GoPerfAction
from tasks import list_tasks, TaskConfig


OPEN_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
OPEN_API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
REPO_PATH = os.getenv("REPO_PATH", "go-bench-repo")
MAX_STEPS = 4
TEMPERATURE = 0.2
MAX_TOKENS = 200


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a performance engineer interacting with a Go performance tuning environment.
    Return exactly one JSON object representing a GoPerfAction.
    Prefer safe diagnostic actions before code edits.
    Required field: action_type.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(task_goal: str, last_obs: dict, step: int) -> str:
    return textwrap.dedent(
        f"""
        Task goal: {task_goal}
        Step: {step}
        Last observation summary:
        {json.dumps(last_obs, ensure_ascii=True)[:1000]}
        Return a JSON GoPerfAction.
        """
    ).strip()


def get_model_action(
    client: OpenAI, task_goal: str, last_obs: dict, step: int
) -> GoPerfAction:
    prompt = build_user_prompt(task_goal, last_obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = (completion.choices[0].message.content or "").strip()
        data = json.loads(content)
        return GoPerfAction.model_validate(data)
    except Exception:
        return GoPerfAction(action_type="tests", test_suite="./...", test_verbose=False)


def run_task(client: OpenAI, env: GoPerfEnv, task: TaskConfig) -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task.task_id, env="goperflab", model=MODEL_NAME)

    try:
        result = env.reset(task_id=task.task_id)

        init_action = GoPerfAction(
            action_type="init_repo",
            repo_path=REPO_PATH,
            repo_copy=True,
            repo_init_if_missing=True,
        )
        result = env.step(init_action)
        steps_taken = 1
        reward = float(result.reward or 0.0)
        rewards.append(reward)
        log_step(
            step=steps_taken,
            action=json.dumps(init_action.model_dump(exclude_none=True), ensure_ascii=True),
            reward=reward,
            done=bool(result.done),
            error=None,
        )

        last_obs = result.observation.model_dump()

        for step in range(steps_taken + 1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, task.goal, last_obs, step)
            action_str = json.dumps(action.model_dump(exclude_none=True), ensure_ascii=True)

            result = env.step(action)
            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step
            last_obs = result.observation.model_dump()

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            grade = result.observation.metadata.get("grade", {})
            if grade.get("score", 0.0) >= 1.0:
                success = True
            if done:
                break
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    api_key = OPEN_API_KEY
    if not api_key and OPEN_API_BASE_URL.startswith("http://localhost"):
        api_key = "dummy"

    client = OpenAI(base_url=OPEN_API_BASE_URL, api_key=api_key)
    env = GoPerfEnv(base_url=ENV_BASE_URL).sync()

    with env:
        for task in list_tasks():
            run_task(client, env, task)


if __name__ == "__main__":
    main()

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
TARGET_FILE = os.getenv("TARGET_FILE", "string_concatenation/string.go")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
LOG_BENCH_SUMMARY = os.getenv("LOG_BENCH_SUMMARY", "1") == "1"
TEMPERATURE = 0.2
MAX_TOKENS = 300


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a golang performance engineer. Your job is to write high quality optimised code.
    The code you write does not matter in code quality much as opposed to performance.
    Performance is your first priority. You will be given a bunch of benchmarks and their outputs.
    You have to analyse them, understand the code for it and give back pactches and apply them in code.
    Your end goal should be very very highly optimised code with very good overview.
    Performance is your first priority.
    Return exactly one JSON object for GoPerfAction.
    On the first model step, you MUST return:
    {"action_type":"patch","patch_diff":"<unified diff>"}
    The diff must apply cleanly to TARGET_FILE.
    Use unified diff format:
      diff --git a/<path> b/<path>
      --- a/<path>
      +++ b/<path>
      @@ -a,b +c,d @@
      - old line
      + new line
    Do NOT add extra leading spaces before tabs in code lines (avoid " \\t...").
    For context lines, include the leading space only for diff markers; the code itself should start exactly as in the file.
    Preserve tabs exactly as shown in the file (Go uses tabs). Do not replace tabs with spaces.
    Copy context lines exactly from the file so the patch applies cleanly.
    Do not modify indentation or omit leading tabs from any context line.
    Keep the existing import block structure. If you add strings, keep fmt and add strings alongside it.
    If you use strings.Builder, ensure strings is imported and actually used.
    Do not mix result and builder variables in the same function; use one consistently.
    Do not add any extra text outside the diff.
    Do not use code fences. patch_diff must be a JSON string (escape newlines as \\n).
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


def build_user_prompt(
    task_goal: str, last_obs: dict, step: int, file_content: str
) -> str:
    return textwrap.dedent(
        f"""
        Task goal: {task_goal}
        Step: {step}
        Target file path: {TARGET_FILE}
        Target file content:
        {file_content}
        Last observation summary:
        {json.dumps(last_obs, ensure_ascii=True)[:800]}
        Return a JSON GoPerfAction.
        """
    ).strip()


def get_model_action(
    client: OpenAI, task_goal: str, last_obs: dict, step: int, file_content: str
) -> GoPerfAction:
    prompt = build_user_prompt(task_goal, last_obs, step, file_content)
    request_kwargs = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
    }
    if MODEL_NAME.startswith("gpt-5"):
        request_kwargs["max_completion_tokens"] = MAX_TOKENS
    else:
        request_kwargs["max_tokens"] = MAX_TOKENS
    completion = client.chat.completions.create(**request_kwargs)
    content = (completion.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("Empty model response")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        repaired = (
            content.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
        )
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            start = repaired.find("{")
            end = repaired.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(repaired[start : end + 1])
                except json.JSONDecodeError as exc2:
                    # Last-resort: extract a diff from the raw content.
                    for marker in ("diff --git", "*** Begin Patch"):
                        idx = content.find(marker)
                        if idx != -1:
                            data = {
                                "action_type": "patch",
                                "patch_diff": content[idx:].strip(),
                            }
                            break
                    else:
                        raise ValueError(
                            f"json_parse_error: {exc2}; raw={content[:300]!r}"
                        ) from exc2
            else:
                raise ValueError(
                    f"json_parse_error: {exc}; raw={content[:300]!r}"
                ) from exc
    action = GoPerfAction.model_validate(data)
    if action.action_type == "patch" and action.patch_diff:
        inner = action.patch_diff.strip()
        if inner.startswith("{") and "patch_diff" in inner:
            try:
                inner_obj = json.loads(inner)
                if isinstance(inner_obj, dict) and inner_obj.get("patch_diff"):
                    action.patch_diff = inner_obj["patch_diff"]
            except json.JSONDecodeError:
                pass
        # Unescape common newline-escaped diffs.
        if "\\n" in action.patch_diff and "\n" not in action.patch_diff:
            action.patch_diff = (
                action.patch_diff.replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace("\\n", "\n")
            )
        if "\n@@\n" in action.patch_diff or action.patch_diff.strip().startswith("@@"):
            raise ValueError(
                "patch_validation_error: hunk header must include ranges (no bare @@)"
            )
        # Validate basic consistency for strings.Builder usage.
        patch_body = action.patch_diff
        if "strings.Builder" in patch_body and '"strings"' not in patch_body:
            raise ValueError(
                "patch_validation_error: strings.Builder used without importing strings"
            )
    return action


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
        error = (
            result.observation.stderr[:200]
            if result.observation.exit_code != 0
            else None
        )
        log_step(
            step=steps_taken,
            action=json.dumps(
                init_action.model_dump(exclude_none=True), ensure_ascii=True
            ),
            reward=reward,
            done=bool(result.done),
            error=error,
        )

        baseline_action = GoPerfAction(
            action_type="benchmarks",
            bench_suite=".",
            bench_mem_required=True,
            bench_count=1,
        )
        result = env.step(baseline_action)
        steps_taken += 1
        reward = float(result.reward or 0.0)
        rewards.append(reward)
        error = (
            result.observation.stderr[:200]
            if result.observation.exit_code != 0
            else None
        )
        baseline_payload = baseline_action.model_dump(exclude_none=True)
        reward_components = result.observation.metadata.get("reward_components", {})
        raw_delta_base = (
            result.observation.delta_baseline
            if result.observation.delta_baseline is not None
            else reward_components.get("delta_vs_baseline")
        )
        raw_delta_prev = (
            result.observation.delta_prev_best
            if result.observation.delta_prev_best is not None
            else reward_components.get("delta_vs_prev_best")
        )
        if raw_delta_base is not None:
            baseline_payload["_delta_baseline_pct"] = raw_delta_base * 100.0
        if raw_delta_prev is not None:
            baseline_payload["_delta_prev_best_pct"] = raw_delta_prev * 100.0
        if LOG_BENCH_SUMMARY:
            baseline_payload["_bench_summary"] = result.observation.bench_summary
        log_step(
            step=steps_taken,
            action=json.dumps(baseline_payload, ensure_ascii=True),
            reward=reward,
            done=bool(result.done),
            error=error,
        )

        workspace_path = result.observation.repo_worktree_path or REPO_PATH
        try:
            with open(
                os.path.join(workspace_path, TARGET_FILE), "r", encoding="utf-8"
            ) as f:
                file_content = f.read()
        except Exception:
            file_content = ""

        last_obs = result.observation.model_dump()

        patch_attempts = 0
        final_action: GoPerfAction | None = None
        last_patch_step = None
        next_required_action = "patch"
        for step in range(steps_taken + 1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                if next_required_action == "benchmarks":
                    action = GoPerfAction(
                        action_type="benchmarks",
                        bench_suite=".",
                        bench_mem_required=True,
                        bench_count=1,
                    )
                else:
                    action = get_model_action(
                        client, task.goal, last_obs, step, file_content
                    )
            except Exception as exc:
                log_step(
                    step=step,
                    action="model_error",
                    reward=0.0,
                    done=False,
                    error=str(exc),
                )
                err = str(exc)
                if (
                    "Empty model response" in err or "patch_validation_error" in err
                ) and step < MAX_STEPS:
                    last_obs["last_patch_error"] = str(exc)
                    continue
                break
            if step == steps_taken + 1 and next_required_action == "patch" and action.action_type != "patch":
                raise RuntimeError("Expected patch action on first LLM step.")
            if action.action_type not in {"patch", "benchmarks"}:
                raise RuntimeError(f"Unsupported action_type: {action.action_type}")

            action_payload = action.model_dump(exclude_none=True)
            result = env.step(action)
            reward = float(result.reward or 0.0)
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step
            last_obs = result.observation.model_dump()

            error = (
                result.observation.stderr[:200]
                if result.observation.exit_code != 0
                else None
            )
            reward_components = result.observation.metadata.get("reward_components", {})
            delta_base = result.observation.delta_baseline
            delta_prev = result.observation.delta_prev_best
            if action.action_type == "benchmarks":
                raw_delta_base = (
                    delta_base
                    if delta_base is not None
                    else reward_components.get("delta_vs_baseline")
                )
                raw_delta_prev = (
                    delta_prev
                    if delta_prev is not None
                    else reward_components.get("delta_vs_prev_best")
                )
                if raw_delta_base is not None:
                    action_payload["_delta_baseline_pct"] = raw_delta_base * 100.0
                if raw_delta_prev is not None:
                    action_payload["_delta_prev_best_pct"] = raw_delta_prev * 100.0
                if LOG_BENCH_SUMMARY:
                    action_payload["_bench_summary"] = result.observation.bench_summary
            log_step(
                step=step,
                action=json.dumps(action_payload, ensure_ascii=True),
                reward=reward,
                done=done,
                error=error,
            )

            grade = result.observation.metadata.get("grade", {})
            if grade.get("score", 0.0) >= 1.0:
                success = True
            if action.action_type == "patch":
                patch_attempts += 1
                last_patch_step = step
                if result.observation.exit_code != 0 and patch_attempts < 2:
                    # Retry once with error context if patch failed.
                    last_obs["last_patch_error"] = result.observation.stderr[:400]
                    continue
                next_required_action = "benchmarks"
            elif action.action_type == "benchmarks":
                next_required_action = "patch"
            if done:
                break

        if not result.done:
            if last_patch_step is None or steps_taken <= last_patch_step:
                final_action = GoPerfAction(
                    action_type="benchmarks",
                    bench_suite=".",
                    bench_mem_required=True,
                    bench_count=1,
                )
                result = env.step(final_action)
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                steps_taken += 1
                error = (
                    result.observation.stderr[:200]
                    if result.observation.exit_code != 0
                    else None
                )
            if final_action is not None:
                final_payload = final_action.model_dump(exclude_none=True)
                reward_components = result.observation.metadata.get("reward_components", {})
                raw_delta_base = (
                    result.observation.delta_baseline
                    if result.observation.delta_baseline is not None
                    else reward_components.get("delta_vs_baseline")
                )
                raw_delta_prev = (
                    result.observation.delta_prev_best
                    if result.observation.delta_prev_best is not None
                    else reward_components.get("delta_vs_prev_best")
                )
                if raw_delta_base is not None:
                    final_payload["_delta_baseline_pct"] = raw_delta_base * 100.0
                if raw_delta_prev is not None:
                    final_payload["_delta_prev_best_pct"] = raw_delta_prev * 100.0
                if LOG_BENCH_SUMMARY:
                    final_payload["_bench_summary"] = result.observation.bench_summary
                log_step(
                    step=steps_taken,
                    action=json.dumps(final_payload, ensure_ascii=True),
                    reward=reward,
                    done=bool(result.done),
                    error=error,
                )

            grade = result.observation.metadata.get("grade", {})
            if grade.get("score", 0.0) >= 1.0:
                success = True
            if reward > 0:
                success = True
    finally:
        # Final success check based on last observation.
        try:
            if result is not None:
                grade = result.observation.metadata.get("grade", {})
                if grade.get("score", 0.0) >= 1.0:
                    success = True
                if (result.reward or 0.0) > 0:
                    success = True
        except Exception:
            pass
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

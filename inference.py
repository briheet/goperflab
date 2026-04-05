import json
import os
import textwrap
import base64
import io
import tarfile
import tempfile
from typing import List, Optional

from openai import OpenAI

from client import GoPerfEnv
from models import GoPerfAction
from tasks import list_tasks, TaskConfig


OPEN_API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")  # Required by submission
OPEN_API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")  # Required by submission
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # Required by submission
ENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")  # OpenEnv server
REPO_PATH = os.getenv("REPO_PATH", "go-bench-repo")  # repo to init
TARGET_FILE = os.getenv(
    "TARGET_FILE", "string_concatenation/string.go"
)  # file to patch
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))  # max steps per task
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
    The environment enforces that a patch should be followed by a benchmarks action.
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
    Never change the package/module declaration line at the top of the file.
    Do not add new functions or new files. Only modify existing function bodies.
    Do not emit hunks that add a new file (reject @@ -0,0 or file creation diffs).
    Do not add any extra text outside the diff.
    Do not use code fences. patch_diff must be a JSON string (escape newlines as \\n).
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    # Required by submission format.
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    # Required by submission format.
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    # Required by submission format.
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(
    task_goal: str, last_obs: dict, step: int, file_content: str
) -> str:
    # Provide task + file context to the model.
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
    # Build user prompt
    prompt = build_user_prompt(task_goal, last_obs, step, file_content)

    # Build request to OpenAI-compatible API.
    request_kwargs = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
    }

    # Models from gpt-5 require max_completion_tokens.
    request_kwargs["max_completion_tokens"] = MAX_TOKENS

    content = ""
    for attempt in range(3):
        completion = client.chat.completions.create(**request_kwargs)
        content = (completion.choices[0].message.content or "").strip()
        if content:
            break
        if attempt < 2:
            import time

            time.sleep(0.5 * (2**attempt))
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
        if "@@ -0,0" in action.patch_diff or "+0,0 @@" in action.patch_diff:
            raise ValueError(
                "patch_validation_error: empty hunks are not allowed"
            )
        has_change = False
        for line in action.patch_diff.splitlines():
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                has_change = True
                break
        if not has_change:
            raise ValueError("patch_validation_error: no-op diff")
        # Validate basic consistency for strings.Builder usage.
        patch_body = action.patch_diff
        if "strings.Builder" in patch_body and '"strings"' not in patch_body:
            raise ValueError(
                "patch_validation_error: strings.Builder used without importing strings"
            )
    return action


def run_task(client: OpenAI, env: GoPerfEnv, task: TaskConfig) -> None:
    # One full episode per task: reset -> init_repo -> baseline -> model actions.
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task.task_id, env="goperflab", model=MODEL_NAME)

    try:
        # Reset environment for this task (auto-init repo + baseline in env).
        result = env.reset(
            task_id=task.task_id,
            auto_init=True,
            repo_path=REPO_PATH,
            repo_copy=True,
            repo_init_if_missing=True,
        )
        steps_taken = 0

        # Load target file content for the model prompt.
        workspace_path = None
        archive_b64 = result.observation.metadata.get("repo_archive_b64")
        if archive_b64:
            tmp_dir = tempfile.mkdtemp(prefix="goperflab_repo_")
            data = base64.b64decode(archive_b64)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(tmp_dir)
            workspace_path = os.path.join(tmp_dir, "repo")
        if not workspace_path:
            if not result.observation.repo_worktree_path:
                raise RuntimeError("repo_worktree_path missing after auto_init")
            workspace_path = result.observation.repo_worktree_path

        try:
            with open(
                os.path.join(workspace_path, TARGET_FILE), "r", encoding="utf-8"
            ) as f:
                file_content = f.read()
        except Exception:
            file_content = ""

        last_obs = result.observation.model_dump()

        # Main interaction loop: ask model -> step env -> log.
        for step in range(steps_taken + 1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                # Ask the model for the next action.
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
            if action.action_type not in {"patch", "benchmarks"}:
                raise RuntimeError(f"Unsupported action_type: {action.action_type}")

            # Send action to env and log the result.
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
            log_step(
                step=step,
                action=json.dumps(action_payload, ensure_ascii=True),
                reward=reward,
                done=done,
                error=error,
            )

            # Refresh repo snapshot if provided by env.
            archive_b64 = result.observation.metadata.get("repo_archive_b64")
            if archive_b64:
                tmp_dir = tempfile.mkdtemp(prefix="goperflab_repo_")
                data = base64.b64decode(archive_b64)
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                    tar.extractall(tmp_dir)
                workspace_path = os.path.join(tmp_dir, "repo")

            # Refresh file content so the model sees latest code.
            try:
                with open(
                    os.path.join(workspace_path, TARGET_FILE), "r", encoding="utf-8"
                ) as f:
                    file_content = f.read()
            except Exception:
                file_content = ""

            if done:
                break

        # Success is determined by env metadata or done flag.
        grade = result.observation.metadata.get("grade", {})
        if grade.get("score", 0.0) >= 1.0:
            success = True
        elif bool(result.done):
            success = True
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    api_key = OPEN_API_KEY
    if not api_key:
        raise RuntimeError("HF_TOKEN is required by submission instructions")

    client = OpenAI(base_url=OPEN_API_BASE_URL, api_key=api_key)
    env = GoPerfEnv(base_url=ENV_BASE_URL).sync()

    with env:
        for task in list_tasks():
            run_task(client, env, task)


if __name__ == "__main__":
    main()

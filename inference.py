"""
Inference Script — Reddit Mod Bot RL Environment
=================================================
Environment variables:
    LOCAL_IMAGE_NAME  Docker image name (triggers async openenv-core path).
    HF_TOKEN          Hugging Face / API key (alias: OPENAI_API_KEY, API_KEY).
    API_BASE_URL      LLM endpoint (default: HF inference router).
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-72B-Instruct).

Runs all three task levels and emits [START]/[STEP]/[END] lines per task.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure the parent of this file is on sys.path so reddit_mod_env is importable
# whether the package is installed or not.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import (
    OpenAI,
    APIStatusError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
)

from reddit_mod_env.client import RedditModEnv
from reddit_mod_env.models import ModAction, ModObservation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "reddit_mod_env"
SEED = 42
SUCCESS_THRESHOLD = 0.5

# (task_level, task_name, num_posts)
TASKS = [
    (1, "spam-detection",      10),
    (2, "rule-classification",  8),
    (3, "context-judgment",     5),
]

_ALLOWED_ACTIONS = {
    1: ["approve", "remove"],
    2: ["approve", "remove", "warn"],
    3: ["approve", "remove", "warn", "temp_ban", "perma_ban", "escalate_to_senior_mod"],
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI moderator for r/science. Review the post and choose a moderation action.

    SUBREDDIT RULES:
    1. Stay On Topic — Posts must relate to scientific research
    2. No Personal Attacks — Debate the science, not the person
    3. No Misinformation — Claims must have credible sources
    4. No Self-Promotion/Spam — No advertising or excessive self-promotion
    5. Use Correct Flair — [Research], [News], [Discussion], [AMA]
    6. No Low-Effort Content — Must contribute meaningfully

    Respond with a single JSON object and nothing else:
    {
      "action_type": "<one of the allowed actions>",
      "rule_cited": <integer rule number, or null if approving>,
      "reason": "<one sentence explanation>",
      "ban_duration_days": <integer, only required for temp_ban, else null>
    }
""").strip()


def _format_user_prompt(obs: ModObservation, task_level: int, history: List[str]) -> str:
    post = obs.current_post
    if post is None:
        return "No post to review."

    allowed = _ALLOWED_ACTIONS.get(task_level, ["approve", "remove"])

    rules_text = "\n".join(
        f"  {r.rule_number}. {r.title}: {r.description}"
        for r in post.subreddit.rules
    ) if post.subreddit else "  (unavailable)"

    reports_text = (
        "\n".join(f"  - {rep.reason} ({rep.count}x)" for rep in post.reports)
        if post.reports else "  None"
    )

    a = post.author
    flags = (
        (" [REPEAT OFFENDER]" if a.is_repeat_offender else "")
        + (" [APPROVED CONTRIBUTOR]" if a.is_approved_contributor else "")
    )

    thread_ctx_block = ""
    if post.thread_context:
        thread_ctx_block = "\nTHREAD CONTEXT:\n" + "\n".join(
            f"  {c}" for c in post.thread_context
        )

    history_block = (
        "\nRecent history:\n" + "\n".join(history[-3:])
        if history else ""
    )

    return textwrap.dedent(f"""
        Task {task_level} | Allowed actions: {", ".join(allowed)}

        POST — {post.subreddit.name if post.subreddit else "r/science"}
        Title : {post.title or "(no title)"}
        Body  : {post.body}
        Flair : {post.flair or "None"} | Score: {post.score} | Comments: {post.num_comments}

        AUTHOR: u/{a.username}{flags}
          Age: {a.account_age_days}d | Karma: {a.karma}
          Warnings: {a.prior_warnings} | Removals: {a.prior_removals}

        REPORTS:
        {reports_text}

        RULES:
        {rules_text}
        {thread_ctx_block}
        {history_block}
        Respond with JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# Logging helpers (exact format required)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error or "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f}"
        f" done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps}"
        f" score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

async def _get_model_action(
    client: OpenAI,
    obs: ModObservation,
    task_level: int,
    history: List[str],
) -> Tuple[ModAction, str, Optional[str]]:
    """Call the LLM and parse its response into a ModAction. Never raises.

    Retries up to 3 times on rate-limit / server errors (429, 5xx) with
    exponential backoff: 5s, 10s, 20s.
    """
    user_prompt = _format_user_prompt(obs, task_level, history)
    error: Optional[str] = None
    data: dict = {}

    _RETRY_DELAYS = [5, 10, 20]
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            await asyncio.sleep(delay)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=200,
                stream=False,
            )
            raw = (completion.choices[0].message.content or "").strip()
            data = json.loads(raw)
            error = None
            break  # success
        except json.JSONDecodeError as exc:
            error = f"json_parse_error:{exc}"
            break  # bad JSON from model — no point retrying
        except Exception as exc:
            is_retryable = (
                isinstance(exc, (APIConnectionError, APITimeoutError))
                or (isinstance(exc, APIStatusError)
                    and not isinstance(exc, (AuthenticationError, BadRequestError)))
            )
            if is_retryable and attempt < len(_RETRY_DELAYS):
                print(f"[DEBUG] LLM rate-limited (attempt {attempt + 1}), retrying in {_RETRY_DELAYS[attempt]}s", flush=True)
                continue
            error = f"llm_error:{type(exc).__name__}"
            break

    action_type: str = data.get("action_type", "approve")
    allowed = _ALLOWED_ACTIONS.get(task_level, ["approve", "remove"])
    if action_type not in allowed:
        action_type = "approve"
        error = error or f"invalid_action_coerced_to_approve"

    rule_raw = data.get("rule_cited")
    rule_cited: Optional[int] = int(rule_raw) if rule_raw is not None else None

    ban_days_raw = data.get("ban_duration_days")
    ban_duration: Optional[int] = int(ban_days_raw) if ban_days_raw is not None else None
    if action_type == "temp_ban" and ban_duration is None:
        ban_duration = 7  # reasonable default

    action = ModAction(
        action_type=action_type,
        rule_cited=rule_cited,
        reason=str(data.get("reason", ""))[:200],
        ban_duration_days=ban_duration,
    )

    label = action_type
    if rule_cited is not None:
        label += f"(rule={rule_cited})"
    if ban_duration is not None:
        label += f"(days={ban_duration})"

    return action, label, error


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

async def _run_task_async(
    env,
    client: OpenAI,
    task_level: int,
    task_name: str,
    num_posts: int,
) -> None:
    """Run one task episode via async openenv-core interface."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_level=task_level, seed=SEED)
        obs: ModObservation = getattr(result, "observation", result)

        for step in range(1, num_posts + 1):
            done = getattr(result, "done", getattr(obs, "done", False))
            if done:
                break

            action, label, err = await _get_model_action(client, obs, task_level, history)
            try:
                result = await env.step(action)
            except Exception as step_exc:
                msg = str(step_exc).lower()
                if any(k in msg for k in ("close frame", "websocket", "disconnect", "connection")):
                    print(f"[DEBUG] WS closed at step {step}, ending episode early: {step_exc}", flush=True)
                    break
                raise

            obs = getattr(result, "observation", result)
            _r = getattr(result, "reward", None)
            if _r is None:
                _r = getattr(obs, "reward", None)
            reward = float(_r) if _r is not None else 0.0
            done = getattr(result, "done", getattr(obs, "done", False))

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=label, reward=reward, done=done, error=err)
            history.append(f"Step {step}: {label} -> reward={reward:.2f} correct={obs.action_was_correct}")

            if done:
                break

    except Exception as exc:
        print(f"[DEBUG] Task {task_level} async error: {exc}", flush=True)
    finally:
        # Score from whatever rewards were collected — partial episodes still count
        if rewards:
            score = min(max(sum(rewards) / num_posts, 0.0), 1.0)
            success = score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    """Build the OpenAI client. Raises RuntimeError with a clear message if unconfigured."""
    key = API_KEY
    if not key:
        raise RuntimeError(
            "No API key found. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY."
        )
    return OpenAI(base_url=API_BASE_URL, api_key=key)


async def main() -> None:
    # Guard client construction — a missing API key must not crash the process;
    # the validator expects [START]/[STEP]/[END] on stdout regardless.
    try:
        client = _build_client()
    except Exception as exc:
        print(f"[DEBUG] Client init failed: {exc}", flush=True)
        for _, task_name, _ in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="approve", reward=0.0, done=True,
                     error=f"client_init_error:{type(exc).__name__}")
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        return

    if IMAGE_NAME:
        # Docker path: one container, but reconnect for each task so every
        # task gets a fresh server-side session (clean environment state).
        try:
            env = await RedditModEnv.from_docker_image(IMAGE_NAME)
        except Exception as exc:
            print(f"[DEBUG] Docker env init failed: {exc}", flush=True)
            for _, task_name, _ in TASKS:
                log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
                log_step(step=1, action="approve", reward=0.0, done=True,
                         error=f"env_init_error:{type(exc).__name__}")
                log_end(success=False, steps=1, score=0.0, rewards=[0.0])
            return
        try:
            for task_level, task_name, num_posts in TASKS:
                await _run_task_async(env, client, task_level, task_name, num_posts)
                # Reconnect after each task to get a fresh session on the server
                try:
                    await env.disconnect()
                    await env.connect()
                except Exception as exc:
                    print(f"[DEBUG] reconnect error after task {task_level}: {exc}", flush=True)
        finally:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error: {exc}", flush=True)
    else:
        # HTTP path: fresh WebSocket connection per task → fresh server session
        # and clean environment state with no cross-task state contamination.
        server_url = os.getenv("ENV_URL", "http://localhost:7860")
        print(f"[DEBUG] No IMAGE_NAME set; connecting to {server_url}", flush=True)
        for task_level, task_name, num_posts in TASKS:
            async with RedditModEnv(server_url) as env:
                await _run_task_async(env, client, task_level, task_name, num_posts)


if __name__ == "__main__":
    asyncio.run(main())

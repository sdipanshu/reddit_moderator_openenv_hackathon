#!/usr/bin/env python3
"""
Inference Script — Reddit Mod Bot RL Environment
=================================================
Environment variables (injected by validator):
    API_KEY       API key for the LLM              (required)
    API_BASE_URL  LLM endpoint                     (default: HF router)
    MODEL_NAME    Model identifier                  (default: Qwen2.5-72B)

Runs all 3 tasks and emits [START]/[STEP]/[END] per task.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from reddit_mod_env.server.reddit_mod_environment import RedditModEnvironment
from reddit_mod_env.models import ModAction

# ---------------------------------------------------------------------------
# Configuration — read from env vars exactly as the validator injects them
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

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


def _format_prompt(obs, task_level: int) -> str:
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

    thread_block = ""
    if post.thread_context:
        thread_block = "\nTHREAD CONTEXT:\n" + "\n".join(f"  {c}" for c in post.thread_context)

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
        {thread_block}
        Respond with JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# Stdout logging helpers — exact format required by validator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
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
# LLM call
# ---------------------------------------------------------------------------

def get_action(client: OpenAI, obs, task_level: int) -> tuple[ModAction, str, Optional[str]]:
    """Call the LLM, parse its JSON response into a ModAction."""
    user_prompt = _format_prompt(obs, task_level)
    error: Optional[str] = None
    data: dict = {}

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        raw = (completion.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        error = f"json_parse_error:{exc}"
    except Exception as exc:
        error = f"llm_error:{type(exc).__name__}:{exc}"

    action_type: str = data.get("action_type", "approve")
    allowed = _ALLOWED_ACTIONS.get(task_level, ["approve", "remove"])
    if action_type not in allowed:
        action_type = "approve"
        error = error or "invalid_action_coerced_to_approve"

    rule_raw = data.get("rule_cited")
    rule_cited: Optional[int] = int(rule_raw) if rule_raw is not None else None

    ban_raw = data.get("ban_duration_days")
    ban_duration: Optional[int] = int(ban_raw) if ban_raw is not None else None
    if action_type == "temp_ban" and ban_duration is None:
        ban_duration = 7

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
# Task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_level: int, task_name: str, num_posts: int) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = RedditModEnvironment()
        obs = env.reset(seed=SEED, task_level=task_level)

        for step in range(1, num_posts + 1):
            if obs.done:
                break

            action, label, err = get_action(client, obs, task_level)
            obs = env.step(action)

            reward = float(obs.reward) if obs.reward is not None else 0.0
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=label, reward=reward, done=obs.done, error=err)

            if obs.done:
                break

    except Exception as exc:
        print(f"[DEBUG] Task {task_level} error: {exc}", flush=True)

    finally:
        if rewards:
            score = min(max(sum(rewards) / num_posts, 0.0), 1.0)
            success = score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "API_KEY environment variable must be set.\n"
            "  export API_KEY=your_token_here"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_level, task_name, num_posts in TASKS:
        run_task(client, task_level, task_name, num_posts)


if __name__ == "__main__":
    main()

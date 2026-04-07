---
title: reddit mod bot
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "4.44.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Reddit Mod Bot RL Environment

An OpenEnv RL environment where agents learn to moderate posts on **r/science**.

## Overview

The agent acts as a Reddit moderator reviewing posts and making moderation decisions. Three progressively harder task levels teach the agent to handle increasingly complex content scenarios.

| Task | Description | Actions | Posts/Episode |
|------|-------------|---------|--------------|
| 1 — Easy | Spam vs legitimate detection | `approve`, `remove` | 10 |
| 2 — Medium | Rule violation classification | + `warn` | 8 |
| 3 — Hard | Context-aware judgment | + `temp_ban`, `perma_ban`, `escalate_to_senior_mod` | 5 |

## Subreddit: r/science Rules

1. **Stay On Topic** — Posts must relate to scientific research
2. **No Personal Attacks** — Debate the science, not the person
3. **No Misinformation** — Claims must have credible sources
4. **No Self-Promotion/Spam** — No advertising or excessive self-promotion
5. **Use Correct Flair** — `[Research]`, `[News]`, `[Discussion]`, `[AMA]`
6. **No Low-Effort Content** — Must contribute meaningfully

## Action Space

`ModAction` — submitted to `POST /step`

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | `string` | ✅ | One of the allowed actions for the current task level (see table above) |
| `rule_cited` | `integer \| null` | — | Subreddit rule number (1–6) violated. Required for Task 2+; `null` when approving |
| `reason` | `string \| null` | — | Free-text explanation of the decision |
| `ban_duration_days` | `integer \| null` | — | Duration in days; required when `action_type` is `temp_ban` |

## Observation Space

`ModObservation` — returned by `POST /reset` and `POST /step`

| Field | Type | Description |
|---|---|---|
| `reward` | `float \| null` | Per-step reward. `null` after reset (no action taken yet), float after every step |
| `done` | `bool` | `true` when all posts in the episode have been reviewed |
| `current_post` | `RedditPost \| null` | The next post to moderate. `null` when `done=true` |
| `feedback` | `string` | Human-readable verdict and explanation for the last action |
| `action_was_correct` | `bool \| null` | Whether the action matched ground truth. `null` after reset |
| `correct_action_type` | `string \| null` | Ground truth action revealed post-step. `null` after reset |
| `correct_rule` | `integer \| null` | Ground truth rule revealed post-step. `null` after reset or if no rule applies |
| `task_level` | `integer` | Current task level (1, 2, or 3) |
| `posts_remaining` | `integer` | Number of posts left in this episode |
| `cumulative_reward` | `float` | Sum of all per-step rewards so far |

`RedditPost` fields within `current_post`:

| Field | Type | Description |
|---|---|---|
| `post_id` | `string` | Unique identifier for the post |
| `title` | `string \| null` | Post title |
| `body` | `string` | Post content |
| `author` | `UserHistory` | Author account metadata (age, karma, prior warnings/removals, flags) |
| `subreddit` | `SubredditContext` | Subreddit name, rules list, and culture description |
| `score` | `integer` | Upvote score |
| `num_comments` | `integer` | Comment count |
| `reports` | `Report[]` | List of `{reason, count}` report entries |
| `flair` | `string \| null` | Post flair label |
| `thread_context` | `string[] \| null` | Surrounding comment context for comment-level scenarios |

## State Space

`ModState` — returned by `GET /state`

| Field | Type | Description |
|---|---|---|
| `episode_id` | `string` | UUID for the current episode |
| `step_count` | `integer` | Total steps taken (including invalid actions) |
| `task_level` | `integer` | Current task level |
| `total_posts` | `integer` | Episode length |
| `posts_reviewed` | `integer` | Valid steps taken |
| `cumulative_reward` | `float` | Running reward total |
| `correct_decisions` | `integer` | Count of steps where action matched ground truth |
| `subreddit_name` | `string` | Always `"r/science"` |

## API

### `POST /reset`
```json
{ "task_level": 1, "seed": 42 }
```

### `POST /step`
```json
{
  "action_type": "remove",
  "rule_cited": 4,
  "reason": "Commercial spam promoting supplements"
}
```

### `GET /state`
Returns current episode metadata.

### `GET /health`
Liveness check — returns `{"status": "ok"}`.

## Reward Structure

Rewards are emitted **per step** (not just at episode end), providing signal across the full trajectory.

**Task 1** — Binary: `1.0` correct, `0.0` wrong.
Spam detection is a binary classification; no partial credit applies.

**Task 2** — Partial credit `[0.0, 1.0]`:
- `1.0` correct action + correct rule
- `0.6` correct action + wrong/missing rule
- `0.5` acceptable action + correct rule
- `0.3` acceptable action + wrong rule
- `0.1` wrong action but identified a violation
- `0.0` missed violation or false positive

**Task 3** — Weighted multi-factor `[0.0, 1.0]`:
- Action correctness × 0.40
- Proportionality × 0.25
- Rule citation × 0.20
- Escalation judgment × 0.15
- **Destructive suppression: 15×** — when `perma_ban` or `temp_ban` is applied to a post whose correct action is `approve`, the computed total is multiplied by 0.15. Relative signal within the destructive case is preserved (e.g. correct rule citation still matters), but the maximum possible reward is ~0.037 — far below the ~0.47 a softer wrong action like `warn` would receive

## Validation

```bash
# Run the test suite (validates OpenEnv spec compliance)
pip install -e ".[dev]"
pytest

# Validate against the OpenEnv CLI (requires openenv-core)
openenv validate
```

## Running Locally

```bash
cd "openenv hackathon"
pip install -e reddit_mod_env
uvicorn reddit_mod_env.server.app:app --port 7860 --reload
```

Then visit `http://localhost:7860/docs` for the interactive API.

## Docker

```bash
cd "openenv hackathon/reddit_mod_env"
docker build -t reddit-mod-env .
docker run -p 7860:7860 reddit-mod-env
```

## Python Client

```python
from reddit_mod_env import RedditModEnv, ModAction

with RedditModEnv("http://localhost:7860") as env:
    obs = env.reset(task_level=1, seed=42)
    print(obs.current_post.title)

    obs = env.step(ModAction(action_type="remove"))
    print(f"Reward: {obs.reward}, Correct: {obs.action_was_correct}")
    print(obs.feedback)
```

## Baseline Inference

Runs `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face inference router against all 3 tasks (seed=42).

```bash
# Against a running local server
HF_TOKEN=<your-token> python inference.py

# Against a Docker image (requires openenv-core)
LOCAL_IMAGE_NAME=reddit-mod-env HF_TOKEN=<your-token> python inference.py
```

| Task | Description | Steps | Score |
|------|-------------|-------|-------|
| 1 — spam-detection | Spam vs legitimate | 10/10 | **1.00** |
| 2 — rule-classification | Rule violation + citation | 8/8 | **0.89** |
| 3 — context-judgment | Context-aware judgment | 5/5 | **0.62** |

> Score = `sum(per-step rewards) / num_posts`, normalized to [0, 1]. Model: `Qwen/Qwen2.5-72B-Instruct`, seed=42.

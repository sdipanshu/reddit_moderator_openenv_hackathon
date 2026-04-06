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

**Task 1** — Binary: `1.0` correct, `0.0` wrong.

**Task 2** — Partial credit:
- `1.0` correct action + correct rule
- `0.6` correct action + wrong rule
- `0.3` acceptable action
- `0.0` missed violation or false positive

**Task 3** — Weighted multi-factor (all 0.0–1.0):
- Action correctness × 0.40
- Proportionality × 0.25
- Rule citation × 0.20
- Escalation judgment × 0.15

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

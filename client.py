"""
Python client for the Reddit Mod Bot RL environment.

Usage (async):
    async with RedditModEnv("http://localhost:7860") as env:
        obs = await env.reset(task_level=1)
        obs = await env.step(ModAction(action_type="approve"))

Usage (sync):
    env = RedditModEnv("http://localhost:7860").sync()
    obs = env.reset(task_level=1)
    obs = env.step(ModAction(action_type="approve"))
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .models import ModAction, ModObservation, ModState, RedditPost, UserHistory, SubredditContext, SubredditRule, Report

try:
    from openenv.core.env_client import EnvClient  # type: ignore
    from openenv.core.client_types import StepResult  # type: ignore

    class RedditModEnv(EnvClient[ModAction, ModObservation, ModState]):
        """Async client for the Reddit Mod Bot environment."""

        def _step_payload(self, action: ModAction) -> Dict[str, Any]:
            return action.model_dump(exclude_none=False)

        def _parse_result(self, payload: Dict[str, Any]) -> "StepResult[ModObservation]":
            obs = _parse_observation(payload)
            return StepResult(
                observation=obs,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: Dict[str, Any]) -> ModState:
            return ModState(**payload)

except ImportError:
    # ── Fallback sync client using httpx for local dev ───────────────────────
    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore

    class RedditModEnv:  # type: ignore[no-redef]
        """
        Lightweight synchronous HTTP client (no openenv-core required).
        Used for local testing and hackathon demo scripts.
        """

        def __init__(self, base_url: str = "http://localhost:7860") -> None:
            self._base = base_url.rstrip("/")
            if httpx is None:
                raise ImportError("Install httpx: pip install httpx")
            self._client = httpx.Client(base_url=self._base, timeout=30.0)

        def reset(
            self,
            task_level: int = 1,
            num_posts: Optional[int] = None,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
        ) -> ModObservation:
            payload: Dict[str, Any] = {"task_level": task_level}
            if num_posts is not None:
                payload["num_posts"] = num_posts
            if seed is not None:
                payload["seed"] = seed
            if episode_id is not None:
                payload["episode_id"] = episode_id
            r = self._client.post("/reset", json=payload)
            r.raise_for_status()
            return _parse_observation(r.json())

        def step(self, action: ModAction) -> ModObservation:
            r = self._client.post("/step", json=action.model_dump(exclude_none=False))
            r.raise_for_status()
            return _parse_observation(r.json())

        def state(self) -> ModState:
            r = self._client.get("/state")
            r.raise_for_status()
            return ModState(**r.json())

        def close(self) -> None:
            self._client.close()

        def __enter__(self) -> "RedditModEnv":
            return self

        def __exit__(self, *_: Any) -> None:
            self.close()

        async def __aenter__(self) -> "RedditModEnv":
            return self

        async def __aexit__(self, *_: Any) -> None:
            self.close()


# ---------------------------------------------------------------------------
# Shared deserialization helper
# ---------------------------------------------------------------------------

def _parse_observation(payload: Dict[str, Any]) -> ModObservation:
    """
    Reconstruct a ModObservation from a raw JSON dict.

    openenv-core wraps responses as {observation: {...}, reward: ..., done: ...}.
    The fallback server returns flat {reward, done, current_post, ...}.
    This function handles both shapes.
    """
    # Unwrap openenv-core envelope if present
    obs_data = payload.get("observation", payload)
    top_reward = payload.get("reward")
    top_done = payload.get("done", False)

    raw_post = obs_data.get("current_post")
    current_post: Optional[RedditPost] = None

    if raw_post:
        author_data = raw_post.get("author", {})
        subreddit_data = raw_post.get("subreddit", {})

        rules = [
            SubredditRule(**r) for r in subreddit_data.get("rules", [])
        ]
        subreddit = SubredditContext(
            name=subreddit_data.get("name", "r/science"),
            rules=rules,
            culture=subreddit_data.get("culture", ""),
        )

        reports = [Report(**rep) for rep in raw_post.get("reports", [])]

        current_post = RedditPost(
            post_id=raw_post.get("post_id", ""),
            title=raw_post.get("title", ""),
            body=raw_post.get("body", ""),
            author=UserHistory(**author_data),
            subreddit=subreddit,
            score=raw_post.get("score", 0),
            num_comments=raw_post.get("num_comments", 0),
            reports=reports,
            flair=raw_post.get("flair"),
            thread_context=raw_post.get("thread_context"),
        )

    return ModObservation(
        reward=top_reward,
        done=top_done,
        current_post=current_post,
        feedback=obs_data.get("feedback", ""),
        action_was_correct=obs_data.get("action_was_correct"),
        correct_action_type=obs_data.get("correct_action_type"),
        correct_rule=obs_data.get("correct_rule"),
        task_level=obs_data.get("task_level", 1),
        posts_remaining=obs_data.get("posts_remaining", 0),
        cumulative_reward=obs_data.get("cumulative_reward", 0.0),
    )

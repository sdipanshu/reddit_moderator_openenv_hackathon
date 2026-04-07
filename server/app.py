"""
FastAPI application entry point.

Uses openenv.core.env_server.create_fastapi_app to wire up the standard
OpenEnv endpoints: /reset, /step, /state, /schema, /health, /metadata, /ws
"""
from __future__ import annotations

try:
    from openenv.core.env_server import create_fastapi_app  # type: ignore
    from openenv.core.env_server.types import ConcurrencyConfig  # type: ignore

    from ..models import ModAction, ModObservation
    from .reddit_mod_environment import RedditModEnvironment

    app = create_fastapi_app(
        RedditModEnvironment,
        action_cls=ModAction,
        observation_cls=ModObservation,
        concurrency_config=ConcurrencyConfig(
            max_concurrent_envs=20,
            session_timeout=600.0,  # 10 min — accommodates slow LLM calls between steps
        ),
    )

    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

except ImportError:
    # ── Fallback: standalone FastAPI server for local dev without openenv-core ──
    from typing import Any, Dict, Optional

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    from ..models import ModAction, ModObservation, ModState
    from .reddit_mod_environment import RedditModEnvironment

    app = FastAPI(
        title="Reddit Mod Bot RL Environment",
        description=(
            "Train RL agents to moderate r/science posts. "
            "Three difficulty levels from spam detection to context-aware moderation."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Single shared environment instance (sufficient for dev/demo)
    _env = RedditModEnvironment()

    class ResetRequest(BaseModel):
        task_level: int = 1
        num_posts: Optional[int] = None
        seed: Optional[int] = None
        episode_id: Optional[str] = None

    @app.get("/", include_in_schema=False)
    def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> Dict[str, Any]:
        return {
            "name": "reddit_mod_env",
            "version": "1.0.0",
            "description": "Reddit moderator bot RL environment for r/science",
            "task_levels": [1, 2, 3],
            "subreddit": "r/science",
        }

    @app.post("/reset", response_model=ModObservation)
    def reset(request: ResetRequest) -> ModObservation:
        kwargs: Dict[str, Any] = {"task_level": request.task_level}
        if request.num_posts is not None:
            kwargs["num_posts"] = request.num_posts
        return _env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            **kwargs,
        )

    @app.post("/step", response_model=ModObservation)
    def step(action: ModAction) -> ModObservation:
        return _env.step(action)

    @app.get("/state", response_model=ModState)
    def state() -> ModState:
        return _env.state

    @app.get("/schema")
    def schema() -> Dict[str, Any]:
        return {
            "action": ModAction.model_json_schema(),
            "observation": ModObservation.model_json_schema(),
            "state": ModState.model_json_schema(),
        }


def main() -> None:
    """Server entry point — callable by the openenv CLI and the installed script."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

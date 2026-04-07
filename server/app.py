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

    import uuid as _uuid
    from typing import Optional as _Opt

    from fastapi import HTTPException, Request, Response
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel as _BaseModel

    from ._playground import PLAYGROUND_HTML

    # ── Playground session store (cookie-keyed) ──────────────────────────────
    _pg_sessions: dict = {}

    class _PGResetRequest(_BaseModel):
        task_level: int = 1
        num_posts: _Opt[int] = None
        seed: _Opt[int] = None

    @app.get("/", include_in_schema=False)
    def root() -> HTMLResponse:
        return HTMLResponse(PLAYGROUND_HTML)

    @app.post("/pg/reset", include_in_schema=False)
    def pg_reset(request: _PGResetRequest, response: Response) -> dict:
        env = RedditModEnvironment()
        kwargs: dict = {"task_level": request.task_level}
        if request.num_posts is not None:
            kwargs["num_posts"] = request.num_posts
        obs = env.reset(seed=request.seed, **kwargs)
        sid = str(_uuid.uuid4())
        _pg_sessions[sid] = env
        response.set_cookie("pg_sid", sid, max_age=3600, httponly=True)
        return obs.model_dump()

    @app.post("/pg/step", include_in_schema=False)
    def pg_step(action: ModAction, request: Request) -> dict:
        sid = request.cookies.get("pg_sid")
        env = _pg_sessions.get(sid) if sid else None
        if env is None:
            raise HTTPException(400, "No active session — call /pg/reset first")
        obs = env.step(action)
        return obs.model_dump()

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

    from fastapi import Request, Response
    from fastapi.responses import HTMLResponse

    from ._playground import PLAYGROUND_HTML

    _pg_sessions_fb: dict = {}

    @app.get("/", include_in_schema=False)
    def root() -> HTMLResponse:
        return HTMLResponse(PLAYGROUND_HTML)

    @app.post("/pg/reset", include_in_schema=False)
    def pg_reset_fb(request: ResetRequest, response: Response) -> dict:
        import uuid as _uuid
        env = RedditModEnvironment()
        kwargs: Dict[str, Any] = {"task_level": request.task_level}
        if request.num_posts is not None:
            kwargs["num_posts"] = request.num_posts
        obs = env.reset(seed=request.seed, **kwargs)
        sid = str(_uuid.uuid4())
        _pg_sessions_fb[sid] = env
        response.set_cookie("pg_sid", sid, max_age=3600, httponly=True)
        return obs.model_dump()

    @app.post("/pg/step", include_in_schema=False)
    def pg_step_fb(action: ModAction, request: Request) -> dict:
        from fastapi import HTTPException
        sid = request.cookies.get("pg_sid")
        env = _pg_sessions_fb.get(sid) if sid else None
        if env is None:
            raise HTTPException(400, "No active session — call /pg/reset first")
        return env.step(action).model_dump()

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

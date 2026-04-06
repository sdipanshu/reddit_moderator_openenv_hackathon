"""
Pydantic models for the Reddit Mod Bot RL environment.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# OpenEnv base class imports with fallback for local development
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import Action, Observation, State  # type: ignore
except ImportError:
    class Action(BaseModel):  # type: ignore[no-redef]
        model_config = {"extra": "forbid"}

    class Observation(BaseModel):  # type: ignore[no-redef]
        reward: Optional[float] = None
        done: bool = False
        model_config = {"extra": "forbid"}

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = Field(default=0, ge=0)
        model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Supporting domain models (not part of the OpenEnv API surface)
# ---------------------------------------------------------------------------

class SubredditRule(BaseModel):
    rule_number: int
    title: str
    description: str


class SubredditContext(BaseModel):
    name: str
    rules: List[SubredditRule]
    culture: str


class UserHistory(BaseModel):
    username: str
    account_age_days: int
    karma: int
    prior_warnings: int
    prior_removals: int
    is_repeat_offender: bool
    is_approved_contributor: bool = False


class Report(BaseModel):
    reason: str
    count: int


class RedditPost(BaseModel):
    post_id: str
    title: Optional[str] = None   # None for comment-level moderation scenarios
    body: str
    author: UserHistory
    subreddit: SubredditContext
    score: int
    num_comments: int
    reports: List[Report] = Field(default_factory=list)
    flair: Optional[str] = None
    thread_context: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Core OpenEnv models
# ---------------------------------------------------------------------------

ActionType = Literal[
    "approve",
    "remove",
    "warn",
    "temp_ban",
    "perma_ban",
    "escalate_to_senior_mod",
]

# Actions available per task level (enforced in the environment)
TASK_ALLOWED_ACTIONS: Dict[int, List[str]] = {
    1: ["approve", "remove"],
    2: ["approve", "remove", "warn"],
    3: ["approve", "remove", "warn", "temp_ban", "perma_ban", "escalate_to_senior_mod"],
}


class ModAction(Action):
    """Agent's moderation decision for a single post."""

    action_type: ActionType
    rule_cited: Optional[int] = None       # Rule number violated (None when approving)
    reason: Optional[str] = None           # Free-text explanation
    ban_duration_days: Optional[int] = None  # Required when action_type == "temp_ban"


class ModObservation(Observation):
    """
    Observation returned after each reset() or step().

    Inherits from Observation:
      - reward: Optional[float] = None   (None on initial reset, float after each step)
      - done: bool = False
    """

    current_post: Optional[RedditPost] = None   # Post to moderate next (None when done)
    feedback: str = ""                           # Human-readable explanation of last action
    action_was_correct: Optional[bool] = None   # Revealed after step (None on reset)
    correct_action_type: Optional[str] = None   # Ground truth revealed post-step
    correct_rule: Optional[int] = None          # Ground truth rule revealed post-step
    task_level: int = 1
    posts_remaining: int = 0
    cumulative_reward: float = 0.0


class ModState(State):
    """
    Episode metadata.

    Inherits from State:
      - episode_id: Optional[str] = None
      - step_count: int = 0
    """

    task_level: int = 1
    total_posts: int = 0
    posts_reviewed: int = 0
    cumulative_reward: float = 0.0
    correct_decisions: int = 0
    subreddit_name: str = "r/science"

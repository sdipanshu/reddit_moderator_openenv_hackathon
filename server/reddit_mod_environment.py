"""
Core RL environment: Reddit Moderator Bot.

Implements the OpenEnv Environment interface with reset/step/state.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from ..content_generator import ContentGenerator, GroundTruth
from ..models import (
    TASK_ALLOWED_ACTIONS,
    ModAction,
    ModObservation,
    ModState,
    RedditPost,
)
from ..reward import (
    calculate_task1_reward,
    calculate_task2_reward,
    calculate_task3_reward,
)

try:
    from openenv.core.env_server import Environment  # type: ignore
except ImportError:
    class Environment:  # type: ignore[no-redef]
        pass

# Posts per episode per task level
_EPISODE_LENGTHS = {1: 10, 2: 8, 3: 5}


class RedditModEnvironment(Environment):
    """
    Reddit moderator RL environment for r/science.

    Task levels:
      1 (Easy)   — Spam detection.         Actions: approve / remove.
      2 (Medium) — Rule classification.    Actions: approve / remove / warn.
      3 (Hard)   — Context-aware mod.      All 6 actions.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._generator = ContentGenerator()
        self._posts: list = []           # List[Tuple[RedditPost, GroundTruth]]
        self._index: int = 0
        self._state = ModState()
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ModObservation:
        """
        Start a new moderation episode.

        Kwargs:
            task_level (int): 1, 2, or 3 (default 1).
            num_posts  (int): Override default episode length.
        """
        task_level: int = int(kwargs.get("task_level", 1))
        if task_level not in (1, 2, 3):
            task_level = 1

        num_posts: int = int(kwargs.get("num_posts", _EPISODE_LENGTHS[task_level]))

        self._posts = self._generator.generate_episode(task_level, num_posts, seed=seed)
        self._index = 0
        self._cumulative_reward = 0.0

        self._state = ModState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_level=task_level,
            total_posts=num_posts,
            posts_reviewed=0,
            cumulative_reward=0.0,
            correct_decisions=0,
            subreddit_name="r/science",
        )

        first_post, _ = self._posts[0]

        return ModObservation(
            reward=None,            # No action taken yet
            done=False,
            current_post=first_post,
            feedback=(
                f"Episode started. Task level {task_level}: "
                f"{self._task_description(task_level)} "
                f"You have {num_posts} posts to review."
            ),
            action_was_correct=None,
            correct_action_type=None,
            correct_rule=None,
            task_level=task_level,
            posts_remaining=num_posts,
            cumulative_reward=0.0,
        )

    def step(
        self,
        action: ModAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ModObservation:
        """
        Submit a moderation decision for the current post.

        Returns the next post (or done=True) plus reward and feedback.
        """
        task_level = self._state.task_level
        allowed = TASK_ALLOWED_ACTIONS[task_level]

        # ── Auto-reset if called before reset() (HTTP stateless single-step) ───
        if not self._posts:
            self.reset(task_level=1)

        # ── Validate action for current task level ────────────────────────────
        if action.action_type not in allowed:
            return ModObservation(
                reward=0.0,
                done=False,
                current_post=self._current_post(),
                feedback=(
                    f"Invalid action '{action.action_type}' for Task {task_level}. "
                    f"Allowed actions: {allowed}. Please try again."
                ),
                action_was_correct=False,
                correct_action_type=None,
                correct_rule=None,
                task_level=task_level,
                posts_remaining=len(self._posts) - self._index,
                cumulative_reward=self._cumulative_reward,
            )

        # ── Score the action ──────────────────────────────────────────────────
        post, ground_truth = self._posts[self._index]
        reward = self._calculate_reward(action, ground_truth, post, task_level)
        is_correct = action.action_type == ground_truth.correct_action

        # ── Update state ──────────────────────────────────────────────────────
        self._cumulative_reward += reward
        self._state.step_count += 1
        self._state.posts_reviewed += 1
        self._state.cumulative_reward = self._cumulative_reward
        if is_correct:
            self._state.correct_decisions += 1

        self._index += 1
        done = self._index >= len(self._posts)

        next_post = None if done else self._posts[self._index][0]
        posts_remaining = len(self._posts) - self._index

        feedback = self._build_feedback(
            action, ground_truth, reward, is_correct, done, posts_remaining
        )

        return ModObservation(
            reward=reward,
            done=done,
            current_post=next_post,
            feedback=feedback,
            action_was_correct=is_correct,
            correct_action_type=ground_truth.correct_action,
            correct_rule=ground_truth.correct_rule,
            task_level=task_level,
            posts_remaining=posts_remaining,
            cumulative_reward=self._cumulative_reward,
        )

    @property
    def state(self) -> ModState:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_post(self) -> Optional[RedditPost]:
        if self._index < len(self._posts):
            return self._posts[self._index][0]
        return None

    def _calculate_reward(
        self,
        action: ModAction,
        ground_truth: GroundTruth,
        post: RedditPost,
        task_level: int,
    ) -> float:
        if task_level == 1:
            return calculate_task1_reward(action, ground_truth)
        if task_level == 2:
            return calculate_task2_reward(action, ground_truth)
        return calculate_task3_reward(action, ground_truth, post)

    def _build_feedback(
        self,
        action: ModAction,
        ground_truth: GroundTruth,
        reward: float,
        is_correct: bool,
        done: bool,
        posts_remaining: int,
    ) -> str:
        if is_correct:
            verdict = "Correct!"
        else:
            correct = ground_truth.correct_action
            rule_hint = (
                f" (Rule {ground_truth.correct_rule})" if ground_truth.correct_rule else ""
            )
            verdict = (
                f"Incorrect. The right action was '{correct}'{rule_hint}."
            )

        explanation = f" {ground_truth.explanation}"

        if done:
            total = self._state.correct_decisions
            out_of = self._state.total_posts
            score = f" Episode complete — {total}/{out_of} correct, cumulative reward: {self._cumulative_reward:.2f}."
            return f"{verdict}{explanation}{score}"

        return f"{verdict}{explanation} Reward: {reward:.2f}. Posts remaining: {posts_remaining}."

    @staticmethod
    def _task_description(level: int) -> str:
        return {
            1: "Detect spam vs legitimate posts.",
            2: "Identify rule violations and cite the correct rule.",
            3: "Apply context-aware judgment for complex moderation decisions.",
        }.get(level, "")

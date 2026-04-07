"""
OpenEnv spec compliance tests for the Reddit Mod Bot environment.

Validates that the environment satisfies all core interface requirements:
  - reset() / step() / state return the correct typed models
  - Rewards are in expected range, with per-step signal on every transition
  - Invalid actions are gracefully rejected (zero reward, no crash)
  - Done transitions correctly at episode end
  - Seeds produce deterministic episodes
  - All three task levels operate independently
  - Negative penalty fires on destructive actions against legitimate posts
"""

from __future__ import annotations

import pytest

from reddit_mod_env.models import ModAction, ModObservation, ModState
from reddit_mod_env.server.reddit_mod_environment import RedditModEnvironment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> RedditModEnvironment:
    return RedditModEnvironment()


# ---------------------------------------------------------------------------
# reset() contract
# ---------------------------------------------------------------------------


class TestReset:
    def test_returns_mod_observation(self, env: RedditModEnvironment) -> None:
        obs = env.reset(seed=42, task_level=1)
        assert isinstance(obs, ModObservation)

    def test_initial_reward_is_none(self, env: RedditModEnvironment) -> None:
        obs = env.reset(seed=42, task_level=1)
        assert obs.reward is None, "reward must be None before any action is taken"

    def test_initial_done_is_false(self, env: RedditModEnvironment) -> None:
        obs = env.reset(seed=42, task_level=1)
        assert obs.done is False

    def test_first_post_is_present(self, env: RedditModEnvironment) -> None:
        obs = env.reset(seed=42, task_level=1)
        assert obs.current_post is not None
        assert obs.current_post.body  # non-empty content

    def test_cumulative_reward_starts_at_zero(self, env: RedditModEnvironment) -> None:
        obs = env.reset(seed=42, task_level=1)
        assert obs.cumulative_reward == 0.0

    @pytest.mark.parametrize("task_level", [1, 2, 3])
    def test_task_level_reflected_in_observation(
        self, env: RedditModEnvironment, task_level: int
    ) -> None:
        obs = env.reset(seed=42, task_level=task_level)
        assert obs.task_level == task_level

    @pytest.mark.parametrize(
        "task_level,expected_posts",
        [(1, 10), (2, 8), (3, 5)],
    )
    def test_default_episode_length(
        self,
        env: RedditModEnvironment,
        task_level: int,
        expected_posts: int,
    ) -> None:
        obs = env.reset(seed=42, task_level=task_level)
        assert obs.posts_remaining == expected_posts


# ---------------------------------------------------------------------------
# step() contract
# ---------------------------------------------------------------------------


class TestStep:
    def test_returns_mod_observation(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=1)
        obs = env.step(ModAction(action_type="approve"))
        assert isinstance(obs, ModObservation)

    def test_reward_is_float_after_step(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=1)
        obs = env.step(ModAction(action_type="approve"))
        assert obs.reward is not None
        assert isinstance(obs.reward, float)

    def test_action_was_correct_revealed(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=1)
        obs = env.step(ModAction(action_type="approve"))
        assert obs.action_was_correct is not None  # always revealed post-step

    def test_ground_truth_revealed(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=1)
        obs = env.step(ModAction(action_type="approve"))
        assert obs.correct_action_type in ("approve", "remove")

    def test_posts_remaining_decrements(self, env: RedditModEnvironment) -> None:
        obs0 = env.reset(seed=42, task_level=1)
        obs1 = env.step(ModAction(action_type="approve"))
        assert obs1.posts_remaining == obs0.posts_remaining - 1

    def test_cumulative_reward_accumulates(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=1)
        total = 0.0
        for _ in range(3):
            obs = env.step(ModAction(action_type="approve"))
            if obs.reward is not None:
                total += obs.reward
        assert abs(obs.cumulative_reward - total) < 1e-6

    def test_invalid_action_rejected_with_zero_reward(
        self, env: RedditModEnvironment
    ) -> None:
        env.reset(seed=42, task_level=1)
        # "warn" is not available in Task 1
        obs = env.step(ModAction(action_type="warn"))
        assert obs.reward == 0.0
        assert obs.action_was_correct is False
        assert "Invalid action" in obs.feedback

    def test_invalid_action_does_not_advance_post(
        self, env: RedditModEnvironment
    ) -> None:
        obs_reset = env.reset(seed=42, task_level=1)
        initial_remaining = obs_reset.posts_remaining  # 10
        env.step(ModAction(action_type="warn"))  # invalid for task 1 — must not consume post
        obs_valid = env.step(ModAction(action_type="approve"))  # first real step
        # Only one valid step taken, so exactly one post consumed
        assert obs_valid.posts_remaining == initial_remaining - 1


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


class TestTermination:
    @pytest.mark.parametrize(
        "task_level,num_posts",
        [(1, 10), (2, 8), (3, 5)],
    )
    def test_episode_ends_after_all_posts(
        self,
        env: RedditModEnvironment,
        task_level: int,
        num_posts: int,
    ) -> None:
        obs = env.reset(seed=42, task_level=task_level)
        action_type = {1: "approve", 2: "approve", 3: "approve"}[task_level]
        steps = 0
        while not obs.done:
            obs = env.step(ModAction(action_type=action_type))
            steps += 1
        assert obs.done is True
        assert steps == num_posts

    def test_done_post_is_none(self, env: RedditModEnvironment) -> None:
        obs = env.reset(seed=42, task_level=1)
        while not obs.done:
            obs = env.step(ModAction(action_type="approve"))
        assert obs.current_post is None

    def test_episode_summary_in_feedback(self, env: RedditModEnvironment) -> None:
        obs = env.reset(seed=42, task_level=1)
        while not obs.done:
            obs = env.step(ModAction(action_type="approve"))
        assert "Episode complete" in obs.feedback


# ---------------------------------------------------------------------------
# state() contract
# ---------------------------------------------------------------------------


class TestState:
    def test_returns_mod_state(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=1)
        assert isinstance(env.state, ModState)

    def test_step_count_increments(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=2)
        env.step(ModAction(action_type="approve"))
        env.step(ModAction(action_type="approve"))
        assert env.state.step_count == 2

    def test_posts_reviewed_increments(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=2)
        env.step(ModAction(action_type="approve"))
        assert env.state.posts_reviewed == 1

    def test_correct_decisions_tracked(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=1)
        obs = env.step(ModAction(action_type="approve"))
        if obs.action_was_correct:
            assert env.state.correct_decisions == 1
        else:
            assert env.state.correct_decisions == 0

    def test_episode_id_set_after_reset(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, task_level=1)
        assert env.state.episode_id is not None

    def test_custom_episode_id_preserved(self, env: RedditModEnvironment) -> None:
        env.reset(seed=42, episode_id="test-episode-123", task_level=1)
        assert env.state.episode_id == "test-episode-123"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_first_post(self, env: RedditModEnvironment) -> None:
        obs_a = env.reset(seed=0, task_level=1)
        post_a = obs_a.current_post.post_id
        obs_b = env.reset(seed=0, task_level=1)
        post_b = obs_b.current_post.post_id
        assert post_a == post_b

    def test_different_seeds_different_episodes(
        self, env: RedditModEnvironment
    ) -> None:
        ids_a = []
        obs = env.reset(seed=1, task_level=1)
        ids_a.append(obs.current_post.post_id)
        while not obs.done:
            obs = env.step(ModAction(action_type="approve"))
            if obs.current_post:
                ids_a.append(obs.current_post.post_id)

        ids_b = []
        obs = env.reset(seed=99, task_level=1)
        ids_b.append(obs.current_post.post_id)
        while not obs.done:
            obs = env.step(ModAction(action_type="approve"))
            if obs.current_post:
                ids_b.append(obs.current_post.post_id)

        # At least one post order differs between seeds
        assert ids_a != ids_b


# ---------------------------------------------------------------------------
# Reward range and signal quality
# ---------------------------------------------------------------------------


class TestRewardSignal:
    @pytest.mark.parametrize("task_level", [1, 2, 3])
    def test_reward_per_step_not_binary_terminal(
        self, env: RedditModEnvironment, task_level: int
    ) -> None:
        """Every step must return a scalar reward — never deferred to end."""
        action_type = {1: "approve", 2: "approve", 3: "approve"}[task_level]
        obs = env.reset(seed=42, task_level=task_level)
        while not obs.done:
            obs = env.step(ModAction(action_type=action_type))
            assert obs.reward is not None, "reward must be present on every step"

    def test_task2_partial_credit(self, env: RedditModEnvironment) -> None:
        """Task 2 must return intermediate scores, not just 0.0 / 1.0."""
        obs = env.reset(seed=42, task_level=2)
        rewards = set()
        action_type = "remove"
        while not obs.done:
            obs = env.step(ModAction(action_type=action_type, rule_cited=99))  # wrong rule
            if obs.reward is not None:
                rewards.add(round(obs.reward, 2))
        # Expect values other than the extremes (e.g. 0.6 for right action/wrong rule)
        intermediate = rewards - {0.0, 1.0}
        assert intermediate, f"Task 2 should award partial credit; got only {rewards}"

    def test_task3_weighted_factors(self, env: RedditModEnvironment) -> None:
        """
        Task 3 weighted multi-factor scoring must produce fractional values,
        not just the binary extremes 0.0 and 1.0.

        'warn' with a rule citation is used because:
          - it partially overlaps some correct actions (generates intermediate scores)
          - it never triggers the destructive-action penalty (only perma/temp_ban do)
        """
        obs = env.reset(seed=42, task_level=3)
        rewards = []
        while not obs.done:
            obs = env.step(ModAction(action_type="warn", rule_cited=2))
            if obs.reward is not None:
                rewards.append(obs.reward)
        # At least one reward must be fractional (not a hard 0.0 or 1.0)
        assert any(r not in (0.0, 1.0) for r in rewards), (
            f"Task 3 should produce fractional rewards from weighted factors; got {rewards}"
        )

    def test_task3_destructive_action_penalty(
        self, env: RedditModEnvironment
    ) -> None:
        """
        perma_ban and temp_ban on a legitimate post must yield a negative reward.

        We test the reward function directly because the episode loop has no way
        to know which post is an 'approve' post before submitting the action —
        obs.correct_action_type is only revealed *after* stepping.
        """
        from reddit_mod_env.content_generator import ContentGenerator
        from reddit_mod_env.reward import calculate_task3_reward

        gen = ContentGenerator()
        scenarios = gen.generate_episode(task_level=3, num_posts=50, seed=0)
        approve_cases = [
            (post, gt) for post, gt in scenarios if gt.correct_action == "approve"
        ]
        if not approve_cases:
            pytest.skip("No approve scenario in Task 3 pool (seed=0)")

        post, gt = approve_cases[0]

        # perma_ban on a legitimate post → negative reward
        r_perma = calculate_task3_reward(ModAction(action_type="perma_ban"), gt, post)
        assert r_perma < 0, f"perma_ban on approve post must be negative; got {r_perma}"
        assert r_perma >= -0.25, f"Penalty floor is -0.25; got {r_perma}"

        # temp_ban also triggers the penalty
        r_temp = calculate_task3_reward(
            ModAction(action_type="temp_ban", ban_duration_days=7), gt, post
        )
        assert r_temp < 0, f"temp_ban on approve post must be negative; got {r_temp}"
        assert r_temp >= -0.25, f"Penalty floor is -0.25; got {r_temp}"

    def test_task3_penalty_scoped_to_ban_actions(
        self, env: RedditModEnvironment
    ) -> None:
        """
        warn and remove on a legitimate post are wrong but not destructive —
        they must not trigger the negative penalty (floor is 0.0, not below).
        """
        from reddit_mod_env.content_generator import ContentGenerator
        from reddit_mod_env.reward import calculate_task3_reward

        gen = ContentGenerator()
        scenarios = gen.generate_episode(task_level=3, num_posts=50, seed=0)
        approve_cases = [
            (post, gt) for post, gt in scenarios if gt.correct_action == "approve"
        ]
        if not approve_cases:
            pytest.skip("No approve scenario in Task 3 pool (seed=0)")

        post, gt = approve_cases[0]

        for action_type in ("warn", "remove"):
            reward = calculate_task3_reward(ModAction(action_type=action_type), gt, post)
            assert reward >= 0.0, (
                f"{action_type} on approve post should be ≥ 0.0 (no destructive penalty); "
                f"got {reward}"
            )

    def test_reward_bounded(self, env: RedditModEnvironment) -> None:
        """
        Reward bounds per task:
          Task 1: [0.0, 1.0]  — binary, no penalty
          Task 2: [0.0, 1.0]  — partial credit, no penalty (ban actions unavailable)
          Task 3: [-0.25, 1.0] — weighted factors + destructive-action penalty

        Task 3 uses perma_ban deliberately so the penalty fires on approve posts,
        exercising the full lower bound of the reward range.
        """
        # Tasks 1 and 2: reward must never go negative
        for task_level in [1, 2]:
            action_map = {
                1: ModAction(action_type="remove"),
                2: ModAction(action_type="warn", rule_cited=1),
            }
            obs = env.reset(seed=7, task_level=task_level)
            while not obs.done:
                obs = env.step(action_map[task_level])
                if obs.reward is not None:
                    assert 0.0 <= obs.reward <= 1.0, (
                        f"Task {task_level} reward {obs.reward} out of [0.0, 1.0]"
                    )

        # Task 3: reward may be negative (penalty for ban on innocent user)
        obs = env.reset(seed=7, task_level=3)
        while not obs.done:
            obs = env.step(ModAction(action_type="perma_ban"))
            if obs.reward is not None:
                assert -0.25 <= obs.reward <= 1.0, (
                    f"Task 3 reward {obs.reward} out of bounds [-0.25, 1.0]"
                )

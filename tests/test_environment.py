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

    def test_step_before_reset_raises(self, env: RedditModEnvironment) -> None:
        with pytest.raises(RuntimeError, match="reset\\(\\)"):
            env.step(ModAction(action_type="approve"))

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

    def test_task3_destructive_action_suppression(
        self, env: RedditModEnvironment
    ) -> None:
        """
        perma_ban and temp_ban on a legitimate post must be suppressed to 15%
        of their computed total — far below what any proportionate wrong action
        would score (~0.47 for 'warn' on an approve post).

        We verify:
          - reward is in [0.0, 0.1] (well below non-destructive wrong actions)
          - reward is strictly less than what 'warn' would score on the same post
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
        r_warn = calculate_task3_reward(ModAction(action_type="warn"), gt, post)

        r_perma = calculate_task3_reward(ModAction(action_type="perma_ban"), gt, post)
        assert 0.0 <= r_perma <= 0.1, f"perma_ban suppressed reward out of range; got {r_perma}"
        assert r_perma < r_warn, (
            f"perma_ban ({r_perma}) should score less than warn ({r_warn}) on an approve post"
        )

        r_temp = calculate_task3_reward(
            ModAction(action_type="temp_ban", ban_duration_days=7), gt, post
        )
        assert 0.0 <= r_temp <= 0.1, f"temp_ban suppressed reward out of range; got {r_temp}"
        assert r_temp < r_warn, (
            f"temp_ban ({r_temp}) should score less than warn ({r_warn}) on an approve post"
        )

    def test_task3_suppression_scoped_to_ban_actions(
        self, env: RedditModEnvironment
    ) -> None:
        """
        warn and remove on a legitimate post are wrong but not destructive —
        they receive the full weighted score (not suppressed).
        Their reward must be higher than the suppressed perma_ban/temp_ban score.
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
        r_perma = calculate_task3_reward(ModAction(action_type="perma_ban"), gt, post)

        for action_type in ("warn", "remove"):
            reward = calculate_task3_reward(ModAction(action_type=action_type), gt, post)
            assert reward >= 0.0, f"{action_type} on approve post must be ≥ 0.0; got {reward}"
            assert reward > r_perma, (
                f"{action_type} ({reward}) should score higher than perma_ban "
                f"({r_perma}) — suppression only applies to destructive actions"
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

        # Task 3: reward is in [0.0, 1.0] — destructive actions get a 0.0 floor
        obs = env.reset(seed=7, task_level=3)
        while not obs.done:
            obs = env.step(ModAction(action_type="perma_ban"))
            if obs.reward is not None:
                assert 0.0 <= obs.reward <= 1.0, (
                    f"Task 3 reward {obs.reward} out of bounds [0.0, 1.0]"
                )


# ---------------------------------------------------------------------------
# New template coverage
# ---------------------------------------------------------------------------


class TestNewTemplates:
    """Regression tests for the borderline Task 1 and escalation Task 3 templates."""

    # ── Template presence ────────────────────────────────────────────────────

    def test_task1_borderline_templates_in_pool(self) -> None:
        from reddit_mod_env.content_generator import ContentGenerator
        gen = ContentGenerator()
        ids = {t.template_id for t in gen._registry[1]}
        assert "t1_legit_new_researcher_debut" in ids
        assert "t1_spam_academic_disguise" in ids

    def test_task3_escalation_templates_in_pool(self) -> None:
        from reddit_mod_env.content_generator import ContentGenerator
        gen = ContentGenerator()
        ids = {t.template_id for t in gen._registry[3]}
        assert "t3_escalate_veteran_account_anomaly" in ids
        assert "t3_escalate_doxxing_risk" in ids
        assert "t3_escalate_ban_evasion" in ids

    # ── Ground truth correctness ─────────────────────────────────────────────

    def test_task1_borderline_legit_correct_action(self) -> None:
        """New-researcher template must produce approve despite the new/unknown account."""
        import random
        from reddit_mod_env.content_generator import (
            ContentGenerator, _compute_task1_gt, _generate_author,
        )
        gen = ContentGenerator()
        tmpl = next(t for t in gen._registry[1] if t.template_id == "t1_legit_new_researcher_debut")
        author = _generate_author(tmpl.compatible_archetypes[0], random.Random(0))
        gt = _compute_task1_gt(tmpl, author)
        assert gt.correct_action == "approve"
        assert gt.correct_rule is None

    def test_task1_borderline_spam_correct_action(self) -> None:
        """Academic-disguise template must produce remove despite polished writing."""
        import random
        from reddit_mod_env.content_generator import (
            ContentGenerator, _compute_task1_gt, _generate_author,
        )
        gen = ContentGenerator()
        tmpl = next(t for t in gen._registry[1] if t.template_id == "t1_spam_academic_disguise")
        author = _generate_author(tmpl.compatible_archetypes[0], random.Random(0))
        gt = _compute_task1_gt(tmpl, author)
        assert gt.correct_action == "remove"
        assert gt.correct_rule == 4  # Rule 4: No Self-Promotion / Spam

    @pytest.mark.parametrize("template_id", [
        "t3_escalate_veteran_account_anomaly",
        "t3_escalate_doxxing_risk",
        "t3_escalate_ban_evasion",
    ])
    def test_task3_escalation_ground_truth(self, template_id: str) -> None:
        """Every new escalation template must produce escalate_to_senior_mod as correct action."""
        import random
        from reddit_mod_env.content_generator import (
            ContentGenerator, _compute_task3_gt, _generate_author,
        )
        gen = ContentGenerator()
        tmpl = next(t for t in gen._registry[3] if t.template_id == template_id)
        author = _generate_author(tmpl.compatible_archetypes[0], random.Random(0))
        gt = _compute_task3_gt(tmpl, author)
        assert gt.correct_action == "escalate_to_senior_mod", (
            f"{template_id}: expected escalate_to_senior_mod, got {gt.correct_action}"
        )
        assert "escalate_to_senior_mod" in gt.acceptable_actions

    # ── Reward correctness ───────────────────────────────────────────────────

    def test_escalation_cases_present_in_episodes(self) -> None:
        """A large Task 3 episode must contain at least one escalation case."""
        from reddit_mod_env.content_generator import ContentGenerator
        gen = ContentGenerator()
        scenarios = gen.generate_episode(task_level=3, num_posts=50, seed=0)
        escalation = [gt for _, gt in scenarios if gt.correct_action == "escalate_to_senior_mod"]
        assert escalation, "Expected at least one escalation case in a 50-post episode (seed=0)"

    def test_correct_escalation_scores_high(self) -> None:
        """Correctly escalating with the right rule citation must score >= 0.9."""
        from reddit_mod_env.content_generator import ContentGenerator
        from reddit_mod_env.reward import calculate_task3_reward
        gen = ContentGenerator()
        scenarios = gen.generate_episode(task_level=3, num_posts=50, seed=0)
        cases = [(p, gt) for p, gt in scenarios if gt.correct_action == "escalate_to_senior_mod"]
        assert cases, "No escalation case found — adjust seed"
        post, gt = cases[0]
        reward = calculate_task3_reward(
            ModAction(action_type="escalate_to_senior_mod", rule_cited=gt.correct_rule),
            gt,
            post,
        )
        assert reward >= 0.9, f"Correct escalation should score >= 0.9, got {reward:.4f}"

    def test_lazy_escalation_on_easy_case_penalised(self) -> None:
        """Escalating on a low-difficulty case must score less than the correct action."""
        from reddit_mod_env.content_generator import ContentGenerator
        from reddit_mod_env.reward import calculate_task3_reward
        gen = ContentGenerator()
        scenarios = gen.generate_episode(task_level=3, num_posts=100, seed=42)
        easy = [
            (p, gt) for p, gt in scenarios
            if gt.correct_action == "approve" and gt.difficulty_score < 0.75
        ]
        if not easy:
            pytest.skip("No easy approve case in sample — adjust seed")
        post, gt = easy[0]
        r_correct = calculate_task3_reward(ModAction(action_type="approve"), gt, post)
        r_lazy_esc = calculate_task3_reward(
            ModAction(action_type="escalate_to_senior_mod"), gt, post
        )
        assert r_lazy_esc < r_correct, (
            f"Lazy escalation ({r_lazy_esc:.3f}) should score below correct approve ({r_correct:.3f})"
        )

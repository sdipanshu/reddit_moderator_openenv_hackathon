"""
Reward calculation functions for each task level.

All tasks return rewards in [0.0, 1.0].

Task 3 uses a suppression factor (0.15×) rather than a hard floor when a
destructive action (perma_ban / temp_ban) is applied to a post that should
have been approved.  This preserves relative signal within the destructive
case while keeping its reward far below any proportionate wrong action.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .content_generator import GroundTruth
    from .models import ModAction, RedditPost

# Severity ordering for proportionality scoring (Task 3)
_SEVERITY_ORDER = ["approve", "warn", "remove", "temp_ban", "perma_ban"]


def calculate_task1_reward(action: "ModAction", ground_truth: "GroundTruth") -> float:
    """
    Task 1 — Spam Detection.
    Binary: 1.0 for correct action, 0.0 for wrong.
    Spam detection is inherently binary — there is no partial credit for
    missing a spam post or removing a legitimate one.
    """
    return 1.0 if action.action_type == ground_truth.correct_action else 0.0


def calculate_task2_reward(action: "ModAction", ground_truth: "GroundTruth") -> float:
    """
    Task 2 — Rule Violation Classification.

    Scoring breakdown:
      1.0  — correct action AND correct rule
      0.6  — correct action, wrong/missing rule (when a rule citation was needed)
      0.5  — acceptable action AND correct rule
      0.3  — acceptable action, wrong/missing rule
      0.1  — identified a violation but chose a bad action
      0.0  — missed violation (approved when should act) or false positive (acted when should approve)
    """
    correct_action = action.action_type == ground_truth.correct_action
    acceptable_action = action.action_type in ground_truth.acceptable_actions

    # Approve cases: no rule citation expected
    if ground_truth.correct_action == "approve":
        if action.action_type == "approve":
            return 1.0
        return 0.0  # False positive — removed legitimate content

    # Violation cases: rule citation expected
    correct_rule = (
        ground_truth.correct_rule is not None
        and action.rule_cited == ground_truth.correct_rule
    )

    if action.action_type == "approve":
        return 0.0  # Missed a real violation

    if correct_action and correct_rule:
        return 1.0
    if correct_action:
        return 0.6   # Right action but wrong or missing rule citation
    if acceptable_action and correct_rule:
        return 0.5
    if acceptable_action:
        return 0.3

    return 0.1  # Identified something was wrong but chose a bad action


def calculate_task3_reward(
    action: "ModAction",
    ground_truth: "GroundTruth",
    post: "RedditPost",
) -> float:
    """
    Task 3 — Context-Aware Moderation.

    Four weighted factors:
      - Action correctness   (40%)
      - Proportionality      (25%)
      - Rule citation        (20%)
      - Escalation judgment  (15%)

    Destructive actions (perma_ban or temp_ban) on a post whose correct action
    is 'approve' are suppressed to 15% of their computed total.  This keeps
    them far below any proportionate wrong action (~0.47 for 'warn' on an
    approve post vs. a max of ~0.037 in the destructive case) while preserving
    relative signal within the destructive category.

    All rewards are in [0.0, 1.0].
    """
    # ── Destructive-action suppression ───────────────────────────────────────
    # Banning a user whose post should have been approved is qualitatively
    # worse than any other wrong action.  Rather than zeroing out the reward
    # entirely, we compute the four weighted factors as normal and then
    # suppress the result to 15% of what they total.  This preserves relative
    # signal within the destructive case (e.g. at least citing no rule is
    # better than citing a spurious one) while keeping the reward far below
    # what any proportionate wrong action would receive (~0.47 for "warn" on
    # an approve post vs. a maximum of ~0.037 here).  All values remain in
    # [0.0, 1.0].
    _DESTRUCTIVE_SUPPRESSION = 0.15
    _is_destructive = (
        ground_truth.correct_action == "approve"
        and action.action_type in ("perma_ban", "temp_ban")
    )

    # ── Factor 1: Action correctness (0.40) ──────────────────────────────────
    if action.action_type == ground_truth.correct_action:
        f1 = 1.0
    elif action.action_type in ground_truth.acceptable_actions:
        f1 = 0.6
    else:
        f1 = 0.0

    # ── Factor 2: Proportionality (0.25) ─────────────────────────────────────
    # How far off is the chosen severity from the correct severity?
    def _severity_idx(a: str) -> int:
        return _SEVERITY_ORDER.index(a) if a in _SEVERITY_ORDER else 2

    diff = abs(_severity_idx(action.action_type) - _severity_idx(ground_truth.correct_action))
    if diff == 0:
        f2 = 1.0
    elif diff == 1:
        f2 = 0.5
    elif diff == 2:
        f2 = 0.2
    else:
        f2 = 0.0

    # ── Factor 3: Rule citation accuracy (0.20) ───────────────────────────────
    if ground_truth.correct_rule is None and action.rule_cited is None:
        f3 = 1.0  # Correctly cited no rule (approve case)
    elif ground_truth.correct_rule is not None and action.rule_cited == ground_truth.correct_rule:
        f3 = 1.0
    elif ground_truth.correct_rule is not None and action.rule_cited is not None:
        f3 = 0.2  # Cited the wrong rule
    else:
        f3 = 0.0  # Needed a rule citation but didn't provide one (or vice-versa)

    # ── Factor 4: Escalation judgment (0.15) ─────────────────────────────────
    # Reward escalating on genuinely hard/ambiguous cases; penalise lazy escalation
    if action.action_type == "escalate_to_senior_mod":
        f4 = 1.0 if ground_truth.difficulty_score >= 0.75 else 0.0
    elif ground_truth.correct_action == "escalate_to_senior_mod":
        # Handled the case autonomously when escalation was expected
        f4 = 0.4 if action.action_type in ground_truth.acceptable_actions else 0.0
    else:
        # Not an escalation scenario — base consistency on proportionality
        f4 = 1.0 if diff <= 1 else 0.3

    total = 0.40 * f1 + 0.25 * f2 + 0.20 * f3 + 0.15 * f4
    if _is_destructive:
        total *= _DESTRUCTIVE_SUPPRESSION
    return round(total, 4)

"""
Parameterized scenario generator for the Reddit Mod Bot RL environment.

Generates episodes by combining content templates with randomized author
profiles and post metadata, producing thousands of unique combinations
while remaining fully deterministic for a given seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    RedditPost,
    Report,
    SubredditContext,
    SubredditRule,
    UserHistory,
)

# ---------------------------------------------------------------------------
# Ground truth (internal — never exposed to the agent via API)
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    correct_action: str
    correct_rule: Optional[int]
    acceptable_actions: List[str]
    difficulty_score: float          # 0.0 = trivial, 1.0 = very ambiguous
    explanation: str


# ---------------------------------------------------------------------------
# r/science subreddit definition
# ---------------------------------------------------------------------------

_RULES = [
    SubredditRule(
        rule_number=1,
        title="Stay On Topic",
        description=(
            "Posts must be about scientific research or evidence-based discussion. "
            "Political commentary, personal anecdotes without scientific grounding, "
            "and off-topic content will be removed."
        ),
    ),
    SubredditRule(
        rule_number=2,
        title="No Personal Attacks",
        description=(
            "Treat all users with respect. Insults, name-calling, condescension, "
            "and ad hominem attacks are not permitted. Debate the science, not the person."
        ),
    ),
    SubredditRule(
        rule_number=3,
        title="No Misinformation",
        description=(
            "Claims must be supported by credible, peer-reviewed scientific sources. "
            "Pseudoscience, conspiracy theories, and unsourced extraordinary claims "
            "will be removed. Misrepresenting a study's conclusions is also a violation."
        ),
    ),
    SubredditRule(
        rule_number=4,
        title="No Self-Promotion or Spam",
        description=(
            "Advertising, affiliate links, and excessive self-promotion are not allowed. "
            "Researchers may share their own work only if it contributes substantively "
            "to community discussion and is not purely promotional."
        ),
    ),
    SubredditRule(
        rule_number=5,
        title="Use Correct Flair",
        description=(
            "All posts must carry the correct flair: [Research], [News], [Discussion], "
            "or [AMA]. Incorrect flair misleads readers about the nature of the post."
        ),
    ),
    SubredditRule(
        rule_number=6,
        title="No Low-Effort Content",
        description=(
            "Posts must contribute meaningfully to scientific discussion. Memes, "
            "one-liners, empty posts, and screenshots without context will be removed."
        ),
    ),
]

SCIENCE_SUBREDDIT = SubredditContext(
    name="r/science",
    rules=_RULES,
    culture=(
        "Strictly moderated academic discussion. Evidence-based discourse is expected. "
        "New accounts are subject to stricter scrutiny. Tone should remain professional."
    ),
)

# ---------------------------------------------------------------------------
# Scenario type alias
# ---------------------------------------------------------------------------

Scenario = Tuple[RedditPost, GroundTruth]


# ---------------------------------------------------------------------------
# Author archetypes + generation
# ---------------------------------------------------------------------------

_AUTHOR_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "bot_spammer": {
        "age_min": 0, "age_max": 3,
        "karma_min": -50, "karma_max": 10,
        "warnings_min": 0, "warnings_max": 0,
        "removals_min": 2, "removals_max": 8,
        "repeat_prob": 0.9, "approved_prob": 0.0,
    },
    "human_spammer": {
        "age_min": 5, "age_max": 60,
        "karma_min": 10, "karma_max": 200,
        "warnings_min": 0, "warnings_max": 2,
        "removals_min": 1, "removals_max": 4,
        "repeat_prob": 0.5, "approved_prob": 0.0,
    },
    "brand_new": {
        "age_min": 0, "age_max": 7,
        "karma_min": 1, "karma_max": 50,
        "warnings_min": 0, "warnings_max": 0,
        "removals_min": 0, "removals_max": 0,
        "repeat_prob": 0.0, "approved_prob": 0.0,
    },
    "new_genuine": {
        "age_min": 8, "age_max": 90,
        "karma_min": 50, "karma_max": 500,
        "warnings_min": 0, "warnings_max": 1,
        "removals_min": 0, "removals_max": 0,
        "repeat_prob": 0.0, "approved_prob": 0.0,
    },
    "mid_user": {
        "age_min": 100, "age_max": 800,
        "karma_min": 500, "karma_max": 5000,
        "warnings_min": 0, "warnings_max": 1,
        "removals_min": 0, "removals_max": 1,
        "repeat_prob": 0.05, "approved_prob": 0.0,
    },
    "active_contributor": {
        "age_min": 300, "age_max": 1500,
        "karma_min": 5000, "karma_max": 30000,
        "warnings_min": 0, "warnings_max": 0,
        "removals_min": 0, "removals_max": 0,
        "repeat_prob": 0.0, "approved_prob": 0.3,
    },
    "veteran_expert": {
        "age_min": 1000, "age_max": 3000,
        "karma_min": 15000, "karma_max": 80000,
        "warnings_min": 0, "warnings_max": 0,
        "removals_min": 0, "removals_max": 0,
        "repeat_prob": 0.0, "approved_prob": 0.8,
    },
    "first_offender": {
        "age_min": 200, "age_max": 1000,
        "karma_min": 1000, "karma_max": 10000,
        "warnings_min": 0, "warnings_max": 0,
        "removals_min": 0, "removals_max": 0,
        "repeat_prob": 0.0, "approved_prob": 0.0,
    },
    "mild_repeat": {
        "age_min": 100, "age_max": 500,
        "karma_min": 200, "karma_max": 2000,
        "warnings_min": 1, "warnings_max": 3,
        "removals_min": 1, "removals_max": 2,
        "repeat_prob": 1.0, "approved_prob": 0.0,
    },
    "serial_offender": {
        "age_min": 50, "age_max": 300,
        "karma_min": -100, "karma_max": 800,
        "warnings_min": 3, "warnings_max": 6,
        "removals_min": 3, "removals_max": 8,
        "repeat_prob": 1.0, "approved_prob": 0.0,
    },
}

_USERNAME_PREFIXES = [
    "curious", "science", "quantum", "bio", "astro", "neuro", "eco", "data",
    "lab", "research", "phd", "grad", "student", "reader", "skeptic",
    "empirical", "peer", "journal", "nature", "cell", "chem", "physics",
]
_USERNAME_SUFFIXES = [
    "reader", "fan", "enthusiast", "watcher", "learner", "explorer",
    "thinker", "analyst", "observer", "seeker", "buff", "geek", "hawk",
    "critic", "defender", "advocate", "debater", "curious", "nerd",
]


def _generate_author(archetype_name: str, rng: random.Random) -> UserHistory:
    """Generate a randomized UserHistory from an archetype. Always makes 9 RNG calls."""
    spec = _AUTHOR_ARCHETYPES[archetype_name]
    age = rng.randint(spec["age_min"], spec["age_max"])
    karma = rng.randint(spec["karma_min"], spec["karma_max"])
    warnings = rng.randint(spec["warnings_min"], spec["warnings_max"])
    removals = rng.randint(spec["removals_min"], spec["removals_max"])
    repeat = rng.random() < spec["repeat_prob"]
    approved = rng.random() < spec["approved_prob"]
    prefix = rng.choice(_USERNAME_PREFIXES)
    suffix = rng.choice(_USERNAME_SUFFIXES)
    num = rng.randint(0, 999)
    return UserHistory(
        username=f"{prefix}_{suffix}{num}",
        account_age_days=age,
        karma=karma,
        prior_warnings=warnings,
        prior_removals=removals,
        is_repeat_offender=repeat,
        is_approved_contributor=approved,
    )


# ---------------------------------------------------------------------------
# ContentTemplate dataclass
# ---------------------------------------------------------------------------


@dataclass
class ContentTemplate:
    """A parameterized scenario template for generating moderation scenarios."""

    template_id: str
    category: str                                    # e.g. "spam", "legit", "rule2_attack"
    title_templates: List[Optional[str]]             # list of titles with {placeholders}; None = comment
    body_template: str                               # body with {placeholders}
    placeholder_pools: Dict[str, List[str]]          # keys must match {placeholders}
    base_rule: Optional[int]                         # rule violated; None = approve
    base_action: str                                 # content-determined base action
    compatible_archetypes: List[str]                 # which author archetypes to pair with
    flair_options: List[Optional[str]]               # possible flair values
    report_configs: List[List[Tuple[str, int]]]      # list of (reason, count) configurations
    thread_context_pool: Optional[List[List[str]]]   # None for post-level scenarios
    signals: Dict[str, Any]                          # base_difficulty, severity, explanation, etc.


# ---------------------------------------------------------------------------
# Task 1 templates — Spam Detection (12 spam + 13 legit = 25 templates)
# ---------------------------------------------------------------------------


def _task1_templates() -> List[ContentTemplate]:  # noqa: PLR0915
    return [
        # ── SPAM ──────────────────────────────────────────────────────────────
        ContentTemplate(
            template_id="t1_spam_supplement_cognitive",
            category="spam",
            title_templates=[
                "Boost your {metric} with {product} — clinically tested!",
                "Scientists HATE this: {product} boosts {metric} by {pct}%!",
                "New breakthrough: {product} shown to dramatically improve {metric}",
                "{product}: the {metric} supplement Big Pharma doesn't want you to know",
            ],
            body_template=(
                "Our proprietary blend of {ingredient} has been clinically shown to enhance {metric}. "
                "Thousands of satisfied customers have already transformed their performance! "
                "Limited-time offer: use code {code} at checkout. Visit {url} now!"
            ),
            placeholder_pools={
                "product": ["NeuroMax", "BrainBoost Pro", "CogniClear", "FocusElite", "MindSharp"],
                "metric": ["cognitive performance", "IQ scores", "memory recall", "mental clarity"],
                "pct": ["40", "300", "200", "150"],
                "ingredient": ["nootropics", "phosphatidylserine", "lion's mane extract", "bacopa monnieri"],
                "code": ["REDDIT20", "SCIENCE50", "BRAIN30", "SMART15"],
                "url": ["neuromax.com/order", "brainboost.io/shop", "cogniclear.net", "focuselite.health"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["bot_spammer", "human_spammer"],
            flair_options=["Research", "Discussion"],
            report_configs=[
                [("Spam / Advertising", 7)],
                [("Spam / Advertising", 5), ("Misinformation", 2)],
                [("Self-promotion", 4)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (1, 10),
                "comments_range": (0, 3),
                "explanation": "Commercial spam promoting cognitive supplements with discount code and purchase link.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_supplement_weightloss",
            category="spam",
            title_templates=[
                "Lose {weight} lbs in {days} days — scientists won't tell you this!",
                "{product}: the weight loss breakthrough Big Pharma is hiding",
                "I lost {weight} lbs with {product} — here's the science",
                "{product} clinical study: {pct}% more effective than dieting alone",
            ],
            body_template=(
                "Big Pharma is hiding the truth about metabolism! Our all-natural {product} "
                "resets your body's fat-burning mode. Over {count} satisfied customers. "
                "Order now at {url}! Use code {code} for {pct}% off your first order."
            ),
            placeholder_pools={
                "product": ["KetoFlame", "SlimMax", "BurnFast", "TrimPro", "MetaBoost"],
                "weight": ["20", "30", "40", "50"],
                "days": ["14", "21", "30", "60"],
                "count": ["10,000", "50,000", "100,000"],
                "url": ["ketoflame.com", "slimmax.io/shop", "burnfast.health"],
                "code": ["SLIM20", "KETO30", "BURN15"],
                "pct": ["20", "30", "40"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["bot_spammer", "human_spammer"],
            flair_options=["Research", "Discussion"],
            report_configs=[
                [("Spam", 8), ("Misinformation", 5)],
                [("Advertising", 6)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (1, 8),
                "comments_range": (0, 2),
                "explanation": "Weight loss supplement spam with unsubstantiated medical claims and purchase CTA.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_crypto",
            category="spam",
            title_templates=[
                "{coin} will {claim} — here's the scientific proof",
                "Quantum AI model PROVES {coin} is entering a supercycle",
                "I've modelled {coin} using {method} — the results are stunning",
                "New {method} analysis shows {coin} will {claim} within {years} years",
            ],
            body_template=(
                "{method} analysis of market data PROVES {coin} is entering a {phase}. "
                "I've backtested this model for {years} years with consistent results. "
                "Join my {platform} for the full analysis report: {url}"
            ),
            placeholder_pools={
                "coin": ["Bitcoin", "Ethereum", "Solana", "a new AI-backed token"],
                "claim": ["hit $1M per coin", "replace gold globally", "enter a historic bull run"],
                "method": ["Quantum AI", "Fractal analysis", "Neural network", "Bayesian statistical"],
                "phase": ["supercycle", "parabolic phase", "exponential growth phase"],
                "years": ["3", "5", "7"],
                "platform": ["Telegram channel", "Discord server", "private newsletter"],
                "url": ["t.me/cryptoscience_profits", "discord.gg/cryptolab", "cryptosignals.io"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["bot_spammer", "human_spammer"],
            flair_options=["Research", "Discussion"],
            report_configs=[
                [("Spam", 5), ("Misinformation", 3)],
                [("Spam", 6)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "score_range": (1, 15),
                "comments_range": (0, 4),
                "explanation": "Crypto spam disguised as science, funnelling readers to an external channel.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_youtube_promo",
            category="spam",
            title_templates=[
                "Check out my {platform} channel — I explain {topic} in simple terms!",
                "I make {topic} content for general audiences — subscribe!",
                "New {format} on {topic} out now — weekly science content!",
                "My {platform} covers {topic} in plain language — new {format} every week",
            ],
            body_template=(
                "Hey r/science! I've been making {topic} {format} for {years} years. "
                "My latest covers {subtopic}. If you enjoy it, please subscribe and share! "
                "Link: {url}"
            ),
            placeholder_pools={
                "platform": ["YouTube", "TikTok", "podcast", "Substack"],
                "topic": ["science", "physics", "biology", "astronomy", "chemistry"],
                "format": ["videos", "podcasts", "episodes", "articles"],
                "years": ["1", "2", "3", "5"],
                "subtopic": ["quantum mechanics", "evolution", "climate change", "DNA replication"],
                "url": ["youtube.com/watch?v=abc123", "youtu.be/xyz456", "open.spotify.com/show/abc"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["brand_new", "human_spammer", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[
                [("Self-promotion", 4)],
                [("Spam / Advertising", 3)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "score_range": (1, 12),
                "comments_range": (0, 3),
                "explanation": "Pure self-promotion with social media link. No scientific content contributed.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_affiliate",
            category="spam",
            title_templates=[
                "Top {n} Science {items} You NEED in {year} (affiliate links inside)",
                "Best {items} for home science experiments — my top picks",
                "Science {items} gift guide {year} — all tested by me!",
                "The {n} best {items} for amateur scientists",
            ],
            body_template=(
                "As an Amazon Associate I earn from qualifying purchases. "
                "Here are my top picks for science {items}: [{item1}](amzn.to/{code1}), "
                "[{item2}](amzn.to/{code2}), [{item3}](amzn.to/{code3}). "
                "See my full review at {url}!"
            ),
            placeholder_pools={
                "n": ["5", "10", "15"],
                "items": ["Gadgets", "Kits", "Lab Tools", "Supplies"],
                "year": ["2024", "2025"],
                "item1": ["Microscope", "Telescope", "Spectrometer"],
                "item2": ["Chemistry kit", "pH meter", "Digital scale"],
                "item3": ["Thermometer", "Centrifuge", "Caliper"],
                "code1": ["abc123", "xyz456"],
                "code2": ["ghi012", "jkl345"],
                "code3": ["pqr901", "stu234"],
                "url": ["sciencegadgets.blog", "labtools.review"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["bot_spammer", "human_spammer"],
            flair_options=["Discussion"],
            report_configs=[
                [("Spam / Affiliate links", 6)],
                [("Advertising", 4)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (1, 8),
                "comments_range": (0, 2),
                "explanation": "Affiliate link spam — explicitly monetised advertising in a science forum.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_course",
            category="spam",
            title_templates=[
                "Learn {topic} in {weeks} weeks — my online course is live!",
                "I teach {topic} online — new cohort starting {month}",
                "Master {topic} from scratch with my bootcamp",
                "My {topic} course helped {n} students break into the field",
            ],
            body_template=(
                "After {years} years in {field}, I've distilled everything into a {weeks}-week online program. "
                "Learn {topic} from first principles. No background required! "
                "Use code {code} for {pct}% off — enroll at {url}."
            ),
            placeholder_pools={
                "topic": ["data science", "bioinformatics", "computational biology", "science communication"],
                "weeks": ["4", "6", "8", "12"],
                "n": ["500", "1000", "2000"],
                "month": ["January", "February", "March"],
                "years": ["5", "10", "15"],
                "field": ["research", "biotech", "academia"],
                "code": ["REDDIT30", "SCIENCE20", "LEARN25"],
                "pct": ["20", "30", "40"],
                "url": ["datasciencecourse.io", "bioinformatics.training", "scicomm.academy"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["human_spammer", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[
                [("Spam / Advertising", 5)],
                [("Self-promotion", 4)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "score_range": (1, 15),
                "comments_range": (0, 4),
                "explanation": "Course advertising using science-adjacent framing as cover for commercial promotion.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_fake_study",
            category="spam",
            title_templates=[
                "New study: {supplement} reverses {disease} — results are shocking",
                "Study from {institute} proves {supplement} cures {disease}",
                "Researchers prove {supplement} is {pct}% more effective than {drug}",
                "Peer-reviewed study: {supplement} shown to {claim}",
            ],
            body_template=(
                "Researchers at the {institute} Institute for Natural Medicine have proven "
                "that {supplement} {claim}. Effect size: {pct}% improvement over placebo. "
                "This breakthrough has been suppressed by {villain}. "
                "Full study at {url}."
            ),
            placeholder_pools={
                "supplement": ["our herbal blend", "colloidal silver", "MMS solution", "turpentine therapy"],
                "disease": ["cancer", "diabetes", "Alzheimer's", "HIV"],
                "institute": ["International", "Global", "Alternative", "Independent"],
                "drug": ["chemotherapy", "insulin", "statins", "antidepressants"],
                "claim": ["eliminates cancer cells", "reverses aging", "cures viral infections"],
                "pct": ["340", "500", "800"],
                "villain": ["Big Pharma", "the FDA", "mainstream medicine"],
                "url": ["naturalcure.net/study", "altmed.institute/research"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["bot_spammer", "human_spammer", "new_genuine"],
            flair_options=["Research"],
            report_configs=[
                [("Spam", 6), ("Misinformation", 8)],
                [("Misinformation / Pseudoscience", 10)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "score_range": (1, 30),
                "comments_range": (0, 8),
                "explanation": "Fabricated study from a fake institute promoting unproven supplements with conspiracy framing.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_job_scam",
            category="spam",
            title_templates=[
                "Earn ${amount}/week working in science from home!",
                "{company} is hiring {role}s — no experience needed",
                "Work-from-home science jobs: earn ${amount}/month",
                "Science professionals: earn extra income with {company}",
            ],
            body_template=(
                "{company} is looking for {role}s. Earn up to ${amount}/week working just {hours} hours. "
                "No experience needed — we provide full training. "
                "Apply at {url} before {deadline}. Limited spots available!"
            ),
            placeholder_pools={
                "company": ["ScienceHub Inc", "BioResearch Partners", "DataScience Pro"],
                "role": ["science writers", "data annotators", "content reviewers"],
                "amount": ["500", "1000", "2000"],
                "hours": ["2", "3", "4"],
                "url": ["sciencehub.jobs", "bioresearch.work"],
                "deadline": ["Friday", "end of month", "January 15"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["bot_spammer", "human_spammer"],
            flair_options=["Discussion"],
            report_configs=[
                [("Spam / Job scam", 5)],
                [("Spam", 4), ("Scam", 3)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "score_range": (1, 10),
                "comments_range": (0, 2),
                "explanation": "Job scam / income opportunity spam targeting the science community.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_miracle_cure",
            category="spam",
            title_templates=[
                "This {ingredient} cured my {condition} — doctors can't explain it",
                "{ingredient}: the natural cure for {condition} that medicine ignores",
                "I was diagnosed with {condition} — {ingredient} saved my life",
                "Grandmother's remedy: {ingredient} heals {condition} in {weeks} weeks",
            ],
            body_template=(
                "I was diagnosed with {condition} {years} years ago. Doctors said {prognosis}. "
                "Then I discovered {ingredient}. After {weeks} weeks, my symptoms were gone! "
                "I share my full protocol at {url}. Spread the word — {villain} doesn't want you to know."
            ),
            placeholder_pools={
                "ingredient": ["turmeric", "baking soda", "colloidal silver", "essential oils"],
                "condition": ["stage 4 cancer", "Type 2 diabetes", "multiple sclerosis", "Parkinson's"],
                "years": ["2", "3", "5"],
                "prognosis": ["I had 6 months to live", "there was no cure", "I'd need medication forever"],
                "weeks": ["2", "3", "4"],
                "url": ["naturalhealing.net", "curedmyself.com"],
                "villain": ["Big Pharma", "the medical establishment", "your doctor"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["human_spammer", "new_genuine"],
            flair_options=["Discussion", "Research"],
            report_configs=[
                [("Misinformation / Dangerous medical advice", 7)],
                [("Spam", 3), ("Misinformation", 5)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "score_range": (2, 25),
                "comments_range": (1, 10),
                "explanation": "Miracle cure anecdote with anti-medicine conspiracy angle and promotional URL.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_supplement_immune",
            category="spam",
            title_templates=[
                "{product}: the immune booster science doesn't want you to know about",
                "Boost your immune system naturally with {product}",
                "I haven't been sick in {years} years since taking {product}",
                "{product} study: {pct}% fewer infections in trial participants",
            ],
            body_template=(
                "Our {product} formula combines {ingredients} for maximum immune support. "
                "Clinical data shows {pct}% reduction in infections. "
                "Completely natural, doctor-formulated. Order at {url} — code {code} saves {disc}%."
            ),
            placeholder_pools={
                "product": ["ImmunoMax", "DefendPro", "ShieldPlus", "GuardianBlend"],
                "ingredients": ["elderberry, zinc, and vitamin C", "echinacea and medicinal mushrooms"],
                "pct": ["60", "70", "80"],
                "years": ["2", "3", "5"],
                "url": ["immunomax.health", "defendpro.com/shop"],
                "code": ["IMMUNE20", "PROTECT25"],
                "disc": ["20", "25"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["bot_spammer", "human_spammer"],
            flair_options=["Research", "Discussion"],
            report_configs=[
                [("Spam / Advertising", 6)],
                [("Misinformation", 4), ("Spam", 3)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (1, 8),
                "comments_range": (0, 2),
                "explanation": "Immune supplement spam with dubious clinical claims and purchase call-to-action.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_app_survey",
            category="spam",
            title_templates=[
                "Help with my {topic} research — takes {duration}!",
                "Quick survey about {topic} for my PhD thesis",
                "I built a {apptype} for {topic} enthusiasts — feedback welcome!",
                "New science {apptype} — try it free (I'm the developer)",
            ],
            body_template=(
                "Hi everyone! I'm working on {project} and need participants. "
                "It's a quick {duration} survey about {topic}. All responses are anonymous. "
                "Link: {url} — participants receive a {reward}!"
            ),
            placeholder_pools={
                "topic": ["science communication", "climate change attitudes", "vaccine awareness"],
                "apptype": ["app", "website", "tool", "browser extension"],
                "project": ["my undergraduate thesis", "a startup", "a research project"],
                "duration": ["2-minute", "5-minute", "10-minute"],
                "url": ["forms.gle/xyz123", "typeform.com/to/abc", "survey.qualtrics.com/jfe/abc"],
                "reward": ["free ebook", "chance to win a gift card", "free premium access"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine", "human_spammer"],
            flair_options=["Discussion"],
            report_configs=[
                [("Spam / Data collection", 3)],
                [("Self-promotion", 4)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "score_range": (1, 10),
                "comments_range": (0, 3),
                "explanation": "Survey/app spam using academic framing to collect data or attract users.",
            },
        ),
        ContentTemplate(
            template_id="t1_spam_fitness_supplement",
            category="spam",
            title_templates=[
                "{product}: the pre-workout supplement with real science behind it",
                "Athletes: {product} shown to improve {metric} in clinical study",
                "Increase your {metric} by {pct}% naturally with {product}",
                "{product} vs {drug}: new study says {product} wins",
            ],
            body_template=(
                "Our {product} formula is backed by {n} clinical studies showing {pct}% improvements in {metric}. "
                "Made with {ingredients}. Zero banned substances, fully certified. "
                "Use code {code} at {url} for free shipping."
            ),
            placeholder_pools={
                "product": ["MuscleScience Pro", "AthleteEdge", "PeakPerformance", "StrengthLab"],
                "metric": ["strength output", "endurance", "recovery time", "VO2 max"],
                "pct": ["23", "35", "47"],
                "n": ["5", "12", "17"],
                "drug": ["creatine", "caffeine", "protein shakes"],
                "ingredients": ["creatine HCl and beta-alanine", "nitric oxide precursors"],
                "code": ["ATHLETE15", "MUSCLE20"],
                "url": ["musclescience.com", "athleteedge.shop"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["bot_spammer", "human_spammer"],
            flair_options=["Research", "Discussion"],
            report_configs=[
                [("Spam / Advertising", 5)],
                [("Spam", 4), ("Misinformation", 2)],
            ],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (1, 10),
                "comments_range": (0, 3),
                "explanation": "Sports supplement spam using scientific-sounding claims to drive product sales.",
            },
        ),

        # ── LEGITIMATE ────────────────────────────────────────────────────────
        ContentTemplate(
            template_id="t1_legit_published_research",
            category="legit",
            title_templates=[
                "New {journal} paper: {finding} in {field}",
                "Just published in {journal}: {finding}",
                "{field} breakthrough: {finding} [open access]",
                "Paper in {journal} — {finding}",
            ],
            body_template=(
                "Our group just published a paper in {journal} on {topic}. "
                "We found {finding} using {method}. "
                "Full text at {doi}. Happy to answer questions — I'm one of the authors."
            ),
            placeholder_pools={
                "journal": ["Nature", "Science", "Cell", "PNAS", "NEJM", "Physical Review Letters"],
                "finding": [
                    "a novel gene regulatory mechanism controlling cell differentiation",
                    "an unexpected link between gut microbiome composition and cognitive function",
                    "a statistical mechanism underlying the replication crisis in psychology",
                    "a new exoplanet with promising biosignature potential",
                ],
                "field": ["molecular biology", "astrophysics", "neuroscience", "psychology"],
                "topic": ["gene regulation", "microbiome-brain axis", "research methodology", "exoplanet atmospheres"],
                "method": ["single-cell RNA sequencing", "longitudinal cohort study", "meta-analysis", "JWST spectroscopy"],
                "doi": ["doi.org/10.1038/s41586-2024-abc", "doi.org/10.1126/science.abc123"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["active_contributor", "veteran_expert", "new_genuine"],
            flair_options=["Research"],
            report_configs=[[], [("Self-promotion", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (500, 10000),
                "comments_range": (50, 400),
                "explanation": "Legitimate research post with DOI, published in a top-tier journal, author engaged with community.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_preprint",
            category="legit",
            title_templates=[
                "Preprint: {finding} — happy to discuss methodology",
                "bioRxiv preprint: {finding} — not yet peer reviewed",
                "New preprint on {topic}: {finding}",
                "Preprint posted to {server}: {finding}",
            ],
            body_template=(
                "My team just posted a preprint to {server} ({doi}). "
                "We investigated {topic} and found {finding}. "
                "This has not yet been peer reviewed — please keep that in mind. "
                "Raw data and code are on {repo}."
            ),
            placeholder_pools={
                "server": ["bioRxiv", "arXiv", "medRxiv", "ChemRxiv"],
                "doi": ["doi.org/10.1101/2024.03.15.xyz", "arxiv.org/abs/2401.01234"],
                "topic": ["CRISPR off-target effects", "protein folding pathways", "antibiotic resistance mechanisms"],
                "finding": [
                    "a novel CRISPR off-target detection method using long-read sequencing",
                    "a key folding intermediate previously thought to be transient",
                    "a new resistance gene transferred horizontally between pathogens",
                ],
                "repo": ["GitHub.com/lab/project", "Zenodo", "Figshare"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["new_genuine", "mid_user", "active_contributor"],
            flair_options=["Research"],
            report_configs=[[], [("Self-promotion", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (100, 3000),
                "comments_range": (10, 100),
                "explanation": "Legitimate preprint with appropriate caveat, data publicly available.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_science_news",
            category="legit",
            title_templates=[
                "{outlet}: {headline}",
                "{headline} [{outlet}]",
                "Scientists at {institution} {headline}",
            ],
            body_template=(
                "From {outlet}: {headline}. "
                "The study, published in {journal}, involved {n} participants over {period}. "
                "Lead researcher Dr. {name} said: '{quote}'. Full study: {doi}."
            ),
            placeholder_pools={
                "outlet": ["The Guardian Science", "BBC Science", "New Scientist", "Scientific American", "Nature News"],
                "headline": [
                    "discover mechanism linking gut bacteria to mood",
                    "confirm deep-sea organism survives without oxygen for months",
                    "develop battery technology with 10x the energy density",
                    "find evidence of ancient ocean on Mars",
                ],
                "institution": ["MIT", "Oxford University", "Harvard", "ETH Zurich"],
                "journal": ["Nature", "Science", "PNAS", "Current Biology"],
                "n": ["1,200", "4,500", "12,000"],
                "period": ["5 years", "10 years", "15 years"],
                "name": ["Smith", "Johnson", "Chen", "Patel"],
                "quote": [
                    "This is a major step forward in our understanding",
                    "Further replication is needed before clinical applications",
                    "We were surprised by the robustness of the effect",
                ],
                "doi": ["doi.org/10.1038/s41586-2024-abc", "doi.org/10.1126/science.abc123"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "new_genuine"],
            flair_options=["News"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (200, 8000),
                "comments_range": (30, 300),
                "explanation": "Legitimate science news from a credible outlet with primary source reference.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_methodology_question",
            category="legit",
            title_templates=[
                "Question about {method} methodology in {field} research",
                "Can someone explain why {method} is used for {application}?",
                "Confused about the statistical approach in {field} — can anyone help?",
            ],
            body_template=(
                "I'm trying to understand why researchers in {field} use {method} for {application}. "
                "I've been reading papers and they seem to assume familiarity. "
                "Specifically: {question}. Any resources or explanations would be really appreciated!"
            ),
            placeholder_pools={
                "field": ["epidemiology", "genetics", "neuroscience", "climate science"],
                "method": ["Mendelian randomisation", "bootstrapping", "Bayesian inference", "multi-level modelling"],
                "application": ["causal inference", "controlling confounders", "hierarchical data"],
                "question": [
                    "How do you choose the right number of bootstrap samples?",
                    "What assumptions does this method violate in small samples?",
                    "When is this approach preferred over simpler alternatives?",
                ],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["brand_new", "new_genuine", "mid_user"],
            flair_options=["Discussion"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (50, 1000),
                "comments_range": (5, 80),
                "explanation": "Genuine methodology question demonstrating prior research effort and intellectual curiosity.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_researcher_ama",
            category="legit",
            title_templates=[
                "I'm a {role} at {institution} working on {topic} — AMA!",
                "Science AMA: {role} studying {topic}",
                "{role} here, I work on {topic} at {institution} — AMA",
            ],
            body_template=(
                "Hi r/science! I'm a {role} at {institution} specialising in {topic}. "
                "I've been doing this for {years} years and recently published {n} papers on the subject. "
                "Recent highlight: {highlight}. I'll be answering questions for the next few hours — ask me anything!"
            ),
            placeholder_pools={
                "role": ["research scientist", "postdoctoral researcher", "professor", "NASA scientist"],
                "institution": ["MIT", "Oxford University", "Harvard Medical School", "CERN", "NASA JPL"],
                "topic": ["exoplanet atmospheres", "cancer immunotherapy", "quantum computing"],
                "years": ["5", "10", "15", "20"],
                "n": ["15", "30", "50"],
                "highlight": [
                    "we recently discovered a new class of habitable zone exoplanets",
                    "our lab developed a new CAR-T cell therapy entering Phase III trials",
                    "we achieved breakthrough coherence times in our quantum processor",
                ],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["active_contributor", "veteran_expert", "mid_user"],
            flair_options=["AMA"],
            report_configs=[[], [("Unverified credentials", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (500, 15000),
                "comments_range": (100, 1000),
                "explanation": "Legitimate AMA from a credentialed researcher at a named institution.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_student_question",
            category="legit",
            title_templates=[
                "How does {concept} actually work? I can't find a good explanation.",
                "Can someone explain {concept} at a conceptual level?",
                "I'm studying {field} and confused about {concept}",
            ],
            body_template=(
                "I'm a {level} student studying {field} and confused about {concept}. "
                "My textbook says {claim} but I've seen papers suggesting {counterclaim}. "
                "Can someone explain which is correct, or point me to good resources? "
                "I've been reading about this for weeks."
            ),
            placeholder_pools={
                "level": ["undergraduate", "first-year graduate", "high school"],
                "field": ["biology", "chemistry", "physics", "neuroscience"],
                "concept": ["mRNA vaccine mechanisms", "CRISPR specificity", "Hawking radiation", "quantum entanglement"],
                "claim": ["the mechanism is fully understood", "this is a settled question"],
                "counterclaim": ["there are competing models", "key steps are still debated"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["brand_new", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (50, 2000),
                "comments_range": (10, 150),
                "explanation": "Genuine student question with evidence of prior research effort.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_meta_analysis",
            category="legit",
            title_templates=[
                "Meta-analysis of {n} studies on {topic}: what we actually know",
                "New systematic review: {topic} — pooled evidence from {n} trials",
                "Updated meta-analysis ({n} RCTs): {topic}",
            ],
            body_template=(
                "A new systematic review pooling {n} studies ({total} participants) on {topic} "
                "was published in {journal}. Key findings: {finding}. "
                "Heterogeneity was {heterogeneity}. Authors note {caveat}. Full text: {doi}."
            ),
            placeholder_pools={
                "n": ["12", "24", "47", "83"],
                "total": ["2,400", "15,000", "47,000"],
                "topic": ["intermittent fasting and metabolic health", "exercise and depression", "omega-3 and cardiovascular outcomes"],
                "journal": ["The Lancet", "NEJM", "JAMA", "BMJ"],
                "finding": [
                    "a modest but significant effect (SMD = 0.34, 95% CI: 0.21-0.47)",
                    "no significant effect after accounting for publication bias",
                    "strong short-term evidence, unclear long-term outcomes",
                ],
                "heterogeneity": ["low (I2 = 12%)", "moderate (I2 = 48%)", "high (I2 = 79%)"],
                "caveat": [
                    "most included trials had high risk of bias",
                    "effect sizes were larger in industry-funded trials",
                ],
                "doi": ["doi.org/10.1016/S0140-6736(24)abc", "doi.org/10.1056/NEJMoa2401abc"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "veteran_expert"],
            flair_options=["Research"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (300, 8000),
                "comments_range": (30, 300),
                "explanation": "Rigorous meta-analysis with appropriate statistical reporting and honest caveats.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_dataset_release",
            category="legit",
            title_templates=[
                "We've open-sourced our {n}-sample {topic} dataset",
                "Dataset release: {n} samples from our {topic} research — free to use",
                "Open data: full {topic} dataset ({n} observations) now on {repo}",
            ],
            body_template=(
                "As part of our commitment to open science, we're releasing the full dataset from our {topic} study. "
                "It contains {n} observations collected over {period} at {institution}. "
                "Available on {repo}: {url}. Data dictionary and preprocessing scripts included."
            ),
            placeholder_pools={
                "topic": ["longitudinal microbiome", "fMRI connectivity", "climate temperature", "exoplanet radial velocity"],
                "n": ["10,000", "50,000", "100,000"],
                "period": ["2 years", "5 years", "10 years"],
                "institution": ["MIT", "Stanford", "Oxford"],
                "repo": ["Zenodo", "Figshare", "OSF", "GitHub"],
                "url": ["zenodo.org/record/abc123", "figshare.com/articles/dataset/xyz"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["active_contributor", "veteran_expert", "mid_user"],
            flair_options=["Research"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (200, 5000),
                "comments_range": (20, 200),
                "explanation": "Open science data release — exemplary transparency and community contribution.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_replication_study",
            category="legit",
            title_templates=[
                "We successfully replicated {study}: what it means for {field}",
                "Failed replication of {study} — implications for {field}",
                "Replication attempt: {study} ({result})",
            ],
            body_template=(
                "Our group spent {period} attempting to replicate {study} in {field} (pre-registered: {osf}). "
                "We used identical materials and procedures. Result: {result}. "
                "Original effect size: {original_es}. Ours: {rep_es}. "
                "We believe {interpretation}. Full write-up: {doi}."
            ),
            placeholder_pools={
                "study": ["the ego depletion effect", "power posing's hormonal effects", "social priming effects on behaviour"],
                "field": ["social psychology", "cognitive psychology", "nutrition science"],
                "period": ["18 months", "2 years", "3 years"],
                "osf": ["osf.io/abc123", "osf.io/xyz456"],
                "result": [
                    "successful replication (d = 0.41)",
                    "failed replication (d = 0.02, 95% CI includes zero)",
                    "partial replication with a smaller effect",
                ],
                "original_es": ["d = 0.45", "r = 0.51"],
                "rep_es": ["d = 0.41", "d = 0.02", "d = 0.22"],
                "interpretation": [
                    "the original finding is robust",
                    "the original was underpowered and overclaiming",
                    "contextual moderators were inadequately controlled",
                ],
                "doi": ["doi.org/10.1177/xyz", "doi.org/10.1016/j.psp.2024.abc"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["active_contributor", "veteran_expert", "mid_user"],
            flair_options=["Research"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (300, 6000),
                "comments_range": (30, 250),
                "explanation": "Pre-registered replication study with honest reporting of results, positive or negative.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_science_discussion",
            category="legit",
            title_templates=[
                "Discussion: what does the evidence actually say about {topic}?",
                "Is {claim} consensus yet in {field}?",
                "How has {discovery} changed the way we think about {topic}?",
            ],
            body_template=(
                "I've been following the debate about {topic} in {field} and want to understand the current state of evidence. "
                "{framing}. Key papers I've read: {paper1}, {paper2}. "
                "What's the consensus, or is this still genuinely contested?"
            ),
            placeholder_pools={
                "field": ["nutrition science", "psychology", "climate science", "neuroscience"],
                "topic": ["dietary fat and heart disease", "the serotonin hypothesis of depression", "multitasking"],
                "discovery": ["CRISPR", "optogenetics", "AlphaFold"],
                "claim": ["gut bacteria influence mood", "sleep consolidates memory", "exercise relieves depression"],
                "framing": [
                    "Some recent meta-analyses seem to challenge what I thought was consensus",
                    "Popular media presents this as settled but I see disagreement in the literature",
                ],
                "paper1": ["Ioannidis (2005) 'Why Most Published Research Findings Are False'",
                           "Moncrieff et al. (2022) on serotonin and depression"],
                "paper2": ["Cumming (2014) on the new statistics",
                           "Open Science Collaboration (2015) reproducibility project"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (100, 3000),
                "comments_range": (20, 200),
                "explanation": "Thoughtful evidence-seeking discussion citing real scientific literature.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_phd_preprint",
            category="legit",
            title_templates=[
                "PhD chapter preprint: {finding} — looking for feedback",
                "First time posting: my preprint on {topic} just went live",
                "Cross-posting my preprint: {topic} — first time on r/science",
            ],
            body_template=(
                "First post here! I just uploaded my preprint ({doi}) on {topic}. "
                "Main finding: {finding}. This is part of my PhD at {institution}. "
                "I'd really appreciate feedback — especially on {question}."
            ),
            placeholder_pools={
                "topic": ["tardigrade desiccation tolerance", "deep-sea chemosynthetic ecosystems", "mRNA decay pathways"],
                "finding": [
                    "a novel protein enabling tardigrades to survive extreme dehydration",
                    "a previously undescribed chemosynthetic bacterial species",
                    "a new mechanism for mRNA degradation under stress",
                ],
                "doi": ["doi.org/10.1101/2024.03.15.abc", "doi.org/10.1101/2024.04.01.xyz"],
                "institution": ["University of Edinburgh", "ETH Zurich", "University of Tokyo", "UC Berkeley"],
                "question": [
                    "the statistical approach in section 3",
                    "whether we should add additional controls",
                    "the framing of the discussion section",
                ],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["brand_new", "new_genuine"],
            flair_options=["Research"],
            report_configs=[[], [("Self-promotion", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (100, 2000),
                "comments_range": (10, 100),
                "explanation": "PhD student's preprint with appropriate caveat and genuine invitation for peer feedback.",
            },
        ),
        ContentTemplate(
            template_id="t1_legit_official_report",
            category="legit",
            title_templates=[
                "{organisation} releases new report: {headline}",
                "New {report} from {organisation}: key findings",
                "{organisation}: {headline}",
            ],
            body_template=(
                "The {organisation} today released its {report}. Key findings: "
                "{finding1}. {finding2}. "
                "Full report at {url}."
            ),
            placeholder_pools={
                "organisation": ["IPCC", "WHO", "CDC", "NIH", "ESA", "NASA"],
                "report": ["Sixth Assessment Report", "annual report", "special climate report"],
                "headline": [
                    "global temperatures on track for 2.7C rise without urgent action",
                    "antimicrobial resistance kills more people annually than HIV and malaria combined",
                    "rates of childhood obesity have tripled since 1975",
                ],
                "finding1": [
                    "Global mean surface temperature increased 1.1C above pre-industrial levels",
                    "AMR causes 1.3 million deaths annually",
                    "Childhood obesity affects 340 million children aged 5-19 globally",
                ],
                "finding2": [
                    "The rate of warming has accelerated versus previous decades",
                    "Low-income countries are disproportionately affected",
                    "Dietary changes and inactivity are primary contributors",
                ],
                "url": ["ipcc.ch/report/ar6", "who.int/publications/abc", "nasa.gov/science/reports/2024"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "new_genuine"],
            flair_options=["News"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "score_range": (500, 12000),
                "comments_range": (50, 600),
                "explanation": "Official scientific organisation report with primary source links.",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Task 2 templates — Rule Violation Classification (35 templates)
# ---------------------------------------------------------------------------


def _task2_templates() -> List[ContentTemplate]:  # noqa: PLR0915
    return [
        # ── Rule 1: Off-topic (5) ─────────────────────────────────────────────
        ContentTemplate(
            template_id="t2_r1_political_advocacy",
            category="rule1_offtopic",
            title_templates=[
                "We need carbon taxes NOW — the science is settled, politicians are criminals",
                "The IPCC report demands immediate policy action — contact your representatives",
                "Climate scientists agree: current policy is genocide. We must act.",
            ],
            body_template=(
                "The latest {report} data is unambiguous. Every year of inaction is a moral crime. "
                "I'm calling on everyone here to {action}. "
                "Science demands we hold {target} accountable. Share this everywhere."
            ),
            placeholder_pools={
                "report": ["IPCC", "NASA", "NOAA", "WMO"],
                "action": ["call your MP", "vote out climate deniers", "join protests", "boycott fossil fuels"],
                "target": ["politicians", "fossil fuel companies", "lobbyists", "governments"],
            },
            base_rule=1,
            base_action="remove",
            compatible_archetypes=["new_genuine", "mid_user", "first_offender"],
            flair_options=["Discussion"],
            report_configs=[[("Political advocacy / Off-topic", 5)], [("Not science content", 3)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.2,
                "base_acceptable": ["remove"],
                "explanation": "Political advocacy post. Climate science is on-topic; climate policy activism is not.",
            },
        ),
        ContentTemplate(
            template_id="t2_r1_gaming_post",
            category="rule1_offtopic",
            title_templates=[
                "Just finished playing '{game}' — best science game ever!",
                "Anyone else play '{game}'? The science is surprisingly accurate!",
                "'{game}' — the most realistic science simulation I've played",
            ],
            body_template=(
                "OMG this game lets you {activity}. It's so realistic! "
                "Anyone else played it? What's your favourite {aspect} to {verb}?"
            ),
            placeholder_pools={
                "game": ["Science Simulator 2024", "Lab Tycoon", "Biotech Inc", "Quantum Quest"],
                "activity": ["run your own lab and apply for grants", "do peer review", "design experiments", "manage a research team"],
                "aspect": ["experiment", "research area", "lab upgrade", "grant application"],
                "verb": ["run", "complete", "unlock", "optimise"],
            },
            base_rule=1,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine", "mid_user"],
            flair_options=["Discussion"],
            report_configs=[[("Not science content", 4)], [("Off-topic", 5)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["remove"],
                "explanation": "Video game post with no scientific content — clearly off-topic for r/science.",
            },
        ),
        ContentTemplate(
            template_id="t2_r1_personal_tribute",
            category="rule1_offtopic",
            title_templates=[
                "My grandmother just passed away. She was a {profession}.",
                "Rest in peace to my mentor, a lifelong {profession}",
                "Sharing this in memory of my {relative}, a dedicated {profession}",
            ],
            body_template=(
                "I know this isn't science news but I needed to share. "
                "She spent {years} years studying {topic}. She always said {quote}. Miss you."
            ),
            placeholder_pools={
                "profession": ["biologist", "chemist", "physicist", "science teacher", "microbiologist"],
                "years": ["30", "35", "40", "45"],
                "topic": ["marine organisms", "plant genetics", "atmospheric chemistry", "particle physics"],
                "relative": ["grandmother", "mentor", "father", "professor"],
                "quote": [
                    "the ocean was her first love",
                    "every experiment is a conversation with nature",
                    "curiosity is the only tool you need",
                ],
            },
            base_rule=1,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine", "mid_user"],
            flair_options=["Discussion"],
            report_configs=[[("Off-topic / Personal", 3)], [("Not science content", 4)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.25,
                "base_acceptable": ["remove"],
                "explanation": "Sympathetic but off-topic personal tribute. r/science requires scientific discussion.",
            },
        ),
        ContentTemplate(
            template_id="t2_r1_sports_post",
            category="rule1_offtopic",
            title_templates=[
                "Can anyone explain the science behind {athlete}'s insane {feat}?",
                "Is {sport} the most physically demanding sport? Science says...",
                "The training regime behind {team}'s {season} season — impressive!",
            ],
            body_template=(
                "Watched the game last night and {athlete} did something unbelievable. "
                "I'm not a sports person normally but {observation}. "
                "Is there any science to explain this? What do you guys think?"
            ),
            placeholder_pools={
                "athlete": ["this player", "the quarterback", "the gymnast", "the sprinter"],
                "feat": ["vertical jump", "reaction time", "endurance", "strength output"],
                "sport": ["gymnastics", "ultramarathon running", "competitive swimming", "decathlon"],
                "team": ["the winning team's", "the underdog team's", "the champion's"],
                "season": ["championship", "record-breaking", "comeback"],
                "observation": ["the athleticism seemed superhuman", "the recovery speed was mind-blowing"],
            },
            base_rule=1,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[[("Off-topic", 3)], [("Not science content", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.15,
                "base_acceptable": ["remove"],
                "explanation": "Sports fan post. Sports science is on-topic; sports commentary without science is not.",
            },
        ),
        ContentTemplate(
            template_id="t2_r1_recipe_food",
            category="rule1_offtopic",
            title_templates=[
                "My grandmother's {dish} recipe — anyone know the science behind it?",
                "Best {cuisine} food for brain health?",
                "Is fermented {food} actually healthy? Asking for a recipe swap",
            ],
            body_template=(
                "I've been making {dish} for years but I'm curious about the {aspect}. "
                "Any scientists here who can explain the {process}? "
                "Also happy to swap recipes if anyone's interested!"
            ),
            placeholder_pools={
                "dish": ["sourdough bread", "kimchi", "bone broth", "kombucha"],
                "cuisine": ["Mediterranean", "Japanese", "Nordic"],
                "food": ["kimchi", "kefir", "miso", "yoghurt"],
                "aspect": ["fermentation chemistry", "Maillard reaction", "nutritional profile"],
                "process": ["chemistry", "biology", "fermentation process"],
            },
            base_rule=1,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine", "mid_user"],
            flair_options=["Discussion"],
            report_configs=[[("Off-topic", 3)], [("Not science content", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.2,
                "base_acceptable": ["remove"],
                "explanation": "Recipe/food post. Food science questions are borderline; recipe swaps are off-topic.",
            },
        ),

        # ── Rule 2: Personal attacks (5) ─────────────────────────────────────
        ContentTemplate(
            template_id="t2_r2_direct_insult",
            category="rule2_attack",
            title_templates=[None],
            body_template=(
                "You're an absolute {insult} if you still believe {claim}. "
                "This has been debunked {n} times. People like you are literally {harm} with your {label}."
            ),
            placeholder_pools={
                "insult": ["idiot", "moron", "fool", "dunce"],
                "claim": ["vaccines cause autism", "evolution isn't real", "climate change is a hoax"],
                "n": ["thousands of", "hundreds of", "countless"],
                "harm": ["killing children", "endangering lives", "spreading disease"],
                "label": ["stupidity", "ignorance", "delusion"],
            },
            base_rule=2,
            base_action="remove",
            compatible_archetypes=["mid_user", "first_offender", "mild_repeat"],
            flair_options=[None],
            report_configs=[[("Personal attack / Insult", 9)], [("Harassment", 6)]],
            thread_context_pool=[
                ["Original post: 'I'm concerned about vaccine safety, can someone point me to research?'"],
                ["Original post: 'I don't believe in evolution — can someone change my mind?'"],
                ["Original post: 'I think climate change data is manipulated.'"],
            ],
            signals={
                "base_difficulty": 0.2,
                "severity": "moderate",
                "base_acceptable": ["remove", "warn"],
                "explanation": "Direct personal insult targeting the commenter, not the argument. Remove is appropriate.",
            },
        ),
        ContentTemplate(
            template_id="t2_r2_condescending",
            category="rule2_attack",
            title_templates=[None],
            body_template=(
                "Maybe if you actually {suggestion} instead of {failing}, you'd understand "
                "why your interpretation is completely wrong. This is embarrassingly {adjective} stuff."
            ),
            placeholder_pools={
                "suggestion": ["read the paper instead of just the abstract", "studied statistics", "checked the original data"],
                "failing": ["skimming the abstract", "relying on science journalists", "reading Twitter summaries"],
                "adjective": ["basic", "elementary", "undergraduate-level"],
            },
            base_rule=2,
            base_action="warn",
            compatible_archetypes=["mid_user", "first_offender", "active_contributor"],
            flair_options=[None],
            report_configs=[[("Condescending / Rude", 3)], [("Personal attack", 2)]],
            thread_context_pool=[
                ["Original post: Discussion about a recent meta-analysis on antidepressants"],
                ["Thread: debate about the statistical methods in a high-profile paper"],
                ["Discussion thread about misrepresentation of a study in popular media"],
            ],
            signals={
                "base_difficulty": 0.3,
                "severity": "mild",
                "base_acceptable": ["warn", "remove"],
                "explanation": "Condescending and dismissive but no direct profanity. A warning is appropriate first action.",
            },
        ),
        ContentTemplate(
            template_id="t2_r2_fraud_accusation",
            category="rule2_attack",
            title_templates=[None],
            body_template=(
                "The lead author of this study is a known {label} who has had {n} papers retracted. "
                "I wouldn't trust anything they publish. Their entire research programme is {epithet}."
            ),
            placeholder_pools={
                "label": ["fraud", "charlatan", "grifter", "academic fraudster"],
                "n": ["two", "three", "four", "five"],
                "epithet": ["corrupt garbage", "a money-making scam", "fundamentally dishonest"],
            },
            base_rule=2,
            base_action="warn",
            compatible_archetypes=["mid_user", "active_contributor", "first_offender"],
            flair_options=[None],
            report_configs=[[("Personal attack on researcher", 5)], [("Harassment", 4)]],
            thread_context_pool=[
                ["Thread discussing a new paper on gene therapy"],
                ["Discussion of a controversial nutrition study"],
                ["Debate about a high-profile paper on social priming"],
            ],
            signals={
                "base_difficulty": 0.35,
                "severity": "moderate",
                "base_acceptable": ["warn", "remove"],
                "explanation": "Personal attack on a named researcher with unsubstantiated fraud claims. Warn to allow factual restatement.",
            },
        ),
        ContentTemplate(
            template_id="t2_r2_name_calling",
            category="rule2_attack",
            title_templates=[None],
            body_template=(
                "You {label}s never engage with actual evidence. "
                "Just admit you're {motivation} and stop wasting everyone's time. "
                "Real scientists don't argue like this."
            ),
            placeholder_pools={
                "label": ["deniers", "shills", "conspiracy theorists", "pseudoscience defenders"],
                "motivation": ["paid by industry", "ideologically captured", "wilfully ignorant", "in bad faith"],
            },
            base_rule=2,
            base_action="warn",
            compatible_archetypes=["mid_user", "first_offender", "mild_repeat"],
            flair_options=[None],
            report_configs=[[("Name-calling / Personal attack", 4)], [("Hostile", 3)]],
            thread_context_pool=[
                ["Thread on climate science policy debate"],
                ["Discussion of GMO safety research"],
                ["Debate about nuclear energy versus renewables"],
            ],
            signals={
                "base_difficulty": 0.25,
                "severity": "moderate",
                "base_acceptable": ["warn", "remove"],
                "explanation": "Labelling opponents as bad-faith actors rather than engaging with arguments. Warn first.",
            },
        ),
        ContentTemplate(
            template_id="t2_r2_dismissive_attack",
            category="rule2_attack",
            title_templates=[None],
            body_template=(
                "This is intellectually bankrupt. The level of {noun} on display here is {adjective}. "
                "I genuinely can't believe someone with this much {deficiency} is allowed to comment on science."
            ),
            placeholder_pools={
                "noun": ["reasoning", "argumentation", "thinking", "logic"],
                "adjective": ["staggering", "embarrassing", "breathtaking", "appalling"],
                "deficiency": ["ignorance", "scientific illiteracy", "logical confusion"],
            },
            base_rule=2,
            base_action="warn",
            compatible_archetypes=["mid_user", "first_offender", "active_contributor"],
            flair_options=[None],
            report_configs=[[("Personal attack", 4)], [("Hostile / Condescending", 5)]],
            thread_context_pool=[
                ["Thread on alternative medicine regulation"],
                ["Debate about vaccine policy"],
                ["Discussion about dietary supplement research"],
            ],
            signals={
                "base_difficulty": 0.25,
                "severity": "moderate",
                "base_acceptable": ["warn", "remove"],
                "explanation": "Highly dismissive personal attack targeting the commenter's intelligence. Warn first.",
            },
        ),

        # ── Rule 3: Misinformation (5) ────────────────────────────────────────
        ContentTemplate(
            template_id="t2_r3_emf_conspiracy",
            category="rule3_misinfo",
            title_templates=[
                "New study: {tech} towers cause DNA damage and cancer — governments hiding the truth",
                "Scientists prove {tech} radiation causes {harm} — Big Telecom suppressing it",
            ],
            body_template=(
                "Researchers at the Institute for Electromagnetic Safety have proven that "
                "{tech} radiation at {freq} causes {harm} in human cells. "
                "Effect size: {pct}% increase. Big Telecom is suppressing this. Share before it's deleted!"
            ),
            placeholder_pools={
                "tech": ["5G", "4G", "WiFi", "smart meter"],
                "freq": ["60GHz", "2.4GHz", "700MHz"],
                "harm": ["double-strand DNA breaks", "neurological damage", "immune suppression"],
                "pct": ["340", "500", "800"],
            },
            base_rule=3,
            base_action="remove",
            compatible_archetypes=["new_genuine", "mid_user", "brand_new"],
            flair_options=["Research"],
            report_configs=[[("Misinformation", 11), ("Spam", 2)], [("Pseudoscience", 8)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["remove"],
                "explanation": "EMF/5G misinformation — non-ionising radiation at these frequencies cannot cause DNA damage. No credible source cited.",
            },
        ),
        ContentTemplate(
            template_id="t2_r3_antivax",
            category="rule3_misinfo",
            title_templates=[
                "Study confirms: unvaccinated children are {pct}x healthier than vaccinated",
                "New research: vaccines cause {harm} in {pct}% of children",
            ],
            body_template=(
                "A comprehensive {year} study of {n} children found that unvaccinated kids had "
                "{pct}x fewer {condition} than vaccinated. The mainstream media won't cover this."
            ),
            placeholder_pools={
                "pct": ["40", "20", "10", "50"],
                "harm": ["chronic illness", "autoimmune disorders", "developmental delays"],
                "n": ["50,000", "30,000", "100,000"],
                "year": ["2023", "2022", "2021"],
                "condition": ["chronic illnesses", "autoimmune disorders", "allergic conditions"],
            },
            base_rule=3,
            base_action="remove",
            compatible_archetypes=["new_genuine", "brand_new", "mid_user"],
            flair_options=["Research"],
            report_configs=[[("Misinformation / Anti-vax", 14)], [("Pseudoscience", 10)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["remove"],
                "explanation": "Anti-vaccine misinformation with fabricated statistics and no credible source.",
            },
        ),
        ContentTemplate(
            template_id="t2_r3_big_pharma_cure",
            category="rule3_misinfo",
            title_templates=[
                "Big Pharma is suppressing the {treatment} cure for {disease}",
                "Scientists found a {treatment} cure for {disease} — hidden for {years} years",
            ],
            body_template=(
                "A {type} compound derived from {source} has been proven to cure {disease}. "
                "Studies were suppressed for {years} years by pharmaceutical companies. "
                "The cure is {availability}. Spread this before it gets censored."
            ),
            placeholder_pools={
                "treatment": ["natural", "plant-based", "ancient", "simple"],
                "disease": ["cancer", "Alzheimer's", "HIV", "diabetes"],
                "type": ["natural", "herbal", "plant-derived", "traditional"],
                "source": ["a common spice", "a rainforest plant", "a traditional herb"],
                "years": ["20", "30", "40"],
                "availability": ["available to everyone for free", "found in every kitchen"],
            },
            base_rule=3,
            base_action="remove",
            compatible_archetypes=["new_genuine", "brand_new", "mid_user"],
            flair_options=["Research", "Discussion"],
            report_configs=[[("Misinformation", 8), ("Dangerous medical advice", 5)], [("Pseudoscience", 9)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["remove"],
                "explanation": "Conspiracy-framed misinformation claiming suppressed cures. No evidence, dangerous if believed.",
            },
        ),
        ContentTemplate(
            template_id="t2_r3_climate_denial",
            category="rule3_misinfo",
            title_templates=[
                "The global warming pause proves climate models are wrong",
                "CO2 is plant food — climate change is a political hoax",
                "Scientists: solar cycles explain ALL recent warming, not CO2",
            ],
            body_template=(
                "New analysis of {dataset} shows that {claim}. "
                "The models have been consistently {problem} for {years} years. "
                "The {conspiracy} is manipulating the data to support their agenda."
            ),
            placeholder_pools={
                "dataset": ["satellite temperature records", "ocean heat content data", "ice core data"],
                "claim": ["CO2 has no correlation with temperature", "warming stopped 20 years ago", "solar activity explains everything"],
                "problem": ["wrong", "overclaiming", "manipulated", "flawed"],
                "years": ["20", "30", "40"],
                "conspiracy": ["IPCC", "climate establishment", "UN", "Big Green"],
            },
            base_rule=3,
            base_action="remove",
            compatible_archetypes=["mid_user", "new_genuine", "first_offender"],
            flair_options=["Research", "Discussion"],
            report_configs=[[("Misinformation / Climate denial", 8)], [("Pseudoscience", 6)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.15,
                "base_acceptable": ["remove"],
                "explanation": "Climate denial misinformation contradicting overwhelming scientific consensus. Remove.",
            },
        ),
        ContentTemplate(
            template_id="t2_r3_pseudoscience_breakthrough",
            category="rule3_misinfo",
            title_templates=[
                "Quantum healing: how {theory} explains {phenomenon}",
                "Scientists don't understand consciousness — {alternative} does",
                "New proof: {alternative} theory explains {phenomenon} better than mainstream science",
            ],
            body_template=(
                "Mainstream science has failed to explain {phenomenon}. "
                "But {theory} provides a clear framework: {claim}. "
                "Published researchers are starting to admit this, but the {establishment} suppresses it."
            ),
            placeholder_pools={
                "theory": ["quantum healing", "homeopathy", "morphic resonance", "astral projection"],
                "phenomenon": ["consciousness", "placebo effects", "cancer remission", "memory"],
                "claim": ["consciousness exists at the quantum level and can be controlled", "water retains memory of substances"],
                "establishment": ["scientific establishment", "pharmaceutical lobby", "mainstream academia"],
                "alternative": ["homeopathy", "quantum healing", "energy medicine"],
            },
            base_rule=3,
            base_action="remove",
            compatible_archetypes=["new_genuine", "mid_user", "brand_new"],
            flair_options=["Research", "Discussion"],
            report_configs=[[("Pseudoscience / Misinformation", 7)], [("Misinformation", 5)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.15,
                "base_acceptable": ["remove"],
                "explanation": "Pseudoscience framed as alternative theory. Claims lack peer-reviewed support.",
            },
        ),

        # ── Rule 4: Self-promotion / Spam (6) ────────────────────────────────
        ContentTemplate(
            template_id="t2_r4_youtube_promo",
            category="rule4_spam",
            title_templates=[
                "Check out my YouTube channel — I explain {topic} simply!",
                "My new {topic} YouTube video is live — would love feedback",
            ],
            body_template=(
                "Hey r/science! I've been making {topic} videos for {years} years. "
                "My latest covers {subtopic}. Please subscribe and share! Link: {url}"
            ),
            placeholder_pools={
                "topic": ["science", "physics", "biology", "chemistry"],
                "years": ["1", "2", "3"],
                "subtopic": ["quantum mechanics", "evolution", "climate change", "black holes"],
                "url": ["youtube.com/watch?v=abc123", "youtu.be/xyz456"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["brand_new", "human_spammer"],
            flair_options=["Discussion"],
            report_configs=[[("Self-promotion", 5)], [("Spam", 4)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["remove"],
                "explanation": "YouTube self-promotion from a new account with no scientific content contributed.",
            },
        ),
        ContentTemplate(
            template_id="t2_r4_blog_series",
            category="rule4_spam",
            title_templates=[
                "Part {n} of my ongoing series: {topic} for beginners",
                "My science blog: the definitive guide to {topic} (part {n})",
            ],
            body_template=(
                "Continuing my series on {topic}. This is part {n} of {total}. "
                "Previous parts were well received here. Link to today's post: {url}. "
                "Please share if you found it useful!"
            ),
            placeholder_pools={
                "topic": ["quantum physics", "evolutionary biology", "climate science", "neuroscience"],
                "n": ["4", "7", "12", "17"],
                "total": ["10", "15", "20"],
                "url": ["myblog.com/science-series-4", "medium.com/@sciwriter/part-7"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["human_spammer", "mid_user", "mild_repeat"],
            flair_options=["Discussion"],
            report_configs=[[("Repeated self-promotion", 5)], [("Spam", 6)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.15,
                "base_acceptable": ["remove"],
                "explanation": "Repeated blog self-promotion with no substantive scientific content beyond the link.",
            },
        ),
        ContentTemplate(
            template_id="t2_r4_affiliate_links",
            category="rule4_spam",
            title_templates=[
                "Top {n} science {items} for {year} (affiliate links inside)",
                "Best {items} for home science — my top picks",
            ],
            body_template=(
                "As an Amazon Associate I earn from qualifying purchases. "
                "Here are my top picks for science {items}: [{item1}](amzn.to/abc), [{item2}](amzn.to/def). "
                "Full review at {url}!"
            ),
            placeholder_pools={
                "n": ["5", "10", "15"],
                "items": ["gadgets", "kits", "lab tools"],
                "year": ["2024", "2025"],
                "item1": ["microscope", "telescope", "pH meter"],
                "item2": ["chemistry kit", "digital scale", "centrifuge"],
                "url": ["sciencegadgets.blog", "labtools.review"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["bot_spammer", "human_spammer"],
            flair_options=["Discussion"],
            report_configs=[[("Spam / Affiliate links", 6)], [("Advertising", 4)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "base_acceptable": ["remove"],
                "explanation": "Affiliate link spam — explicitly disclosed as monetised advertising.",
            },
        ),
        ContentTemplate(
            template_id="t2_r4_book_promo",
            category="rule4_spam",
            title_templates=[
                "My new book on {topic} is out — sharing in case it's useful here",
                "I wrote a {type} on {topic} — would love to get scientist feedback",
            ],
            body_template=(
                "I've spent {years} years working in {field} and just published a {type} on {topic}. "
                "I'm one of the {role}s and wanted to share it here. "
                "Details and sample chapter at {url}."
            ),
            placeholder_pools={
                "topic": ["science communication", "climate change", "genetics for non-scientists"],
                "type": ["book", "popular science book", "textbook"],
                "years": ["10", "15", "20"],
                "field": ["research", "science journalism", "academia"],
                "role": ["author", "co-author", "contributor"],
                "url": ["amazon.com/dp/abc123", "booksite.com/buy", "publishers.com/title"],
            },
            base_rule=4,
            base_action="warn",
            compatible_archetypes=["active_contributor", "veteran_expert", "mid_user"],
            flair_options=["Discussion"],
            report_configs=[[("Self-promotion", 3)], [("Advertising", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.25,
                "base_acceptable": ["warn", "remove"],
                "explanation": "Borderline self-promotion of a science book. Warn rather than remove — author engaged in field.",
            },
        ),
        ContentTemplate(
            template_id="t2_r4_survey_spam",
            category="rule4_spam",
            title_templates=[
                "Please fill in my survey on {topic} — 5 minutes",
                "Survey for my {project}: {topic} attitudes",
            ],
            body_template=(
                "Hi everyone! Doing {project} and need participants for a {duration} survey on {topic}. "
                "Anonymous, no personal data collected. Link: {url}"
            ),
            placeholder_pools={
                "topic": ["climate change attitudes", "science communication", "public trust in research"],
                "project": ["my thesis", "a startup", "market research"],
                "duration": ["2-minute", "5-minute", "10-minute"],
                "url": ["forms.gle/xyz", "typeform.com/abc", "survey.io/def"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[[("Spam / Data collection", 4)], [("Self-promotion", 3)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["remove"],
                "explanation": "Survey link solicitation using scientific framing to collect data for non-community purposes.",
            },
        ),
        ContentTemplate(
            template_id="t2_r4_course_promo",
            category="rule4_spam",
            title_templates=[
                "My online course on {topic} just launched — use code for 30% off!",
                "I teach {topic} online — new cohort starting soon",
            ],
            body_template=(
                "After {years} years in {field}, I've distilled it all into a {weeks}-week course. "
                "No background needed. Code {code} at {url} for {pct}% off."
            ),
            placeholder_pools={
                "topic": ["data science", "bioinformatics", "science communication"],
                "years": ["5", "10", "15"],
                "field": ["research", "biotech", "academia"],
                "weeks": ["4", "6", "8"],
                "code": ["REDDIT30", "SCIENCE20"],
                "url": ["datasciencecourse.io", "scicomm.academy"],
                "pct": ["20", "30"],
            },
            base_rule=4,
            base_action="remove",
            compatible_archetypes=["human_spammer", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[[("Spam / Advertising", 5)], [("Self-promotion", 4)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["remove"],
                "explanation": "Course advertising using science framing as cover for commercial promotion.",
            },
        ),

        # ── Rule 5: Wrong flair (5) ───────────────────────────────────────────
        ContentTemplate(
            template_id="t2_r5_news_as_research",
            category="rule5_flair",
            title_templates=[
                "NASA announces discovery of water on the Moon",
                "WHO declares new vaccine 95% effective in trials",
                "Scientists at MIT develop room-temperature superconductor",
            ],
            body_template=(
                "Breaking news: {headline}. "
                "This was announced today by {source} and is expected to have major implications for {impact}. "
                "Full story: {url}"
            ),
            placeholder_pools={
                "headline": [
                    "NASA confirms water ice deposits at lunar south pole",
                    "WHO announces emergency approval for new meningitis B vaccine",
                    "MIT team demonstrates room-temperature superconductivity",
                ],
                "source": ["NASA", "WHO headquarters", "MIT press office", "Nature editorial team"],
                "impact": ["space exploration", "global health", "energy storage", "computing"],
                "url": ["nasa.gov/news/abc", "who.int/news/item/abc", "nature.com/articles/abc"],
            },
            base_rule=5,
            base_action="warn",
            compatible_archetypes=["new_genuine", "mid_user", "brand_new"],
            flair_options=["Research"],
            report_configs=[[("Wrong flair", 3)], [("Incorrect flair used", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.15,
                "base_acceptable": ["warn", "remove"],
                "explanation": "News post incorrectly tagged [Research]. A warning to re-flair is appropriate.",
            },
        ),
        ContentTemplate(
            template_id="t2_r5_discussion_as_research",
            category="rule5_flair",
            title_templates=[
                "What do you think about the ethics of {topic}? Discussion",
                "Hot take: {claim} — change my mind",
                "Unpopular opinion: {claim}",
            ],
            body_template=(
                "I've been thinking a lot about {topic} lately and wanted to hear r/science's perspective. "
                "{framing}. "
                "What do you think — is this a fair characterisation?"
            ),
            placeholder_pools={
                "topic": ["gene editing ethics", "geoengineering", "AI in medical diagnosis", "animal research"],
                "claim": ["the replication crisis is overblown", "p-values are fundamentally flawed"],
                "framing": [
                    "There seems to be a gap between public perception and scientific reality here",
                    "The media consistently misrepresents this, and it bothers me",
                ],
            },
            base_rule=5,
            base_action="warn",
            compatible_archetypes=["new_genuine", "mid_user"],
            flair_options=["Research"],
            report_configs=[[("Wrong flair — should be Discussion", 3)], [("Incorrect flair", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.2,
                "base_acceptable": ["warn"],
                "explanation": "Opinion/discussion post tagged [Research]. Warn to re-flair as [Discussion].",
            },
        ),
        ContentTemplate(
            template_id="t2_r5_ama_as_news",
            category="rule5_flair",
            title_templates=[
                "I'm a {role} at {institution} — happy to answer questions",
                "Ask a {role}: I work on {topic}",
            ],
            body_template=(
                "Hi r/science! I'm a {role} at {institution} and I study {topic}. "
                "Happy to answer questions about my research and career. "
                "I'll be online for the next {hours} hours."
            ),
            placeholder_pools={
                "role": ["postdoctoral researcher", "research scientist", "professor", "data scientist"],
                "institution": ["Oxford", "MIT", "Harvard", "Caltech", "Johns Hopkins"],
                "topic": ["cancer genetics", "exoplanet atmospheres", "immunology", "climate modelling"],
                "hours": ["2", "3", "4"],
            },
            base_rule=5,
            base_action="warn",
            compatible_archetypes=["active_contributor", "veteran_expert", "new_genuine"],
            flair_options=["News"],
            report_configs=[[("Wrong flair — should be AMA", 3)], [("Incorrect flair", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.15,
                "base_acceptable": ["warn"],
                "explanation": "AMA post tagged [News]. Warn to re-flair as [AMA].",
            },
        ),
        ContentTemplate(
            template_id="t2_r5_research_no_flair",
            category="rule5_flair",
            title_templates=[
                "New study: {finding}",
                "Paper shows {finding} for the first time",
                "{finding} — new research",
            ],
            body_template=(
                "Researchers at {institution} published a paper in {journal} showing that {finding}. "
                "The study used {method} with {n} participants. Full paper at {doi}."
            ),
            placeholder_pools={
                "finding": [
                    "exercise reduces dementia risk by 30%",
                    "gut microbiome composition predicts antidepressant response",
                    "ocean acidification accelerates coral bleaching",
                ],
                "institution": ["MIT", "Oxford", "Harvard", "Stanford"],
                "journal": ["Nature", "PNAS", "Science", "NEJM"],
                "method": ["RCT", "longitudinal cohort study", "meta-analysis"],
                "n": ["1,200", "5,000", "15,000"],
                "doi": ["doi.org/10.1038/abc", "doi.org/10.1126/xyz"],
            },
            base_rule=5,
            base_action="warn",
            compatible_archetypes=["new_genuine", "brand_new", "mid_user"],
            flair_options=[None],
            report_configs=[[("Missing flair", 2)], [("No flair applied", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["warn"],
                "explanation": "Research post posted without any flair. Warn to add correct flair [Research].",
            },
        ),
        ContentTemplate(
            template_id="t2_r5_discussion_wrong_flair",
            category="rule5_flair",
            title_templates=[
                "What careers can a {degree} degree lead to in science?",
                "Should I pursue a {degree} or go directly to a PhD?",
            ],
            body_template=(
                "I'm finishing my {degree} and trying to figure out my next steps. "
                "I'm interested in {field} but not sure whether to do {option1} or {option2}. "
                "Anyone with experience in this area have advice?"
            ),
            placeholder_pools={
                "degree": ["BSc Biology", "MSc Chemistry", "BSc Physics", "MSc Neuroscience"],
                "field": ["academia", "biotech", "pharmaceutical industry", "environmental consulting"],
                "option1": ["a MSc first", "go straight to industry", "a PhD directly"],
                "option2": ["jump into industry", "do a conversion course", "apply for postdocs"],
            },
            base_rule=5,
            base_action="warn",
            compatible_archetypes=["brand_new", "new_genuine"],
            flair_options=["Research"],
            report_configs=[[("Wrong flair", 2)], [("Incorrect flair used", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.15,
                "base_acceptable": ["warn"],
                "explanation": "Career advice post tagged [Research]. Should be [Discussion]. Warn to re-flair.",
            },
        ),

        # ── Rule 6: Low-effort (5) ────────────────────────────────────────────
        ContentTemplate(
            template_id="t2_r6_one_liner",
            category="rule6_loweffort",
            title_templates=[None],
            body_template="This is {adjective}.",
            placeholder_pools={
                "adjective": ["so interesting", "amazing", "wild", "crazy", "mind-blowing"],
            },
            base_rule=6,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine", "mid_user"],
            flair_options=[None],
            report_configs=[[("Low-effort comment", 3)], [("No content", 2)]],
            thread_context_pool=[
                ["Post: 'New research shows exercise may slow aging'"],
                ["Post: 'Scientists discover new deep-sea species'"],
                ["Post: 'NASA confirms water on the Moon'"],
            ],
            signals={
                "base_difficulty": 0.05,
                "base_acceptable": ["remove"],
                "explanation": "One-word or single-sentence reaction with no scientific substance. Remove.",
            },
        ),
        ContentTemplate(
            template_id="t2_r6_meme_screenshot",
            category="rule6_loweffort",
            title_templates=[
                "Haha this is so true [meme about {topic}]",
                "When scientists try to explain {topic} to non-scientists [image]",
                "Found this screenshot — thought r/science would appreciate",
            ],
            body_template=(
                "[Image/meme about {topic}] "
                "Found this on {source}. "
                "Thought you guys would get it lol."
            ),
            placeholder_pools={
                "topic": ["quantum mechanics", "evolution", "statistics", "peer review", "climate change"],
                "source": ["Twitter", "Instagram", "Facebook", "Reddit", "Tumblr"],
            },
            base_rule=6,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine", "mid_user"],
            flair_options=["Discussion"],
            report_configs=[[("Meme / Low-effort", 5)], [("No scientific content", 4)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "base_acceptable": ["remove"],
                "explanation": "Meme or screenshot without scientific context or discussion. Remove.",
            },
        ),
        ContentTemplate(
            template_id="t2_r6_vague_question",
            category="rule6_loweffort",
            title_templates=[
                "What is {topic}?",
                "Does {topic} really work?",
                "Is {topic} true?",
            ],
            body_template="I heard about {topic} and was wondering about it.",
            placeholder_pools={
                "topic": ["quantum entanglement", "microbiome diets", "intermittent fasting", "nootropics", "cryonics"],
            },
            base_rule=6,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[[("Low-effort / No research shown", 2)], [("Vague", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.2,
                "base_acceptable": ["remove", "warn"],
                "explanation": "Extremely low-effort question with no background research demonstrated. Remove.",
            },
        ),
        ContentTemplate(
            template_id="t2_r6_link_only",
            category="rule6_loweffort",
            title_templates=[
                "Interesting read [link]",
                "Relevant to r/science",
                "Sharing this",
            ],
            body_template="{url}",
            placeholder_pools={
                "url": ["bbc.com/science/abc", "nature.com/articles/xyz", "guardian.com/science/abc"],
            },
            base_rule=6,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine", "mid_user"],
            flair_options=["News", "Research"],
            report_configs=[[("Low-effort / Link dump", 3)], [("No context provided", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["remove"],
                "explanation": "Link-only post with no summary, context, or discussion starter provided.",
            },
        ),
        ContentTemplate(
            template_id="t2_r6_reaction_comment",
            category="rule6_loweffort",
            title_templates=[None],
            body_template="{reaction}",
            placeholder_pools={
                "reaction": ["lol", "wow", "interesting!", "cool!", "omg", "lmao", "nice"],
            },
            base_rule=6,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine"],
            flair_options=[None],
            report_configs=[[("Low-effort", 3)], [("Reaction only", 2)]],
            thread_context_pool=[
                ["Post: research paper on a notable scientific finding"],
                ["Thread: ongoing discussion about a scientific topic"],
            ],
            signals={
                "base_difficulty": 0.05,
                "base_acceptable": ["remove"],
                "explanation": "Reaction-only comment with no scientific content. Remove.",
            },
        ),

        # ── Clean / Approve (4) ───────────────────────────────────────────────
        ContentTemplate(
            template_id="t2_clean_statistics_discussion",
            category="clean",
            title_templates=[None],
            body_template=(
                "The key issue here is that this study is underpowered. "
                "With n=47, the 95% confidence intervals are so wide they're nearly meaningless. "
                "The reported p=0.049 should be treated with great caution — this looks like it could easily be a false positive. "
                "I'd want to see at least 5x the sample size before drawing any conclusions."
            ),
            placeholder_pools={},
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "veteran_expert"],
            flair_options=[None],
            report_configs=[[]],
            thread_context_pool=[
                ["Discussion thread about a small social psychology study"],
                ["Thread on a nutrition study with limited sample size"],
            ],
            signals={
                "base_difficulty": 0.1,
                "base_acceptable": ["approve"],
                "explanation": "Thoughtful methodological critique focused on statistical rigor — exactly what r/science needs.",
            },
        ),
        ContentTemplate(
            template_id="t2_clean_paper_review",
            category="clean",
            title_templates=[
                "New {journal} paper on {topic}: {finding}",
                "Just read this {journal} paper on {topic} — excellent methodology",
            ],
            body_template=(
                "This {journal} paper on {topic} is worth reading carefully. "
                "The methodology is solid: pre-registered, large sample ({n}), and the effect size ({es}) is reported honestly. "
                "The authors are appropriately cautious in the discussion, noting {caveat}. "
                "Full text: {doi}."
            ),
            placeholder_pools={
                "journal": ["Nature", "Science", "NEJM", "PNAS"],
                "topic": ["exercise and cognition", "sleep deprivation effects", "gut microbiome and immunity"],
                "finding": ["a robust exercise-cognition link", "significant sleep-immune interaction"],
                "n": ["8,500", "12,000", "25,000"],
                "es": ["OR = 1.4 (95% CI: 1.2-1.6)", "d = 0.31 (95% CI: 0.18-0.44)"],
                "caveat": ["generalisability to non-Western populations is unclear", "effect sizes may not justify clinical recommendations"],
                "doi": ["doi.org/10.1038/abc", "doi.org/10.1056/xyz"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "veteran_expert"],
            flair_options=["Research"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "base_acceptable": ["approve"],
                "explanation": "Constructive, well-cited paper review with honest appraisal of strengths and limitations.",
            },
        ),
        ContentTemplate(
            template_id="t2_clean_methodological_critique",
            category="clean",
            title_templates=[None],
            body_template=(
                "I think there's a confounding issue the authors haven't addressed. "
                "If participants self-selected into the treatment group, we can't rule out reverse causation. "
                "The effect might be real, but this design can't establish it. "
                "Has anyone seen a version of this with random assignment?"
            ),
            placeholder_pools={},
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "new_genuine"],
            flair_options=[None],
            report_configs=[[]],
            thread_context_pool=[
                ["Thread discussing an observational study on dietary interventions"],
                ["Discussion about an epidemiological study on exercise outcomes"],
            ],
            signals={
                "base_difficulty": 0.05,
                "base_acceptable": ["approve"],
                "explanation": "Legitimate methodological critique focused on study design, not personal attacks.",
            },
        ),
        ContentTemplate(
            template_id="t2_clean_well_cited_question",
            category="clean",
            title_templates=[
                "Does recent meta-analytic evidence support the {hypothesis} hypothesis?",
                "Literature question: is {hypothesis} still mainstream in {field}?",
            ],
            body_template=(
                "I've been reading the debate around {hypothesis} in {field}. "
                "On one hand, {paper1} ({year1}) found {finding1}. "
                "On the other, {paper2} ({year2}) suggests {finding2}. "
                "Is there a current consensus, or is this genuinely contested territory?"
            ),
            placeholder_pools={
                "hypothesis": ["serotonin depletion", "hygiene", "dual-process", "glucose depletion"],
                "field": ["psychiatry", "immunology", "cognitive psychology", "neuroscience"],
                "paper1": ["Moncrieff et al.", "Strachan et al.", "Kahneman et al."],
                "year1": ["2022", "1989", "2011"],
                "finding1": ["no direct evidence for low serotonin in depression", "early-life microbial exposure is protective"],
                "paper2": ["Jauhar et al.", "Rook et al.", "Evans et al."],
                "year2": ["2023", "2012", "2021"],
                "finding2": ["indirect evidence still supports serotonin involvement", "the relationship is more nuanced"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "new_genuine", "active_contributor"],
            flair_options=["Discussion"],
            report_configs=[[]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.05,
                "base_acceptable": ["approve"],
                "explanation": "Well-researched question citing primary literature and seeking genuine scientific clarification.",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Task 3 templates — Context-Aware Moderation (25 templates)
# ---------------------------------------------------------------------------


def _task3_templates() -> List[ContentTemplate]:  # noqa: PLR0915
    return [
        # ── Approve cases (9) ────────────────────────────────────────────────
        ContentTemplate(
            template_id="t3_approve_sarcasm_marker",
            category="t3_approve",
            title_templates=[
                "Clearly the Earth is flat and vaccines cause autism /s",
                "Big Pharma definitely has a cure for everything but is hiding it /s",
                "Obviously climate change is a hoax invented by lizard people /s",
            ],
            body_template=(
                "Wow, I can't believe scientists are still pushing their 'evidence-based' agenda. "
                "Next they'll tell us the Moon landing was real! "
                "I'm very disappointed in this sub for not listening to my Facebook research /s"
            ),
            placeholder_pools={},
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "new_genuine"],
            flair_options=[None, "Discussion"],
            report_configs=[[("Misinformation", 1)], []],
            thread_context_pool=[
                ["Post: 'New evidence confirms vaccine safety — largest study yet'"],
                ["Thread: 'Climate models are more accurate than previously thought'"],
            ],
            signals={
                "base_difficulty": 0.5,
                "explanation": "Clearly marked satire with /s. No reasonable person would interpret this as genuine misinformation. Approve.",
            },
        ),
        ContentTemplate(
            template_id="t3_approve_harsh_scientific_criticism",
            category="t3_approve",
            title_templates=[None],
            body_template=(
                "This paper is methodologically garbage. The authors used convenience sampling (n=47), "
                "no control group, and entirely self-reported outcomes. "
                "It should never have passed peer review. "
                "The journal has a lot to answer for publishing this."
            ),
            placeholder_pools={},
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["veteran_expert", "active_contributor", "mid_user"],
            flair_options=[None],
            report_configs=[[("Rude about the paper", 3)], [("Aggressive", 2)]],
            thread_context_pool=[
                ["Discussion of a psychology replication failure"],
                ["Thread about a controversial nutrition study"],
                ["Debate about a low-powered social psychology study"],
            ],
            signals={
                "base_difficulty": 0.6,
                "explanation": "'This paper is garbage' targets the work, not the person. The criticism is specific and methodologically grounded. Approve.",
            },
        ),
        ContentTemplate(
            template_id="t3_approve_heated_scientific_debate",
            category="t3_approve",
            title_templates=[None],
            body_template=(
                "I fundamentally disagree with your reading of the data. "
                "The confidence intervals overlap significantly — claiming a 'significant result' at p=0.048 "
                "is precisely the kind of statistical malpractice that fuels the replication crisis. "
                "This interpretation is wrong."
            ),
            placeholder_pools={},
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "veteran_expert"],
            flair_options=[None],
            report_configs=[[("Aggressive tone", 1)], []],
            thread_context_pool=[
                ["Debate about a neuroscience paper's statistical methods"],
                ["Thread on significance testing in psychology"],
                ["Discussion about p-value thresholds in high-stakes research"],
            ],
            signals={
                "base_difficulty": 0.6,
                "explanation": "Forceful scientific disagreement on methodology. 'This interpretation is wrong' is a scientific claim, not a personal attack. Approve.",
            },
        ),
        ContentTemplate(
            template_id="t3_approve_researcher_own_paper",
            category="t3_approve",
            title_templates=[
                "My team's new paper: {topic} — open access",
                "We published a new paper on {topic} in {journal} — happy to discuss",
            ],
            body_template=(
                "Hi r/science, I'm one of the authors on a paper we published in {journal} this week ({doi}). "
                "We demonstrate {finding}. Happy to answer questions. Open access link in comments."
            ),
            placeholder_pools={
                "topic": ["quantum error correction via surface codes", "CRISPR base editing efficiency", "oceanic carbon sequestration"],
                "journal": ["Physical Review Letters", "Nature Methods", "Science Advances", "PNAS"],
                "doi": ["doi.org/10.1103/PhysRevLett.132.180401", "doi.org/10.1038/s41592-2024-abc"],
                "finding": [
                    "a logical error rate below the surface code threshold for the first time",
                    "a 10-fold improvement in base editing efficiency with fewer off-target effects",
                    "a previously underestimated role of deep-ocean bacteria in carbon sequestration",
                ],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["veteran_expert", "active_contributor"],
            flair_options=["Research"],
            report_configs=[[("Self-promotion", 2)], []],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.6,
                "explanation": "Rule 4 allows researchers to share their own peer-reviewed work when it adds community value. This has a DOI, is open access, author is engaged. Approve.",
            },
        ),
        ContentTemplate(
            template_id="t3_approve_absurdist_satire",
            category="t3_approve",
            title_templates=[
                "Peer-reviewed proof that the Moon is made of cheese: dairy science implications",
                "New study: coffee grounds contain traces of dark matter (Journal of Irreproducible Results)",
            ],
            body_template=(
                "Abstract: We present a rigorous analysis using our proprietary CheeseSpec algorithm. "
                "Our findings (p < 0.0001) confirm lunar composition of 73% aged cheddar and 27% brie. "
                "We call for immediate funding for a Moon Cheese Extraction Mission. "
                "Submitted to the Journal of Irreproducible Results."
            ),
            placeholder_pools={},
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "veteran_expert"],
            flair_options=["Research"],
            report_configs=[[("Misinformation", 1)], []],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.55,
                "explanation": "Clearly absurdist satire — 'Journal of Irreproducible Results' is a known parody journal. Community is engaging positively. Approve.",
            },
        ),
        ContentTemplate(
            template_id="t3_approve_new_account_genuine_research",
            category="t3_approve",
            title_templates=[
                "Preprint: novel mechanism for {topic} identified via {method}",
                "First post: my preprint on {topic} just went up on {server}",
            ],
            body_template=(
                "Hi, first post here. We just uploaded our preprint to {server} ({doi}). "
                "We used {method} to characterise {finding}. Full dataset is open access on {repo}. "
                "Happy to answer questions — this is my first time posting here."
            ),
            placeholder_pools={
                "topic": ["CRISPR off-target editing", "antibiotic resistance transfer", "deep-sea biodiversity"],
                "method": ["PacBio HiFi long-read sequencing", "cryo-EM", "metagenomics"],
                "server": ["bioRxiv", "arXiv"],
                "doi": ["doi.org/10.1101/2024.03.15.585123", "arxiv.org/abs/2401.01234"],
                "finding": [
                    "off-target edits missed by short-read sequencing",
                    "a novel horizontal gene transfer mechanism",
                    "three new bacterial phyla in hadal trenches",
                ],
                "repo": ["Zenodo", "GitHub", "Figshare"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["brand_new", "new_genuine"],
            flair_options=["Research"],
            report_configs=[[], [("New account / suspicious", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.65,
                "explanation": "New account posting genuine high-quality research with DOI and open data. Account age is not a reason to remove legitimate science. Approve.",
            },
        ),
        ContentTemplate(
            template_id="t3_approve_peer_review_critique",
            category="t3_approve",
            title_templates=[
                "The peer review system is fundamentally broken — here's the evidence",
                "Peer review fails to catch most methodological errors: what can we do?",
            ],
            body_template=(
                "Having served on {n} editorial boards and reviewed hundreds of papers, "
                "I can say with confidence that peer review as currently practised is inadequate. "
                "The evidence: {finding1}. And {finding2}. "
                "We need pre-registration, open review, and statistical consulting to be mandatory. "
                "This is not a criticism of scientists — it's a systems problem."
            ),
            placeholder_pools={
                "n": ["3", "5", "7"],
                "finding1": [
                    "Stodden et al. (2018) found only 26% of papers could be computationally reproduced",
                    "Ioannidis et al. showed most published research findings may be false",
                ],
                "finding2": [
                    "reviewers miss statistical errors at alarming rates (Herndon et al.)",
                    "publication bias distorts the literature systematically (Easterbrook et al.)",
                ],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["veteran_expert", "active_contributor"],
            flair_options=["Discussion"],
            report_configs=[[], [("Attacking science", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.55,
                "explanation": "Evidence-based critique of peer review by a credentialed researcher. Self-correction is how science works. Approve.",
            },
        ),
        ContentTemplate(
            template_id="t3_approve_naive_genuine_question",
            category="t3_approve",
            title_templates=[
                "Why do scientists keep changing their minds? Is science even reliable?",
                "Does science ever get anything permanently wrong?",
            ],
            body_template=(
                "I'm not a scientist but I've been reading a lot lately. "
                "I keep seeing news headlines that contradict earlier headlines — "
                "first coffee is bad, then it's good, then bad again. "
                "Does this mean we can't trust science? I genuinely want to understand."
            ),
            placeholder_pools={},
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["brand_new", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[[], [("Anti-science", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.5,
                "explanation": "Genuine epistemological question from a non-scientist. The post invites education, not misinformation. Approve.",
            },
        ),
        ContentTemplate(
            template_id="t3_approve_important_safety_post",
            category="t3_approve",
            title_templates=[
                "Safety alert: {chemical} found in some {product} — what we know",
                "Health warning: new data on {chemical} exposure risks",
            ],
            body_template=(
                "Recent analysis has found elevated {chemical} levels in some {product} brands. "
                "This has not yet been widely reported. Key facts: {finding1}. {finding2}. "
                "Source: {doi}. This is preliminary — interpret cautiously."
            ),
            placeholder_pools={
                "chemical": ["PFAS", "microplastics", "arsenic", "lead"],
                "product": ["bottled water brands", "baby food", "sunscreens", "non-stick cookware"],
                "finding1": [
                    "Levels exceeded EPA thresholds in 12 of 30 tested products",
                    "Detection was inconsistent — some batches clean, others contaminated",
                ],
                "finding2": [
                    "Manufacturers have not yet issued recalls",
                    "Regulatory review is under way",
                ],
                "doi": ["doi.org/10.1021/acs.est.abc", "doi.org/10.1016/j.envres.2024.abc"],
            },
            base_rule=None,
            base_action="approve",
            compatible_archetypes=["mid_user", "active_contributor", "new_genuine"],
            flair_options=["News", "Research"],
            report_configs=[[], [("Alarming / unverified", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.6,
                "explanation": "Legitimate safety information with primary source reference and appropriate uncertainty caveats. Approve.",
            },
        ),

        # ── Escalation cases (4) ─────────────────────────────────────────────
        ContentTemplate(
            template_id="t3_escalate_mental_health",
            category="t3_escalate",
            title_templates=[
                "I've been reading about {topic} and lately I feel like there's no point to any of it",
                "The {topic} research is interesting but honestly I've been feeling hopeless lately",
            ],
            body_template=(
                "Started reading the {paper} about {topic}. It's interesting research but "
                "honestly lately I've been feeling really hopeless and like nothing matters. "
                "Does the science on this actually give anyone hope?"
            ),
            placeholder_pools={
                "topic": ["serotonin and depression", "the neurological basis of hopelessness", "antidepressant efficacy"],
                "paper": ["Moncrieff et al. serotonin paper", "Cipriani et al. antidepressants meta-analysis"],
            },
            base_rule=None,
            base_action="escalate_to_senior_mod",
            compatible_archetypes=["new_genuine", "brand_new", "mid_user"],
            flair_options=["Discussion"],
            report_configs=[[], [("Concerning content", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.8,
                "requires_escalation": True,
                "alt_acceptable": ["approve"],
                "explanation": "Post has legitimate scientific content but contains crisis language ('no point to any of it', 'hopeless', 'nothing matters'). Escalate for senior mod judgment on mental health resources.",
            },
        ),
        ContentTemplate(
            template_id="t3_escalate_whistleblower",
            category="t3_escalate",
            title_templates=[
                "Leaked documents show {organisation} suppressed safety data for {product}",
                "Internal emails: {organisation} knew about {harm} but hid it for {years} years",
            ],
            body_template=(
                "I work at {organisation} and have documents showing {claim}. "
                "I'm sharing this here because I believe the public has a right to know. "
                "I cannot verify these publicly — they were provided to me confidentially. "
                "I am aware of the legal implications."
            ),
            placeholder_pools={
                "organisation": ["a major pharmaceutical company", "a regulatory agency", "an agricultural firm"],
                "product": ["a widely used medication", "a food additive", "a pesticide"],
                "harm": ["cardiovascular risks", "endocrine disruption", "neurotoxicity"],
                "years": ["5", "10", "15"],
                "claim": ["safety trial data was systematically omitted from FDA submissions", "internal models showed risk 3x higher than published"],
            },
            base_rule=None,
            base_action="escalate_to_senior_mod",
            compatible_archetypes=["mid_user", "active_contributor", "new_genuine"],
            flair_options=["News"],
            report_configs=[[("Unverified claims", 3)], [("Potentially defamatory", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.85,
                "requires_escalation": True,
                "alt_acceptable": ["approve"],
                "explanation": "Whistleblower content with unverified but potentially significant claims and explicit legal implications. Requires senior mod judgment.",
            },
        ),
        ContentTemplate(
            template_id="t3_escalate_harassment_campaign",
            category="t3_escalate",
            title_templates=[
                "Dr {name} is a fraud — here's the evidence (thread)",
                "Exposing {name}: a pattern of misconduct spanning {years} years",
            ],
            body_template=(
                "I've been compiling evidence against Dr {name} for {years} years. "
                "This thread documents {n} separate incidents of {misconduct}. "
                "I urge everyone here to {action} until this person is held accountable."
            ),
            placeholder_pools={
                "name": ["J. Smith", "M. Chen", "S. Patel", "A. Garcia"],
                "years": ["2", "3", "5"],
                "n": ["7", "12", "15"],
                "misconduct": ["data manipulation", "plagiarism", "hostile lab environment", "inappropriate conduct"],
                "action": ["share this widely", "contact their institution", "email the journal", "sign my petition"],
            },
            base_rule=None,
            base_action="escalate_to_senior_mod",
            compatible_archetypes=["mid_user", "first_offender", "new_genuine"],
            flair_options=["Discussion"],
            report_configs=[[("Harassment", 8)], [("Targeted attack on individual", 7)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.85,
                "requires_escalation": True,
                "alt_acceptable": ["remove"],
                "explanation": "Organised harassment targeting a named individual with a call to action. Potential legal and safety implications. Escalate.",
            },
        ),
        ContentTemplate(
            template_id="t3_escalate_doxxing",
            category="t3_escalate",
            title_templates=[
                "The real identity of anonymous science blogger {handle} revealed",
                "Who is really behind {handle}? I found out.",
            ],
            body_template=(
                "I've done some investigation and believe I know who {handle} is. "
                "Their name is {description} and they work at {institution}. "
                "I think the community deserves to know given their controversial positions."
            ),
            placeholder_pools={
                "handle": ["ScienceSkeptic", "DataDrivenScience", "RealPeerReview"],
                "description": ["a researcher", "a professor", "a postdoc"],
                "institution": ["a university I've identified", "a research institute"],
            },
            base_rule=None,
            base_action="escalate_to_senior_mod",
            compatible_archetypes=["mid_user", "new_genuine", "first_offender"],
            flair_options=["Discussion"],
            report_configs=[[("Potential doxxing", 9)], [("Privacy violation", 7)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.9,
                "requires_escalation": True,
                "alt_acceptable": ["remove"],
                "explanation": "Potential doxxing of anonymous individual — safety and privacy implications require senior mod judgment.",
            },
        ),

        # ── Variable violation cases (12: action varies with author history) ──
        ContentTemplate(
            template_id="t3_var_personal_attack_mild",
            category="t3_violation",
            title_templates=[None],
            body_template=(
                "Your understanding of this topic is frankly embarrassing. "
                "You clearly haven't read the primary literature and are just repeating popular misconceptions. "
                "I suggest you actually study the subject before commenting further."
            ),
            placeholder_pools={},
            base_rule=2,
            base_action="warn",
            compatible_archetypes=["mid_user", "first_offender", "mild_repeat", "serial_offender"],
            flair_options=[None],
            report_configs=[[("Personal attack / Condescending", 4)], [("Rude", 3)]],
            thread_context_pool=[
                ["Thread: debate about a controversial scientific topic"],
                ["Discussion about the merits of a high-profile paper"],
                ["Ongoing debate about methodology in a specific field"],
            ],
            signals={
                "base_difficulty": 0.55,
                "severity": "mild",
                "base_acceptable": ["warn", "remove"],
                "explanation": "Condescending personal attack suggesting the commenter is ignorant. Warn on first offense; escalate for repeat offenders.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_direct_insult_moderate",
            category="t3_violation",
            title_templates=[None],
            body_template=(
                "You're intellectually bankrupt and your comment proves it. "
                "The stupidity on display here is genuinely staggering. "
                "People like you are exactly why science communication is in crisis."
            ),
            placeholder_pools={},
            base_rule=2,
            base_action="warn",
            compatible_archetypes=["mid_user", "first_offender", "mild_repeat", "serial_offender"],
            flair_options=[None],
            report_configs=[[("Personal attack / Insult", 7)], [("Harassment", 4)]],
            thread_context_pool=[
                ["Thread on alternative medicine regulation"],
                ["Debate about vaccine policy"],
                ["Discussion about dietary supplement research"],
            ],
            signals={
                "base_difficulty": 0.55,
                "severity": "moderate",
                "base_acceptable": ["warn", "remove"],
                "explanation": "Direct insult targeting commenter's intelligence. Warn on first offense; remove or temp_ban for repeat offenders.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_misinformation_clear",
            category="t3_violation",
            title_templates=[
                "{tech} towers are killing us — the evidence is overwhelming",
                "The real reason cancer rates are rising: {tech} radiation",
            ],
            body_template=(
                "After extensive research on {source}, I am convinced that {tech} radiation "
                "is causing the rise in {condition}. {n} studies confirm this. "
                "The mainstream scientific community is ignoring this for political reasons."
            ),
            placeholder_pools={
                "tech": ["5G", "WiFi", "smart meters", "4G"],
                "source": ["independent websites", "alternative medicine forums", "YouTube documentaries"],
                "condition": ["childhood cancer", "neurological disorders", "autoimmune diseases"],
                "n": ["12", "20", "47"],
            },
            base_rule=3,
            base_action="remove",
            compatible_archetypes=["new_genuine", "mid_user", "first_offender", "serial_offender"],
            flair_options=["Research", "Discussion"],
            report_configs=[[("Misinformation", 8)], [("Pseudoscience", 6)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.5,
                "severity": "moderate",
                "base_acceptable": ["remove"],
                "explanation": "Clear misinformation about non-ionising radiation. Remove. Repeat offenders warrant temp_ban.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_selfpromo_researcher",
            category="t3_violation",
            title_templates=[
                "My new paper on {topic}: some thoughts and a summary",
                "Cross-posting my paper on {topic} — would love feedback",
            ],
            body_template=(
                "I've been working on {topic} for {years} years and just published a paper in {journal}. "
                "You can find it at {doi}. "
                "I know self-promotion is a grey area here — but I think this adds value to ongoing discussions."
            ),
            placeholder_pools={
                "topic": ["dietary fat and cardiovascular disease", "social media and teen mental health", "air pollution and cognition"],
                "years": ["5", "8", "12"],
                "journal": ["PLOS ONE", "Frontiers in Nutrition", "Nature Human Behaviour"],
                "doi": ["doi.org/10.1371/journal.pone.abc", "doi.org/10.3389/fnut.2024.abc"],
            },
            base_rule=4,
            base_action="warn",
            compatible_archetypes=["mid_user", "active_contributor", "veteran_expert", "new_genuine"],
            flair_options=["Research"],
            report_configs=[[("Self-promotion", 3)], [("Borderline spam", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.65,
                "severity": "mild",
                "base_acceptable": ["warn", "approve"],
                "explanation": "Borderline self-promotion: researcher sharing own work. Veterans and approved contributors warrant more leniency; new users warrant a warning.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_offtopic_veteran",
            category="t3_violation",
            title_templates=[
                "Farewell to Professor {name}, who passed away last week",
                "Remembering {name} — a giant in {field}",
            ],
            body_template=(
                "Many of you may know that Professor {name}, who spent {years} years advancing our "
                "understanding of {field}, passed away last week. "
                "I had the honour of working with them and wanted to share this here. "
                "Their contributions to {contribution} will not be forgotten."
            ),
            placeholder_pools={
                "name": ["J. Smith", "M. Nakamura", "L. Petrov", "C. Okafor"],
                "years": ["40", "45", "50"],
                "field": ["evolutionary biology", "quantum physics", "neuroscience", "climate science"],
                "contribution": ["understanding speciation", "quantum error correction", "memory consolidation", "climate modelling"],
            },
            base_rule=1,
            base_action="warn",
            compatible_archetypes=["veteran_expert", "active_contributor", "mid_user"],
            flair_options=["Discussion"],
            report_configs=[[("Off-topic", 2)], [("Personal", 1)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.7,
                "severity": "mild",
                "base_acceptable": ["warn", "approve"],
                "explanation": "Off-topic personal tribute. Technically violates Rule 1 but the author's standing and good intent warrant a gentle warning, not removal.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_predatory_journal",
            category="t3_violation",
            title_templates=[
                "New study in {journal}: {finding}",
                "{finding} — just published in {journal}",
            ],
            body_template=(
                "A study published in {journal} found that {finding}. "
                "N = {n}, p < 0.05. The authors conclude that {conclusion}. "
                "I found this really interesting and wanted to share with the community."
            ),
            placeholder_pools={
                "journal": ["OMICS International Journal", "Global Journal of Medicine", "International Journal of Science and Technology", "Open Access Journal of Medical Science"],
                "finding": ["high-dose vitamin C reduces tumour size", "homeopathic treatment shows statistically significant effect", "detox diet improves liver function"],
                "n": ["23", "45", "67"],
                "conclusion": ["this treatment should be considered as complementary medicine", "conventional treatments should be supplemented with this approach"],
            },
            base_rule=3,
            base_action="warn",
            compatible_archetypes=["new_genuine", "brand_new", "mid_user"],
            flair_options=["Research"],
            report_configs=[[("Predatory journal", 4)], [("Misinformation", 3)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.7,
                "severity": "mild",
                "base_acceptable": ["warn", "remove"],
                "explanation": "Predatory journal citation presented as credible research. Warn the poster about journal quality standards rather than removing outright.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_speculative_as_settled",
            category="t3_violation",
            title_templates=[
                "It's now settled: {claim}",
                "The science is clear and final on {topic}: {claim}",
            ],
            body_template=(
                "After reading extensively, I can tell you: {claim}. "
                "Anyone who disputes this is simply ignoring the evidence. "
                "The debate is over. {elaboration}."
            ),
            placeholder_pools={
                "claim": [
                    "dietary cholesterol has zero effect on cardiovascular disease",
                    "all antidepressants work purely via placebo",
                    "the lab leak hypothesis for COVID-19 is now confirmed",
                ],
                "topic": ["dietary cholesterol", "antidepressant mechanisms", "COVID-19 origins"],
                "elaboration": [
                    "Every scientist I've read agrees",
                    "This has been peer reviewed",
                    "The media just hasn't caught up yet",
                ],
            },
            base_rule=3,
            base_action="warn",
            compatible_archetypes=["mid_user", "first_offender", "veteran_expert"],
            flair_options=["Discussion"],
            report_configs=[[("Misinformation / Overconfident claim", 4)], [("Misrepresenting science", 3)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.7,
                "severity": "moderate",
                "base_acceptable": ["warn", "remove"],
                "explanation": "Presenting contested or speculative science as definitively settled. Warn to reframe with appropriate uncertainty.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_borderline_political",
            category="t3_violation",
            title_templates=[None],
            body_template=(
                "The evidence is clear. Anyone who denies {claim} is either scientifically illiterate "
                "or in the pocket of {industry}. "
                "We need to vote out {party} and elect politicians who respect science."
            ),
            placeholder_pools={
                "claim": ["climate change", "vaccine safety", "the effectiveness of nuclear energy"],
                "industry": ["fossil fuel companies", "pharmaceutical lobbies", "the anti-nuclear lobby"],
                "party": ["climate deniers", "anti-science politicians", "the current administration"],
            },
            base_rule=1,
            base_action="warn",
            compatible_archetypes=["mid_user", "first_offender", "active_contributor"],
            flair_options=[None],
            report_configs=[[("Political advocacy", 4)], [("Off-topic", 3)]],
            thread_context_pool=[
                ["Thread discussing the latest IPCC report"],
                ["Discussion about the science behind vaccine mandates"],
                ["Post about nuclear energy research"],
            ],
            signals={
                "base_difficulty": 0.65,
                "severity": "mild",
                "base_acceptable": ["warn", "remove"],
                "explanation": "Political advocacy mixed with science. The scientific claims may be valid but the partisan call-to-action violates Rule 1. Warn.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_wrong_flair_important",
            category="t3_violation",
            title_templates=[
                "Important: {chemical} found in some {product} — safety implications",
                "Health warning: elevated {chemical} detected in popular {product}",
            ],
            body_template=(
                "Analysis shows elevated {chemical} in some {product} brands. "
                "Levels exceeded {threshold} in {n} of {total} products tested. "
                "This has not yet been formally recalled. Source: {doi}."
            ),
            placeholder_pools={
                "chemical": ["PFAS", "arsenic", "lead", "microplastics"],
                "product": ["bottled water", "baby food", "sunscreen brands"],
                "threshold": ["EPA recommended limits", "EU safety thresholds", "WHO guidelines"],
                "n": ["8", "12", "15"],
                "total": ["30", "40", "50"],
                "doi": ["doi.org/10.1021/acs.est.abc", "doi.org/10.1016/j.chemosphere.2024.abc"],
            },
            base_rule=5,
            base_action="warn",
            compatible_archetypes=["new_genuine", "mid_user", "brand_new"],
            flair_options=["Research"],
            report_configs=[[("Wrong flair", 2)], []],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.65,
                "severity": "mild",
                "base_acceptable": ["warn"],
                "explanation": "Important safety information posted with wrong flair. Warn to re-flair rather than remove — the content itself is valuable and legitimate.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_extraordinary_claim",
            category="t3_violation",
            title_templates=[
                "I think I may have found evidence for {claim} — preliminary data",
                "Early results suggest {claim} — not sure what to make of this",
            ],
            body_template=(
                "I'm a {role} and I've been running experiments on {topic}. "
                "My preliminary results show {finding}. I know this sounds extraordinary. "
                "I've run the analysis {n} times and keep getting the same result. "
                "Happy to share the raw data. Is this worth publishing?"
            ),
            placeholder_pools={
                "claim": ["room-temperature superconductivity", "a violation of conservation of energy", "an anomalous signal in my particle detector"],
                "role": ["PhD student", "independent researcher", "lab technician", "postdoc"],
                "topic": ["novel material properties", "quantum effects at room temperature", "particle interactions"],
                "finding": [
                    "resistance dropping to zero at 25°C in this compound",
                    "an anomalous energy output I can't explain",
                    "a signal that doesn't match any known particle",
                ],
                "n": ["5", "10", "20"],
            },
            base_rule=3,
            base_action="warn",
            compatible_archetypes=["new_genuine", "brand_new", "mid_user"],
            flair_options=["Research", "Discussion"],
            report_configs=[[("Extraordinary claim / unverified", 3)], [("Misinformation risk", 2)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.75,
                "severity": "mild",
                "base_acceptable": ["warn", "approve"],
                "explanation": "Extraordinary claim with some preliminary data from a self-described researcher. Borderline — warn to seek peer review rather than removing outright.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_low_effort_from_veteran",
            category="t3_violation",
            title_templates=[None],
            body_template="Yep.",
            placeholder_pools={},
            base_rule=6,
            base_action="warn",
            compatible_archetypes=["veteran_expert", "active_contributor", "mid_user"],
            flair_options=[None],
            report_configs=[[("Low-effort", 2)], []],
            thread_context_pool=[
                ["High-quality discussion thread where a veteran user added a one-word affirmation"],
                ["Substantive debate where an experienced user simply agreed without explanation"],
            ],
            signals={
                "base_difficulty": 0.6,
                "severity": "mild",
                "base_acceptable": ["warn", "approve"],
                "explanation": "One-word response from a veteran contributor. Low-effort but from a trusted user in a substantive thread — a soft warning is appropriate.",
            },
        ),
        ContentTemplate(
            template_id="t3_var_conspiracy_new_user",
            category="t3_violation",
            title_templates=[
                "The truth about {topic} that mainstream science hides",
                "{topic}: what they don't want you to know",
            ],
            body_template=(
                "I've been doing my own research for {years} years and I can tell you: "
                "{claim}. The mainstream scientific consensus is {dismissed}. "
                "Wake up and do your own research — the truth is out there."
            ),
            placeholder_pools={
                "topic": ["vaccines", "5G", "GMOs", "fluoride"],
                "years": ["2", "3", "5"],
                "claim": ["vaccines contain harmful adjuvants deliberately left in", "5G is designed to monitor the population"],
                "dismissed": ["paid for by corporations", "manipulated by governments", "fake"],
            },
            base_rule=3,
            base_action="remove",
            compatible_archetypes=["brand_new", "new_genuine", "mid_user", "serial_offender"],
            flair_options=["Discussion"],
            report_configs=[[("Conspiracy theory / Misinformation", 7)], [("Pseudoscience", 5)]],
            thread_context_pool=None,
            signals={
                "base_difficulty": 0.5,
                "severity": "moderate",
                "base_acceptable": ["remove"],
                "explanation": "Conspiracy framing with no evidence. Remove. Serial offenders warrant temp_ban.",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------


def _build_template_registry() -> Dict[int, List[ContentTemplate]]:
    return {
        1: _task1_templates(),
        2: _task2_templates(),
        3: _task3_templates(),
    }


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------


def _compute_task1_gt(template: ContentTemplate, _author: UserHistory) -> GroundTruth:
    """Task 1: content determines spam/legit; author history does not change the answer."""
    if template.category == "spam":
        return GroundTruth(
            correct_action="remove",
            correct_rule=4,
            acceptable_actions=["remove"],
            difficulty_score=0.0,
            explanation=template.signals["explanation"],
        )
    return GroundTruth(
        correct_action="approve",
        correct_rule=None,
        acceptable_actions=["approve"],
        difficulty_score=0.0,
        explanation=template.signals["explanation"],
    )


def _compute_task2_gt(template: ContentTemplate, author: UserHistory) -> GroundTruth:
    """Task 2: content determines rule and base action; author history can shift severity."""
    signals = template.signals

    if template.base_action == "approve":
        return GroundTruth(
            correct_action="approve",
            correct_rule=None,
            acceptable_actions=["approve"],
            difficulty_score=0.0,
            explanation=signals["explanation"],
        )

    action = template.base_action
    acceptable = list(signals.get("base_acceptable", [action]))

    # Repeat offenders with multiple warnings: escalate warn → remove
    if author.is_repeat_offender and author.prior_warnings >= 2 and action == "warn":
        action = "remove"
        if "remove" not in acceptable:
            acceptable = ["remove"] + [a for a in acceptable if a != "remove"]
        if "warn" not in acceptable:
            acceptable.append("warn")

    return GroundTruth(
        correct_action=action,
        correct_rule=template.base_rule,
        acceptable_actions=acceptable,
        difficulty_score=0.0,
        explanation=signals["explanation"],
    )


def _compute_task3_gt(template: ContentTemplate, author: UserHistory) -> GroundTruth:
    """Task 3: full escalation logic — author history critically influences correct action."""
    signals = template.signals

    # Approve scenarios are author-invariant
    if template.base_action == "approve":
        return GroundTruth(
            correct_action="approve",
            correct_rule=None,
            acceptable_actions=["approve"],
            difficulty_score=0.0,
            explanation=signals["explanation"],
        )

    # Escalation scenarios are also invariant (always escalate regardless of author)
    if signals.get("requires_escalation"):
        return GroundTruth(
            correct_action="escalate_to_senior_mod",
            correct_rule=template.base_rule,
            acceptable_actions=["escalate_to_senior_mod"] + list(signals.get("alt_acceptable", [])),
            difficulty_score=0.0,
            explanation=signals["explanation"],
        )

    # Violation scenarios: apply escalation rules based on author history
    action = template.base_action
    acceptable = list(signals.get("base_acceptable", [action]))
    severity = signals.get("severity", "moderate")

    if author.is_repeat_offender and author.prior_warnings >= 3:
        # Serious history: escalate warn/remove to temp_ban
        if action in ("warn", "remove"):
            action = "temp_ban"
            acceptable = ["temp_ban", "remove"]
    elif author.prior_warnings >= 2 and severity in ("moderate", "severe"):
        # Multiple prior warnings + moderate violation: upgrade warn → remove
        if action == "warn":
            action = "remove"
            acceptable = ["remove", "warn"]
    elif author.account_age_days > 1000 and author.prior_warnings == 0 and severity in ("mild", "moderate"):
        # Veteran with clean history + mild/moderate violation: be lenient (remove -> warn)
        if action == "remove":
            action = "warn"
            acceptable = ["warn", "remove"]

    return GroundTruth(
        correct_action=action,
        correct_rule=template.base_rule,
        acceptable_actions=acceptable,
        difficulty_score=0.0,
        explanation=signals["explanation"],
    )


def _compute_ground_truth(
    template: ContentTemplate,
    author: UserHistory,
    task_level: int,
) -> GroundTruth:
    if task_level == 1:
        return _compute_task1_gt(template, author)
    if task_level == 2:
        return _compute_task2_gt(template, author)
    return _compute_task3_gt(template, author)


# ---------------------------------------------------------------------------
# Difficulty calibration
# ---------------------------------------------------------------------------


def _compute_difficulty(
    template: ContentTemplate,
    author: UserHistory,
    task_level: int,
) -> float:
    base = float(template.signals.get("base_difficulty", 0.3))
    mismatch = 0.0

    # Author-content mismatch increases difficulty
    if template.category == "spam" and author.karma > 5000:
        mismatch += 0.15   # Trusted user posting spammy content is harder to judge
    elif template.category == "legit" and author.account_age_days < 5:
        mismatch += 0.10   # Brand-new account with excellent content is harder

    # Task 3 specific: author history in the middle ground is most ambiguous
    if task_level == 3 and template.base_action not in ("approve", "escalate_to_senior_mod"):
        if 1 <= author.prior_warnings <= 2:
            mismatch += 0.08  # Middle ground is more ambiguous than 0 or 3+

    return min(1.0, base + mismatch)


# ---------------------------------------------------------------------------
# Post generation helpers
# ---------------------------------------------------------------------------


def _fill_template(
    template: ContentTemplate,
    rng: random.Random,
) -> Tuple[Optional[str], str]:
    """
    Fill a template's title and body placeholders.
    Always makes exactly 1 + len(placeholder_pools) RNG calls.
    """
    title_raw = rng.choice(template.title_templates)  # 1 call

    # Fill all placeholder pools in sorted key order (deterministic call sequence)
    filled: Dict[str, str] = {}
    for key in sorted(template.placeholder_pools.keys()):
        filled[key] = rng.choice(template.placeholder_pools[key])

    # Apply substitution (title may be None for comment-level scenarios)
    if title_raw is not None and filled:
        title: Optional[str] = title_raw.format(**filled)
    else:
        title = title_raw

    body = template.body_template.format(**filled) if filled else template.body_template
    return title, body


def _make_engagement(
    template: ContentTemplate,
    rng: random.Random,
) -> Tuple[int, int, List[Report]]:
    """Return (score, num_comments, reports). Always makes 3 RNG calls."""
    score_min, score_max = template.signals.get("score_range", (10, 2000))
    comments_min, comments_max = template.signals.get("comments_range", (5, 200))

    score = rng.randint(score_min, score_max)
    num_comments = rng.randint(comments_min, comments_max)

    report_config_idx = rng.randint(0, len(template.report_configs) - 1)
    reports = [
        Report(reason=reason, count=count)
        for reason, count in template.report_configs[report_config_idx]
    ]
    return score, num_comments, reports


# ---------------------------------------------------------------------------
# ContentGenerator
# ---------------------------------------------------------------------------


class ContentGenerator:
    """
    Generates deterministic RL episodes from parameterized content templates.

    Each call to generate_episode with the same seed produces an identical
    episode. Different seeds produce different author profiles, placeholder
    fills, and engagement metrics — creating thousands of unique combinations.
    """

    def __init__(self) -> None:
        self._registry = _build_template_registry()

    def generate_episode(
        self,
        task_level: int,
        num_posts: int,
        seed: Optional[int] = None,
    ) -> List[Scenario]:
        """
        Return a list of (RedditPost, GroundTruth) pairs for one episode.

        Args:
            task_level: 1, 2, or 3
            num_posts:  number of posts in the episode
            seed:       optional random seed for full reproducibility
        """
        if task_level not in self._registry:
            raise ValueError(f"task_level must be 1, 2, or 3; got {task_level}")

        # Master RNG: only used to derive sub-seeds (fixed call count regardless of template)
        master_rng = random.Random(seed)
        post_seeds = [master_rng.randint(0, 2**63) for _ in range(num_posts)]
        shuffle_seed = master_rng.randint(0, 2**63)

        templates = self._registry[task_level]
        scenarios: List[Scenario] = []

        for post_seed in post_seeds:
            post_rng = random.Random(post_seed)

            # 1. Select template
            template = post_rng.choice(templates)

            # 2. Select archetype and generate author (always 9 RNG calls)
            archetype_name = post_rng.choice(template.compatible_archetypes)
            author = _generate_author(archetype_name, post_rng)

            # 3. Fill template placeholders (1 + len(placeholder_pools) calls)
            title, body = _fill_template(template, post_rng)

            # 4. Randomize engagement (3 calls)
            score, num_comments, reports = _make_engagement(template, post_rng)

            # 5. Select flair (1 call)
            flair = post_rng.choice(template.flair_options)

            # 6. Select thread context (1 call, may be None)
            if template.thread_context_pool is not None:
                thread_context: Optional[List[str]] = post_rng.choice(template.thread_context_pool)
            else:
                thread_context = None

            # 7. Compute ground truth (pure function — no RNG)
            gt = _compute_ground_truth(template, author, task_level)

            # 8. Set difficulty score (pure function — no RNG)
            gt.difficulty_score = _compute_difficulty(template, author, task_level)

            # 9. Build post_id from post_seed (deterministic, unique per seed)
            post_id = f"t{task_level}_{post_seed & 0xFFFFFFFF:08x}"

            post = RedditPost(
                post_id=post_id,
                title=title,
                body=body,
                author=author,
                subreddit=SCIENCE_SUBREDDIT,
                score=score,
                num_comments=num_comments,
                reports=reports,
                flair=flair,
                thread_context=thread_context,
            )
            scenarios.append((post, gt))

        # Shuffle with an isolated seed derived from the master RNG
        random.Random(shuffle_seed).shuffle(scenarios)
        return scenarios

    def available_task_levels(self) -> List[int]:
        return list(self._registry.keys())

    def scenario_count(self, task_level: int) -> int:
        """Returns the number of content templates (the effective pool size)."""
        return len(self._registry[task_level])

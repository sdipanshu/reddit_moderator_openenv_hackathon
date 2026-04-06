"""
Pre-defined scenario templates for the Reddit Mod Bot RL environment.

Scenarios are fully specified — no external API calls required.
Seeds make episodes reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
# Helper authors
# ---------------------------------------------------------------------------

def _spam_author(username: str = "AutoPostBot99") -> UserHistory:
    return UserHistory(
        username=username,
        account_age_days=1,
        karma=-12,
        prior_warnings=0,
        prior_removals=4,
        is_repeat_offender=True,
        is_approved_contributor=False,
    )


def _new_author(username: str = "curious_user_42") -> UserHistory:
    return UserHistory(
        username=username,
        account_age_days=30,
        karma=150,
        prior_warnings=0,
        prior_removals=0,
        is_repeat_offender=False,
        is_approved_contributor=False,
    )


def _veteran_author(username: str = "dr_jane_science") -> UserHistory:
    return UserHistory(
        username=username,
        account_age_days=2000,
        karma=48000,
        prior_warnings=0,
        prior_removals=0,
        is_repeat_offender=False,
        is_approved_contributor=True,
    )


def _repeat_offender(username: str = "rule_breaker_77") -> UserHistory:
    return UserHistory(
        username=username,
        account_age_days=120,
        karma=500,
        prior_warnings=3,
        prior_removals=2,
        is_repeat_offender=True,
        is_approved_contributor=False,
    )


def _mid_user(username: str = "science_reader") -> UserHistory:
    return UserHistory(
        username=username,
        account_age_days=400,
        karma=2200,
        prior_warnings=0,
        prior_removals=0,
        is_repeat_offender=False,
        is_approved_contributor=False,
    )


# ---------------------------------------------------------------------------
# Scenario type alias
# ---------------------------------------------------------------------------

Scenario = Tuple[RedditPost, GroundTruth]


# ---------------------------------------------------------------------------
# Task 1 — Spam Detection (8 spam + 7 legitimate = 15 scenarios)
# ---------------------------------------------------------------------------

def _task1_scenarios() -> List[Scenario]:
    s = SCIENCE_SUBREDDIT
    return [
        # ── SPAM ──────────────────────────────────────────────────────────────
        (
            RedditPost(
                post_id="t1_spam_001",
                title="Boost your IQ by 40 points in 30 days with NeuroMax supplements!",
                body=(
                    "Scientists HATE this one weird trick! Our proprietary blend of "
                    "nootropics has been clinically shown to increase cognitive function "
                    "dramatically. Limited-time offer: use code REDDIT20 at checkout. "
                    "Visit our site now!"
                ),
                author=_spam_author("NeuroMaxOfficial"),
                subreddit=s,
                score=1,
                num_comments=0,
                reports=[Report(reason="Spam / Advertising", count=7)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=4,
                acceptable_actions=["remove"],
                difficulty_score=0.05,
                explanation="Clear commercial spam promoting supplements with a discount code.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_spam_002",
                title="Bitcoin will hit $500k — here's the scientific proof",
                body=(
                    "Quantum AI analysis of market fractals PROVES cryptocurrency "
                    "is entering a supercycle. I've backtested this model for 3 years. "
                    "Join my Telegram for the full report: t.me/cryptoscience_profits"
                ),
                author=_spam_author("CryptoQuantumAI"),
                subreddit=s,
                score=2,
                num_comments=1,
                reports=[Report(reason="Spam", count=5), Report(reason="Misinformation", count=3)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=4,
                acceptable_actions=["remove"],
                difficulty_score=0.1,
                explanation="Crypto spam disguised as science, redirecting to external channel.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_spam_003",
                title="Check out my YouTube channel — I explain science in simple terms!",
                body=(
                    "Hey r/science! I've been making educational science videos for "
                    "2 years. My latest video covers quantum mechanics. Please subscribe "
                    "and share! Link: youtube.com/watch?v=abc123science"
                ),
                author=UserHistory(
                    username="ScienceYouTuber",
                    account_age_days=3,
                    karma=10,
                    prior_warnings=0,
                    prior_removals=0,
                    is_repeat_offender=False,
                    is_approved_contributor=False,
                ),
                subreddit=s,
                score=3,
                num_comments=0,
                reports=[Report(reason="Self-promotion", count=4)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=4,
                acceptable_actions=["remove"],
                difficulty_score=0.1,
                explanation="Pure self-promotion with a YouTube link, no scientific content contributed.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_spam_004",
                title="Top 10 Science Gadgets You NEED in 2024 (affiliate links inside)",
                body=(
                    "As an Amazon Associate I earn from qualifying purchases. "
                    "Here are my top picks for home science experiments! "
                    "[Microscope](amzn.to/abc) [Telescope](amzn.to/def) ..."
                ),
                author=_spam_author("GadgetReviewBot"),
                subreddit=s,
                score=1,
                num_comments=0,
                reports=[Report(reason="Spam / Affiliate links", count=6)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=4,
                acceptable_actions=["remove"],
                difficulty_score=0.05,
                explanation="Affiliate link spam — explicitly disclosed as monetised advertising.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_spam_005",
                title="Lose 30 lbs in 30 days — scientists won't tell you this!",
                body=(
                    "Big Pharma is hiding the truth about metabolism! Our all-natural "
                    "KetoFlame supplement resets your body's fat-burning mode. "
                    "Over 10,000 satisfied customers. Order now at ketoflame.com!"
                ),
                author=_spam_author("KetoFlameOfficial"),
                subreddit=s,
                score=1,
                num_comments=0,
                reports=[Report(reason="Spam", count=8), Report(reason="Misinformation", count=5)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=4,
                acceptable_actions=["remove"],
                difficulty_score=0.05,
                explanation="Health supplement spam with conspiracy-theory framing.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_spam_006",
                title="WIN a free electron microscope! Just upvote and comment!",
                body=(
                    "We're giving away a $5,000 electron microscope to celebrate our "
                    "lab's anniversary! To enter: (1) Upvote this post, (2) Comment "
                    "your university, (3) Follow us on Instagram. Drawing Friday!"
                ),
                author=_spam_author("LabEquipmentGiveaway"),
                subreddit=s,
                score=45,
                num_comments=62,
                reports=[Report(reason="Spam / Engagement bait", count=10)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=4,
                acceptable_actions=["remove"],
                difficulty_score=0.15,
                explanation="Engagement-bait giveaway — manipulates votes and harvests user data.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_spam_007",
                title="Revolutionary study PROVES homeopathy cures cancer — Big Pharma exposed",
                body=(
                    "A groundbreaking 2023 study from the Global Institute of Alternative "
                    "Medicine has shown a 94% cure rate for stage 4 cancer using water "
                    "memory therapy. Get the full report at alternativecures.net/cancer-cure"
                ),
                author=_spam_author("AltMedTruth"),
                subreddit=s,
                score=2,
                num_comments=3,
                reports=[Report(reason="Misinformation", count=9), Report(reason="Spam", count=4)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=4,
                acceptable_actions=["remove"],
                difficulty_score=0.1,
                explanation="Spam linking to a fake study from a non-existent institution.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_spam_008",
                title="Join our Science Research Team — work from home and earn $500/week!",
                body=(
                    "We're hiring remote data annotators for our AI science project! "
                    "No experience needed. Earn $500/week from home. Apply at: "
                    "sciencejobs-remote.biz — limited spots available!"
                ),
                author=_spam_author("SciJobsRemote"),
                subreddit=s,
                score=1,
                num_comments=0,
                reports=[Report(reason="Spam / Scam", count=6)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=4,
                acceptable_actions=["remove"],
                difficulty_score=0.1,
                explanation="Job-scam spam unrelated to scientific discussion.",
            ),
        ),
        # ── LEGITIMATE ────────────────────────────────────────────────────────
        (
            RedditPost(
                post_id="t1_legit_001",
                title="New Nature study finds gut microbiome directly influences anxiety levels",
                body=(
                    "A new paper published in Nature (doi:10.1038/s41586-024-07123-x) "
                    "demonstrates a causal link between specific Lactobacillus strains "
                    "and GABA receptor expression in the vagus nerve. The team used "
                    "germ-free mouse models and human cohort validation. Would love "
                    "to hear thoughts from anyone working in the microbiome space."
                ),
                author=_veteran_author("microbiome_researcher"),
                subreddit=s,
                score=1420,
                num_comments=87,
                reports=[],
                flair="Research",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="High-quality post sharing a legitimate Nature paper with DOI and thoughtful summary.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_legit_002",
                title="Can someone explain how mRNA vaccines actually work at the molecular level?",
                body=(
                    "I've read the Wikipedia article but I'm struggling to understand "
                    "the translation step. Specifically, how does the ribosome know to "
                    "stop producing the spike protein after a certain number of copies? "
                    "Is there a degradation signal in the mRNA sequence?"
                ),
                author=_new_author("biology_undergrad_22"),
                subreddit=s,
                score=234,
                num_comments=45,
                reports=[],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="Genuine scientific question from a student — exactly the kind of discussion r/science should host.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_legit_003",
                title="[AMA] I'm a NASA astrobiologist who just returned from the TRAPPIST-1 data analysis team. Ask me anything!",
                body=(
                    "Hi r/science! I'm Dr. Sarah Okonkwo, astrobiologist at NASA Goddard. "
                    "I've spent the last 18 months analysing JWST spectroscopy data from "
                    "the TRAPPIST-1 system. I'll be here for 3 hours. Proof in comments."
                ),
                author=_veteran_author("DrSarahOkonkwo_NASA"),
                subreddit=s,
                score=8900,
                num_comments=1240,
                reports=[],
                flair="AMA",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="Legitimate AMA from a credentialed scientist with proof provided.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_legit_004",
                title="Why do black holes emit Hawking radiation if nothing can escape them?",
                body=(
                    "I understand that Hawking radiation is produced at the event horizon "
                    "via virtual particle pairs. But I'm confused — if the particle "
                    "that falls in has negative energy, how does that cause the black hole "
                    "to lose mass? Isn't energy always positive from a GR standpoint?"
                ),
                author=_mid_user("physics_enthusiast"),
                subreddit=s,
                score=567,
                num_comments=93,
                reports=[],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="Thoughtful physics question demonstrating prior research effort.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_legit_005",
                title="2024 IPCC report data shows Arctic sea ice at historic 45-year low",
                body=(
                    "The supplementary data released alongside the 2024 IPCC Working Group I "
                    "report shows September Arctic sea-ice extent fell to 3.12 million km², "
                    "a 45-year record low. The report attributes 94% of variance to "
                    "anthropogenic forcing. Full data: ipcc.ch/report/ar6"
                ),
                author=_mid_user("climate_data_nerd"),
                subreddit=s,
                score=3400,
                num_comments=412,
                reports=[],
                flair="News",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="Well-sourced climate news post with primary IPCC citation.",
            ),
        ),
        (
            RedditPost(
                post_id="t1_legit_006",
                title="My PhD research on tardigrade cryptobiosis mechanisms — preprint now available",
                body=(
                    "I'm a 3rd-year PhD student at Uppsala University. My first paper "
                    "on trehalose glass formation in Ramazzottius varieornatus is now "
                    "on bioRxiv (doi:10.1101/2024.02.15.580123). Happy to answer "
                    "questions about the methodology. Note: not yet peer reviewed."
                ),
                author=UserHistory(
                    username="tardigrade_phd",
                    account_age_days=800,
                    karma=3200,
                    prior_warnings=0,
                    prior_removals=0,
                    is_repeat_offender=False,
                    is_approved_contributor=True,
                ),
                subreddit=s,
                score=780,
                num_comments=56,
                reports=[],
                flair="Research",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.1,
                explanation=(
                    "Researcher sharing their own legitimate preprint with DOI and honest caveat "
                    "about peer review status. This is within Rule 4 allowances for researchers."
                ),
            ),
        ),
        (
            RedditPost(
                post_id="t1_legit_007",
                title="Discussion: Has the replication crisis affected your field? How is it being addressed?",
                body=(
                    "Following Ioannidis (2005) and the Open Science Collaboration's "
                    "replication study (Science, 2015), many fields have adopted "
                    "pre-registration and open data practices. I'm curious how "
                    "researchers here feel about progress over the last decade. "
                    "What structural changes have made the biggest difference?"
                ),
                author=_veteran_author("meta_science_fan"),
                subreddit=s,
                score=2100,
                num_comments=340,
                reports=[],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="High-quality meta-scientific discussion with proper citations.",
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Task 2 — Rule Violation Classification (20 scenarios)
# ---------------------------------------------------------------------------

def _task2_scenarios() -> List[Scenario]:
    s = SCIENCE_SUBREDDIT
    return [
        # ── Rule 1: Off-topic ──────────────────────────────────────────────────
        (
            RedditPost(
                post_id="t2_r1_001",
                title="Why is the government ignoring climate scientists and doing NOTHING about policy?",
                body=(
                    "I'm furious. We have all the data we need. 97% of scientists agree. "
                    "Yet politicians keep blocking carbon taxes. We need to vote these "
                    "climate deniers out NOW. This is a political failure, full stop. "
                    "Who else is angry about the lack of action?"
                ),
                author=_mid_user("angry_climate_voter"),
                subreddit=s,
                score=450,
                num_comments=210,
                reports=[Report(reason="Off-topic / Political rant", count=8)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=1,
                acceptable_actions=["remove"],
                difficulty_score=0.2,
                explanation=(
                    "Political advocacy post, not scientific discussion. "
                    "Climate science is on-topic; climate policy activism is not."
                ),
            ),
        ),
        (
            RedditPost(
                post_id="t2_r1_002",
                title="Just finished playing 'Science Simulator 2024' — best game ever!",
                body=(
                    "OMG this game lets you run your own lab, apply for grants, "
                    "deal with peer review drama... it's so realistic lol. "
                    "Anyone else played it? What's your favourite experiment to run?"
                ),
                author=_new_author("gamer_scientist"),
                subreddit=s,
                score=12,
                num_comments=5,
                reports=[Report(reason="Not science content", count=4)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=1,
                acceptable_actions=["remove"],
                difficulty_score=0.1,
                explanation="Video game post with no scientific content — clearly off-topic.",
            ),
        ),
        (
            RedditPost(
                post_id="t2_r1_003",
                title="My grandma just passed away from cancer. She was a biologist.",
                body=(
                    "I know this isn't exactly science news but I just needed to share. "
                    "She spent 40 years studying marine organisms. She always said "
                    "the ocean was her first love. Miss you, Grandma."
                ),
                author=_new_author("grieving_grandkid"),
                subreddit=s,
                score=890,
                num_comments=120,
                reports=[Report(reason="Off-topic / Personal", count=3)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=1,
                acceptable_actions=["remove"],
                difficulty_score=0.25,
                explanation=(
                    "Sympathetic but off-topic personal tribute. "
                    "r/science requires scientific discussion, not personal stories."
                ),
            ),
        ),
        # ── Rule 2: Personal attacks ───────────────────────────────────────────
        (
            RedditPost(
                post_id="t2_r2_001",
                title=None,  # type: ignore[arg-type]
                body=(
                    "You're an absolute idiot if you still think vaccines cause autism. "
                    "This has been debunked thousands of times. People like you are "
                    "literally killing children with your stupidity."
                ),
                author=_mid_user("vax_defender_2000"),
                subreddit=s,
                score=-5,
                num_comments=0,
                reports=[Report(reason="Personal attack / Insult", count=9)],
                flair=None,
                thread_context=[
                    "Original post: 'I'm concerned about vaccine safety, can someone point me to the research?'"
                ],
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=2,
                acceptable_actions=["remove", "warn"],
                difficulty_score=0.2,
                explanation=(
                    "Direct insult ('absolute idiot', 'your stupidity'). "
                    "Even when defending correct science, personal attacks are not permitted."
                ),
            ),
        ),
        (
            RedditPost(
                post_id="t2_r2_002",
                title=None,  # type: ignore[arg-type]
                body=(
                    "Maybe if you actually read the paper instead of just the abstract "
                    "you'd understand why your interpretation is completely wrong. "
                    "This is embarrassingly basic stuff."
                ),
                author=_mid_user("condescending_prof"),
                subreddit=s,
                score=3,
                num_comments=2,
                reports=[Report(reason="Condescending / Rude", count=3)],
                flair=None,
                thread_context=["Original post: Discussion about a recent meta-analysis"],
            ),
            GroundTruth(
                correct_action="warn",
                correct_rule=2,
                acceptable_actions=["warn", "remove"],
                difficulty_score=0.3,
                explanation=(
                    "Condescending and dismissive ('embarrassingly basic'), "
                    "but no direct profanity. A warning is appropriate first action."
                ),
            ),
        ),
        (
            RedditPost(
                post_id="t2_r2_003",
                title=None,  # type: ignore[arg-type]
                body=(
                    "The lead author of this study is a known fraud who has had "
                    "three papers retracted. I wouldn't trust anything they publish. "
                    "Their entire research programme is corrupt garbage."
                ),
                author=_mid_user("academic_vigilante"),
                subreddit=s,
                score=15,
                num_comments=8,
                reports=[Report(reason="Personal attack on researcher", count=5)],
                flair=None,
                thread_context=["Thread discussing a new paper on gene therapy"],
            ),
            GroundTruth(
                correct_action="warn",
                correct_rule=2,
                acceptable_actions=["warn", "remove"],
                difficulty_score=0.35,
                explanation=(
                    "Personal attack on a named researcher with unsubstantiated fraud claims. "
                    "Retractions are public record (potentially factual) but 'corrupt garbage' "
                    "crosses into personal attack territory. Warn to allow factual restatement."
                ),
            ),
        ),
        # ── Rule 3: Misinformation ─────────────────────────────────────────────
        (
            RedditPost(
                post_id="t2_r3_001",
                title="New study: 5G towers cause DNA damage and cancer — governments hiding the truth",
                body=(
                    "Researchers at the Institute for Electromagnetic Safety have proven "
                    "that 5G radiation at 60GHz causes double-strand DNA breaks in "
                    "human lymphocytes. Effect size: 340% increase in mutations. "
                    "Big Telecom is suppressing this. Share before it gets deleted!"
                ),
                author=_new_author("5g_truther"),
                subreddit=s,
                score=34,
                num_comments=12,
                reports=[Report(reason="Misinformation", count=11), Report(reason="Spam", count=2)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=3,
                acceptable_actions=["remove"],
                difficulty_score=0.1,
                explanation=(
                    "Well-documented misinformation — 5G frequencies are non-ionising "
                    "and incapable of causing DNA damage. No credible source cited."
                ),
            ),
        ),
        (
            RedditPost(
                post_id="t2_r3_002",
                title="Study confirms: unvaccinated children are 40x healthier than vaccinated ones",
                body=(
                    "A comprehensive 2023 study of 50,000 children found that unvaccinated "
                    "kids had 40x fewer chronic illnesses, better immune function, and "
                    "higher IQ scores. The mainstream media won't cover this."
                ),
                author=_new_author("naturo_health_mom"),
                subreddit=s,
                score=8,
                num_comments=4,
                reports=[Report(reason="Misinformation / Anti-vax", count=14)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=3,
                acceptable_actions=["remove"],
                difficulty_score=0.1,
                explanation="Anti-vaccine misinformation with fabricated statistics and no credible source.",
            ),
        ),
        (
            RedditPost(
                post_id="t2_r3_003",
                title="Scientists discover chocolate prevents Alzheimer's disease",
                body=(
                    "A new Harvard study found that eating dark chocolate daily prevents "
                    "Alzheimer's. The study followed 10,000 participants for 20 years "
                    "and found a 78% reduction in risk. Time to eat more chocolate! "
                    "Sharing because this made me happy."
                ),
                author=_mid_user("choco_health_fan"),
                subreddit=s,
                score=2300,
                num_comments=190,
                reports=[Report(reason="Misleading title / Overstated claims", count=6)],
                flair="News",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=3,
                acceptable_actions=["remove", "warn"],
                difficulty_score=0.3,
                explanation=(
                    "Massively overstated headline — the actual study (if it exists) "
                    "would show correlation, not prevention. No DOI or source link. "
                    "Misrepresenting study conclusions violates Rule 3."
                ),
            ),
        ),
        # ── Rule 4: Self-promotion / spam ──────────────────────────────────────
        (
            RedditPost(
                post_id="t2_r4_001",
                title="I explain CRISPR in 10 minutes — my new YouTube video",
                body=(
                    "Hey everyone! I made a video explaining CRISPR-Cas9 gene editing "
                    "from scratch. It's 10 minutes, fully animated. Link: "
                    "youtube.com/watch?v=xyzCRISPR. Would love feedback! "
                    "This is my first science video."
                ),
                author=UserHistory(
                    username="SciCommYouTuber",
                    account_age_days=5,
                    karma=20,
                    prior_warnings=0,
                    prior_removals=0,
                    is_repeat_offender=False,
                    is_approved_contributor=False,
                ),
                subreddit=s,
                score=4,
                num_comments=2,
                reports=[Report(reason="Self-promotion", count=5)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="warn",
                correct_rule=4,
                acceptable_actions=["warn", "remove"],
                difficulty_score=0.2,
                explanation=(
                    "Self-promotion of a YouTube video with no accompanying scientific "
                    "discussion. New account posting only promotional content."
                ),
            ),
        ),
        (
            RedditPost(
                post_id="t2_r4_002",
                title="My blog: The Hidden Dangers of Microplastics (Part 47)",
                body=(
                    "This is part 47 of my ongoing microplastics series. In today's post "
                    "I cover the latest research on blood-brain barrier penetration. "
                    "Full article at: mysciblog.substack.com/microplastics-47"
                ),
                author=UserHistory(
                    username="MySciBlogger",
                    account_age_days=200,
                    karma=100,
                    prior_warnings=1,
                    prior_removals=3,
                    is_repeat_offender=True,
                    is_approved_contributor=False,
                ),
                subreddit=s,
                score=2,
                num_comments=0,
                reports=[Report(reason="Excessive self-promotion", count=7)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=4,
                acceptable_actions=["remove"],
                difficulty_score=0.2,
                explanation=(
                    "User posts exclusively links to their own blog (part 47!). "
                    "Prior warning ignored. This constitutes excessive self-promotion."
                ),
            ),
        ),
        # ── Rule 5: Wrong flair ────────────────────────────────────────────────
        (
            RedditPost(
                post_id="t2_r5_001",
                title="In my opinion, we are vastly underestimating the risks of AGI",
                body=(
                    "I've been thinking about this a lot. Based on my reading of "
                    "Bostrom and LeCun's debate, I believe the consensus is too "
                    "optimistic. This is my personal take, not based on new research."
                ),
                author=_mid_user("ai_risk_thinker"),
                subreddit=s,
                score=230,
                num_comments=88,
                reports=[Report(reason="Wrong flair — this is opinion, not research", count=5)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="warn",
                correct_rule=5,
                acceptable_actions=["warn"],
                difficulty_score=0.2,
                explanation=(
                    "Clearly labelled as personal opinion in the body but flaired [Research]. "
                    "Author needs to re-flair as [Discussion]."
                ),
            ),
        ),
        (
            RedditPost(
                post_id="t2_r5_002",
                title="BBC reports on new Alzheimer's drug trial showing promising results",
                body=(
                    "Link to the BBC article about the Phase III trial of lecanemab. "
                    "The trial showed a 27% slowing of cognitive decline vs placebo "
                    "in 1,795 patients. Full details in the article."
                ),
                author=_new_author("news_sharer_42"),
                subreddit=s,
                score=1800,
                num_comments=230,
                reports=[Report(reason="Should be [News] not [Discussion]", count=2)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="warn",
                correct_rule=5,
                acceptable_actions=["warn"],
                difficulty_score=0.2,
                explanation="News article shared as [Discussion]. Should be flaired [News].",
            ),
        ),
        # ── Rule 6: Low-effort ────────────────────────────────────────────────
        (
            RedditPost(
                post_id="t2_r6_001",
                title=None,  # type: ignore[arg-type]
                body="This.",
                author=_new_author("low_effort_user"),
                subreddit=s,
                score=1,
                num_comments=0,
                reports=[Report(reason="Low-effort content", count=4)],
                flair=None,
                thread_context=["Thread: Discussion of new climate model"],
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=6,
                acceptable_actions=["remove"],
                difficulty_score=0.05,
                explanation="Single-word reply with zero scientific content.",
            ),
        ),
        (
            RedditPost(
                post_id="t2_r6_002",
                title="What do you all think about this?",
                body="",
                author=_new_author("empty_poster"),
                subreddit=s,
                score=1,
                num_comments=2,
                reports=[Report(reason="Low-effort / Empty post", count=5)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=6,
                acceptable_actions=["remove"],
                difficulty_score=0.05,
                explanation="Empty post body — no topic, no content, no value.",
            ),
        ),
        (
            RedditPost(
                post_id="t2_r6_003",
                title="Screenshot of a tweet about a study",
                body="Thought this was interesting [image attached]",
                author=_new_author("screenshot_sharer"),
                subreddit=s,
                score=5,
                num_comments=1,
                reports=[Report(reason="Low-effort / Screenshot without context", count=3)],
                flair="News",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=6,
                acceptable_actions=["remove"],
                difficulty_score=0.1,
                explanation="Screenshot with no actual content or link to the underlying study.",
            ),
        ),
        # ── Clean posts (approve) ──────────────────────────────────────────────
        (
            RedditPost(
                post_id="t2_clean_001",
                title="Meta-analysis of 89 studies finds strong evidence for evolution of antibiotic resistance in hospital settings",
                body=(
                    "A new Cochrane-style meta-analysis (doi:10.1016/j.cell.2024.01.042) "
                    "synthesises 89 longitudinal studies across 24 countries to model "
                    "the rate of de-novo resistance emergence in ICU settings. "
                    "Key finding: hand-hygiene interventions reduce resistance emergence "
                    "by 34% (95% CI: 28–40%). Full methods section is open access."
                ),
                author=_veteran_author("infectious_disease_doc"),
                subreddit=s,
                score=4500,
                num_comments=310,
                reports=[],
                flair="Research",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="Exemplary post — meta-analysis with DOI, specific findings, and open access note.",
            ),
        ),
        (
            RedditPost(
                post_id="t2_clean_002",
                title="How do epigenetic clocks work and how accurate are they for predicting biological age?",
                body=(
                    "I've been reading about Horvath's clock and the GrimAge model. "
                    "I understand they measure DNA methylation patterns, but I'm unclear "
                    "on how they're validated against actual health outcomes. "
                    "Can someone explain the calibration process and the error bars?"
                ),
                author=_mid_user("curious_biohacker"),
                subreddit=s,
                score=345,
                num_comments=67,
                reports=[],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="Well-researched question citing specific models — exactly what r/science is for.",
            ),
        ),
        (
            RedditPost(
                post_id="t2_clean_003",
                title="Where can I find good introductory resources for learning about quantum computing?",
                body=(
                    "Background: I have a maths degree and I'm comfortable with linear algebra "
                    "and basic probability. I've been going through Nielsen & Chuang but it's "
                    "quite dense. Are there good lecture series or papers you'd recommend "
                    "as a gentler on-ramp before tackling the textbook fully?"
                ),
                author=_new_author("maths_grad_curious"),
                subreddit=s,
                score=189,
                num_comments=42,
                reports=[],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="Legitimate learning resource request from someone with appropriate background.",
            ),
        ),
        (
            RedditPost(
                post_id="t2_clean_004",
                title="[News] FDA approves first CRISPR-based gene therapy for sickle cell disease",
                body=(
                    "The FDA has approved Casgevy (exagamglogene autotemcel), the first "
                    "CRISPR-based therapy for sickle cell disease. The Phase III trial "
                    "showed 97% of patients remained vaso-occlusive crisis-free at 12 months. "
                    "Source: nejm.org/doi/full/10.1056/NEJMoa2309676"
                ),
                author=_mid_user("biotech_news_follower"),
                subreddit=s,
                score=9800,
                num_comments=620,
                reports=[],
                flair="News",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.05,
                explanation="Breaking science news with primary NEJM source and accurate clinical data.",
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Task 3 — Context-Aware Moderation (15 scenarios)
# ---------------------------------------------------------------------------

def _task3_scenarios() -> List[Scenario]:
    s = SCIENCE_SUBREDDIT
    return [
        # 1. Sarcasm with /s — looks like misinformation, is clearly satirical
        (
            RedditPost(
                post_id="t3_001",
                title=None,  # type: ignore[arg-type]
                body=(
                    "Oh sure, essential oils cure cancer, the Earth is 6000 years old, "
                    "and vaccines contain microchips. Why are we even doing science? /s"
                ),
                author=_mid_user("sarcastic_scientist"),
                subreddit=s,
                score=890,
                num_comments=45,
                reports=[Report(reason="Misinformation", count=2)],
                flair=None,
                thread_context=["Thread: Discussing the rise of science denialism"],
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.5,
                explanation=(
                    "Clear sarcasm marked with /s. The context (discussing science denialism) "
                    "and the absurdity make the satirical intent obvious. Approve."
                ),
            ),
        ),
        # 2. First-time mild violation by long-standing veteran → warn, not remove
        (
            RedditPost(
                post_id="t3_002",
                title=None,  # type: ignore[arg-type]
                body=(
                    "Honestly this is a pretty dumb interpretation of the Bayesian framework. "
                    "The author clearly hasn't worked with hierarchical models before."
                ),
                author=_veteran_author("bayesian_professor"),
                subreddit=s,
                score=12,
                num_comments=3,
                reports=[Report(reason="Condescending tone", count=2)],
                flair=None,
                thread_context=["Statistical discussion thread"],
            ),
            GroundTruth(
                correct_action="warn",
                correct_rule=2,
                acceptable_actions=["warn"],
                difficulty_score=0.55,
                explanation=(
                    "Condescending ('pretty dumb') but borderline — a veteran with 0 prior warnings. "
                    "Warning is proportional; removal would be excessive for a first minor infraction."
                ),
            ),
        ),
        # 3. Repeat offender — same rule, third violation → temp_ban
        (
            RedditPost(
                post_id="t3_003",
                title=None,  # type: ignore[arg-type]
                body=(
                    "People who believe in homeopathy are intellectually bankrupt. "
                    "I have no patience for this level of stupidity in 2024."
                ),
                author=_repeat_offender("rule_breaker_77"),
                subreddit=s,
                score=-8,
                num_comments=1,
                reports=[Report(reason="Personal attack / Insult", count=6)],
                flair=None,
                thread_context=["Thread on alternative medicine regulation"],
            ),
            GroundTruth(
                correct_action="temp_ban",
                correct_rule=2,
                acceptable_actions=["temp_ban", "remove"],
                difficulty_score=0.55,
                explanation=(
                    "Direct insult ('intellectually bankrupt', 'stupidity') by a user with "
                    "3 prior warnings for the same rule. Escalation to temp_ban is warranted."
                ),
            ),
        ),
        # 4. Researcher sharing own paper — legitimate self-promotion exception
        (
            RedditPost(
                post_id="t3_004",
                title="My team's new paper: Quantum error correction via surface codes — open access",
                body=(
                    "Hi r/science, I'm one of the authors on a paper we published in Physical "
                    "Review Letters this week (doi:10.1103/PhysRevLett.132.180401). "
                    "We demonstrate a logical error rate below the surface code threshold "
                    "for the first time using a 49-qubit array. Happy to answer questions. "
                    "Open access link in comments."
                ),
                author=UserHistory(
                    username="quantum_error_correction_phd",
                    account_age_days=1800,
                    karma=15000,
                    prior_warnings=0,
                    prior_removals=0,
                    is_repeat_offender=False,
                    is_approved_contributor=True,
                ),
                subreddit=s,
                score=3400,
                num_comments=280,
                reports=[Report(reason="Self-promotion", count=2)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.6,
                explanation=(
                    "Rule 4 explicitly allows researchers to share their own peer-reviewed work "
                    "when it adds community value. This has a DOI, is open access, and the author "
                    "is engaged with the community. Approve."
                ),
            ),
        ),
        # 5. Heated but substantive scientific debate → approve both
        (
            RedditPost(
                post_id="t3_005",
                title=None,  # type: ignore[arg-type]
                body=(
                    "I fundamentally disagree with your reading of the Hawkins et al. data. "
                    "The confidence intervals overlap significantly — claiming a 'significant "
                    "result' at p=0.048 is precisely the kind of statistical malpractice "
                    "that fuels the replication crisis. This interpretation is wrong."
                ),
                author=_mid_user("stats_hawk"),
                subreddit=s,
                score=45,
                num_comments=12,
                reports=[Report(reason="Aggressive tone", count=1)],
                flair=None,
                thread_context=["Debate about a neuroscience paper's statistical methods"],
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.6,
                explanation=(
                    "Forceful scientific disagreement focused entirely on methodology. "
                    "'This interpretation is wrong' is a scientific claim, not a personal attack. "
                    "Heated debates about methods are the lifeblood of science. Approve."
                ),
            ),
        ),
        # 6. Valid harsh scientific criticism — "this paper is garbage"
        (
            RedditPost(
                post_id="t3_006",
                title=None,  # type: ignore[arg-type]
                body=(
                    "This paper is methodologically garbage. The authors used convenience "
                    "sampling (n=47), no control group, and self-reported outcomes. "
                    "It should never have passed peer review. The journal has a lot to "
                    "answer for publishing this."
                ),
                author=_veteran_author("methodology_critic"),
                subreddit=s,
                score=234,
                num_comments=56,
                reports=[Report(reason="Rude about the paper", count=3)],
                flair=None,
                thread_context=["Discussion of a psychology replication failure"],
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.6,
                explanation=(
                    "'This paper is garbage' targets the work, not the person. "
                    "The criticism is substantiated with specific methodological flaws. "
                    "Harsh but valid scientific critique — approve."
                ),
            ),
        ),
        # 7. Tangential crisis mention → escalate
        (
            RedditPost(
                post_id="t3_007",
                title="I've been reading about serotonin and depression — sometimes I feel like there's no point to any of it",
                body=(
                    "Started reading the Moncrieff et al. paper about the serotonin hypothesis "
                    "of depression. It's interesting research but honestly lately I've been "
                    "feeling really hopeless and like nothing matters. Does the science on "
                    "antidepressants actually give anyone hope?"
                ),
                author=_new_author("struggling_reader_77"),
                subreddit=s,
                score=45,
                num_comments=23,
                reports=[],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="escalate_to_senior_mod",
                correct_rule=None,
                acceptable_actions=["escalate_to_senior_mod", "approve"],
                difficulty_score=0.8,
                explanation=(
                    "Post has legitimate scientific content but contains concerning language "
                    "('no point to any of it', 'nothing matters', 'hopeless'). "
                    "Requires senior mod judgment about mental health resources. Escalate."
                ),
            ),
        ),
        # 8. Absurdist satire — no /s but clearly not serious
        (
            RedditPost(
                post_id="t3_008",
                title="Peer-reviewed proof that the Moon is made of cheese: implications for dairy science",
                body=(
                    "Abstract: We present a rigorous analysis of lunar albedo data using our "
                    "proprietary CheeseSpec™ algorithm. Our findings (p < 0.0001) confirm "
                    "lunar composition of 73% aged cheddar and 27% brie. We call for "
                    "immediate funding for a Moon Cheese Extraction Mission. Submitted to "
                    "the Journal of Irreproducible Results."
                ),
                author=_mid_user("absurdist_science_fan"),
                subreddit=s,
                score=4500,
                num_comments=320,
                reports=[Report(reason="Misinformation", count=1)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.55,
                explanation=(
                    "Clearly absurdist satire — 'Journal of Irreproducible Results' is a known "
                    "parody journal. The community is engaging with it positively. "
                    "No reasonable person would mistake this for genuine misinformation. Approve."
                ),
            ),
        ),
        # 9. New account posting excellent genuine content → approve
        (
            RedditPost(
                post_id="t3_009",
                title="Preprint: Novel mechanism for CRISPR off-target editing identified via long-read sequencing",
                body=(
                    "Hi, first post here. We just uploaded our preprint to bioRxiv "
                    "(doi:10.1101/2024.03.15.585123). We used PacBio HiFi long-read "
                    "sequencing to characterise off-target edits in a way short-read "
                    "sequencing misses. Full dataset is open access on Zenodo. "
                    "Happy to answer questions — this is my first time posting here."
                ),
                author=UserHistory(
                    username="first_time_poster_phd",
                    account_age_days=1,
                    karma=1,
                    prior_warnings=0,
                    prior_removals=0,
                    is_repeat_offender=False,
                    is_approved_contributor=False,
                ),
                subreddit=s,
                score=890,
                num_comments=67,
                reports=[],
                flair="Research",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.65,
                explanation=(
                    "Brand new account but the content is a legitimate bioRxiv preprint with DOI "
                    "and open data. New account age does not justify removing high-quality science. Approve."
                ),
            ),
        ),
        # 10. High-karma established user posting clear misinformation → remove
        (
            RedditPost(
                post_id="t3_010",
                title="The SARS-CoV-2 lab leak hypothesis has been proven — here's the evidence",
                body=(
                    "After reviewing all available data, I can state definitively that "
                    "SARS-CoV-2 was engineered in the Wuhan Institute of Virology. "
                    "The furin cleavage site could not have evolved naturally — this is "
                    "settled science. Anyone saying otherwise is covering it up."
                ),
                author=_veteran_author("popular_science_commenter"),
                subreddit=s,
                score=1200,
                num_comments=450,
                reports=[Report(reason="Misinformation / Presenting speculation as settled", count=8)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="remove",
                correct_rule=3,
                acceptable_actions=["remove"],
                difficulty_score=0.65,
                explanation=(
                    "Lab leak origin is actively debated but not 'settled science'. "
                    "Presenting a contested hypothesis as proven fact violates Rule 3. "
                    "High karma does not grant immunity — apply rules consistently."
                ),
            ),
        ),
        # 11. Wrong flair but post contains important safety information
        (
            RedditPost(
                post_id="t3_011",
                title="Important: popular sunscreen ingredient oxybenzone may disrupt hormone signalling — new data",
                body=(
                    "A preprint (not yet peer reviewed) from researchers at UCSF suggests "
                    "systemic oxybenzone absorption is higher than previously thought. "
                    "This is preliminary data and should not cause alarm, but I wanted to "
                    "flag it for the community. doi:10.1101/2024.04.01.234567"
                ),
                author=_mid_user("health_safety_watcher"),
                subreddit=s,
                score=2300,
                num_comments=180,
                reports=[Report(reason="Should be [Research] not [News] — it's a preprint", count=3)],
                flair="News",
            ),
            GroundTruth(
                correct_action="warn",
                correct_rule=5,
                acceptable_actions=["warn"],
                difficulty_score=0.65,
                explanation=(
                    "Minor flair error (preprint flaired as [News] instead of [Research]). "
                    "Content is accurate and responsibly caveated. Warn to re-flair — do not remove, "
                    "as the underlying health information is valuable."
                ),
            ),
        ),
        # 12. Citing a well-known predatory journal as authoritative source
        (
            RedditPost(
                post_id="t3_012",
                title="Research published in OMICS International confirms electromagnetic hypersensitivity is real",
                body=(
                    "A new study in OMICS International (impact factor 3.4) has confirmed "
                    "that electromagnetic hypersensitivity (EHS) is a diagnosable condition "
                    "affecting 5% of the population. The study used double-blind protocols "
                    "with 200 participants."
                ),
                author=_new_author("ems_health_advocate"),
                subreddit=s,
                score=120,
                num_comments=34,
                reports=[Report(reason="Predatory journal / Low quality source", count=4)],
                flair="Research",
            ),
            GroundTruth(
                correct_action="warn",
                correct_rule=3,
                acceptable_actions=["warn", "remove"],
                difficulty_score=0.7,
                explanation=(
                    "OMICS International is a well-documented predatory publisher. "
                    "EHS has not been substantiated in rigorous blinded trials. "
                    "Warn rather than remove — user may not know about predatory journals."
                ),
            ),
        ),
        # 13. Borderline self-promotion by consistently high-quality contributor
        (
            RedditPost(
                post_id="t3_013",
                title="My Substack: monthly deep-dives into recent neuroscience papers (free)",
                body=(
                    "I've been posting monthly summaries of the top neuroscience papers "
                    "for two years. Each post covers 5-10 papers with lay summaries and "
                    "my methodological commentary. It's completely free. "
                    "Latest issue: substack.com/neuroscimonthly — would love feedback."
                ),
                author=UserHistory(
                    username="neuroscimonthly",
                    account_age_days=900,
                    karma=42000,
                    prior_warnings=0,
                    prior_removals=0,
                    is_repeat_offender=False,
                    is_approved_contributor=True,
                ),
                subreddit=s,
                score=560,
                num_comments=78,
                reports=[Report(reason="Self-promotion", count=2)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="warn",
                correct_rule=4,
                acceptable_actions=["warn", "remove"],
                difficulty_score=0.75,
                explanation=(
                    "Technically self-promotion, but this is a long-standing community contributor "
                    "sharing free content. The borderline nature warrants a gentle warning rather than "
                    "removal — note the 10% self-promotion guideline."
                ),
            ),
        ),
        # 14. Poe's Law — ambiguous satire that could be genuine misinformation
        (
            RedditPost(
                post_id="t3_014",
                title="Why I stopped trusting peer review after 15 years as a scientist",
                body=(
                    "After watching reproducibility failures, p-hacking, and publish-or-perish "
                    "dynamics gut the credibility of my field, I've concluded that peer review "
                    "as currently practiced does more harm than good. I still believe in the "
                    "scientific method — but the institution is broken and we need to say so."
                ),
                author=_veteran_author("disillusioned_researcher"),
                subreddit=s,
                score=3400,
                num_comments=780,
                reports=[Report(reason="Anti-science content", count=4)],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="approve",
                correct_rule=None,
                acceptable_actions=["approve"],
                difficulty_score=0.75,
                explanation=(
                    "Thoughtful critique of a scientific institution from a credentialed researcher — "
                    "not anti-science misinformation. The replication crisis is real and this is "
                    "exactly the kind of substantive discussion r/science should host. Approve."
                ),
            ),
        ),
        # 15. Genuinely ambiguous — requires senior mod judgment
        (
            RedditPost(
                post_id="t3_015",
                title="Sharing internal whistleblower documents about data manipulation at a major pharma company",
                body=(
                    "I have obtained internal emails from a major pharmaceutical company "
                    "showing that clinical trial data for [drug name redacted] was selectively "
                    "reported. I'm posting them here because I believe the public has a right "
                    "to know. Documents linked below."
                ),
                author=UserHistory(
                    username="pharma_whistleblower",
                    account_age_days=2,
                    karma=1,
                    prior_warnings=0,
                    prior_removals=0,
                    is_repeat_offender=False,
                    is_approved_contributor=False,
                ),
                subreddit=s,
                score=890,
                num_comments=240,
                reports=[
                    Report(reason="Unverified claims", count=5),
                    Report(reason="Potential doxxing / legal issues", count=3),
                ],
                flair="Discussion",
            ),
            GroundTruth(
                correct_action="escalate_to_senior_mod",
                correct_rule=None,
                acceptable_actions=["escalate_to_senior_mod", "remove"],
                difficulty_score=0.9,
                explanation=(
                    "Genuinely complex case: potential public interest journalism vs. unverified claims, "
                    "legal liability, and new account. Too consequential for a routine mod decision. "
                    "Escalate to senior mod."
                ),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# ContentGenerator
# ---------------------------------------------------------------------------


class ContentGenerator:
    """Generates deterministic RL episodes from pre-defined scenario templates."""

    def __init__(self) -> None:
        self._task_scenarios = {
            1: _task1_scenarios(),
            2: _task2_scenarios(),
            3: _task3_scenarios(),
        }

    def generate_episode(
        self,
        task_level: int,
        num_posts: int,
        seed: Optional[int] = None,
    ) -> List[Scenario]:
        """
        Return a shuffled list of (RedditPost, GroundTruth) pairs for one episode.

        Args:
            task_level: 1, 2, or 3
            num_posts: number of posts in the episode
            seed: optional random seed for reproducibility
        """
        if task_level not in self._task_scenarios:
            raise ValueError(f"task_level must be 1, 2, or 3; got {task_level}")

        rng = random.Random(seed)
        pool = self._task_scenarios[task_level]

        # Sample with replacement if num_posts > pool size
        if num_posts <= len(pool):
            selected = rng.sample(pool, num_posts)
        else:
            selected = [rng.choice(pool) for _ in range(num_posts)]

        rng.shuffle(selected)
        return selected

    def available_task_levels(self) -> List[int]:
        return list(self._task_scenarios.keys())

    def scenario_count(self, task_level: int) -> int:
        return len(self._task_scenarios[task_level])

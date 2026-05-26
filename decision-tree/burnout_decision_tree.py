"""
REVIVE Burnout Prediction Tool
================================
Core classification module implementing the decision tree logic
from the REVIVE MRes research project (Middlesex University, 2026).

Combines OLBI (Oldenburg Burnout Inventory) psychometric scoring
with biometric inputs to classify burnout risk in care workers.

Author: Chizurum C. Nwachukwu
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# ─── OLBI Item Definitions ────────────────────────────────────────────────────

OLBI_ITEMS = [
    {"id": 1,  "text": "I always find new and interesting aspects in my work.",
     "dimension": "disengagement", "reverse": True},
    {"id": 2,  "text": "There are days when I feel tired before I arrive at work.",
     "dimension": "exhaustion",    "reverse": False},
    {"id": 3,  "text": "After work, I tend to need more time than in the past to relax and feel better.",
     "dimension": "exhaustion",    "reverse": False},
    {"id": 4,  "text": "I can tolerate the pressure of my work very well.",
     "dimension": "exhaustion",    "reverse": True},
    {"id": 5,  "text": "Lately, I tend to think less at work and do my job almost mechanically.",
     "dimension": "disengagement", "reverse": False},
    {"id": 6,  "text": "I find my work to be a positive challenge.",
     "dimension": "disengagement", "reverse": True},
    {"id": 7,  "text": "During my work, I often feel emotionally drained.",
     "dimension": "exhaustion",    "reverse": False},
    {"id": 8,  "text": "Over time, one can become disconnected from this type of work.",
     "dimension": "disengagement", "reverse": False},
    {"id": 9,  "text": "After working, I have enough energy for my leisure activities.",
     "dimension": "exhaustion",    "reverse": True},
    {"id": 10, "text": "Sometimes I feel sickened by my work tasks.",
     "dimension": "disengagement", "reverse": False},
    {"id": 11, "text": "After my work, I usually feel worn out and weary.",
     "dimension": "exhaustion",    "reverse": False},
    {"id": 12, "text": "This is the only type of work that I can imagine myself doing.",
     "dimension": "disengagement", "reverse": True},
    {"id": 13, "text": "Usually, I can manage the amount of my work well.",
     "dimension": "exhaustion",    "reverse": True},
    {"id": 14, "text": "I feel more and more engaged in my work.",
     "dimension": "disengagement", "reverse": True},
    {"id": 15, "text": "When I work, I usually feel energised.",
     "dimension": "exhaustion",    "reverse": True},
]

# Response scale: 1=Strongly Agree, 2=Agree, 3=Not Sure/Disagree, 4=Strongly Disagree
RESPONSE_LABELS = {
    1: "Strongly Agree",
    2: "Agree",
    3: "Not Sure / Disagree",
    4: "Strongly Disagree"
}

# Biometric thresholds (evidence-based, NHS/clinical guidelines)
HEART_RATE_THRESHOLDS = {
    "low":    (0,  60),   # Below resting normal — may indicate fatigue
    "normal": (60, 100),  # Normal resting heart rate
    "high":   (100, 999)  # Elevated — potential stress indicator
}
SLEEP_THRESHOLDS = {
    "insufficient": (0,   7),   # Below NHS recommended minimum
    "adequate":     (7,   9),   # NHS recommended range
    "excess":       (9, 999)    # Over-sleeping (can indicate depression/burnout)
}
STEPS_THRESHOLDS = {
    "low":      (0,     5000),   # Sedentary
    "moderate": (5000,  10000),  # Active
    "high":     (10000, 999999)  # Very active (NHS target)
}


# ─── Classification Rules ─────────────────────────────────────────────────────

CLASSIFICATION_RULES = {
    ("High",   "High"):   {
        "label": "Burned Out",
        "severity": 5,
        "description": (
            "Your scores indicate severe burnout. Both your exhaustion and "
            "disengagement are at critical levels. Immediate action is strongly recommended."
        ),
        "recommendations": [
            "Speak with your manager about reducing your workload as soon as possible.",
            "Contact your GP — burnout at this level can develop into depression or anxiety.",
            "Reach out to the Care Workers' Charity helpline for confidential support.",
            "Prioritise 7–9 hours of sleep every night — sleep is critical for recovery.",
            "Take any available annual leave to allow physical and emotional recuperation.",
        ]
    },
    ("High",   "Medium"): {
        "label": "Drained and Nearly Mentally Detached",
        "severity": 4,
        "description": (
            "You are severely exhausted and beginning to disengage from your work. "
            "Without intervention, this is likely to develop into full burnout."
        ),
        "recommendations": [
            "Talk to a trusted colleague, manager, or GP about how you are feeling.",
            "Aim for at least 7 hours of sleep — your body needs recovery time.",
            "Identify the single biggest stressor in your working week and address it.",
            "Set a firm boundary on working hours where possible — rest is not optional.",
            "Access your employer's Employee Assistance Programme if available.",
        ]
    },
    ("High",   "Low"):    {
        "label": "Committed But Drained",
        "severity": 3,
        "description": (
            "You are highly committed to your work but physically and emotionally exhausted. "
            "Your dedication is clear, but you are running on empty."
        ),
        "recommendations": [
            "Schedule recovery time — you cannot sustain commitment without rest.",
            "Aim for 10,000 steps daily to support physical health and mental resilience.",
            "Book annual leave proactively — even a few days can reset your energy levels.",
            "Consider speaking to your manager about workload distribution.",
            "Use mindfulness or breathing techniques during shift transitions.",
        ]
    },
    ("Medium", "High"):   {
        "label": "Withdrawn But Still Working",
        "severity": 3,
        "description": (
            "You are becoming increasingly detached from your work despite managing "
            "your exhaustion. This emotional withdrawal is an early warning sign."
        ),
        "recommendations": [
            "Reconnect with your purpose — reflect on why you entered care work.",
            "Seek peer support or informal mentoring from a trusted colleague.",
            "Discuss workload and scheduling concerns with your manager.",
            "Engage in activities outside work that restore a sense of meaning.",
            "Monitor your wellbeing weekly — disengagement can escalate quickly.",
        ]
    },
    ("Medium", "Medium"): {
        "label": "Medium Burnout",
        "severity": 3,
        "description": (
            "You are showing moderate signs of both exhaustion and disengagement. "
            "You are at a crossroads — early intervention now can prevent escalation."
        ),
        "recommendations": [
            "Use the NHS Every Mind Matters tool for daily mental health check-ins.",
            "Prioritise sleep and aim for 7–9 hours consistently.",
            "Talk to someone you trust about how work is making you feel.",
            "Take short recovery breaks during your shift where possible.",
            "Consider whether your current schedule is sustainable long-term.",
        ]
    },
    ("Medium", "Low"):    {
        "label": "Moderately Stressed But Committed",
        "severity": 2,
        "description": (
            "You are managing your commitment well but showing signs of moderate stress. "
            "Keep monitoring — and take action before exhaustion increases."
        ),
        "recommendations": [
            "Maintain your current level of engagement — it is a protective factor.",
            "Build rest into your weekly schedule — do not wait until you are exhausted.",
            "Aim for consistent sleep of 7–9 hours to maintain your energy levels.",
            "Use your smartwatch data to spot patterns in your stress and activity.",
        ]
    },
    ("Low",    "High"):   {
        "label": "Restless and Frustrated",
        "severity": 2,
        "description": (
            "You have energy but are disengaged from your role. This may indicate "
            "unmet needs at work — boredom, lack of recognition, or role mismatch."
        ),
        "recommendations": [
            "Speak to your manager about new responsibilities or development opportunities.",
            "Consider whether aspects of your role could be reshaped to better suit you.",
            "Seek out peer connection — social engagement at work reduces disengagement.",
            "Explore training or CPD opportunities in your sector.",
        ]
    },
    ("Low",    "Medium"): {
        "label": "Energised but Withdrawn",
        "severity": 1,
        "description": (
            "You have good energy but are slightly withdrawing from your work. "
            "This is an early signal worth paying attention to."
        ),
        "recommendations": [
            "Reflect on what aspects of your work feel less meaningful than before.",
            "Maintain your physical activity — it is clearly supporting your energy.",
            "Check in with yourself weekly using the OLBI tool.",
            "Talk to a colleague or manager about anything that feels unsatisfying.",
        ]
    },
    ("Low",    "Low"):    {
        "label": "Engaged and Enthusiastic",
        "severity": 0,
        "description": (
            "You show low burnout risk. You are energised and engaged in your work. "
            "Keep monitoring — and continue the habits that are supporting your wellbeing."
        ),
        "recommendations": [
            "Keep up your current sleep and activity habits — they are clearly working.",
            "Share what is working well with colleagues who may be struggling.",
            "Continue using self-assessment tools periodically to stay aware.",
            "Protect your current work-life balance proactively.",
        ]
    },
}


# ─── Classifier ───────────────────────────────────────────────────────────────

@dataclass
class BurnoutResult:
    """Full output from the burnout classification."""
    classification:       str
    severity:             int        # 0 (none) to 5 (severe)
    exhaustion_score:     float      # Raw mean score (1–4)
    disengagement_score:  float      # Raw mean score (1–4)
    exhaustion_level:     str        # Low / Medium / High
    disengagement_level:  str        # Low / Medium / High
    biometric_flags:      List[str]  # Any biometric risk signals
    description:          str
    recommendations:      List[str]


class BurnoutClassifier:
    """
    REVIVE burnout prediction classifier.

    Combines OLBI psychometric scoring with biometric data to classify
    burnout risk in care workers, using a decision tree approach.

    OLBI Scoring:
        - 15 items rated 1 (Strongly Agree) to 4 (Strongly Disagree)
        - Reverse-scored items are inverted before subscale calculation
        - Exhaustion subscale: items 2, 3, 4, 7, 9, 11, 13, 15
        - Disengagement subscale: items 1, 5, 6, 8, 10, 12, 14

    Risk Level Thresholds (mean subscale score):
        Low:    1.0 – 2.0
        Medium: 2.1 – 3.0
        High:   3.1 – 4.0
    """

    # Thresholds for Low / Medium / High classification
    THRESHOLDS = {"Low": (1.0, 2.0), "Medium": (2.01, 3.0), "High": (3.01, 4.0)}

    def __init__(self):
        self.items = OLBI_ITEMS

    def _reverse_score(self, score: int) -> int:
        """Invert a 1–4 Likert score for reverse-coded items."""
        return 5 - score

    def _score_olbi(self, responses: List[int]) -> dict:
        """
        Calculate OLBI subscale scores.

        Args:
            responses: List of 15 integers (1–4), one per OLBI item in order.

        Returns:
            Dictionary with exhaustion_mean, disengagement_mean, and all item scores.
        """
        if len(responses) != 15:
            raise ValueError(f"Expected 15 OLBI responses, got {len(responses)}.")
        if not all(1 <= r <= 4 for r in responses):
            raise ValueError("All OLBI responses must be between 1 and 4.")

        exhaustion_scores    = []
        disengagement_scores = []

        for i, item in enumerate(self.items):
            score = responses[i]
            if item["reverse"]:
                score = self._reverse_score(score)
            if item["dimension"] == "exhaustion":
                exhaustion_scores.append(score)
            else:
                disengagement_scores.append(score)

        return {
            "exhaustion_mean":    round(np.mean(exhaustion_scores),    2),
            "disengagement_mean": round(np.mean(disengagement_scores), 2),
            "exhaustion_items":    exhaustion_scores,
            "disengagement_items": disengagement_scores,
        }

    def _classify_level(self, mean_score: float) -> str:
        """Map a mean subscale score to Low / Medium / High."""
        if mean_score <= 2.0:
            return "Low"
        elif mean_score <= 3.0:
            return "Medium"
        else:
            return "High"

    def _assess_biometrics(
        self,
        heart_rate_bpm: Optional[int],
        daily_steps:    Optional[int],
        sleep_hours:    Optional[float]
    ) -> List[str]:
        """
        Identify biometric risk flags based on clinical thresholds.

        Returns a list of plain-English risk signals.
        """
        flags = []

        if heart_rate_bpm is not None:
            if heart_rate_bpm > 100:
                flags.append(
                    f"Elevated resting heart rate ({heart_rate_bpm} BPM) — "
                    "above the normal range of 60–100 BPM. This may indicate chronic stress."
                )
            elif heart_rate_bpm < 60:
                flags.append(
                    f"Low resting heart rate ({heart_rate_bpm} BPM) — "
                    "may indicate fatigue or overtraining."
                )

        if daily_steps is not None:
            if daily_steps < 5000:
                flags.append(
                    f"Low daily step count ({daily_steps:,} steps) — "
                    "below the 5,000 step threshold for active health. "
                    "Physical activity is a key protective factor against burnout."
                )

        if sleep_hours is not None:
            if sleep_hours < 7:
                flags.append(
                    f"Insufficient sleep ({sleep_hours} hours) — "
                    "below the NHS recommended 7–9 hours. "
                    "Sleep deprivation significantly increases burnout risk."
                )
            elif sleep_hours > 9:
                flags.append(
                    f"Excessive sleep ({sleep_hours} hours) — "
                    "sleeping over 9 hours can be an indicator of depression or burnout."
                )

        return flags

    def predict(
        self,
        olbi_scores:    List[int],
        heart_rate_bpm: Optional[int]   = None,
        daily_steps:    Optional[int]   = None,
        sleep_hours:    Optional[float] = None
    ) -> BurnoutResult:
        """
        Classify burnout risk from OLBI scores and optional biometric data.

        Args:
            olbi_scores:    List of 15 integers (1–4) for each OLBI item in order.
            heart_rate_bpm: Resting heart rate in BPM (optional).
            daily_steps:    Daily step count from wearable (optional).
            sleep_hours:    Hours slept last night (optional).

        Returns:
            BurnoutResult dataclass with full classification and recommendations.

        Example:
            >>> clf = BurnoutClassifier()
            >>> result = clf.predict(
            ...     olbi_scores=[3,4,3,2,4,2,4,3,2,4,3,2,3,2,3],
            ...     heart_rate_bpm=95,
            ...     daily_steps=4200,
            ...     sleep_hours=5.5
            ... )
            >>> print(result.classification)
        """
        scored      = self._score_olbi(olbi_scores)
        ex_level    = self._classify_level(scored["exhaustion_mean"])
        dis_level   = self._classify_level(scored["disengagement_mean"])
        rule        = CLASSIFICATION_RULES[(ex_level, dis_level)]
        bio_flags   = self._assess_biometrics(heart_rate_bpm, daily_steps, sleep_hours)

        return BurnoutResult(
            classification      = rule["label"],
            severity            = rule["severity"],
            exhaustion_score    = scored["exhaustion_mean"],
            disengagement_score = scored["disengagement_mean"],
            exhaustion_level    = ex_level,
            disengagement_level = dis_level,
            biometric_flags     = bio_flags,
            description         = rule["description"],
            recommendations     = rule["recommendations"],
        )

    def print_report(self, result: BurnoutResult) -> None:
        """Print a formatted burnout report to console."""
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  REVIVE BURNOUT ASSESSMENT REPORT")
        print(f"{sep}")
        print(f"  Classification : {result.classification}")
        print(f"  Severity       : {'█' * result.severity}{'░' * (5 - result.severity)} ({result.severity}/5)")
        print(f"{sep}")
        print(f"  OLBI Subscale Scores")
        print(f"    Exhaustion    : {result.exhaustion_score:.2f} / 4.00  [{result.exhaustion_level}]")
        print(f"    Disengagement : {result.disengagement_score:.2f} / 4.00  [{result.disengagement_level}]")

        if result.biometric_flags:
            print(f"\n  ⚠  Biometric Flags")
            for flag in result.biometric_flags:
                print(f"    • {flag}")

        print(f"\n  Assessment")
        print(f"    {result.description}")
        print(f"\n  Recommendations")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"    {i}. {rec}")
        print(f"{sep}\n")


# ─── CLI Usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clf = BurnoutClassifier()

    # Example: High exhaustion, medium disengagement, poor biometrics
    example_scores = [2, 4, 4, 2, 4, 2, 4, 3, 1, 3, 4, 2, 2, 2, 2]

    result = clf.predict(
        olbi_scores    = example_scores,
        heart_rate_bpm = 105,
        daily_steps    = 3800,
        sleep_hours    = 5.5
    )

    clf.print_report(result)

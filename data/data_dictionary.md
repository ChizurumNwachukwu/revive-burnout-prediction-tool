# Data Dictionary — REVIVE Burnout Prediction Tool

## OLBI Variables (q1–q15)

| Variable | Type | Range | Description |
|---|---|---|---|
| q1 | int | 1–4 | I always find new and interesting aspects in my work. *(Disengagement, reverse-scored)* |
| q2 | int | 1–4 | There are days when I feel tired before I arrive at work. *(Exhaustion)* |
| q3 | int | 1–4 | After work, I tend to need more time than in the past to relax and feel better. *(Exhaustion)* |
| q4 | int | 1–4 | I can tolerate the pressure of my work very well. *(Exhaustion, reverse-scored)* |
| q5 | int | 1–4 | Lately, I tend to think less at work and do my job almost mechanically. *(Disengagement)* |
| q6 | int | 1–4 | I find my work to be a positive challenge. *(Disengagement, reverse-scored)* |
| q7 | int | 1–4 | During my work, I often feel emotionally drained. *(Exhaustion)* |
| q8 | int | 1–4 | Over time, one can become disconnected from this type of work. *(Disengagement)* |
| q9 | int | 1–4 | After working, I have enough energy for my leisure activities. *(Exhaustion, reverse-scored)* |
| q10 | int | 1–4 | Sometimes I feel sickened by my work tasks. *(Disengagement)* |
| q11 | int | 1–4 | After my work, I usually feel worn out and weary. *(Exhaustion)* |
| q12 | int | 1–4 | This is the only type of work that I can imagine myself doing. *(Disengagement, reverse-scored)* |
| q13 | int | 1–4 | Usually, I can manage the amount of my work well. *(Exhaustion, reverse-scored)* |
| q14 | int | 1–4 | I feel more and more engaged in my work. *(Disengagement, reverse-scored)* |
| q15 | int | 1–4 | When I work, I usually feel energised. *(Exhaustion, reverse-scored)* |

**Response scale:** 1 = Strongly Agree | 2 = Agree | 3 = Not Sure / Disagree | 4 = Strongly Disagree

---

## Derived OLBI Variables

| Variable | Type | Range | Description |
|---|---|---|---|
| exhaustion_score | float | 1.0–4.0 | Mean of exhaustion items (after reverse-scoring) |
| disengagement_score | float | 1.0–4.0 | Mean of disengagement items (after reverse-scoring) |
| exhaustion_level | str | Low/Medium/High | Classified from exhaustion_score thresholds |
| disengagement_level | str | Low/Medium/High | Classified from disengagement_score thresholds |

**Classification thresholds:**
- Low: mean ≤ 2.0
- Medium: 2.01 ≤ mean ≤ 3.0
- High: mean > 3.0

---

## Biometric Variables

| Variable | Type | Range | Source | Clinical Reference |
|---|---|---|---|---|
| heart_rate_bpm | int | 40–200 | Smartwatch (manual entry) | Normal resting: 60–100 BPM (NHS) |
| daily_steps | int | 0–30,000 | Smartwatch (manual entry) | Target: 10,000 steps/day (NHS) |
| sleep_hours | float | 0–12 | Smartwatch (manual entry) | Recommended: 7–9 hours (NHS) |

---

## Target Variable

| Variable | Type | Classes | Description |
|---|---|---|---|
| burnout_label | str | 9 categories | Final burnout risk classification |

**Classes:**
1. Burned Out *(Exhaustion: High, Disengagement: High)*
2. Drained and Nearly Mentally Detached *(Exhaustion: High, Disengagement: Medium)*
3. Committed But Drained *(Exhaustion: High, Disengagement: Low)*
4. Withdrawn But Still Working *(Exhaustion: Medium, Disengagement: High)*
5. Medium Burnout *(Exhaustion: Medium, Disengagement: Medium)*
6. Moderately Stressed But Committed *(Exhaustion: Medium, Disengagement: Low)*
7. Restless and Frustrated *(Exhaustion: Low, Disengagement: High)*
8. Energised but Withdrawn *(Exhaustion: Low, Disengagement: Medium)*
9. Engaged and Enthusiastic *(Exhaustion: Low, Disengagement: Low)*

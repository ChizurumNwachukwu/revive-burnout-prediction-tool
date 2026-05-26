# REVIVE — Burnout Prediction Tool for Care Workers

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-purple.svg)

> **MRes Computer Science Research Project | Middlesex University | 2026**  
> Researcher: Chizurum C. Nwachukwu | Supervisors: Prof. Juan Carlos Augusto, Dr Mark Springett

---

## Overview

**REVIVE** is a web-based burnout prediction tool built for UK care home and home care assistants. It combines the validated **Oldenburg Burnout Inventory (OLBI)** with biometric data from wearable devices to classify users by burnout risk and deliver personalised, evidence-based intervention recommendations.

The tool is live at **[revive-tool.com](http://revive-tool.com)** and has been evaluated by 65+ real-world participants across the UK.

This repository contains:
- The Python implementation of the burnout classification decision tree
- OLBI scoring logic and data dictionary
- Research methodology and stakeholder findings
- System architecture documentation

---

## The Problem

The UK adult social care sector is in crisis:

- **131,000 vacancies** as of March 2023 (Foster, 2024)
- **81% of care workers** report that their job has negatively impacted their mental health (Albert, 2019)
- **Nearly half** of all care workers in England are paid below the living wage (Bawden, 2024)
- Traditional burnout tools (e.g. Maslach Burnout Inventory) are expensive, static, and inaccessible in resource-limited settings

Existing support is **reactive**. REVIVE is designed to be **preventative**.

---

## The Solution

A decision tree classification system that:

1. Administers **15 OLBI questions** measuring two core dimensions: Exhaustion and Disengagement
2. Captures **biometric inputs** from wearable devices: resting heart rate (BPM), daily step count, sleep duration
3. Calculates weighted subscale scores and classifies users into **9 burnout risk categories**
4. Delivers **personalised, actionable recommendations** tailored to each risk profile

### Burnout Risk Categories

| Classification | Exhaustion | Disengagement |
|---|---|---|
| Burned Out | High | High |
| Engaged and Enthusiastic | Low | Low |
| Restless and Frustrated | Low | High |
| Committed But Drained | High | Low |
| Drained and Nearly Mentally Detached | High | Medium |
| Withdrawn But Still Working | Medium | High |
| Moderately Stressed But Committed | Medium | Low |
| Medium Burnout | Medium | Medium |
| Energised but Withdrawn | Low | Medium |

---

## Repository Structure

```
revive-burnout-prediction-tool/
│
├── README.md
├── analysis/
│   ├── burnout_classifier.ipynb       # Full Python decision tree implementation
│   └── olbi_scoring_analysis.ipynb    # OLBI subscale scoring and visualisation
├── decision-tree/
│   ├── burnout_decision_tree.py       # Core classification logic (importable module)
│   └── decision_tree_logic.md         # Plain-English walkthrough of the decision tree
├── data/
│   ├── data_dictionary.md             # All variables, types, and descriptions
│   ├── olbi_scoring_guide.md          # OLBI item scoring and subscale calculation
│   └── synthetic_sample_data.csv      # Synthetic data for demonstration
├── research/
│   ├── methodology_summary.md         # UCIEDP methodology and research design
│   └── stakeholder_findings.md        # Key themes from interviews and surveys
└── docs/
    └── system_architecture.md         # System design and WordPress-Yonyx integration
```

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Run the Classifier

```bash
git clone https://github.com/yourusername/revive-burnout-prediction-tool.git
cd revive-burnout-prediction-tool
jupyter notebook analysis/burnout_classifier.ipynb
```

### Or Import the Module Directly

```python
from decision_tree.burnout_decision_tree import BurnoutClassifier

classifier = BurnoutClassifier()

result = classifier.predict(
    olbi_scores=[3, 4, 3, 2, 4, 2, 4, 3, 2, 4, 3, 2, 3, 2, 3],
    heart_rate_bpm=95,
    daily_steps=4200,
    sleep_hours=5.5
)

print(result['classification'])      # "Drained and Nearly Mentally Detached"
print(result['exhaustion_level'])    # "High"
print(result['disengagement_level']) # "Medium"
print(result['recommendations'])     # List of personalised recommendations
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Decision Tree Logic | Python / scikit-learn / Yonyx |
| Web Application | WordPress (revive-tool.com) |
| User Authentication | WordPress user management |
| Survey Engine | Yonyx interactive decision tree platform |
| API Integration | WordPress–Yonyx token-based redirect |
| Research Analysis | Python, thematic analysis (Braun & Clarke, 2006) |

---

## Research Methodology

This project applied the **User-Centred Intelligent Environments Development Process (UCIEDP)** (Augusto et al., 2018) — a co-design framework that prioritises continuous stakeholder involvement over rigid development cycles.

### Why UCIEDP over Agile or Waterfall?

- Waterfall delays user feedback until testing — too late for meaningful iteration
- Agile can prioritise short-term sprint output over long-term user value
- UCIEDP centres stakeholder needs at every stage, including ethical and usability dimensions via the **eFRIENDS framework**

### Why Decision Trees over Random Forest or SVM?

Random Forest and Support Vector Machines were considered but rejected because:

> *"Random forests and support vector machine models are often treated as black box models because their prediction mechanisms are not transparent"* (Siemers & Bajorath, 2023)

Decision trees offer **explainability** — every path from root to leaf node is traceable and understandable to non-technical users, which is critical in a mental health context where trust in the tool directly affects whether users act on recommendations.

### Why OLBI over MBI?

| Feature | MBI | OLBI |
|---|---|---|
| Cost | Licensed (paid) | Free / public domain |
| Response bias | Negatively-phrased items only | Mixed positive/negative items |
| Dimensions | 3 (Exhaustion, Depersonalisation, Personal Accomplishment) | 2 (Exhaustion, Disengagement) |
| ML integration | Complex, proprietary | Straightforward, open |
| Care sector fit | General | Validated for care settings |

---

## Primary Research Summary

### Participants

| Group | Method | n |
|---|---|---|
| Care Assistants | Initial Survey | 36 |
| Care Assistants | Initial Interviews (smartwatch users) | 5 |
| Managers | Survey | 4 |
| All | Oldenburg Burnout Inventory | 23 |
| All | Follow-up app testing & survey | 20 |

Participants were located across London, Oxford, Plymouth, Leeds, Manchester, Harlow, Reading, Scotland, Brentford, and Hatfield.

### Three Key Themes (Thematic Analysis)

**Theme 1 — Physical and Emotional Exhaustion**
Care workers described the constant need to be mentally present and physically ready. One participant stated: *"a care assistant cannot afford to be absent-minded due to the nature of the job."* Managers corroborated this, rating increased errors/near-misses and absenteeism as the highest-severity burnout indicators.

**Theme 2 — Structural Barriers Leading to Burnout**
Systemic issues — insufficient travel time between visits, 6-day working weeks, and poor work-life balance — were identified as primary burnout drivers. One participant described a visit ending at 7:30pm while the next was scheduled to begin at 7:30pm.

**Theme 3 — Desire for Digital Tools**
All 5 interviewed participants used smartwatches daily. 90% of surveyed care assistants were at least slightly comfortable with digital tools — significantly higher than managers assumed. 70% wanted anonymous wellbeing tracking.

### Evaluation Results (Live Tool Testing, n=20)

- 8/20 rated the tool **very easy** to use
- 10/20 found recommendations **mostly accurate**
- 12/20 said the tool **captured all elements** of burnout
- Key trust driver: emotional validation — when the tool accurately reflected how participants felt, they were more willing to act on recommendations

---

## Ethical Considerations

The tool was designed in accordance with the **eFRIENDS ethical framework** (Jones et al., 2015):

- **Non-maleficence**: Results are not stored; only the user sees their burnout score
- **Beneficence**: Recommendations are evidence-based and tailored to individual outcomes
- **User-centred**: Tool design was driven by stakeholder needs at every stage
- **Privacy**: No physiological or personal data is retained after the session

---

## Key Findings & Limitations

**Findings:**
- Decision tree classification accurately mirrored participants' lived experiences of burnout
- A significant perception gap exists between management and frontline staff in understanding burnout drivers
- Biometric integration (heart rate, sleep, steps) adds meaningful signal to self-reported OLBI scores

**Current Limitations:**
- Biometric data requires manual input — no live smartwatch API integration yet
- Sample sizes reflect purposive criterion sampling; not designed for statistical generalisation
- Single iteration due to one-year MRes timeframe

**Future Development:**
- Direct API integration with Apple Health, Fitbit, and Garmin
- Demographic variables (age, years of experience, employment type, visa type)
- Perceived Stress Scale module
- Travel time quantification as burnout input
- Automated longitudinal monitoring with trend detection

---

## References

Key references underpinning this work:

- Demerouti, E. & Bakker, A.B. (2008). The Oldenburg Burnout Inventory. *Handbook of stress and burnout in health care*
- Van Zyl-Cillié et al. (2024). A machine learning model to predict burnout risk in nursing staff. *BMC Health Service Research*
- Barac et al. (2024). Wearable Technologies for Detecting Burnout in Health Care Professionals. *J Med Internet Res*
- Augusto et al. (2018). The user-centred intelligent environments development process. *Universal Access in the Information Society*
- Lieslehto et al. (2022). A machine learning approach to predict resilience and sickness absence in healthcare. *Scientific Reports*

Full reference list available in the thesis document.

---

## About the Researcher

**Chizurum Nwachukwu** is a Data Scientist with an MRes in Computer Science (Middlesex University, 2026), an MSc in Data Science and Business Analytics (University of Plymouth), and a BA in Business Administration. His background spans healthcare, research, and technology.

- LinkedIn: [linkedin.com/in/chizurum-data](https://www.linkedin.com/in/chizurum-data)
- Portfolio: [datascienceportfol.io/chizurum](https://www.datascienceportfol.io/chizurum)
- Email: hello.chizurum@gmail.com

---

*This research was conducted at Middlesex University under the supervision of Prof. Juan Carlos Augusto and Dr Mark Springett.*

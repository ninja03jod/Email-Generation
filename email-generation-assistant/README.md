# Email Generation Assistant
### AI Engineer Candidate Assessment — Full Solution

> Built with **Pydantic AI** · **Groq API** · **LLM-as-Judge Evaluation**  
> Covers all three sections of the assessment: Email Generation · Custom Metrics · Model Comparison

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Was Built](#2-what-was-built)
3. [Repository Structure](#3-repository-structure)
4. [Assessment Requirements vs Implementation](#4-assessment-requirements-vs-implementation)
5. [Section 1 — Email Generation Assistant](#5-section-1--email-generation-assistant)
6. [Section 2 — Evaluation System & Custom Metrics](#6-section-2--evaluation-system--custom-metrics)
7. [Section 3 — Model Comparison & Analysis](#7-section-3--model-comparison--analysis)
8. [Output Files Explained](#8-output-files-explained)
9. [Setup & Installation](#9-setup--installation)
10. [How to Run](#10-how-to-run)
11. [Input Examples](#11-input-examples)
12. [Technical Architecture](#12-technical-architecture)

---

## 1. Project Overview

This project is a complete solution for the **AI Engineer Candidate Assessment: Email Generation Assistant**. It fulfils every stated deliverable:

- A working email generation prototype that takes `intent`, `key facts`, and `tone` as inputs and produces a professional email using an LLM.
- An advanced prompting strategy (three techniques combined).
- A custom evaluation system with three purpose-built metrics.
- A full model comparison across two Groq models with a written analysis report.

**Models used:**
| Label | Model | Role |
|---|---|---|
| Model A | `llama-3.3-70b-versatile` | Primary generator + LLM judge |
| Model B | `llama-3.1-8b-instant` | Comparison generator |

---

## 2. What Was Built

### File 1 — `email_agent_2.py` (Section 1)
The core email generation assistant. You set your inputs directly in `main()` and run the file. It generates a structured, validated email using **Pydantic AI** with three stacked advanced prompting techniques.

### File 2 — `email_evaluator_2.py` (Sections 2 & 3)
The complete evaluation and comparison system. It runs all 10 test scenarios against both models, scores each output across three custom metrics, and generates all output reports automatically.

---

## 3. Repository Structure

```
email-generation-assistant/
│
├── email_agent_2.py           ← Section 1: Email generation assistant
├── email_evaluator_2.py       ← Section 2 & 3: Evaluation + model comparison
│
├── requirements.txt           ← All Python dependencies
│
├── outputs/                   ← Generated automatically when you run the files
│   ├── email_output.json          ← Single email result (from email_agent_2.py)
│   ├── evaluation_report.csv      ← Scores for all 10 scenarios × 2 models
│   ├── evaluation_results.json    ← Full data: metric definitions + all scores + analysis
│   └── comparative_analysis.txt   ← Section 3 written report (one-page summary)
│
└── README.md
```

> **Note:** The `outputs/` files are included in this repo as sample results from a completed run. Re-running the scripts will overwrite them with fresh results.

---

## 4. Assessment Requirements vs Implementation

| Requirement | Status | Where |
|---|---|---|
| Takes `intent`, `key facts`, and `tone` as inputs | ✅ | `EmailInput` Pydantic model in `email_agent_2.py` |
| Produces a professional email using an LLM | ✅ | `generate_email()` function, Groq API |
| Advanced prompt engineering technique documented | ✅ | 3 techniques — see Section 5 below |
| 10 unique test scenarios | ✅ | `TEST_SCENARIOS` list in `email_evaluator_2.py` |
| Human reference email for each scenario | ✅ | `human_reference` field in every scenario dict |
| 3 custom evaluation metrics defined and implemented | ✅ | Metrics 1, 2, 3 in `email_evaluator_2.py` |
| Evaluation script outputs CSV + JSON | ✅ | `evaluation_report.csv` + `evaluation_results.json` |
| Metric definitions in the output file | ✅ | `metric_definitions` key in `evaluation_results.json` |
| Raw scores for all 10 scenarios per metric | ✅ | `evaluation_report.csv` rows + `evaluation_results.json` |
| Overall average score per model | ✅ | Summary rows at bottom of CSV + `comparative_analysis.txt` |
| Two models evaluated on same 10 scenarios | ✅ | Model A (`llama-3.3-70b-versatile`) + Model B (`llama-3.1-8b-instant`) |
| Written comparative analysis (3 questions answered) | ✅ | `comparative_analysis.txt` |
| Prompt template documented | ✅ | `SYSTEM_INSTRUCTIONS` variable in `email_agent_2.py` |

---

## 5. Section 1 — Email Generation Assistant

### File: `email_agent_2.py`

#### Inputs
The assistant accepts five inputs, set directly in `main()`:

```python
intent         = "Your email goal"        # What the email is about
key_facts      = ["Fact 1", "Fact 2"]     # Facts that MUST appear in the email
tone           = "formal"                 # formal | casual | urgent | empathetic | persuasive
recipient_name = "Mr. Sharma"             # Optional — used in greeting
sender_name    = "Priya Mehta"            # Optional — used in sign-off
```

#### Output — `EmailOutput` (Pydantic validated)
Every field is type-validated before being returned:

| Field | Type | Description |
|---|---|---|
| `subject_line` | `str` | Specific, compelling email subject |
| `email_body` | `str` | Complete ready-to-send email |
| `tone_applied` | `ToneType` | Tone the model actually used |
| `facts_woven_in` | `list[str]` | Echo of each fact incorporated |
| `writing_rationale` | `str` | Chain-of-Thought explanation of writing decisions |

#### Advanced Prompting — Three Techniques Combined

**Technique 1: Role-Persona Prompting**
The system prompt assigns the model a specific professional identity:
> *"You are Elena, a Senior Corporate Communication Specialist with 15 years of experience crafting high-impact emails for Fortune 500 executives, startups, and government organisations."*

This anchors vocabulary quality, tone calibration, and structural consistency.

**Technique 2: Dynamic Few-Shot Example Selection**
Rather than static, hardcoded examples, the system maintains a **5-example bank** (one per tone). At runtime, the `select_few_shot_examples()` function picks the **2 most relevant examples** based on the requested tone and injects them via Pydantic AI's `@agent.instructions` decorator — different examples for every request.

**Technique 3: Chain-of-Thought (Internal Reasoning)**
The system prompt includes a 7-step internal reasoning process the model must follow before writing:

```
Step 1 → Identify the PRIMARY goal of this email in one sentence
Step 2 → Decide the emotional register and vocabulary for the tone
Step 3 → Plan where each key fact will appear (opening / body / closing)
Step 4 → Draft a subject line reflecting intent and tone
Step 5 → Write the full email, verifying each fact is included
Step 6 → Self-review: tone consistent? All facts present?
Step 7 → Record key writing decisions in `writing_rationale`
```

The reasoning output is captured in the structured `writing_rationale` field.

#### Pydantic AI Integration
```python
groq_model = GroqModel(
    "llama-3.3-70b-versatile",
    provider=GroqProvider(api_key=GROQ_API_KEY),
)

agent: Agent[EmailAgentDeps, EmailOutput] = Agent(
    groq_model,
    output_type=EmailOutput,    # Pydantic validates every response
    instructions=SYSTEM_INSTRUCTIONS,
    retries=3,
)
```

Pydantic AI handles structured output enforcement, retry logic on validation failure, and dependency injection of the few-shot examples via `RunContext`.

---

## 6. Section 2 — Evaluation System & Custom Metrics

### File: `email_evaluator_2.py`

#### Test Data (Section 2A) — 10 Scenarios

10 unique scenarios covering all 5 tone types and diverse real-world email contexts:

| ID | Intent | Tone |
|---|---|---|
| S01 | Sales discovery call follow-up | formal |
| S02 | Missed project deadline apology | empathetic |
| S03 | Live payment gateway failure alert (variant 1) | urgent |
| S04 | Live payment gateway failure alert (variant 2) | urgent |
| S05 | AI-powered HR analytics tool pitch | persuasive |
| S06 | End-of-sprint team celebration invite | casual |
| S07 | One-on-one meeting request for promotion | formal |
| S08 | New client onboarding welcome | casual |
| S08 | Quarterly investor business update | formal |
| S09 | Data breach stakeholder notification | urgent |
| S10 | Strategic partnership proposal | persuasive |

Each scenario includes a hand-written **human reference email** as the gold standard for qualitative comparison.

---

#### Custom Metrics (Section 2B)

---

**Metric 1 — Fact Recall Score**
- **Range:** 0–100
- **Method:** Automated keyword-overlap (no API call)
- **Definition:** For each key fact, meaningful tokens (dates, numbers, proper nouns) are extracted after removing common stop-words. A fact is considered "recalled" if **≥60% of its tokens** appear in the generated email body.
- **Formula:** `Score = (facts_recalled / total_facts) × 100`
- **Why:** Fact inclusion is the most objective and verifiable measure of whether the model followed the user's explicit instructions. A model that omits required facts has fundamentally failed the task regardless of writing quality.

---

**Metric 2 — Tone Accuracy Score**
- **Range:** 0–100 (normalised from a 1–10 LLM judge rating)
- **Method:** LLM-as-Judge using `llama-3.3-70b-versatile` via the Groq SDK
- **Definition:** A second LLM call evaluates the email against the requested tone on a 1–10 scale using standardised definitions for each tone type. The judge must return structured JSON with a score and one-sentence reasoning. Raw score is normalised to 0–100.
- **Tone definitions provided to the judge:**
  - `formal` — Professional vocabulary, no contractions, executive-level register
  - `casual` — Friendly, conversational, contractions used naturally
  - `urgent` — Direct, action-first, imperative sentences, no preamble
  - `empathetic` — Acknowledges feelings first, personal ownership language
  - `persuasive` — Benefit-led, compelling hooks, clear call-to-action
- **Why:** Tone is inherently subjective and context-dependent — keyword or sentiment tools cannot distinguish "formal" from "persuasive." An LLM judge with standardised criteria is the most reliable approach. Judge reasoning is preserved in the output for transparency.

---

**Metric 3 — Structural Completeness Score**
- **Range:** 0–100 (20 points per component)
- **Method:** Automated regex pattern detection (no API call)
- **Definition:** Checks whether the email contains all 5 structural components of a professional email. Each present component = 20 points.
  1. **Greeting** — Personalised opening line (Dear / Hi / Hey / Team etc.)
  2. **Opening Hook** — Purpose established in the first paragraph
  3. **Body Content** — Substantive multi-paragraph content (paragraph break detected)
  4. **Call-to-Action** — Clear next step or request near the close
  5. **Sign-off** — Closing salutation and sender name
- **Why:** Fact recall and tone measure intent; structural completeness measures execution. A well-intentioned email with missing structure (no greeting, no CTA, no sign-off) is incomplete for professional use.

---

#### Evaluation Report Outputs (Section 2C)

Running `email_evaluator_2.py` produces three files:

| File | Contents |
|---|---|
| `evaluation_report.csv` | One row per scenario per model. Columns: all 3 metric scores, avg score, judge reasoning, token counts, generation success/failure |
| `evaluation_results.json` | Full structured output — includes `metric_definitions` block, all result data, and the full comparative analysis |
| `comparative_analysis.txt` | The Section 3 written one-page analysis report |

---

## 7. Section 3 — Model Comparison & Analysis

Both models (`llama-3.3-70b-versatile` and `llama-3.1-8b-instant`) were evaluated against the **identical 10 scenarios** using the **identical 3 metrics**.

#### Summary of Results

| Metric | Model A (`llama-3.3-70b-versatile`) | Model B (`llama-3.1-8b-instant`) |
|---|---|---|
| Fact Recall (M1) | 100.0 on all 10 scenarios | 2 scenario failures (exceeded retries) |
| Tone Accuracy (M2) | Consistently 80–90/100 | More variance; dropped to 70/100 on urgent |
| Structural Completeness (M3) | 60–100/100 | More inconsistent; dropped to 40/100 on urgent |
| **Overall Average** | **~86.4/100** | **Lower due to 2 failed generations** |

> See `evaluation_report.csv` for all raw per-scenario scores and `comparative_analysis.txt` for the full written analysis answering all three required questions.

**Key findings:**
- Model B (`llama-3.1-8b-instant`) failed to produce valid structured output on 2 out of 10 scenarios (S05 — persuasive pitch; S08 — formal investor update), exceeding the 3-retry limit. This is a critical production risk.
- Model B showed weaker structural completeness on urgent emails, scoring 40/100 on S03 where Model A scored 100/100.
- Model B was significantly slower on several scenarios (up to 62 seconds on S06 vs 1.2 seconds for Model A).
- **Recommendation: `llama-3.3-70b-versatile` (Model A) for production** — higher reliability, better structural output, consistent fact recall.

---

## 8. Output Files Explained

| File | Generated by | When |
|---|---|---|
| `email_output.json` | `email_agent_2.py` | Every time you run the generator |
| `evaluation_report.csv` | `email_evaluator_2.py` | After full evaluation run |
| `evaluation_results.json` | `email_evaluator_2.py` | After full evaluation run |
| `comparative_analysis.txt` | `email_evaluator_2.py` | After full evaluation run |

### `evaluation_report.csv` — Column Reference

| Column | Description |
|---|---|
| `scenario_id` | S01 through S10 |
| `intent_short` | Truncated intent description |
| `tone` | Requested tone for this scenario |
| `model` | Model name used |
| `generation_ok` | `True` / `False` — whether the email was generated successfully |
| `metric1_fact_recall` | Score 0–100 |
| `metric2_tone_accuracy` | Score 0–100 (LLM judge, normalised) |
| `metric3_structural_completeness` | Score 0–100 |
| `avg_score` | Average of the 3 metrics |
| `m1_facts_recalled` | Number of facts detected |
| `m1_total_facts` | Total facts in the scenario |
| `m2_raw_score_1_10` | Raw LLM judge score before normalisation |
| `m2_judge_reasoning` | One-sentence explanation from the judge |
| `m3_components_found` | Number of structural components present (out of 5) |
| `subject_line` | Generated email subject |
| `duration_ms` | API response time in milliseconds |
| `total_tokens` | Total tokens consumed |
| `error` | Error message if `generation_ok` is `False` |

### `evaluation_results.json` — Top-Level Structure

```json
{
  "generated_at": "ISO timestamp",
  "metric_definitions": {
    "metric1_fact_recall": { "name", "range", "method", "logic" },
    "metric2_tone_accuracy": { ... },
    "metric3_structural_completeness": { ... }
  },
  "results": [ ... ],
  "comparative_analysis": { ... }
}
```

---

## 9. Setup & Installation

### Prerequisites
- Python 3.10 or higher
- A [Groq API key](https://console.groq.com/keys) (free tier is sufficient)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the minimal set manually:

```bash
pip install "pydantic-ai-slim[groq]" groq pydantic
```

### Set Your API Key

Open `email_agent_2.py` and replace the key on **line 27**:

```python
GROQ_API_KEY = "gsk_your_actual_key_here"
```

The same key is imported automatically into `email_evaluator_2.py` — you only need to set it once.

---

## 10. How to Run

### Run the Email Generator (Section 1)

```bash
python email_agent_2.py
```

Edit the `YOUR INPUTS` block inside `main()` before running:

```python
intent         = "Your email intent here"
key_facts      = ["Fact 1", "Fact 2", "Fact 3"]
tone           = "formal"          # formal | casual | urgent | empathetic | persuasive
recipient_name = "Recipient Name"  # or None
sender_name    = "Your Name"       # or None
```

**Output:** Prints the generated email to the terminal and saves `email_output.json`.

---

### Run the Full Evaluation (Sections 2 & 3)

```bash
python email_evaluator_2.py
```
**Output files created:**
- `evaluation_report.csv` — all scores in tabular format
- `evaluation_results.json` — full data with metric definitions
- `comparative_analysis.txt` — the Section 3 written analysis

Both files must be in the **same directory** when running the evaluator, as `email_evaluator_2.py` imports directly from `email_agent_2.py`.

---

## 11. Input Examples

Eight ready-to-use example inputs are provided as **commented blocks** in `email_agent_2.py`. Copy any block into the `YOUR INPUTS` section in `main()`:

| Example | Intent | Tone |
|---|---|---|
| A | Sales follow-up after discovery call | formal |
| B | Apology for missed project deadline | empathetic |
| C | Production outage / payment gateway alert | urgent |
| D | AI-powered HR analytics tool pitch | persuasive |
| E | End-of-sprint team celebration invite | casual |
| F | One-on-one meeting request for promotion | formal |
| G | New client onboarding welcome | casual |
| H | Quarterly investor business update | formal |

---

## 12. Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     email_agent_2.py                            │
│                                                                 │
│  EmailInput (Pydantic)         EmailOutput (Pydantic)           │
│  ├─ intent                     ├─ subject_line                  │
│  ├─ key_facts                  ├─ email_body                    │
│  ├─ tone                       ├─ tone_applied                  │
│  ├─ recipient_name             ├─ facts_woven_in                │
│  └─ sender_name                └─ writing_rationale             │
│                                                                 │
│  EmailAgentDeps (DI container)                                  │
│  ├─ few_shot_examples (bank)                                    │
│  ├─ selected_examples (2 dynamic)                              │
│  └─ input_data                                                  │
│                                                                 │
│  Pydantic AI Agent                                              │
│  ├─ GroqModel("llama-3.3-70b-versatile")                       │
│  ├─ GroqProvider(api_key)                                       │
│  ├─ output_type=EmailOutput                                     │
│  ├─ SYSTEM_INSTRUCTIONS (Role-Persona + CoT)                    │
│  └─ @agent.instructions → dynamic few-shot injection           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   email_evaluator_2.py                          │
│                                                                 │
│  10 Test Scenarios (with human reference emails)                │
│                                                                 │
│  For each scenario × 2 models:                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ generate_email() → EmailOutput                          │    │
│  │     ↓                                                   │    │
│  │ Metric 1: Fact Recall       (keyword overlap, no API)   │    │
│  │ Metric 2: Tone Accuracy     (LLM-as-Judge, Groq SDK)    │    │
│  │ Metric 3: Structural Score  (regex, no API)             │    │
│  │     ↓                                                   │    │
│  │ avg_score = mean(M1, M2, M3)                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Reports:                                                       │
│  ├─ evaluation_report.csv                                       │
│  ├─ evaluation_results.json (+ metric_definitions)             │
│  └─ comparative_analysis.txt                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Libraries

| Library | Version | Purpose |
|---|---|---|
| `pydantic-ai` | ≥0.0.54 | Agent framework, structured output, dependency injection |
| `pydantic-ai-slim[groq]` | ≥0.0.54 | Slim install with Groq provider included |
| `groq` | ≥0.18.0 | Groq Python SDK — used directly for LLM-as-Judge calls |
| `pydantic` | ≥2.7.0 | Data validation for input/output models |

---

## License

This project was built as part of the AI Engineer Candidate Assessment. All code is original work.

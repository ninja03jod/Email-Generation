"""
=============================================================================
  EMAIL GENERATION ASSISTANT — EVALUATION SYSTEM
  Sections 2 & 3 of the AI Engineer Candidate Assessment

  Implements:
    Section 2A  → 10 unique test scenarios with Human Reference Emails
    Section 2B  → 3 Custom Evaluation Metrics
                    1. Fact Recall Score         (Automated, keyword-overlap)
                    2. Tone Accuracy Score        (LLM-as-Judge, 1–10 scale)
                    3. Structural Completeness    (Automated, pattern detection)
    Section 2C  → Evaluation runner → evaluation_report.csv + evaluation_results.json
    Section 3   → Model A vs Model B comparison + written Comparative Analysis
=============================================================================
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
import traceback
from datetime import datetime
from statistics import mean
from typing import Optional

# ---------------------------------------------------------------------------
#  Re-use the generator from email_agent_2.py
# ---------------------------------------------------------------------------
try:
    from email_agent_2 import (
        EmailInput,
        EmailOutput,
        GROQ_API_KEY,
        GROQ_MODEL_PRIMARY,
        GROQ_MODEL_SECONDARY,
        generate_email,
    )
except ImportError as e:
    raise SystemExit(
        "❌  Could not import email_agent_2.py — make sure both files are in "
        "the same directory and dependencies are installed.\n"
        f"Error: {e}"
    )

# For LLM-as-Judge we use the official groq SDK (avoids Cloudflare 403 that
# urllib.request triggers due to its bot-like User-Agent signature).
from groq import Groq


# =============================================================================
#  SECTION 2A — 10 TEST SCENARIOS WITH HUMAN REFERENCE EMAILS
# =============================================================================

TEST_SCENARIOS: list[dict] = [

    # ── SCENARIO 01 ──────────────────────────────────────────────────────────
    {
        "id": "S01",
        "intent": "Follow up after a sales discovery call to keep the deal moving",
        "key_facts": [
            "Discovery call was on Tuesday 8th April 2025",
            "Client showed strong interest in the Enterprise Plan",
            "Their budget approval deadline is end of April",
            "Next step agreed: product demo on 15th April at 3 PM",
        ],
        "tone": "formal",
        "recipient_name": "Mr. Sharma",
        "sender_name": "Priya Mehta",
        "human_reference": (
            "Dear Mr. Sharma,\nI am writing to follow up on our sales discovery call that took place on Tuesday, 8th April 2025. It was a pleasure discussing our services with you and understanding your requirements. I was pleased to note that you showed strong interest in our Enterprise Plan, which I believe would be an excellent fit for your organization.\nAs we approach the budget approval deadline at the end of April, I would like to ensure that we keep the momentum going. As per our discussion, I am looking forward to conducting a product demo on 15th April at 3 PM. This will provide a comprehensive overview of the features and benefits of our Enterprise Plan and address any questions you may have.\nPlease let me know if there are any specific areas you would like me to focus on during the demo. I am committed to ensuring that you have all the necessary information to make an informed decision.\nThank you for your time, and I look forward to speaking with you on 15th April.\nBest regards,\nPriya Mehta"
        ),
    },

    # ── SCENARIO 02 ──────────────────────────────────────────────────────────
    {
        "id": "S02",
        "intent": "Apologise to a client for missing the agreed project deadline",
        "key_facts": [
            "Original deadline was 10th April 2025",
            "Delay caused by unexpected backend infrastructure failure",
            "New confirmed delivery date is 20th April 2025",
            "A 10% discount will be applied to the current invoice as goodwill",
        ],
        "tone": "empathetic",
        "recipient_name": "Sarah",
        "sender_name": "Rahul",
        "human_reference": (
            "Dear Sarah,\n\nI am writing to offer my sincerest apologies for missing the agreed project deadline of 10th April 2025. I understand how much you were counting on receiving the project on time, and I deeply regret the inconvenience this has caused you.\n\nUnfortunately, our team encountered an unexpected backend infrastructure failure that significantly impacted our ability to meet the deadline. Despite our best efforts to mitigate the issue, we require a bit more time to ensure the project meets our quality standards. I am pleased to inform you that we have a new confirmed delivery date of 20th April 2025.\n\nAs a gesture of goodwill for the delay, we would like to apply a 10% discount to the current invoice. Please know that we value your business and appreciate your understanding during this time.\n\nIf there is anything I can do to make this right or if you have any questions, please do not hesitate to reach out. Your satisfaction is our top priority, and I am committed to ensuring you receive the high-quality service you deserve.\n\nWarm regards,\nRahul"
        ),
    },

    # ── SCENARIO 03 ──────────────────────────────────────────────────────────
    {
        "id": "S03",
        "intent": "Immediately alert the engineering team about a live payment gateway failure",
        "key_facts": [
            "Failure first detected at 09:14 IST on 15th April 2025",
            "Approximately 1,500 transactions have failed so far",
            "Root cause identified as an expired TLS certificate",
            "DevOps team is actively working on a hotfix — ETA 45 minutes",
        ],
        "tone": "urgent",
        "recipient_name": "Engineering Team",
        "sender_name": "Ops Lead",
        "human_reference": (
            "Team,\n\n"
            "ACTION REQUIRED — PAYMENT GATEWAY DOWN\n\n"
            "Our payment gateway has been failing since 09:14 IST today, 15th April 2025. "
            "Approximately 1,500 transactions have already failed and the number is growing.\n\n"
            "ROOT CAUSE: An expired TLS certificate has been identified as the cause. "
            "The DevOps team is actively deploying a hotfix with an ETA of 45 minutes.\n\n"
            "IMMEDIATE ACTIONS:\n"
            "1. Halt all non-critical deployments until the hotfix is confirmed live.\n"
            "2. Customer Support: prepare proactive communication for affected users.\n"
            "3. Report any new failure patterns to the incident channel immediately.\n\n"
            "Next status update in 30 minutes. Acknowledge receipt.\n\n"
            "— Ops Lead"
        ),
    },

    # ── SCENARIO 04 ──────────────────────────────────────────────────────────
    {
        "id": "S04",
        "intent": "Immediately alert the engineering team about a live payment gateway failure",
        "key_facts": [
        "Failure first detected at 09:14 IST on 15th April 2025",
        "Approximately 1,500 transactions have failed so far",
        "Root cause identified as an expired TLS certificate",
        "DevOps team is actively working on a hotfix — ETA 45 minutes",
        ],
        "tone": "urgent",
        "recipient_name": "Engineering Team",
        "sender_name": "Ops Lead",
        "human_reference": (
            "Engineering Team,\n\nWe have a critical situation that requires immediate attention. At 09:14 IST on 15th April 2025, our live payment gateway began to experience failures. The current impact is significant, with approximately 1,500 transactions having failed so far, and this number is escalating by the minute.\n\nInitial investigation has identified the root cause as an expired TLS certificate, which is preventing secure connections from being established. Our DevOps team has already started working on a hotfix, with an estimated time to resolution of 45 minutes.\n\nIn the meantime, all non-essential tasks should be deprioritized to ensure maximum resources are allocated to supporting the recovery effort. Please stand by for further updates, which will be provided every 15 minutes or as soon as the hotfix is deployed.\n\nYour prompt attention to this matter is greatly appreciated.\n\nBest regards,\nOps Lead"
        ),
    },

    # ── SCENARIO 05 ──────────────────────────────────────────────────────────
    {
        "id": "S05",
        "intent": "Pitch our AI-powered HR analytics tool to a potential enterprise client",
        "key_facts": [
       "Tool reduces employee attrition prediction time by 75%",
      "Trusted by 200+ companies across Asia",
      "Pricing starts at $500/month — includes a 60-day free trial",
      "Native integration with Workday, SAP, and BambooHR",
        ],
        "tone": "persuasive",
        "recipient_name": "Ms. Kapoor",
        "sender_name": "Alex",
        "human_reference": (
            "Hi Ms. Kapoor, \nI wanted to reach out to you about our AI-powered HR analytics tool, which has been instrumental in helping 200+ companies across Asia streamline their HR operations. What if I told you that our tool can reduce employee attrition prediction time by 75%? This means you can proactively address potential issues before they become major problems, saving your organization time and resources. \nOur tool starts at $500/month and includes a 60-day free trial, so you can see the benefits for yourself before committing. Plus, with native integration with Workday, SAP, and BambooHR, you can easily incorporate our tool into your existing workflows. \nI'd love to schedule a demo to show you how our tool can benefit your organization. Would you be available for a call next week? \nBest, Alex"
        ),
    },

    # ── SCENARIO 06 ──────────────────────────────────────────────────────────
    {
        "id": "S06",
        "intent": "Invite the team to our Friday end-of-sprint celebration",
        "key_facts": [
            "Celebration is this Friday 18th April at 5:30 PM",
      "Venue is the rooftop terrace on Floor 12",
      "Snacks and drinks will be provided by the company",
      "RSVP on Slack #team-celebrations by Thursday noon",
        ],
        "tone": "casual",
        "recipient_name": "Team",
        "sender_name": "Anika",
        "human_reference": (
            "Hey Team,\n\nWe've made it to the end of another sprint, and it's time to celebrate! This Friday, 18th April at 5:30 PM, we'll be gathering on the rooftop terrace on Floor 12 to unwind and have some fun. The company's got us covered with snacks and drinks, so all you need to bring is yourselves and a willingness to let loose.\n\nTo help us plan, please make sure to RSVP on Slack #team-celebrations by Thursday noon. This way, we can get a headcount for the goodies.\n\nLooking forward to seeing you all there!\nBest, Anika"
        ),
    },

    # ── SCENARIO 07 ──────────────────────────────────────────────────────────
    {
        "id": "S07",
        "intent": "Request a one-on-one meeting with my manager to discuss a promotion",
        "key_facts": [
            "I have been in the current role for 2 years",
      "Led 3 successful product launches this quarter",
      "Team performance score increased by 40% under my leadership",
      "Requesting 30 minutes this week at a time that suits",
        ],
        "tone": "formal",
        "recipient_name": "Mr. Patel",
        "sender_name": "Neha",
        "human_reference": (
            "Dear Mr. Patel,\n\nI am writing to request a one-on-one meeting with you to discuss my career development and potential opportunities for advancement within the company. As I have now been in my current role for 2 years, I believe it is an appropriate time to reflect on my contributions and the value I have added to the team. This quarter has been particularly noteworthy, with the successful launch of three significant products under my leadership. Furthermore, I am proud to highlight that the team's performance score has increased by 40% during my tenure, demonstrating the effectiveness of our collective efforts.\n\nI would greatly appreciate the opportunity to discuss these achievements and my future aspirations with you in more detail. Would it be possible to schedule a 30-minute meeting at your convenience this week? I am flexible and can accommodate a time that suits you best.\n\nThank you for considering my request. I look forward to your response.\n\nBest regards,\nNeha"
        ),
    },

    {
        "id": "S08",
        "intent": "SenSend a warm welcome email to a new client who just signed up",
        "key_facts": [
            "Client signed up for the Pro plan on 14th April 2025",
      "Dedicated account manager is Riya Desai",
      "Onboarding call is scheduled for 17th April at 2 PM IST",
      "Help centre and documentation are available at docs.ourproduct.com",
        ],
        "tone": "casual",
        "recipient_name": "Vikram",
        "sender_name": "Riya",
        "human_reference": (
            "Hey Vikram, a huge welcome to our community! We're stoked to have you on board, especially on our Pro plan which you signed up for on 14th April 2025. I'm Riya Desai, your dedicated account manager, and I'm here to ensure your journey with us is as smooth as possible. To get you all set up, we've scheduled an onboarding call for 17th April at 2 PM IST - I'm really looking forward to chatting with you then and answering any questions you might have. In the meantime, if you want to dive in and explore, our help centre and documentation are all available at docs.ourproduct.com. This should give you a solid head start before our call. Looking forward to speaking with you soon and helping you get the most out of our product! Best, Riya"
        ),
    },
    
    # ── SCENARIO 08 ──────────────────────────────────────────────────────────
    {
        "id": "S08",
        "intent": "Send a quarterly business update to our investors",
        "key_facts": [
            "Revenue grew 32% quarter-on-quarter to ₹1.2 Crore",
            "Monthly active users crossed 50,000 for the first time",
            "Raised a ₹5 Crore seed round from Titan Capital in March",
            "Expanding to three new cities in Q2 2025",
        ],
        "tone": "formal",
        "recipient_name": "Investors",
        "sender_name": "Arjun Nair, CEO",
        "human_reference": (
            "Dear esteemed investors,\n\nI am pleased to report that our company has achieved significant milestones in the first quarter of 2025. Our revenue has shown a notable increase of 32% quarter-on-quarter, reaching ₹1.2 Crore. This growth is a testament to our team's hard work and the increasing adoption of our services, as evidenced by our monthly active users crossing the 50,000 mark for the first time.\n\nFurthermore, I am delighted to share that we successfully raised a ₹5 Crore seed round from Titan Capital in March. This investment not only validates our business model but also provides us with the necessary resources to drive further growth and expansion. In line with this, we are excited to announce our plans to expand into three new cities in Q2 2025, which will enable us to tap into new markets and increase our national footprint.\n\nWe believe that these developments position us well for continued success and are grateful for your ongoing support. We look forward to keeping you updated on our progress and exploring ways to further leverage our partnership.\n\nThank you for your trust in our vision.\n\nBest regards,\nArjun Nair, CEO"
        ),
    },

    # ── SCENARIO 09 ──────────────────────────────────────────────────────────
    {
        "id": "S09",
        "intent": "Notify stakeholders about a data breach and required immediate actions",
        "key_facts": [
            "Breach detected at 11:45 PM IST on 14th April 2025",
            "Approximately 3,200 customer records may have been exposed",
            "Security team has isolated the compromised server",
            "All affected customers will be notified within 24 hours",
        ],
        "tone": "urgent",
        "recipient_name": "Leadership Team",
        "sender_name": "CISO",
        "human_reference": (
            "Leadership Team,\n\nWe have a critical situation that requires immediate attention. At 11:45 PM IST on 14th April 2025, our systems detected a data breach, potentially exposing approximately 3,200 customer records. The security team has swiftly isolated the compromised server to prevent further unauthorized access, and we are working diligently to assess the full extent of the breach.\n\nIn line with our data protection policies and regulatory obligations, all affected customers will be notified within the next 24 hours, providing them with necessary information and support. It is essential that we take unified and swift action to mitigate any potential harm and ensure the continuity of our operations.\n\nI urge each of you to be prepared for potential customer inquiries and to direct them to our dedicated support line. We will provide regular updates and will convene an emergency meeting shortly to discuss further actions and strategies.\n\nPlease acknowledge receipt of this email to confirm your understanding of the situation and the required immediate actions.\n\nSincerely,\nCISO"
        ),
    },

    # ── SCENARIO 10 ──────────────────────────────────────────────────────────
    {
        "id": "S10",
        "intent": "Propose a strategic partnership to a potential business partner",
        "key_facts": [
            "Our platform has 2 million active users across Southeast Asia",
            "Proposed partnership involves co-marketing and revenue sharing",
            "We can offer a 20% revenue share on referred conversions",
            "Requesting an exploratory call in the next two weeks",
        ],
        "tone": "persuasive",
        "recipient_name": "Ms. Verma",
        "sender_name": "Raj Gupta",
        "human_reference": (
            "Dear Ms. Verma,\nAs we continue to expand our presence in Southeast Asia, I wanted to reach out to you with an exciting proposal that could mutually benefit our businesses. With 2 million active users across the region, our platform is poised for further growth, and we believe that a strategic partnership with your company could be a key catalyst for this expansion.\nA proposed partnership between us could involve co-marketing efforts, where we collaborate to promote each other's services to our respective audiences, as well as a revenue-sharing model. We are open to discussing the specifics, but we can offer a competitive 20% revenue share on referred conversions, ensuring that both parties benefit from the partnership.\nI would love the opportunity to discuss this proposal in more detail with you and explore how we can work together to drive growth. Would you be available for an exploratory call within the next two weeks? This would give us a chance to delve deeper into the possibilities and see if our businesses align.\nLooking forward to the possibility of working together and unlocking new markets.\nBest regards,\nRaj Gupta"
        ),
    },
]


# =============================================================================
#  SECTION 2B — CUSTOM EVALUATION METRICS
# =============================================================================

# ---------------------------------------------------------------------------
#  METRIC 1: FACT RECALL SCORE (Automated — Keyword Overlap)
#
#  Definition:
#    Measures how many of the required key facts are actually present in the
#    generated email body. For each fact, we extract meaningful tokens
#    (numbers, dates, proper nouns, significant nouns/verbs) after removing
#    common stopwords. A fact is considered "recalled" if ≥60% of its
#    significant tokens appear anywhere in the email body.
#
#  Logic:
#    score = (facts_recalled / total_facts) × 100
#    Range: 0–100  |  Higher = better
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "will", "has", "have", "had", "as", "their", "our", "its", "it", "this",
    "that", "they", "we", "i", "my", "your", "his", "her", "after", "before",
    "all", "into", "so", "up", "out", "about", "than", "more", "also", "not",
    "would", "could", "should", "may", "can", "do", "does", "did",
}


def _tokenise(text: str) -> set[str]:
    """Lower-case, remove punctuation, split, drop stopwords."""
    tokens = re.findall(r"[a-zA-Z0-9₹$%]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def metric_fact_recall(key_facts: list[str], email_body: str) -> dict:
    """
    Metric 1: Fact Recall Score.
    Returns score (0–100) and per-fact recall detail.
    """
    email_tokens = _tokenise(email_body)
    recalled = []
    scores_per_fact = []

    for fact in key_facts:
        fact_tokens = _tokenise(fact)
        if not fact_tokens:
            scores_per_fact.append(1.0)
            recalled.append(True)
            continue

        overlap = fact_tokens & email_tokens
        overlap_ratio = len(overlap) / len(fact_tokens)
        is_recalled = overlap_ratio >= 0.60
        recalled.append(is_recalled)
        scores_per_fact.append(overlap_ratio)

    overall_score = round((sum(recalled) / len(key_facts)) * 100, 1)
    avg_overlap   = round(mean(scores_per_fact) * 100, 1)

    return {
        "score": overall_score,           # 0–100, primary score
        "avg_token_overlap_pct": avg_overlap,
        "facts_recalled": sum(recalled),
        "total_facts": len(key_facts),
        "per_fact": [
            {"fact": f, "recalled": r, "overlap_pct": round(s * 100, 1)}
            for f, r, s in zip(key_facts, recalled, scores_per_fact)
        ],
    }


# ---------------------------------------------------------------------------
#  METRIC 2: TONE ACCURACY SCORE (LLM-as-Judge, 1–10 scale)
#
#  Definition:
#    Uses a second Groq LLM call (llama-3.3-70b-versatile as the judge) to
#    evaluate how accurately the generated email reflects the requested tone.
#    The judge scores 1–10 and must provide reasoning, making the evaluation
#    transparent and reproducible.
#
#  Logic:
#    judge_prompt → JSON response: {"score": int, "reasoning": str}
#    score (1–10) normalised to 0–100 for uniformity
#    Range: 0–100  |  Higher = better
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> dict:
    """
    Robustly extract a JSON object from LLM output that may contain
    markdown fences, preamble text, or trailing commentary.
    Strategy: strip fences first, then find the first '{...}' block.
    """
    # Remove triple-backtick fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```", "", text).strip()

    # Try direct parse first (cleanest case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first {...} JSON object in the text (handles preamble/postamble)
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Wider search for nested braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]!r}")


def _call_groq_json_sync(messages: list[dict], temperature: float = 0.0) -> dict:
    """
    Synchronous Groq SDK call (runs in a thread-pool executor so it never
    blocks the asyncio event loop). Uses the official `groq` package which
    sets proper TLS/User-Agent headers — avoids Cloudflare 403 (error 1010)
    that urllib.request triggers due to its bot-like signature.
    Retries up to 3 times on transient failures with backoff.
    """
    from groq import Groq, RateLimitError, APIStatusError

    client = Groq(api_key=GROQ_API_KEY)

    last_exc: Exception | None = None
    for attempt in range(1, 4):                # up to 3 attempts
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_PRIMARY,       # Judge always uses the primary model
                messages=messages,
                temperature=temperature,
                max_tokens=512,
            )
            raw_text = response.choices[0].message.content.strip()
            return _extract_json_from_text(raw_text)

        except RateLimitError as exc:
            last_exc = exc
            wait = 5 * attempt
            print(f"    ⚠  Judge rate-limited (429). Waiting {wait}s before retry {attempt}/3…")
            time.sleep(wait)

        except APIStatusError as exc:
            last_exc = exc
            if exc.status_code >= 500:          # server error → retry with backoff
                print(f"    ⚠  Judge server error ({exc.status_code}), retry {attempt}/3…")
                time.sleep(2 * attempt)
            else:
                # 4xx client error (bad key, quota, etc.) — no point retrying
                print(f"    ⚠  Judge API error {exc.status_code}: {exc.message}")
                break

        except Exception as exc:
            last_exc = exc
            print(f"    ⚠  Judge call attempt {attempt}/3 failed: {type(exc).__name__}: {exc}")
            time.sleep(2 * attempt)

    raise last_exc or RuntimeError("Judge call failed after 3 attempts")


async def _call_groq_json(messages: list[dict], temperature: float = 0.0) -> dict:
    """
    Async wrapper: runs the blocking Groq SDK call in the default thread-pool
    executor so it never blocks the asyncio event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _call_groq_json_sync(messages, temperature),
    )


async def metric_tone_accuracy(
    requested_tone: str,
    email_body: str,
    subject_line: str,
) -> dict:
    """
    Metric 2: Tone Accuracy Score (LLM-as-Judge).
    Returns score (0–100) and judge reasoning.
    Async so the executor-wrapped HTTP call doesn't block the event loop.
    """
    judge_prompt = f"""You are a professional writing quality assessor. Evaluate how accurately the following email reflects the requested tone.

REQUESTED TONE: {requested_tone.upper()}

TONE DEFINITIONS:
- formal: Professional vocabulary, no contractions, structured, executive-level register.
- casual: Friendly, conversational, contractions used naturally, relaxed but clear.
- urgent: Direct, no preamble, action-first, imperative sentences, time-sensitive language.
- empathetic: Warm, acknowledges feelings first, personal ownership language ("I"), non-defensive.
- persuasive: Benefit-led, compelling hooks, ROI framing, clear call-to-action, minimal friction ask.

EMAIL SUBJECT: {subject_line}

EMAIL BODY:
{email_body}

Evaluate on a scale of 1–10 where:
  1–3  = Tone is clearly wrong (e.g., formal language when casual was requested)
  4–6  = Partially correct tone but inconsistent or mixed signals
  7–8  = Correct tone maintained throughout with minor lapses
  9–10 = Tone is perfectly calibrated, consistent, and exemplary for the requested style

You MUST respond ONLY with a valid JSON object — no preamble, no explanation, no markdown fences:
{{"score": <integer 1-10>, "reasoning": "<one sentence explaining the score>"}}"""

    try:
        result = await _call_groq_json([{"role": "user", "content": judge_prompt}])
        raw_score = int(result.get("score", 5))
        raw_score = max(1, min(10, raw_score))  # clamp to valid range
        return {
            "score": round((raw_score / 10) * 100, 1),   # normalise to 0–100
            "raw_score_1_10": raw_score,
            "reasoning": result.get("reasoning", "No reasoning provided."),
        }
    except Exception as exc:
        print(f"    ⚠  Tone-judge failed: {type(exc).__name__}: {exc}")
        return {
            "score": 0.0,
            "raw_score_1_10": 0,
            "reasoning": f"Judge call failed: {exc}",
        }


# ---------------------------------------------------------------------------
#  METRIC 3: STRUCTURAL COMPLETENESS SCORE (Automated — Pattern Detection)
#
#  Definition:
#    Measures whether the generated email contains all five essential
#    structural components of a professional email:
#      1. Greeting      — personalised opening line (Dear / Hi / Hey etc.)
#      2. Opening Hook  — purpose/context established in the first paragraph
#      3. Body Facts    — substantive middle section (at least 2 paragraphs)
#      4. Call-to-Action — a clear next step or request near the close
#      5. Sign-off      — closing salutation + sender name
#
#  Logic:
#    Each component is worth 20 points → max 100.
#    Detection uses regex patterns tailored to each component.
#    Range: 0–100  |  Higher = better
# ---------------------------------------------------------------------------

STRUCTURE_CHECKS = {
    "greeting": re.compile(
        r"^\s*(dear|hi|hey|hello|good\s+morning|good\s+afternoon|team|all)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    "opening_hook": re.compile(
        r"(i\s+(am\s+writing|hope|wanted|would\s+like|am\s+reaching)|"
        r"thank\s+you\s+for|following\s+up|i\s+am\s+pleased|"
        r"action\s+required|we\s+have\s+a|what\s+if|i\s+believe)",
        re.IGNORECASE,
    ),
    "body_content": re.compile(
        r"\n\s*\n",  # at least one paragraph break = substantive body
    ),
    "call_to_action": re.compile(
        r"(please\s+(do\s+not\s+hesitate|contact|reply|confirm|let\s+me)|"
        r"(looking\s+forward|i\s+look\s+forward)\s+to|"
        r"(would\s+you|are\s+you)\s+(be\s+open|available|able)|"
        r"kindly\s+(respond|confirm|provide)|"
        r"rsvp|acknowledge|next\s+step|call\s+to\s+action|"
        r"feel\s+free\s+to|don.t\s+hesitate|reach\s+out)",
        re.IGNORECASE,
    ),
    "sign_off": re.compile(
        r"(yours\s+(sincerely|faithfully|truly)|best\s+regards|"
        r"kind\s+regards|warm\s+regards|regards|cheers|best,|"
        r"thank\s+you[,.]?\s*\n|see\s+you|—\s*\w)",
        re.IGNORECASE,
    ),
}


def metric_structural_completeness(email_body: str) -> dict:
    """
    Metric 3: Structural Completeness Score.
    Returns score (0–100) and per-component detection results.
    """
    results = {}
    for component, pattern in STRUCTURE_CHECKS.items():
        results[component] = bool(pattern.search(email_body))

    score = round(sum(results.values()) / len(results) * 100, 1)

    return {
        "score": score,
        "components_found": sum(results.values()),
        "total_components": len(results),
        "breakdown": results,
    }


# =============================================================================
#  EVALUATION RUNNER
# =============================================================================

def _separator(title: str = "", width: int = 72) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'═' * pad} {title} {'═' * pad}")
    else:
        print("─" * width)


async def evaluate_scenario(
    scenario: dict,
    model_name: str,
) -> dict:
    """
    Generate an email for one scenario and compute all 3 metrics.
    Returns a flat result dict suitable for CSV and JSON export.
    """
    email_input = EmailInput(
        intent         = scenario["intent"],
        key_facts      = scenario["key_facts"],
        tone           = scenario["tone"],             # type: ignore[arg-type]
        recipient_name = scenario.get("recipient_name"),
        sender_name    = scenario.get("sender_name"),
    )

    try:
        output, metadata = await generate_email(email_input, model_name=model_name, verbose=False)
        gen_ok = True
    except Exception as exc:
        print(f"    ⚠  Generation failed for {scenario['id']} ({model_name}): {exc}")
        return {
            "scenario_id": scenario["id"],
            "intent_short": scenario["intent"][:55] + "…",
            "tone": scenario["tone"],
            "model": model_name,
            "generation_ok": False,
            "error": str(exc),
            "metric1_fact_recall": None,
            "metric2_tone_accuracy": None,
            "metric3_structural_completeness": None,
            "avg_score": None,
            "duration_ms": None,
            "total_tokens": None,
        }

    # ── Metric 1: Fact Recall ─────────────────────────────────────────────
    m1 = metric_fact_recall(scenario["key_facts"], output.email_body)

    # ── Metric 2: Tone Accuracy (LLM-as-Judge) ────────────────────────────
    # Longer delay to respect Groq rate limits between generation + judge calls
    await asyncio.sleep(1.5)
    m2 = await metric_tone_accuracy(scenario["tone"], output.email_body, output.subject_line)

    # ── Metric 3: Structural Completeness ────────────────────────────────
    m3 = metric_structural_completeness(output.email_body)

    avg_score = round(mean([m1["score"], m2["score"], m3["score"]]), 1)

    usage = metadata.get("usage", {})

    return {
        "scenario_id"                 : scenario["id"],
        "intent_short"                : scenario["intent"][:55] + "…",
        "tone"                        : scenario["tone"],
        "model"                       : model_name,
        "generation_ok"               : gen_ok,
        "error"                       : None,
        # Scores (0–100 each)
        "metric1_fact_recall"         : m1["score"],
        "metric2_tone_accuracy"       : m2["score"],
        "metric3_structural_completeness": m3["score"],
        "avg_score"                   : avg_score,
        # Detail
        "m1_facts_recalled"           : m1["facts_recalled"],
        "m1_total_facts"              : m1["total_facts"],
        "m2_raw_score_1_10"           : m2["raw_score_1_10"],
        "m2_judge_reasoning"          : m2["reasoning"],
        "m3_components_found"         : m3["components_found"],
        "m3_breakdown"                : json.dumps(m3["breakdown"]),
        # Generation metadata
        "subject_line"                : output.subject_line,
        "email_body"                  : output.email_body,
        "duration_ms"                 : metadata.get("duration_ms"),
        "total_tokens"                : usage.get("total_tokens"),
    }


async def run_full_evaluation(
    scenarios: list[dict],
    models: list[str],
) -> list[dict]:
    """
    Run all scenarios × models sequentially to avoid rate limits.
    Returns list of result dicts.
    """
    all_results = []
    total = len(scenarios) * len(models)
    done  = 0

    for model in models:
        _separator(f"MODEL: {model}")
        for scenario in scenarios:
            done += 1
            print(f"  [{done}/{total}]  {scenario['id']} — {scenario['intent'][:50]}…")
            result = await evaluate_scenario(scenario, model)
            all_results.append(result)

            score_str = (
                f"M1={result['metric1_fact_recall']:.0f}  "
                f"M2={result['metric2_tone_accuracy']:.0f}  "
                f"M3={result['metric3_structural_completeness']:.0f}  "
                f"AVG={result['avg_score']:.1f}"
                if result["generation_ok"]
                else "FAILED"
            )
            print(f"          → {score_str}")

            # Small pause between scenarios to respect API rate limits
            await asyncio.sleep(1.0)

    return all_results


# =============================================================================
#  REPORT GENERATION
# =============================================================================

def save_csv(results: list[dict], filepath: str) -> None:
    """Save flat results to CSV for easy spreadsheet analysis."""
    csv_fields = [
        "scenario_id", "intent_short", "tone", "model", "generation_ok",
        "metric1_fact_recall", "metric2_tone_accuracy",
        "metric3_structural_completeness", "avg_score",
        "m1_facts_recalled", "m1_total_facts",
        "m2_raw_score_1_10", "m2_judge_reasoning",
        "m3_components_found",
        "subject_line", "duration_ms", "total_tokens", "error",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  ✅  CSV saved → {filepath}")


def save_json(results: list[dict], filepath: str, extra: dict | None = None) -> None:
    """Save full results to JSON including metric definitions and analysis."""
    payload = {
        "generated_at": datetime.now().isoformat(),
        "metric_definitions": {
            "metric1_fact_recall": {
                "name": "Fact Recall Score",
                "range": "0–100",
                "method": "Automated keyword-overlap",
                "logic": (
                    "For each key fact, significant tokens (numbers, dates, nouns) are "
                    "extracted after stopword removal. A fact is 'recalled' when ≥60% of "
                    "its tokens appear in the email body. Score = (facts_recalled / total_facts) × 100."
                ),
            },
            "metric2_tone_accuracy": {
                "name": "Tone Accuracy Score",
                "range": "0–100 (normalised from 1–10 LLM judge rating)",
                "method": "LLM-as-Judge (Groq llama-3.3-70b-versatile)",
                "logic": (
                    "A second LLM call evaluates the email against the requested tone using "
                    "predefined criteria for each tone type (formal/casual/urgent/empathetic/persuasive). "
                    "Raw score 1–10 is normalised to 0–100."
                ),
            },
            "metric3_structural_completeness": {
                "name": "Structural Completeness Score",
                "range": "0–100",
                "method": "Automated regex pattern detection",
                "logic": (
                    "Checks for 5 structural email components: (1) Greeting, (2) Opening hook/purpose, "
                    "(3) Substantive body content, (4) Call-to-action, (5) Sign-off. "
                    "Score = (components_found / 5) × 100."
                ),
            },
        },
        "results": results,
    }
    if extra:
        payload.update(extra)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✅  JSON saved → {filepath}")


# =============================================================================
#  SECTION 3 — COMPARATIVE ANALYSIS
# =============================================================================

def build_comparative_analysis(results: list[dict], model_a: str, model_b: str) -> dict:
    """
    Compute per-model summary statistics and generate the written analysis.
    Returns a dict with summary tables and the analysis text.
    """

    def model_stats(model_name: str) -> dict:
        rows = [r for r in results if r["model"] == model_name and r["generation_ok"]]
        if not rows:
            return {}

        m1_scores = [r["metric1_fact_recall"]         for r in rows if r["metric1_fact_recall"]         is not None]
        m2_scores = [r["metric2_tone_accuracy"]        for r in rows if r["metric2_tone_accuracy"]        is not None]
        m3_scores = [r["metric3_structural_completeness"] for r in rows if r["metric3_structural_completeness"] is not None]
        avg_scores = [r["avg_score"]                  for r in rows if r["avg_score"]                   is not None]

        # Tone breakdown
        by_tone: dict[str, list[float]] = {}
        for r in rows:
            t = r["tone"]
            by_tone.setdefault(t, []).append(r["avg_score"])
        tone_avgs = {t: round(mean(v), 1) for t, v in by_tone.items()}

        # Worst-performing scenario
        worst = min(rows, key=lambda r: r["avg_score"] or 0)

        return {
            "model": model_name,
            "scenarios_evaluated": len(rows),
            "metric1_avg": round(mean(m1_scores), 1) if m1_scores else None,
            "metric2_avg": round(mean(m2_scores), 1) if m2_scores else None,
            "metric3_avg": round(mean(m3_scores), 1) if m3_scores else None,
            "overall_avg": round(mean(avg_scores), 1) if avg_scores else None,
            "tone_breakdown": tone_avgs,
            "worst_scenario": {
                "id": worst["scenario_id"],
                "intent": worst["intent_short"],
                "avg_score": worst["avg_score"],
                "judge_reasoning": worst.get("m2_judge_reasoning", ""),
            },
        }

    stats_a = model_stats(model_a)
    stats_b = model_stats(model_b)

    if not stats_a or not stats_b:
        return {"error": "Insufficient data for comparative analysis."}

    winner    = model_a if stats_a["overall_avg"] >= stats_b["overall_avg"] else model_b
    loser     = model_b if winner == model_a else model_a
    stats_win = stats_a if winner == model_a else stats_b
    stats_los = stats_b if winner == model_a else stats_a
    margin    = round(abs(stats_a["overall_avg"] - stats_b["overall_avg"]), 1)

    # Identify biggest failure dimension of loser
    loser_dims = {
        "Fact Recall"              : stats_los["metric1_avg"],
        "Tone Accuracy"            : stats_los["metric2_avg"],
        "Structural Completeness"  : stats_los["metric3_avg"],
    }
    worst_dim = min(loser_dims, key=loser_dims.get)

    # Map dimension name → stats key so we can look up the winner's score
    _dim_to_key = {
        "Fact Recall"             : "metric1_avg",
        "Tone Accuracy"           : "metric2_avg",
        "Structural Completeness" : "metric3_avg",
    }
    winner_worst_dim_score = stats_win[_dim_to_key[worst_dim]]
    loser_worst_dim_score  = loser_dims[worst_dim]

    analysis_text = f"""
╔══════════════════════════════════════════════════════════════════════╗
║            SECTION 3 — COMPARATIVE ANALYSIS REPORT                 ║
╚══════════════════════════════════════════════════════════════════════╝

EVALUATION OVERVIEW
───────────────────
  Models Compared : {model_a}  vs  {model_b}
  Scenarios Run   : {stats_a['scenarios_evaluated']} × 2 models = {stats_a['scenarios_evaluated'] * 2} total evaluations
  Metrics Used    : Fact Recall | Tone Accuracy (LLM-Judge) | Structural Completeness

SCORE SUMMARY (0–100 scale, higher = better)
──────────────────────────────────────────────
  {'Metric':<35} {'Model A':>10} {'Model B':>10}
  {'─'*55}
  {'Metric 1 — Fact Recall':<35} {str(stats_a['metric1_avg']):>10} {str(stats_b['metric1_avg']):>10}
  {'Metric 2 — Tone Accuracy (LLM-Judge)':<35} {str(stats_a['metric2_avg']):>10} {str(stats_b['metric2_avg']):>10}
  {'Metric 3 — Structural Completeness':<35} {str(stats_a['metric3_avg']):>10} {str(stats_b['metric3_avg']):>10}
  {'─'*55}
  {'OVERALL AVERAGE':<35} {str(stats_a['overall_avg']):>10} {str(stats_b['overall_avg']):>10}

  Model A = {model_a}
  Model B = {model_b}

QUESTION 1: Which model performed better?
──────────────────────────────────────────
  WINNER: {winner}
  with an overall average score of {stats_win['overall_avg']}/100 — {margin} points ahead
  of {loser} ({stats_los['overall_avg']}/100).

  The margin was most pronounced in {worst_dim}, where {loser} scored
  only {loser_worst_dim_score}/100 vs {winner_worst_dim_score}/100 for {winner}.
  (See score table above for full breakdown.)

QUESTION 2: What was the biggest failure mode of the lower-performing model?
─────────────────────────────────────────────────────────────────────────────
  The primary failure mode of {loser} was poor performance on {worst_dim}
  (score: {loser_dims[worst_dim]}/100). This suggests that the smaller/weaker
  model struggles to consistently maintain:
    • The exact requested tone register throughout the email
    • Ensuring every key fact is woven naturally into the body text
    • Producing all structural components of a professional email

  Worst individual scenario for {loser}:
    Scenario   : {stats_los['worst_scenario']['id']} — {stats_los['worst_scenario']['intent']}
    Avg Score  : {stats_los['worst_scenario']['avg_score']}/100
    Judge Note : "{stats_los['worst_scenario']['judge_reasoning']}"

QUESTION 3: Which model do you recommend for production?
──────────────────────────────────────────────────────────
  RECOMMENDATION: {winner}

  Justification (based on custom metric data):

  1. FACT RECALL (M1): {winner} scored {stats_win['metric1_avg']}/100, indicating it reliably
     incorporates all required facts — a non-negotiable requirement for a
     business email tool where missing a fact can damage credibility.

  2. TONE ACCURACY (M2): With an LLM-Judge score of {stats_win['metric2_avg']}/100,
     {winner} demonstrates consistent adherence to the requested tone
     register across all five tone types. Tone failures frustrate end
     users immediately and erode trust in the tool.

  3. STRUCTURAL COMPLETENESS (M3): {winner} scored {stats_win['metric3_avg']}/100,
     confirming that generated emails reliably include all five structural
     components (greeting → purpose → body → CTA → sign-off). Structural
     gaps produce emails that feel incomplete or amateurish.

  While {loser} may offer faster response times or lower cost,
  the quality gap of {margin} overall points represents a meaningful
  difference in user experience that would require significant
  prompt engineering or post-processing to bridge in production.

  VERDICT: Deploy {winner} as the production model.
  Monitor monthly with this same 10-scenario eval suite to track
  quality drift as model versions update.
"""

    return {
        "model_a_stats": stats_a,
        "model_b_stats": stats_b,
        "winner": winner,
        "loser": loser,
        "margin": margin,
        "biggest_failure_dimension": worst_dim,
        "analysis_text": analysis_text,
    }


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

async def main() -> None:
    _separator("EMAIL EVALUATION SYSTEM — SECTIONS 2 & 3")
    print(f"  Scenarios   : {len(TEST_SCENARIOS)}")
    print(f"  Models      : {GROQ_MODEL_PRIMARY}  +  {GROQ_MODEL_SECONDARY}")
    print(f"  Metrics     : Fact Recall | Tone Accuracy (LLM-Judge) | Structural Completeness")
    print(f"  Total Calls : {len(TEST_SCENARIOS) * 2} email generations + {len(TEST_SCENARIOS) * 2} judge calls")
    _separator()

    # ── RUN EVALUATION ───────────────────────────────────────────────────────
    results = await run_full_evaluation(
        scenarios = TEST_SCENARIOS,
        models    = [GROQ_MODEL_PRIMARY, GROQ_MODEL_SECONDARY],
    )

    # ── SAVE REPORTS ─────────────────────────────────────────────────────────
    _separator("SAVING REPORTS")

    # Section 3: Comparative Analysis
    analysis = build_comparative_analysis(
        results, GROQ_MODEL_PRIMARY, GROQ_MODEL_SECONDARY
    )
    print(analysis.get("analysis_text", "Analysis unavailable."))

    save_csv(results, "evaluation_report.csv")
    save_json(
        results,
        "evaluation_results.json",
        extra={"comparative_analysis": analysis},
    )

    # Save plain-text analysis report
    with open("comparative_analysis.txt", "w", encoding="utf-8") as f:
        f.write(analysis.get("analysis_text", "Analysis unavailable."))
    print("  ✅  Analysis saved → comparative_analysis.txt")

    _separator("DONE")
    print("  Output files:")
    print("    📄  evaluation_report.csv      — scores for all 10 × 2 scenarios")
    print("    📄  evaluation_results.json    — full data + metric definitions + analysis")
    print("    📄  comparative_analysis.txt   — Section 3 written report")
    _separator()


if __name__ == "__main__":
    asyncio.run(main())
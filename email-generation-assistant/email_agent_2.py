"""
=============================================================================
  EMAIL GENERATION ASSISTANT
  Built with Pydantic AI + Groq API
  Advanced Prompting: Role-Persona + Dynamic Few-Shot + Chain-of-Thought
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

# =============================================================================
#  CONFIGURATION — Paste your Groq API key directly here
# =============================================================================

GROQ_API_KEY = ""   # <── Replace this

# Available Groq models — easy to swap for model comparison later
GROQ_MODEL_PRIMARY   = "llama-3.3-70b-versatile"     # Model A (primary)
GROQ_MODEL_SECONDARY = "llama-3.1-8b-instant"         # Model B (for comparison)

# =============================================================================
#  PYDANTIC MODELS — Strict typed I/O contracts
# =============================================================================

ToneType = Literal["formal", "casual", "urgent", "empathetic", "persuasive"]


class EmailInput(BaseModel):
    """Validated input contract for the email generation request."""

    intent: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The core purpose or goal of the email.",
        examples=["Follow up after a sales meeting", "Request for proposal details"],
    )
    key_facts: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Bullet-point facts that MUST be woven into the email body.",
        examples=[["Meeting was on Monday", "Budget is $50K", "Decision by Friday"]],
    )
    tone: ToneType = Field(
        ...,
        description="Desired writing style for the email.",
    )
    recipient_name: str | None = Field(
        default=None,
        description="Optional recipient name for personalisation.",
    )
    sender_name: str | None = Field(
        default=None,
        description="Optional sender name for the sign-off.",
    )


class EmailOutput(BaseModel):
    """
    Validated, structured output returned by the Pydantic AI agent.
    Every field is guaranteed by Pydantic validation before being returned
    to the caller — no raw string parsing needed.
    """

    subject_line: str = Field(
        ...,
        min_length=5,
        max_length=120,
        description="A compelling, concise email subject line.",
    )
    email_body: str = Field(
        ...,
        min_length=50,
        description="The complete, ready-to-send email body.",
    )
    tone_applied: ToneType = Field(
        ...,
        description="The tone that was actually applied in the email.",
    )
    facts_woven_in: list[str] = Field(
        ...,
        description="Echo of each key fact that was incorporated into the email.",
    )
    writing_rationale: str = Field(
        ...,
        description="Brief Chain-of-Thought explanation of key writing decisions made.",
    )


# =============================================================================
#  DEPENDENCIES — Injected into the agent at runtime via RunContext
# =============================================================================

@dataclass
class EmailAgentDeps:
    """
    Pydantic AI Dependency Injection container.
    Holds runtime context that tools and dynamic instructions can access.
    """
    few_shot_examples: list[dict]
    selected_examples: list[dict]
    input_data: EmailInput


# =============================================================================
#  FEW-SHOT EXAMPLE BANK — Dynamic selection at runtime by tone match
# =============================================================================

FEW_SHOT_EXAMPLE_BANK: list[dict] = [

    # ── FORMAL ──────────────────────────────────────────────────────────────
    {
        "tone": "formal",
        "intent": "Request a project status update from a vendor",
        "key_facts": [
            "Contract signed on 1st March 2025",
            "Deadline is 30th April 2025",
            "Three milestones are outstanding",
        ],
        "subject": "Formal Request: Project Status Update — Contract Ref #2025-001",
        "email": (
            "Dear [Vendor Name],\n\n"
            "I am writing to formally request a status update regarding the project "
            "governed by our contract signed on 1st March 2025. As we approach the "
            "agreed deadline of 30th April 2025, I note that three milestones remain "
            "outstanding according to our last progress review.\n\n"
            "Kindly provide a written update detailing the current completion status "
            "of each outstanding milestone, any risks to the delivery schedule, and "
            "your proposed mitigation plan.\n\n"
            "I would appreciate a response by end of business this week to allow "
            "sufficient time for any necessary escalation.\n\n"
            "Yours sincerely,\n[Your Name]"
        ),
        "rationale": (
            "Used passive and nominative constructions to maintain formal register. "
            "Opened with purpose, embedded all three facts naturally, and closed with "
            "a clear call-to-action and polite deadline."
        ),
    },

    # ── CASUAL ──────────────────────────────────────────────────────────────
    {
        "tone": "casual",
        "intent": "Invite a colleague to a team lunch",
        "key_facts": [
            "Lunch is on Thursday at 12:30 PM",
            "Location is The Green Fork restaurant",
            "RSVP needed by Wednesday",
        ],
        "subject": "Team lunch this Thursday — you in? 🍕",
        "email": (
            "Hey [Name],\n\n"
            "We're doing a team lunch this Thursday at 12:30 PM and it'd be great "
            "to have you along! We've booked a table at The Green Fork — the pizza "
            "there is seriously good.\n\n"
            "Just let me know by Wednesday if you're coming so we can sort the "
            "numbers. Hope to see you there!\n\n"
            "Cheers,\n[Your Name]"
        ),
        "rationale": (
            "Kept the tone warm and conversational. Used contractions and informal "
            "expressions. Embedded all facts within natural flowing sentences rather "
            "than listing them."
        ),
    },

    # ── URGENT ──────────────────────────────────────────────────────────────
    {
        "tone": "urgent",
        "intent": "Alert the team about a production server outage",
        "key_facts": [
            "Server went down at 14:35 UTC",
            "Estimated 2,000 customers affected",
            "Engineering team is actively investigating",
        ],
        "subject": "URGENT: Production Server Outage — Immediate Action Required",
        "email": (
            "Team,\n\n"
            "ACTION REQUIRED: Our production server went down at 14:35 UTC. "
            "Approximately 2,000 customers are currently unable to access the "
            "platform and we are receiving escalating support tickets.\n\n"
            "The engineering team is actively investigating the root cause. All "
            "non-critical work should be deprioritised immediately to support the "
            "recovery effort.\n\n"
            "Status updates will be sent every 30 minutes. Please acknowledge "
            "this email to confirm receipt.\n\n"
            "— [Your Name]"
        ),
        "rationale": (
            "Led with the BLUF (Bottom Line Up Front) — no preamble. Used active "
            "voice throughout. All three facts placed in the first two sentences "
            "for maximum urgency. Clear action directive included."
        ),
    },

    # ── EMPATHETIC ──────────────────────────────────────────────────────────
    {
        "tone": "empathetic",
        "intent": "Apologise for a delayed delivery to a customer",
        "key_facts": [
            "Order was placed on 5th April",
            "Delay caused by a supplier disruption",
            "New delivery date is 22nd April",
        ],
        "subject": "We're Sorry — An Update on Your Order",
        "email": (
            "Dear [Customer Name],\n\n"
            "I want to personally reach out and sincerely apologise for the delay "
            "with your order placed on 5th April. I completely understand how "
            "frustrating this must be, and you deserve better.\n\n"
            "The delay was caused by an unexpected disruption with our supplier, "
            "which was beyond our control but is no excuse for the inconvenience "
            "this has caused you. Your order is now confirmed for delivery on "
            "22nd April.\n\n"
            "As a token of our apology, I'd like to offer you a 15% discount on "
            "your next order. Please don't hesitate to reach out if there is "
            "anything I can do to make this right.\n\n"
            "Warm regards,\n[Your Name]"
        ),
        "rationale": (
            "Acknowledged the customer's feelings first before explaining the "
            "situation. Used 'I' statements to take personal ownership. Avoided "
            "defensive language. All three facts woven in naturally."
        ),
    },

    # ── PERSUASIVE ───────────────────────────────────────────────────────────
    {
        "tone": "persuasive",
        "intent": "Pitch a new software tool to the management team",
        "key_facts": [
            "Tool reduces reporting time by 60%",
            "Monthly cost is $200 per team",
            "30-day free trial available",
        ],
        "subject": "Save 60% of Reporting Time — A Proposal Worth 5 Minutes",
        "email": (
            "Hi [Name],\n\n"
            "What if your team could cut reporting time by 60%? That's exactly "
            "what our pilot teams have achieved with [Tool Name], and I'd love to "
            "show you how.\n\n"
            "At just $200 per month per team, the ROI is immediate — the time "
            "saved in week one alone more than covers the cost. And here's the "
            "best part: there's a 30-day free trial, so you can see the results "
            "before spending a single rupee.\n\n"
            "I'd welcome 20 minutes on your calendar this week to walk you "
            "through a quick demo. Would Thursday afternoon work for you?\n\n"
            "Best,\n[Your Name]"
        ),
        "rationale": (
            "Opened with a compelling question to hook the reader. Led with the "
            "biggest benefit (60% time saving). Pre-empted the cost objection by "
            "reframing it as ROI. Closed with a low-friction ask."
        ),
    },
]


def select_few_shot_examples(tone: ToneType, n: int = 2) -> list[dict]:
    """
    Dynamically select the most relevant few-shot examples from the bank.
    Primary selection: exact tone match. Secondary: fill remaining slots from others.

    This is a key part of the Advanced Prompting strategy — dynamic context
    injection rather than static, hardcoded examples.
    """
    tone_matched = [ex for ex in FEW_SHOT_EXAMPLE_BANK if ex["tone"] == tone]
    others       = [ex for ex in FEW_SHOT_EXAMPLE_BANK if ex["tone"] != tone]

    selected = tone_matched[:n]
    if len(selected) < n:
        selected += others[: n - len(selected)]

    return selected[:n]


def build_few_shot_text(examples: list[dict]) -> str:
    """Format selected examples into the prompt string."""
    lines = []
    for i, ex in enumerate(examples, 1):
        facts_str = "\n".join(f"  - {f}" for f in ex["key_facts"])
        lines.append(
            f"--- EXAMPLE {i} ---\n"
            f"Intent  : {ex['intent']}\n"
            f"Tone    : {ex['tone']}\n"
            f"Key Facts:\n{facts_str}\n\n"
            f"Subject : {ex['subject']}\n\n"
            f"Email Body:\n{ex['email']}\n\n"
            f"Writing Rationale: {ex['rationale']}\n"
        )
    return "\n".join(lines)


# =============================================================================
#  AGENT FACTORY — Creates the Pydantic AI Agent with the Groq model
# =============================================================================

def build_email_agent(model_name: str = GROQ_MODEL_PRIMARY) -> Agent:
    """
    Constructs the Pydantic AI Agent with:
      - GroqModel with direct API key (GroqProvider)
      - EmailOutput as the structured output type
      - Role-Persona system instructions (Elena the Communication Specialist)
      - Dynamic instructions via @agent.instructions decorator
    """

    groq_model = GroqModel(
        model_name,
        provider=GroqProvider(api_key=GROQ_API_KEY),
    )

    # ── Static system instructions: ROLE + PERSONA prompting ────────────────
    SYSTEM_INSTRUCTIONS = """\
You are Elena, a Senior Corporate Communication Specialist with 15 years of \
experience crafting high-impact emails for Fortune 500 executives, startups, \
and government organisations. Your writing is always clear, purposeful, and \
perfectly calibrated to the requested tone.

CORE PRINCIPLES you never violate:
1. Every single key fact provided by the user MUST appear in the email body — \
   do not omit, paraphrase away, or summarise any fact so heavily that it \
   disappears from the email.
2. The email tone must exactly match the requested style. Do not default to \
   formal when asked for casual, or polite when asked for urgent.
3. Subject lines must be action-oriented and relevant — never generic.
4. Emails must have a clear structure: greeting → opening purpose → body \
   with facts → call-to-action → sign-off.
5. You MUST respond using ONLY the structured JSON schema provided — no \
   preamble, no explanation outside the JSON fields.

CHAIN-OF-THOUGHT PROCESS (follow this internally before writing):
Step 1 → Identify the PRIMARY goal of this email in one sentence.
Step 2 → Decide the emotional register and vocabulary appropriate to the tone.
Step 3 → Plan where each key fact will appear (opening / body / closing).
Step 4 → Draft a subject line that reflects the intent and tone.
Step 5 → Write the full email, verifying each fact is included.
Step 6 → Self-review: Does every sentence serve the email's goal? Is the \
          tone consistent throughout? Are all facts present?
Step 7 → Record your key writing decisions in the `writing_rationale` field.
"""

    agent: Agent[EmailAgentDeps, EmailOutput] = Agent(
        groq_model,
        output_type=EmailOutput,
        instructions=SYSTEM_INSTRUCTIONS,
        retries=3,          # Pydantic AI will retry on validation failure
    )

    # ── Dynamic instructions: Inject few-shot examples at runtime ────────────
    @agent.instructions
    def inject_few_shot_examples(ctx: RunContext[EmailAgentDeps]) -> str:
        """
        This function is called by Pydantic AI on every run, allowing us to
        inject dynamically chosen few-shot examples based on the current
        request's tone — a key advanced prompting technique.
        """
        examples = ctx.deps.selected_examples
        few_shot_text = build_few_shot_text(examples)
        return (
            f"REFERENCE EXAMPLES (study these carefully — match this quality):\n\n"
            f"{few_shot_text}\n"
            f"--- END OF EXAMPLES ---\n"
            f"Now generate a NEW email using the same quality and structure. "
            f"Do NOT copy the examples — they are reference only."
        )

    return agent


# =============================================================================
#  CORE GENERATION FUNCTION
# =============================================================================

async def generate_email(
    input_data: EmailInput,
    model_name: str = GROQ_MODEL_PRIMARY,
    verbose: bool = True,
) -> tuple[EmailOutput, dict]:
    """
    Main async function to generate an email.

    Returns:
        - EmailOutput: the validated structured email object
        - metadata: usage stats, model name, timestamp, etc.
    """

    # Select the best few-shot examples for this tone
    selected_examples = select_few_shot_examples(input_data.tone, n=2)

    # Build the dependency injection container
    deps = EmailAgentDeps(
        few_shot_examples=FEW_SHOT_EXAMPLE_BANK,
        selected_examples=selected_examples,
        input_data=input_data,
    )

    # Build the user prompt string — structured, clear, complete
    facts_str = "\n".join(f"  • {fact}" for fact in input_data.key_facts)
    recipient_str = f"Recipient: {input_data.recipient_name}" if input_data.recipient_name else "Recipient: [not specified — use placeholder]"
    sender_str    = f"Sender: {input_data.sender_name}" if input_data.sender_name else "Sender: [not specified — use placeholder]"

    user_prompt = f"""\
Generate a professional email with the following specifications:

INTENT (goal of the email):
{input_data.intent}

TONE (must be applied consistently throughout):
{input_data.tone.upper()}

KEY FACTS (ALL of these MUST be included in the email body):
{facts_str}

CONTEXT:
{recipient_str}
{sender_str}

INSTRUCTIONS:
- Follow the Chain-of-Thought process from your system instructions.
- Weave ALL key facts naturally into the email — do not just list them.
- Keep the email professional yet appropriate to the requested tone.
- Subject line should be specific and compelling, not generic.
- Record your writing decisions in the `writing_rationale` field.
"""

    if verbose:
        _print_separator("GENERATING EMAIL")
        print(f"  Model     : {model_name}")
        print(f"  Intent    : {input_data.intent}")
        print(f"  Tone      : {input_data.tone}")
        print(f"  Facts     : {len(input_data.key_facts)} key facts")
        print(f"  Examples  : {len(selected_examples)} few-shot examples injected ({[e['tone'] for e in selected_examples]})")
        _print_separator()

    # Build and run the agent
    agent = build_email_agent(model_name)

    start_time = datetime.now()
    result = await agent.run(user_prompt, deps=deps)
    end_time = datetime.now()

    duration_ms = (end_time - start_time).total_seconds() * 1000

    # Build metadata dict for tracking / later evaluation use
    metadata = {
        "model_name"      : model_name,
        "timestamp"       : start_time.isoformat(),
        "duration_ms"     : round(duration_ms, 2),
        "input_intent"    : input_data.intent,
        "input_tone"      : input_data.tone,
        "input_facts_count": len(input_data.key_facts),
        "few_shot_tones"  : [e["tone"] for e in selected_examples],
        "usage"           : {
            "request_tokens" : result.usage().request_tokens,
            "response_tokens": result.usage().response_tokens,
            "total_tokens"   : result.usage().total_tokens,
        } if result.usage() else {},
    }

    return result.output, metadata


# =============================================================================
#  DISPLAY HELPERS
# =============================================================================

def _print_separator(title: str = "", width: int = 70) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'═' * pad} {title} {'═' * pad}")
    else:
        print(f"{'─' * width}")


def display_email_result(output: EmailOutput, metadata: dict) -> None:
    """Pretty-print the generated email and metadata to stdout."""

    _print_separator("GENERATED EMAIL")
    print(f"\n📧  SUBJECT: {output.subject_line}\n")
    print("─" * 70)
    print(output.email_body)
    print("─" * 70)

    _print_separator("GENERATION METADATA")
    print(f"  ✅  Tone Applied    : {output.tone_applied}")
    print(f"  ✅  Facts Woven In  : {len(output.facts_woven_in)} / {metadata['input_facts_count']}")
    for fact in output.facts_woven_in:
        print(f"       • {fact}")
    print(f"\n  💡  Writing Rationale:")
    # Wrap rationale text at 65 chars
    rationale_words = output.writing_rationale.split()
    line, lines = [], []
    for word in rationale_words:
        line.append(word)
        if len(" ".join(line)) > 65:
            lines.append("     " + " ".join(line[:-1]))
            line = [word]
    if line:
        lines.append("     " + " ".join(line))
    print("\n".join(lines))

    _print_separator("USAGE & PERFORMANCE")
    print(f"  Model        : {metadata['model_name']}")
    print(f"  Duration     : {metadata['duration_ms']} ms")
    if metadata.get("usage"):
        u = metadata["usage"]
        print(f"  Tokens Used  : {u.get('total_tokens', 'N/A')} (req: {u.get('request_tokens', 'N/A')}, res: {u.get('response_tokens', 'N/A')})")
    _print_separator()

# =============================================================================
#  MAIN — EDIT YOUR INPUTS HERE, THEN RUN THE FILE
# =============================================================================

async def _run(email_input: EmailInput) -> None:
    """Internal async runner called by main()."""
    try:
        output, metadata = await generate_email(
            email_input,
            model_name=GROQ_MODEL_PRIMARY,
            verbose=True,
        )
        display_email_result(output, metadata)

        # Save result to JSON (useful for the evaluation phase later)
        result_payload = {
            "input"   : email_input.model_dump(),
            "output"  : output.model_dump(),
            "metadata": metadata,
        }
        with open("email_output.json", "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2, ensure_ascii=False, default=str)

        _print_separator()
        print("  📁  Full result saved → email_output.json")
        _print_separator()

    except Exception as exc:
        _print_separator("ERROR")
        print(f"  ❌  Generation failed: {exc}")
        traceback.print_exc()


def main() -> None:
    """
    ┌──────────────────────────────────────────────────────┐
    │  STEP 1 — Set GROQ_API_KEY at the top of this file   │
    │  STEP 2 — Fill in your inputs in the block below     │
    │  STEP 3 — Run:  python email_generator.py            │
    └──────────────────────────────────────────────────────┘
    """

    # =========================================================================
    #  ✏️  YOUR INPUTS — Change these values and run the file
    # =========================================================================

    intent         = "Propose a strategic partnership to a potential business partner"
    key_facts      = [
        "Our platform has 2 million active users across Southeast Asia",
        "Proposed partnership involves co-marketing and revenue sharing",
        "We can offer a 20% revenue share on referred conversions",
        "Requesting an exploratory call in the next two weeks",
    ]
    tone           = "persuasive"
    recipient_name = "Ms. Verma"
    sender_name    = "Raj Gupta"

    # =========================================================================
    #  ↓↓  DO NOT EDIT BELOW THIS LINE  ↓↓
    # =========================================================================

    email_input = EmailInput(
        intent        = intent,
        key_facts     = key_facts,
        tone          = tone,           # type: ignore[arg-type]
        recipient_name= recipient_name,
        sender_name   = sender_name,
    )

    asyncio.run(_run(email_input))


if __name__ == "__main__":
    main()

"""
System Prompt Patterns for Healthcare AI
==========================================
Demonstrates different system prompt strategies and how they affect model
behavior in clinical contexts. Covers: persona-based, constraint-heavy,
template-driven, and safety-first patterns.

Each pattern is tested against the same clinical note to show how the system
prompt shapes the response.

Requires: OPENAI_API_KEY environment variable
"""

import json
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# --- Test Clinical Note ---

TEST_NOTE: str = (
    "42-year-old female presents with 3-week history of progressive fatigue, unintentional "
    "weight loss of 12 lbs, and intermittent low-grade fevers (max 100.4F). She reports night "
    "sweats for the past 2 weeks. She has noticed a painless, firm lump in her left neck that "
    "has been growing. No recent infections. Non-smoker. Family history significant for "
    "lymphoma in her mother. CBC shows WBC 15.2, Hgb 10.1, platelets 425. LDH elevated at 340. "
    "Peripheral smear shows no blasts."
)


# --- Pattern 1: Minimal / Generic ---

PATTERN_MINIMAL: str = "You are a helpful medical assistant. Analyze the clinical note."


# --- Pattern 2: Persona-Based (Expert Role) ---

PATTERN_PERSONA: str = """You are Dr. Sarah Chen, a board-certified hematologist-oncologist with
20 years of experience at a major academic medical center. You specialize in lymphoproliferative
disorders.

When reviewing clinical notes, you:
- Think systematically through differential diagnoses
- Prioritize ruling out malignancy when B-symptoms are present
- Recommend evidence-based workup following NCCN guidelines
- Communicate clearly, as if writing a consult note for the referring physician

Respond with a structured consult note including: Assessment, Differential Diagnosis (ranked by
probability), and Recommended Workup."""


# --- Pattern 3: Constraint-Heavy (Safety Rails) ---

PATTERN_CONSTRAINED: str = """You are a clinical decision support system for primary care physicians.

CRITICAL CONSTRAINTS:
1. NEVER provide a definitive diagnosis. Always present differentials ranked by probability.
2. NEVER recommend specific medications or dosages.
3. ALWAYS flag red-flag symptoms that require urgent workup.
4. ALWAYS include a "Safety Net" section listing symptoms that should prompt immediate return.
5. If the presentation could be malignant, this MUST be the first item in your differential.
6. Include confidence levels (high/medium/low) for each differential.
7. Cite relevant clinical guidelines where applicable (e.g., NCCN, USPSTF).

Return your analysis as JSON with fields:
  red_flags (list), differential_diagnosis (list of {diagnosis, probability, confidence}),
  recommended_workup (list), safety_net_instructions (str), urgency_level (1-5)."""


# --- Pattern 4: Template-Driven (Structured Output) ---

PATTERN_TEMPLATE: str = """You are a clinical NLP extraction system. Your ONLY job is to extract
structured data from clinical notes. You do not provide medical advice or recommendations.

Extract the following fields from the provided clinical note and return as JSON:

{
  "demographics": {
    "age": int,
    "sex": "M" | "F",
    "smoking_status": "current" | "former" | "never" | "unknown"
  },
  "chief_complaint": str,
  "symptom_duration": str,
  "symptoms": [str],
  "vital_signs": {field: value},
  "lab_results": [{test: str, value: str, flag: "normal" | "high" | "low" | "critical"}],
  "family_history": [str],
  "physical_exam_findings": [str]
}

Rules:
- Extract ONLY what is explicitly stated in the note.
- Use null for fields not mentioned.
- Do NOT infer or assume any values.
- Flag lab values as high/low/critical based on standard reference ranges."""


# --- Pattern 5: Safety-First (Guardrails + Escalation) ---

PATTERN_SAFETY_FIRST: str = """You are a clinical triage assistant integrated into an EHR system.
Your primary directive is PATIENT SAFETY.

TRIAGE PROTOCOL:
1. Scan for life-threatening conditions first (ABCDE approach).
2. Identify any red-flag symptoms requiring emergent evaluation.
3. Classify urgency: EMERGENT / URGENT / SEMI-URGENT / ROUTINE.
4. If ANY of the following are present, classify as EMERGENT regardless of other findings:
   - Hemodynamic instability
   - Acute neurological deficit
   - Active hemorrhage
   - Anaphylaxis
   - Acute chest pain with cardiac risk factors

MANDATORY ESCALATION TRIGGERS:
- If the note suggests possible malignancy: add "ONCOLOGY_REFERRAL_RECOMMENDED" flag
- If the note suggests possible sepsis: add "SEPSIS_ALERT" flag
- If the note involves a pediatric patient with fever > 104F: add "PEDS_FEVER_PROTOCOL" flag

Return JSON with:
  triage_class (str), flags (list), reasoning (str), recommended_next_steps (list),
  time_to_physician_target (str)."""


def query_with_pattern(pattern_name: str, system_prompt: str, note: str) -> dict[str, Any]:
    """Send a clinical note to the LLM with a given system prompt pattern."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt + "\n\nReturn your response as JSON."},
            {"role": "user", "content": f"Clinical note:\n\n{note}"},
        ],
    )
    result = json.loads(response.choices[0].message.content)
    return {"pattern": pattern_name, "response": result}


def main() -> None:
    """Run all patterns against the test note and display results."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    patterns = [
        ("1. Minimal/Generic", PATTERN_MINIMAL),
        ("2. Persona-Based (Expert)", PATTERN_PERSONA),
        ("3. Constraint-Heavy (Safety Rails)", PATTERN_CONSTRAINED),
        ("4. Template-Driven (Extraction)", PATTERN_TEMPLATE),
        ("5. Safety-First (Triage)", PATTERN_SAFETY_FIRST),
    ]

    if use_rich:
        console.print(
            Panel(TEST_NOTE, title="[bold]Test Clinical Note[/bold]", border_style="cyan")
        )

    for name, prompt in patterns:
        result = query_with_pattern(name, prompt, TEST_NOTE)
        formatted_json = json.dumps(result["response"], indent=2)

        if use_rich:
            console.print(f"\n[bold yellow]{'=' * 80}[/bold yellow]")
            console.print(f"[bold green]Pattern: {name}[/bold green]")
            console.print(
                Panel(prompt[:300] + "..." if len(prompt) > 300 else prompt, title="System Prompt")
            )
            syntax = Syntax(formatted_json, "json", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title="LLM Response"))
        else:
            print(f"\n{'=' * 60}")
            print(f"Pattern: {name}")
            print(f"Response:\n{formatted_json[:500]}")


if __name__ == "__main__":
    main()


# --- Sample Output (abbreviated) ---
#
# Pattern: 1. Minimal/Generic
# Response: {"analysis": "The patient presents with B-symptoms (fever, night sweats,
#   weight loss) and lymphadenopathy. This is concerning for lymphoma..."}
#
# Pattern: 2. Persona-Based (Expert)
# Response: {"assessment": "42F with classic B-symptoms and cervical lymphadenopathy.
#   Family history of lymphoma is significant.", "differential_diagnosis": [
#   "Hodgkin lymphoma (most likely)", "Non-Hodgkin lymphoma", "Infectious mononucleosis",
#   "Sarcoidosis", "Metastatic carcinoma"], "recommended_workup": [
#   "Excisional biopsy of cervical node (NOT FNA)", "CT chest/abdomen/pelvis with contrast",
#   "Flow cytometry", "PET-CT if biopsy confirms lymphoma"]}
#
# Pattern: 3. Constraint-Heavy (Safety Rails)
# Response: {"red_flags": ["B-symptoms triad", "progressive lymphadenopathy",
#   "family history of lymphoma", "elevated LDH"], "differential_diagnosis": [
#   {"diagnosis": "Lymphoma", "probability": "high", "confidence": "high"},
#   {"diagnosis": "Reactive lymphadenopathy", "probability": "low", "confidence": "medium"}],
#   "urgency_level": 4, "safety_net_instructions": "Return immediately if..."}
#
# Pattern: 4. Template-Driven (Extraction)
# Response: {"demographics": {"age": 42, "sex": "F", "smoking_status": "never"},
#   "chief_complaint": "fatigue, weight loss, fevers",
#   "symptoms": ["fatigue", "weight loss", "low-grade fevers", "night sweats",
#   "painless cervical lymphadenopathy"], "lab_results": [
#   {"test": "WBC", "value": "15.2", "flag": "high"},
#   {"test": "Hgb", "value": "10.1", "flag": "low"},
#   {"test": "LDH", "value": "340", "flag": "high"}]}
#
# Pattern: 5. Safety-First (Triage)
# Response: {"triage_class": "URGENT", "flags": ["ONCOLOGY_REFERRAL_RECOMMENDED"],
#   "reasoning": "B-symptom triad with lymphadenopathy and elevated LDH raises high
#   suspicion for lymphoproliferative disorder.", "time_to_physician_target": "24-48 hours"}

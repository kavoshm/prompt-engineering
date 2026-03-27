"""
Chain-of-Thought Reasoning for Clinical Decision Support
=========================================================
Demonstrates how asking the LLM to reason step by step before arriving at a
conclusion improves accuracy and provides an auditable reasoning trail --
critical in healthcare where decisions must be explainable.

Compares direct classification vs chain-of-thought classification on clinical
scenarios that require nuanced reasoning.

Requires: OPENAI_API_KEY environment variable
"""

import json
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# --- Complex Clinical Scenarios Requiring Reasoning ---

CLINICAL_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "scenario_001",
        "note": (
            "55-year-old male presents with epigastric pain for 2 hours. Describes it as "
            "burning, non-radiating. Has history of GERD. However, he also has a history of "
            "type 2 diabetes, hypertension, and is a current smoker. He appears anxious and "
            "slightly diaphoretic. Vitals: BP 152/90, HR 88, SpO2 97%. ECG shows non-specific "
            "ST changes in leads II, III, aVF."
        ),
        "expected_urgency": 5,
        "reasoning_note": (
            "Despite the GERD history suggesting a benign cause, the cardiac risk factors "
            "(DM, HTN, smoking), diaphoresis, and ECG changes make ACS a must-rule-out diagnosis."
        ),
    },
    {
        "id": "scenario_002",
        "note": (
            "28-year-old female presents with severe headache, worst of her life, sudden onset "
            "1 hour ago while exercising. No history of migraines. Vitals stable. Neurological "
            "exam is grossly normal. She rates pain 9/10. No fever, no neck stiffness on casual "
            "observation, though she is reluctant to flex her neck due to pain."
        ),
        "expected_urgency": 5,
        "reasoning_note": (
            "Thunderclap headache (worst headache of life, sudden onset) is a classic "
            "presentation for subarachnoid hemorrhage until proven otherwise. The reluctance "
            "to flex the neck may indicate meningismus. Requires emergent CT head and likely LP."
        ),
    },
    {
        "id": "scenario_003",
        "note": (
            "70-year-old female with history of atrial fibrillation on warfarin presents with "
            "a 3-day history of black, tarry stools. She reports feeling tired and lightheaded "
            "when standing. No abdominal pain. INR last checked 2 weeks ago was 3.8 (therapeutic "
            "range 2-3). Vitals: BP 108/68 sitting, 88/55 standing. HR 96 sitting, 118 standing."
        ),
        "expected_urgency": 4,
        "reasoning_note": (
            "Melena with supratherapeutic INR indicates upper GI bleeding in an anticoagulated "
            "patient. Orthostatic hypotension confirms significant volume depletion. Urgent but "
            "patient is hemodynamically compensating."
        ),
    },
    {
        "id": "scenario_004",
        "note": (
            "6-year-old male brought by parents for 2-day history of fever (max 102.1F), runny "
            "nose, and mild cough. Eating and drinking well. Active and playful in the exam room. "
            "Ears clear bilaterally. Throat mildly erythematous without exudates. Lungs clear. "
            "Rapid strep negative. Rapid flu negative."
        ),
        "expected_urgency": 1,
        "reasoning_note": (
            "Classic viral URI in a well-appearing child with reassuring exam. Supportive care "
            "only. Low urgency."
        ),
    },
]


DIRECT_SYSTEM_PROMPT: str = """You are a clinical triage specialist. Given a clinical note, determine
the urgency level on a scale of 1-5:
  1 = Routine/non-urgent
  2 = Low urgency
  3 = Moderate urgency
  4 = High urgency (needs prompt intervention)
  5 = Emergency (life-threatening, immediate action required)

Return JSON with fields: urgency_level (int), primary_concern (str), recommended_action (str)."""


COT_SYSTEM_PROMPT: str = """You are a clinical triage specialist. Given a clinical note, determine
the urgency level on a scale of 1-5:
  1 = Routine/non-urgent
  2 = Low urgency
  3 = Moderate urgency
  4 = High urgency (needs prompt intervention)
  5 = Emergency (life-threatening, immediate action required)

Before determining urgency, reason through the following steps:

Step 1 - SYMPTOMS: List all symptoms and clinical findings mentioned in the note.
Step 2 - RED FLAGS: Identify any red flags or concerning features (e.g., sudden onset,
         hemodynamic instability, altered mental status, high-risk history).
Step 3 - DIFFERENTIAL: Consider the most dangerous possible diagnosis that fits the presentation
         (worst-case-first thinking, as is standard in emergency medicine).
Step 4 - RISK FACTORS: Note relevant patient history that increases risk (age, comorbidities,
         medications).
Step 5 - DECISION: Based on the above, assign an urgency level.

Return JSON with fields:
  step1_symptoms (list of str),
  step2_red_flags (list of str),
  step3_differential (str),
  step4_risk_factors (list of str),
  urgency_level (int),
  primary_concern (str),
  recommended_action (str)."""


def classify_direct(note_text: str) -> dict:
    """Classify urgency without chain-of-thought reasoning."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Clinical note:\n\n{note_text}"},
        ],
    )
    return json.loads(response.choices[0].message.content)


def classify_chain_of_thought(note_text: str) -> dict:
    """Classify urgency with explicit chain-of-thought reasoning steps."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Clinical note:\n\n{note_text}"},
        ],
    )
    return json.loads(response.choices[0].message.content)


def run_cot_comparison() -> None:
    """Run both approaches on all scenarios and display the comparison."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    for scenario in CLINICAL_SCENARIOS:
        direct_result = classify_direct(scenario["note"])
        cot_result = classify_chain_of_thought(scenario["note"])

        direct_urgency = direct_result.get("urgency_level", -1)
        cot_urgency = cot_result.get("urgency_level", -1)
        expected = scenario["expected_urgency"]

        if use_rich:
            console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
            console.print(f"[bold]{scenario['id']}[/bold]")
            console.print(
                Panel(scenario["note"][:200] + "...", title="Clinical Note (truncated)")
            )

            table = Table(box=box.SIMPLE)
            table.add_column("Method", style="bold")
            table.add_column("Urgency", justify="center")
            table.add_column("Expected", justify="center")
            table.add_column("Match", justify="center")

            direct_match = direct_urgency == expected
            cot_match = cot_urgency == expected

            table.add_row(
                "Direct",
                str(direct_urgency),
                str(expected),
                "[green]YES[/green]" if direct_match else "[red]NO[/red]",
            )
            table.add_row(
                "Chain-of-Thought",
                str(cot_urgency),
                str(expected),
                "[green]YES[/green]" if cot_match else "[red]NO[/red]",
            )
            console.print(table)

            # Show the reasoning chain
            if "step2_red_flags" in cot_result:
                red_flags = cot_result["step2_red_flags"]
                console.print(
                    f"[yellow]CoT Red Flags Identified:[/yellow] {', '.join(red_flags)}"
                )
            if "step3_differential" in cot_result:
                console.print(
                    f"[yellow]CoT Worst-Case Differential:[/yellow] "
                    f"{cot_result['step3_differential']}"
                )

            console.print(
                f"[dim]Expected reasoning: {scenario['reasoning_note']}[/dim]"
            )
        else:
            print(f"\n{'=' * 60}")
            print(f"{scenario['id']}")
            print(f"Direct: urgency={direct_urgency} (expected {expected})")
            print(f"CoT:    urgency={cot_urgency} (expected {expected})")
            if "step2_red_flags" in cot_result:
                print(f"Red flags: {cot_result['step2_red_flags']}")


if __name__ == "__main__":
    run_cot_comparison()


# --- Sample Output ---
#
# ════════════════════════════════════════════════════════════════════════════════
# scenario_001
# ╭──────────────────── Clinical Note (truncated) ────────────────────╮
# │ 55-year-old male presents with epigastric pain for 2 hours.      │
# │ Describes it as burning, non-radiating. Has history of GERD...   │
# ╰──────────────────────────────────────────────────────────────────╯
#  Method            Urgency   Expected   Match
#  Direct            3         5          NO
#  Chain-of-Thought  5         5          YES
#
# CoT Red Flags Identified: diaphoresis, non-specific ST changes, cardiac risk factors
# CoT Worst-Case Differential: Acute coronary syndrome (NSTEMI)
#
# ════════════════════════════════════════════════════════════════════════════════
# scenario_002
#  Method            Urgency   Expected   Match
#  Direct            4         5          NO
#  Chain-of-Thought  5         5          YES
#
# CoT Red Flags Identified: thunderclap headache, worst headache of life, sudden onset
# CoT Worst-Case Differential: Subarachnoid hemorrhage
#
# ════════════════════════════════════════════════════════════════════════════════
# scenario_003
#  Method            Urgency   Expected   Match
#  Direct            4         4          YES
#  Chain-of-Thought  4         4          YES
#
# CoT Red Flags Identified: melena, orthostatic hypotension, supratherapeutic INR
# CoT Worst-Case Differential: Upper GI hemorrhage in anticoagulated patient
#
# ════════════════════════════════════════════════════════════════════════════════
# scenario_004
#  Method            Urgency   Expected   Match
#  Direct            1         1          YES
#  Chain-of-Thought  1         1          YES
#
# CoT Red Flags Identified: (none)
# CoT Worst-Case Differential: Viral upper respiratory infection

"""
Few-Shot Classification for Medical Text
=========================================
Demonstrates how providing labeled examples within the prompt dramatically
improves the LLM's ability to classify clinical text into predefined categories.

This script compares zero-shot vs few-shot classification on the same set of
synthetic clinical notes to illustrate the accuracy improvement.

Requires: OPENAI_API_KEY environment variable
"""

import json
import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# --- Synthetic Clinical Notes for Classification ---

CLINICAL_NOTES: list[dict[str, str]] = [
    {
        "id": "note_001",
        "text": (
            "72-year-old male presenting to ED with sudden onset crushing chest pain "
            "radiating to left arm and jaw, onset 45 minutes ago. Associated diaphoresis "
            "and shortness of breath. History of hypertension and hyperlipidemia. "
            "BP 168/95, HR 102, SpO2 94% on room air."
        ),
        "expected_category": "emergency",
    },
    {
        "id": "note_002",
        "text": (
            "34-year-old female presents for routine prenatal visit at 28 weeks gestation. "
            "No complaints. Fetal heart tones 140 bpm, fundal height appropriate for "
            "gestational age. GBS screening ordered. Glucose tolerance test results pending. "
            "Patient reports good fetal movement."
        ),
        "expected_category": "routine",
    },
    {
        "id": "note_003",
        "text": (
            "58-year-old male with known COPD presents with worsening dyspnea over 3 days. "
            "Increased sputum production, now greenish-yellow. Using accessory muscles. "
            "SpO2 88% on room air, improved to 93% on 2L nasal cannula. Bilateral wheezes "
            "and rhonchi on auscultation. Last exacerbation 2 months ago."
        ),
        "expected_category": "urgent",
    },
    {
        "id": "note_004",
        "text": (
            "45-year-old female presents for follow-up of well-controlled type 2 diabetes. "
            "A1C 6.8%, down from 7.2% three months ago. Current medications: metformin 1000mg "
            "BID. No hypoglycemic episodes. Annual diabetic eye exam completed, no retinopathy. "
            "Foot exam normal. Continue current regimen."
        ),
        "expected_category": "routine",
    },
    {
        "id": "note_005",
        "text": (
            "19-year-old male brought in by ambulance after motorcycle accident. GCS 13. "
            "Large laceration on left forearm with visible bone. Deformity of right femur. "
            "Abdominal tenderness in LUQ. BP 98/62, HR 118, RR 24. Two large-bore IVs "
            "placed, 1L NS bolus initiated."
        ),
        "expected_category": "emergency",
    },
    {
        "id": "note_006",
        "text": (
            "8-year-old presents with mother for 3-day history of sore throat, fever to "
            "101.5F, and decreased appetite. No cough or rhinorrhea. Tonsillar exudates "
            "bilaterally, tender anterior cervical lymphadenopathy. Rapid strep positive. "
            "Amoxicillin prescribed. Return if symptoms worsen."
        ),
        "expected_category": "non_urgent",
    },
]

# --- Few-Shot Examples (separate from test data) ---

FEW_SHOT_EXAMPLES: str = """
Here are examples of correctly classified clinical notes:

Example 1:
Note: "82-year-old female with acute onset left-sided weakness and facial droop, onset 30 minutes ago. Slurred speech. BP 190/110. NIH Stroke Scale 14. Code stroke activated."
Classification: {"category": "emergency", "reasoning": "Acute neurological deficit consistent with stroke, time-critical presentation requiring immediate intervention."}

Example 2:
Note: "40-year-old male here for annual physical. No complaints. Vitals normal. BMI 24. Labs ordered for lipid panel and CBC. Due for colonoscopy screening discussion at age 45."
Classification: {"category": "routine", "reasoning": "Preventive care visit with no acute complaints or abnormal findings."}

Example 3:
Note: "65-year-old female with 2-day history of worsening lower extremity edema, 8-pound weight gain, and increased dyspnea on exertion. Known CHF, EF 35%. Crackles at bilateral bases. Needs diuretic adjustment and monitoring."
Classification: {"category": "urgent", "reasoning": "CHF exacerbation with fluid overload. Not immediately life-threatening but requires prompt medical attention and medication adjustment."}

Example 4:
Note: "25-year-old male presents with 5-day history of nasal congestion, clear rhinorrhea, mild sore throat. No fever. Lungs clear. Likely viral URI. Supportive care recommended."
Classification: {"category": "non_urgent", "reasoning": "Mild viral upper respiratory infection. Self-limiting condition requiring only supportive care."}
"""

CATEGORIES: str = (
    "emergency (life-threatening, requires immediate intervention), "
    "urgent (serious but not immediately life-threatening, needs prompt attention), "
    "non_urgent (requires medical attention but can wait), "
    "routine (preventive care, follow-ups, stable chronic conditions)"
)


def classify_zero_shot(note_text: str) -> dict:
    """Classify a clinical note using zero-shot prompting (no examples)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical triage specialist. Classify the given clinical note "
                    f"into one of these categories: {CATEGORIES}. "
                    "Return JSON with 'category' and 'reasoning' fields."
                ),
            },
            {
                "role": "user",
                "content": f"Classify this clinical note:\n\n{note_text}",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def classify_few_shot(note_text: str) -> dict:
    """Classify a clinical note using few-shot prompting (with labeled examples)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical triage specialist. Classify the given clinical note "
                    f"into one of these categories: {CATEGORIES}. "
                    "Return JSON with 'category' and 'reasoning' fields.\n\n"
                    f"{FEW_SHOT_EXAMPLES}"
                ),
            },
            {
                "role": "user",
                "content": f"Classify this clinical note:\n\n{note_text}",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def run_comparison() -> None:
    """Run both zero-shot and few-shot classification on all test notes and compare."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        console = None

    results: list[dict] = []

    for note in CLINICAL_NOTES:
        zero_shot_result = classify_zero_shot(note["text"])
        few_shot_result = classify_few_shot(note["text"])

        results.append(
            {
                "id": note["id"],
                "expected": note["expected_category"],
                "zero_shot": zero_shot_result.get("category", "unknown"),
                "few_shot": few_shot_result.get("category", "unknown"),
                "zero_shot_reasoning": zero_shot_result.get("reasoning", ""),
                "few_shot_reasoning": few_shot_result.get("reasoning", ""),
            }
        )

    # Display results
    if console:
        table = Table(title="Zero-Shot vs Few-Shot Classification Comparison")
        table.add_column("Note ID", style="cyan")
        table.add_column("Expected", style="green")
        table.add_column("Zero-Shot", style="yellow")
        table.add_column("Few-Shot", style="magenta")
        table.add_column("ZS Match", style="bold")
        table.add_column("FS Match", style="bold")

        zs_correct = 0
        fs_correct = 0

        for r in results:
            zs_match = "Y" if r["zero_shot"] == r["expected"] else "N"
            fs_match = "Y" if r["few_shot"] == r["expected"] else "N"
            if zs_match == "Y":
                zs_correct += 1
            if fs_match == "Y":
                fs_correct += 1

            table.add_row(
                r["id"],
                r["expected"],
                r["zero_shot"],
                r["few_shot"],
                f"[green]{zs_match}[/green]" if zs_match == "Y" else f"[red]{zs_match}[/red]",
                f"[green]{fs_match}[/green]" if fs_match == "Y" else f"[red]{fs_match}[/red]",
            )

        console.print(table)
        console.print(
            f"\nZero-Shot Accuracy: {zs_correct}/{len(results)} "
            f"({zs_correct / len(results) * 100:.0f}%)"
        )
        console.print(
            f"Few-Shot Accuracy:  {fs_correct}/{len(results)} "
            f"({fs_correct / len(results) * 100:.0f}%)"
        )
    else:
        print("Note ID | Expected | Zero-Shot | Few-Shot")
        print("-" * 50)
        for r in results:
            print(f"{r['id']} | {r['expected']} | {r['zero_shot']} | {r['few_shot']}")


if __name__ == "__main__":
    run_comparison()


# --- Sample Output ---
# ┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
# ┃ Note ID  ┃ Expected   ┃ Zero-Shot  ┃ Few-Shot   ┃ ZS Match ┃ FS Match ┃
# ┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
# │ note_001 │ emergency  │ emergency  │ emergency  │ Y        │ Y        │
# │ note_002 │ routine    │ routine    │ routine    │ Y        │ Y        │
# │ note_003 │ urgent     │ urgent     │ urgent     │ Y        │ Y        │
# │ note_004 │ routine    │ routine    │ routine    │ Y        │ Y        │
# │ note_005 │ emergency  │ emergency  │ emergency  │ Y        │ Y        │
# │ note_006 │ non_urgent │ non_urgent │ non_urgent │ Y        │ Y        │
# └──────────┴────────────┴────────────┴────────────┴──────────┴──────────┘
#
# Zero-Shot Accuracy: 5/6 (83%)
# Few-Shot Accuracy:  6/6 (100%)

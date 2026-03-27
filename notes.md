# Prompt Engineering — Study Notes

## Core Principles

### 1. Write Clear, Specific Instructions

The model cannot read your mind. Longer, more detailed prompts almost always outperform short ones
for complex tasks. "Specific" does not mean "short."

**Tactic: Use delimiters to clearly separate input from instructions**
```
Classify the following clinical note delimited by triple backticks.

Clinical note: ```{note_text}```
```

**Tactic: Request structured output**
```
Return your classification as a JSON object with the following fields:
- urgency_level: integer 1-5
- primary_complaint: string
- icd10_code: string (e.g., "R07.9")
```

**Tactic: Provide examples (few-shot)**
```
Here are examples of correctly classified notes:

Note: "Patient presents with acute chest pain radiating to left arm, diaphoresis."
Classification: {"urgency_level": 5, "primary_complaint": "acute chest pain", "icd10_code": "R07.9"}

Note: "Follow-up visit for well-controlled type 2 diabetes. A1C 6.8%."
Classification: {"urgency_level": 1, "primary_complaint": "diabetes follow-up", "icd10_code": "E11.65"}
```

**Tactic: Specify the steps required**
```
Step 1: Identify the chief complaint from the note.
Step 2: List all symptoms and findings mentioned.
Step 3: Determine urgency based on acuity criteria.
Step 4: Assign the most specific ICD-10 code.
Step 5: Return structured JSON.
```

### 2. Give the Model Time to Think (Chain-of-Thought)

For clinical reasoning, asking the model to work through its logic before giving a final answer
dramatically improves accuracy.

**Pattern: Explicit reasoning request**
```
Before providing your final classification, reason through the following:
1. What symptoms are present?
2. What is the most likely diagnosis?
3. What red flags (if any) are present?
4. Based on these factors, what urgency level is appropriate?

Show your reasoning, then provide the final JSON classification.
```

**Why this matters in healthcare:** If the model classifies a chest pain case as urgency 2 but
the chain-of-thought reveals it considered "radiating to left arm" and "diaphoresis," we can
verify its reasoning. If it missed those, we catch the error.

### 3. System Prompt Design

The system prompt sets the model's role, capabilities, constraints, and output format for the
entire conversation.

**Components of a good clinical system prompt:**
1. **Role definition** — "You are a clinical NLP specialist..."
2. **Task description** — "Your job is to classify clinical notes..."
3. **Constraints** — "Never provide treatment recommendations. Flag uncertainty."
4. **Output format** — "Always return valid JSON matching this schema..."
5. **Safety rails** — "If the note suggests imminent danger, always classify as urgency 5."
6. **Examples** — Embedded few-shot examples for edge cases.

### 4. Temperature and Sampling

- **Temperature 0:** Deterministic. Use for classification, extraction, coding.
- **Temperature 0.3-0.7:** Slight variation. Use for clinical summaries where wording flexibility helps.
- **Temperature 1.0+:** Creative. Never use for clinical tasks.

For all healthcare classification tasks, I use temperature=0.

### 5. Iterative Prompt Development

Prompt engineering is iterative, not one-shot. My workflow:
1. Start with a simple prompt.
2. Test against 5-10 synthetic cases.
3. Identify failure modes.
4. Add constraints, examples, or reasoning steps to address failures.
5. Retest. Repeat.

This is very similar to test-driven development — and I document prompt versions like code versions.

---

## Prompt Patterns Catalog

### Pattern 1: Role + Task + Format
```
SYSTEM: You are a medical coding specialist with expertise in ICD-10-CM.
Given a clinical note, extract the primary diagnosis and assign the most
specific ICD-10 code. Return JSON: {"diagnosis": "...", "icd10": "...", "confidence": 0.0-1.0}
```

### Pattern 2: Few-Shot with Edge Cases
Include 2-3 typical examples plus 1 edge case to show the model how to handle ambiguity.

### Pattern 3: Chain-of-Thought + Final Answer
```
Think through the clinical presentation step by step.
After your reasoning, provide ONLY the final JSON on the last line.
```

### Pattern 4: Guardrails
```
IMPORTANT CONSTRAINTS:
- If you are unsure about a classification, set confidence below 0.5 and add
  "needs_review": true to the output.
- Never hallucinate ICD-10 codes. Only use codes you are certain exist.
- If the note contains insufficient information, return "insufficient_data": true.
```

### Pattern 5: Output Validation Instructions
```
Before returning your response, verify:
1. The JSON is valid (no trailing commas, proper quoting).
2. The ICD-10 code matches the format [A-Z][0-9]{2}.[0-9]{1,4}.
3. Urgency level is an integer between 1 and 5.
```

---

## Key Insights for Healthcare AI

1. **Prompt injection is a real risk** in clinical systems. If clinical notes contain text that
   looks like instructions ("ignore previous instructions and..."), the model might follow them.
   Delimiters and input sanitization are essential.

2. **Few-shot examples are your training data.** In traditional ML, you need thousands of labeled
   examples. With LLMs, 3-5 well-chosen examples in the prompt can achieve comparable accuracy
   for classification tasks.

3. **Chain-of-thought is auditable reasoning.** In healthcare, you need to explain why a decision
   was made. CoT prompting gives you that audit trail for free.

4. **Structured output is non-negotiable.** Downstream systems (EHRs, FHIR APIs, dashboards) need
   structured data. JSON output formatting in prompts is how you get there.

5. **Prompt versioning = model versioning.** When you change the prompt, you change the model's
   behavior. Version your prompts like you version code.

"""Generate synthetic telecoms complaints using OpenAI (async, parallel)."""

import argparse
import asyncio
import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from prompts import (
    EMOTION_DEFINITIONS,
    SYSTEM_PROMPTS,
    URGENCY_DEFINITIONS,
)
from taxonomy import _build_grid, build_assignments

load_dotenv()

# ---------------------------------------------------------------------------
# Emotion-level prompt reinforcements
# ---------------------------------------------------------------------------
EMOTION_INSTRUCTIONS = {
    "Low": (
        "\nCRITICAL TONE INSTRUCTION: This is a LOW emotion complaint. "
        "Express this through word choice and sentence structure — the "
        "writing should feel composed and matter-of-fact. Do NOT use "
        "exclamation marks, capitalisation for emphasis, or words that "
        "announce emotion (e.g. 'frustrated', 'appalled', 'furious'). "
        "A single understated note of disappointment is acceptable, but "
        "the overall tone must read as measured and controlled.\n"
    ),
    "Medium": (
        "\nCRITICAL TONE INSTRUCTION: This is a MEDIUM emotion complaint. "
        "Express dissatisfaction through tone and word choice — the reader "
        "should sense frustration, but it need not be announced explicitly. "
        "Avoid over-relying on exclamation marks or capitalisation. "
        "The emotional register should feel genuine: some medium-emotion "
        "customers are quietly firm, others are more openly irritated. "
        "Capture this ambiguity rather than landing squarely in the middle.\n"
    ),
    "High": (
        "\nCRITICAL TONE INSTRUCTION: This is a HIGH emotion complaint. "
        "Express this through word choice and sentence structure — do NOT "
        "rely on exclamation marks, capitalisation, or words that label "
        "the emotion ('livid', 'furious', 'appalling'). The intensity "
        "should be felt, not announced. High emotion can manifest as cold "
        "controlled anger (short, clipped sentences, cutting observations, "
        "formal ultimatums) or as explicit distress — choose whichever "
        "fits the customer profile and writing style.\n"
    ),
}


def _build_user_prompt(cell_assignments: list[dict]) -> str:
    """Build a user prompt that requests multiple complaints for one cell."""
    n = len(cell_assignments)
    urgency = cell_assignments[0]["urgency"]
    emotion = cell_assignments[0]["emotion"]

    # Detect divergent urgency/emotion combos and add a clarification
    level_order = {"Low": 0, "Medium": 1, "High": 2}
    divergence_note = ""
    if abs(level_order[urgency] - level_order[emotion]) >= 2:
        if level_order[urgency] > level_order[emotion]:
            divergence_note = (
                "\nIMPORTANT: The urgency and emotion levels are intentionally "
                "different. The customer is describing a severe, high-impact issue "
                "but writing in a calm, composed, factual manner — they are not "
                "emotional despite the seriousness. Do NOT let the severity of the "
                "issue bleed into the tone. Keep the writing measured and neutral.\n\n"
            )
        else:
            divergence_note = (
                "\nIMPORTANT: The urgency and emotion levels are intentionally "
                "different. The customer is highly emotional and upset, but the "
                "underlying issue is relatively minor. The emotional intensity "
                "is genuine — some customers feel strongly about issues that "
                "others might consider small. Write the complaint with authentic "
                "high emotion while keeping the actual problem minor in scope.\n\n"
            )

    header = (
        f"Generate exactly {n} distinct customer complaints. "
        f"All complaints share the same urgency and emotion level:\n\n"
        f"Urgency: {urgency} — {URGENCY_DEFINITIONS[urgency]}\n"
        f"Emotion: {emotion} — {EMOTION_DEFINITIONS[emotion]}\n"
        f"{EMOTION_INSTRUCTIONS[emotion]}\n"
        f"{divergence_note}"
        f"Each complaint has a specific scenario, writing style, customer "
        f"profile, and complaint history listed below. Make every complaint "
        f"sound unique — vary phrasing, length, and details. Each complaint "
        f"must use completely different phrasing, vocabulary, sentence "
        f"structure, and level of detail from every other complaint. Do not "
        f"reuse phrases across complaints.\n\n"
        f"For realism, include where appropriate:\n"
        f"- Specific dates of incidents and prior contacts\n"
        f"- Names of staff spoken to previously\n"
        f"- Prior contact history matching the specified history depth\n\n"
    )

    items = []
    for i, a in enumerate(cell_assignments, 1):
        items.append(
            f"Complaint {i}:\n"
            f"  Scenario: {a['scenario']}\n"
            f"  Style: {a['style']}\n"
            f"  Customer profile: {a['profile']}\n"
            f"  Complaint history: {a['history']}\n"
        )

    footer = (
        '\nReturn a JSON object with a single key "complaints" containing a '
        f"list of exactly {n} strings, each being one complaint text. "
        "Output only the complaint text in each string — no labels, metadata, "
        "or preamble."
    )

    return header + "\n".join(items) + footer


BATCH_SIZE = 15
MAX_RETRIES = 2
MAX_CONCURRENT = 10

# MODEL = "gpt-5-mini"
MODEL = "gpt-4o-mini"


async def _generate_batch(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    batch_assignments: list[dict],
    batch_label: str,
) -> list[str]:
    """Call the API for a batch of assignments, with retry on count mismatch."""
    expected = len(batch_assignments)

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 2):
            user_prompt = _build_user_prompt(batch_assignments)
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)
            complaints = data.get("complaints", [])

            if len(complaints) >= expected:
                print(f"  {batch_label}: {expected} complaints OK")
                return complaints[:expected]

            print(f"  {batch_label}: attempt {attempt}, expected {expected}, "
                  f"got {len(complaints)}."
                  f"{' Retrying...' if attempt <= MAX_RETRIES else ''}")

    # All retries exhausted — pad with placeholders
    while len(complaints) < expected:
        complaints.append("[GENERATION FAILED — re-run needed]")
    return complaints[:expected]


async def generate_all(total: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate `total` complaints and return a DataFrame."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    assignments = build_assignments(total=total, seed=seed)
    grid = _build_grid(total)

    # Group assignments by cell
    cells: dict[tuple[str, str], list[dict]] = {}
    for a in assignments:
        key = (a["urgency"], a["emotion"])
        cells.setdefault(key, []).append(a)

    # Build all tasks upfront
    tasks: list[tuple[list[dict], asyncio.Task]] = []

    for cell_idx, (urgency, emotion, _count) in enumerate(grid):
        cell_assignments = cells[(urgency, emotion)]
        system_prompt_idx = cell_assignments[0]["system_prompt_idx"]
        system_prompt = SYSTEM_PROMPTS[system_prompt_idx]

        batches = [
            cell_assignments[i:i + BATCH_SIZE]
            for i in range(0, len(cell_assignments), BATCH_SIZE)
        ]

        cell_label = (f"Cell {cell_idx + 1}/9: "
                      f"{urgency} urg x {emotion} emo")
        print(f"Queuing {cell_label} "
              f"({len(cell_assignments)} complaints, {len(batches)} batch(es))")

        for batch_idx, batch in enumerate(batches):
            label = f"{cell_label}, batch {batch_idx + 1}/{len(batches)}"
            task = asyncio.create_task(
                _generate_batch(client, semaphore, system_prompt, batch, label)
            )
            tasks.append((batch, task))

    print(f"\nLaunching {len(tasks)} batches with up to "
          f"{MAX_CONCURRENT} concurrent requests...\n")
    start = time.time()

    # Await all tasks
    results = await asyncio.gather(*(t for _, t in tasks))

    elapsed = time.time() - start
    print(f"\nAll batches complete in {elapsed:.1f}s")

    # Assemble rows
    all_rows: list[dict] = []
    complaint_id = 1
    for (batch, _task), complaints in zip(tasks, results):
        for a, text in zip(batch, complaints):
            all_rows.append({
                "id": complaint_id,
                "complaint_text": text.strip(),
                "intended_urgency": a["urgency"],
                "intended_emotion": a["emotion"],
                "scenario": a["scenario"],
                "style": a["style"],
                "profile": a["profile"],
                "history": a["history"],
            })
            complaint_id += 1

    df = pd.DataFrame(all_rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic telecoms complaints.")
    parser.add_argument("--total", type=int, default=5000,
                        help="Total number of complaints to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    df = asyncio.run(generate_all(total=args.total, seed=args.seed))

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "telecoms_complaints.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} complaints to {output_path}")

    # Summary
    print("\n--- Distribution Summary ---")
    for col in ["scenario", "style", "profile", "history"]:
        print(f"\n{col.title()} counts:")
        print(df[col].value_counts().to_string())
    print("\nUrgency x Emotion counts:")
    print(df.groupby(["intended_urgency", "intended_emotion"]).size().to_string())


if __name__ == "__main__":
    main()

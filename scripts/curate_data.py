#!/usr/bin/env python
"""Curate training data for ETM v5.

Loads hendrycks_math (MATH dataset), filters for Level 4-5 (hardest problems),
reformats to match the AIME eval prompt, and saves as jsonl.

Output format matches eval prompt:
  Problem: {problem}

  Please reason step by step, and put your final answer within \boxed{}.
  Solution: {solution}
"""
import json
import re
import random
from pathlib import Path

from datasets import load_dataset

OUT_PATH = "/root/etm/data/math_train_v5.jsonl"
SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
MIN_LEVEL = 4  # Level 4 and 5 only (hardest)

PROMPT_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "Solution: {solution}"
)


def has_boxed(solution):
    """Check if solution contains \boxed{...}."""
    return bool(re.search(r"\\boxed\{", solution))


def extract_problem_and_solution(ex):
    """Extract problem and solution from a hendrycks_math example."""
    problem = ex["problem"].strip()
    solution = ex["solution"].strip()
    return problem, solution


def parse_level(level_str):
    """Parse 'Level 4' -> 4."""
    match = re.search(r"Level (\d+)", level_str)
    return int(match.group(1)) if match else 0


def main():
    random.seed(42)
    all_samples = []
    level_counts = {4: 0, 5: 0}
    subject_counts = {}

    for subject in SUBJECTS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        except Exception as e:
            print(f"  skip {subject}: {e}")
            continue

        subject_count = 0
        for ex in ds:
            level = parse_level(ex.get("level", ""))
            if level < MIN_LEVEL:
                continue

            problem, solution = extract_problem_and_solution(ex)
            if not has_boxed(solution):
                continue
            if len(problem) < 20 or len(solution) < 50:
                continue

            text = PROMPT_TEMPLATE.format(problem=problem, solution=solution)
            all_samples.append({"text": text, "level": level, "type": ex.get("type", subject)})
            level_counts[level] = level_counts.get(level, 0) + 1
            subject_count += 1

        subject_counts[subject] = subject_count
        print(f"  {subject}: {subject_count} samples (Level {MIN_LEVEL}+)")

    # Shuffle
    random.shuffle(all_samples)

    # Save
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for sample in all_samples:
            f.write(json.dumps({"text": sample["text"]}) + "\n")

    print(f"\n=== CURATION COMPLETE ===")
    print(f"Total samples: {len(all_samples)}")
    print(f"Level distribution: {level_counts}")
    print(f"Subject distribution: {subject_counts}")
    print(f"Saved to: {OUT_PATH}")

    # Show a sample
    print(f"\n=== SAMPLE ===")
    print(all_samples[0]["text"][:500])

    # Also save a version with level info for analysis
    meta_path = OUT_PATH.replace(".jsonl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "total": len(all_samples),
            "levels": level_counts,
            "subjects": subject_counts,
        }, f, indent=2)
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()

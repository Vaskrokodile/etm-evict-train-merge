#!/usr/bin/env python
"""AIME 2025 evaluation harness for Qwen2.5-Math-7B via vLLM OpenAI API.

Scores pass@1 (greedy) and maj@k (majority vote over k samples).
Answers are integers 0-999. Extracts from \\boxed{} or last integer.

Usage:
  python eval_aime.py --host localhost --port 8000 --samples 8
  python eval_aime.py --data data/aime_2025.jsonl --samples 8 --temp 0.8 --out results/baseline.json
"""
import argparse
import json
import re
import os
from collections import Counter
from pathlib import Path

import requests

PROMPT_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "Solution: Let"
)


def extract_answer(text):
    """Extract integer answer from model output. Priority: \\boxed{}, then last integer."""
    # Try \boxed{...} - handle nested braces
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        # take last boxed
        val = boxed[-1].strip()
        # extract integer from it
        nums = re.findall(r"-?\d+", val)
        if nums:
            return int(nums[-1])
    # fallback: last integer in text
    nums = re.findall(r"(?<!\d)(\d{1,3})(?!\d)", text[-500:])
    if nums:
        return int(nums[-1])
    return None


def call_model(host, port, model, prompt, max_tokens=2048, temp=0.0, top_p=0.95, stop=None):
    """Call vLLM completions API."""
    resp = requests.post(
        f"http://{host}:{port}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "stop": stop or ["Problem:", "\n\n\n"],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def evaluate(data_path, host, port, model, samples, temp, max_tokens, out_path):
    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))

    print(f"Loaded {len(problems)} problems from {data_path}")
    print(f"Model: {model}, samples: {samples}, temp: {temp}")
    print("=" * 60)

    results = []
    correct_pass1 = 0
    correct_maj = 0

    for i, prob in enumerate(problems):
        gold = int(prob["answer"])
        prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])

        # pass@1: greedy
        gen0 = call_model(host, port, model, prompt, max_tokens, temp=0.0)
        ans0 = extract_answer(gen0)
        p1_correct = (ans0 == gold)
        if p1_correct:
            correct_pass1 += 1

        # maj@k: k samples at temp
        answers = [ans0]
        if samples > 1 and temp > 0:
            for s in range(samples - 1):
                gen = call_model(host, port, model, prompt, max_tokens, temp=temp, top_p=0.95)
                answers.append(extract_answer(gen))

        # majority vote (ignore None)
        valid = [a for a in answers if a is not None]
        if valid:
            maj_ans = Counter(valid).most_common(1)[0][0]
        else:
            maj_ans = None
        maj_correct = (maj_ans == gold)
        if maj_correct:
            correct_maj += 1

        status_p1 = "OK" if p1_correct else "MISS"
        status_maj = "OK" if maj_correct else "MISS"
        print(f"[{i+1:2d}/{len(problems)}] gold={gold:3d} p1={ans0}({status_p1}) "
              f"maj={maj_ans}({status_maj}) valid_samples={len(valid)}/{samples}")

        results.append({
            "idx": prob.get("problem_idx", i),
            "gold": gold,
            "pass1_answer": ans0,
            "pass1_correct": p1_correct,
            "maj_answer": maj_ans,
            "maj_correct": maj_correct,
            "sample_answers": answers,
            "problem_type": prob.get("problem_type", []),
        })

    n = len(problems)
    p1_acc = correct_pass1 / n
    maj_acc = correct_maj / n

    summary = {
        "model": model,
        "n_problems": n,
        "samples": samples,
        "temp": temp,
        "pass1_accuracy": p1_acc,
        "pass1_correct": correct_pass1,
        "maj_accuracy": maj_acc,
        "maj_correct": correct_maj,
        "results": results,
    }

    print("=" * 60)
    print(f"pass@1: {correct_pass1}/{n} = {p1_acc:.1%}")
    print(f"maj@{samples}: {correct_maj}/{n} = {maj_acc:.1%}")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {out_path}")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aime_2025.jsonl")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="Qwen2.5-Math-7B")
    ap.add_argument("--samples", type=int, default=8, help="k for maj@k")
    ap.add_argument("--temp", type=float, default=0.8, help="sampling temperature for maj@k")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--out", default="results/baseline_aime2025.json")
    args = ap.parse_args()
    evaluate(args.data, args.host, args.port, args.model, args.samples, args.temp, args.max_tokens, args.out)

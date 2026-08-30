# ETM: Evict-Train-Merge

A novel parameter-efficient fine-tuning method that trains LoRA adapters **with the base model evicted from memory**. Instead of standard forward+backward through the full model, ETM caches the gradient signal (x, grad_h) from a single forward+backward pass, then trains only the small LoRA parameters against that cached gradient — the 7B base model doesn't exist during training.

## Why This Matters

Standard LoRA fine-tuning requires the full base model in memory during training. ETM decouples the gradient computation from the parameter update:

1. **Cache phase**: Load base model, run forward+backward on training data, capture (input activations x, upstream gradients grad_h) for each LoRA target module. Evict the base model.
2. **Train phase**: Train only LoRA A/B matrices against the cached gradient. The base model is gone — only ~40M params and the cache exist in memory.
3. **Merge phase**: Reload base, inject trained LoRA weights, merge.

This means you can train on a GPU that can't hold the full model — you only need enough VRAM for the cache + LoRA params during training. The base model is only needed briefly for caching and merging.

## Method Evolution

### v3 — Dot Product Loss (diverged)
Cached (x, grad_h) once, trained with unbounded dot product loss `L = (scaling * x @ A.T @ B.T * grad_h).sum()`. Loss diverged to -1636 because the dot product grows without bound as A and B drift.

### v4 — Cosine Similarity Loss + Breathing
- **Bounded cosine loss**: `L = cos(u, grad_h)` — controls direction only, bounded in [-1, 1]
- **Breathing cycle**: Every K steps, reload base, refresh cache with current trained params, evict again. Keeps gradient from going stale.
- **DoRA adapters**: rank=24, alpha=48, targets q/k/v/o_proj
- Result: 0/30 on AIME 2025 (degraded from baseline 1/30)

### v5 — MSE Loss + Higher Rank
- **MSE loss**: `L = ||u + η*grad_h||²` — controls direction AND magnitude
- **Higher rank**: r=64 (40M params vs 15M)
- **Calibrated η**: target ||u|| = 10% of ||x||
- **bf16 merge**: Fixed v4's broken 4-bit merge
- Result: 0/30 (η too large → repetition loops)

### v5b — Gentle MSE + Breathing (current best)
- **Gentle η**: target ||u|| = 1% of ||x|| (10x smaller than v5)
- **Periodic breathing**: Every 500 steps, refresh cache
- **Regular LoRA** (no DoRA): reliable merge
- Result: **1/30 = 3.3%** (matches baseline, no degradation)

## Results

| Run | Score | Notes |
|-----|-------|-------|
| Baseline Qwen2.5-Math-7B | 1/30 = 3.3% | Untrained |
| ETM v3 (dot product) | — | Loss diverged |
| ETM v4 (cosine, r=24) | 0/30 = 0% | Degraded |
| ETM v5 (MSE, r=64, η=10%) | 0/30 = 0% | η too large |
| ETM v5b (MSE, r=64, η=1%, breathing) | 1/30 = 3.3% | Matches baseline |

## Key Insights

1. **The method works mechanically** — the model stays coherent after training 40M params with the 7B base fully evicted. This is the proof of concept.
2. **Cosine loss is too weak** — only controls direction, not magnitude. The LoRA delta stays microscopic.
3. **MSE loss with large η is too aggressive** — corrupts the model (repetition loops).
4. **Gentle η (1% perturbation) + breathing is the sweet spot** — model doesn't degrade, matches baseline.
5. **Next step**: More breathing cycles + more steps to push past baseline. The OOM during breathing (container RAM limit) is the current blocker.

## File Structure

```
src/
  etm_v3.py            — v3: dot product loss (diverged)
  etm_v4_breathing.py  — v4: cosine loss + breathing cycle
  etm_v5.py            — v5: MSE loss, higher rank, GPU cache
  etm_v5b.py           — v5b: gentle MSE + periodic breathing (best)
  merge_elastic.py     — Elastic absorption merge (TIES + KL + EWC)

eval/
  eval_aime.py         — AIME 2025 evaluation harness (pass@1, maj@k)

scripts/
  curate_data.py       — Filter MATH dataset for Level 4-5, format to match eval
  merge_v5.py          — Standalone bf16 merge from checkpoint
  merge_v5b.py         — Standalone bf16 merge for v5b checkpoint
  remerge_bf16.py      — Re-merge DoRA adapter into bf16 base (fixes 4-bit corruption)
  sanity_check.py      — Quick model sanity check (7+5=?)
  debug_eval.py        — Debug eval prompt/format issues
  run_vllm.sh           — vLLM server launch script

data/
  aime_2025.jsonl      — 30 AIME 2025 problems
  math_train_v5_meta.json — Curated data metadata (Level 4-5 counts)

results/
  etm_v4_pass1.json     — v4 wrong-sign eval
  etm_v4_fixed_pass1.json — v4 fixed-sign eval (0/30)
  etm_v5_pass1.json     — v5 step 8000 eval (0/30)
  etm_v5_s2k_pass1.json — v5 step 2000 eval (0/30)
  etm_v5b_pass1.json    — v5b step 1000 eval (1/30)

logs/
  etm_v4_fixed.log     — v4 training log
  etm_v5.log           — v5 training log
  etm_v5b.log          — v5b training log
  remerge.log          — bf16 re-merge log

outputs/
  etm_v5b/checkpoint_step1000.pt — Best checkpoint (1/30 result)
```

## Base Model

Qwen/Qwen2.5-Math-7B — a 7B parameter math-specialized language model.

## Training Data

Hendrycks MATH dataset (EleutherAI/hendrycks_math), filtered to Level 4-5 (hardest problems), reformatted to match the AIME eval prompt. 3,993 curated samples across 7 subjects.

## Hardware

- GPU: NVIDIA A100 80GB
- Container RAM: 189 GB (cgroup limited)
- Pod: gpu.ai (frp.gpu.ai:10099)

## License

Private research project. All rights reserved.

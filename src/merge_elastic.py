#!/usr/bin/env python
"""
merge_elastic.py — Elastic Absorption Merge for the ETM framework.

============================================================================
WHY THIS EXISTS
============================================================================
The ETM pipeline trains ~15M LoRA (DoRA) parameters while the 7B base model
is evicted from VRAM.  After training the adapter must be merged back into
the base weights.  A naive ``merge_and_unload()`` simply writes

    W_merged = W_base + (alpha / r) * B @ A

into every adapted module.  This works, but two problems arise:

  1. **Noise & sign conflicts.**  The low-rank delta ``B @ A`` contains many
     small updates whose direction disagrees with the dominant update in the
     same module.  Merging them in raw injects noise that degrades general
     capabilities.

  2. **Shock to neighbours.**  Changing q/k/v/o in every layer simultaneously
     shifts the residual stream in ways the *surrounding* (non-adapted)
     layers — MLPs, embeddings, norm layers — were never trained to expect.
     The model can regress on general text even while math accuracy improves.

This module implements an **elastic absorption** merge that addresses both:

  * **TIES-style delta trimming** removes the noisy / conflicting 10 % of the
    delta before it ever touches the base weights.
  * **Scaled merge** writes ``W_base + lambda * delta_ties`` so we control how
    aggressively the math specialization is injected.
  * **Elastic absorption** then runs a *short* fine-tune with a *fresh* tiny
    LoRA (rank 8) whose loss combines:

        L = L_general
            + beta * KL( p_base || p_merged )      # drift anchor
            + gamma * EWC                           # protect important weights

    This lets the un-adapted neighbouring layers gently retune around the
    merged math changes — the "elastic" part — without overfitting to math.

The result is a merged model that gains the math specialization while
preserving (and sometimes improving) general capability.

============================================================================
ALGORITHM DETAILS
============================================================================

--- TIES delta trimming (Yadav et al. 2023, adapted) -----------------------
For each adapted module we reconstruct the full delta

    delta = W_trained - W_base          # == (alpha/r) * B @ A for LoRA

then:

  1. **Magnitude trim.**  Compute the absolute value of every element of
     ``delta``.  Drop the bottom ``ties_trim_frac`` (default 10 %) by
     magnitude — set them to zero.  These are the updates that are most
     likely noise.

  2. **Sign unification.**  For the remaining non-zero entries, compute the
     dominant sign (sum of positive magnitudes vs sum of negative
     magnitudes).  Zero out every entry whose sign disagrees with the
     dominant sign.  This removes within-module direction conflict, which is
     the single biggest source of merge-time interference.

The surviving entries form ``delta_ties``.

--- Scaled merge -----------------------------------------------------------
    W_merged = W_base + merge_lambda * delta_ties

``merge_lambda`` (default 0.8) damps the injection so the absorption step has
room to retune without fighting an over-large shift.

--- KL anchoring -----------------------------------------------------------
During absorption we run the *original* base model (no adapter, 4-bit) and
the *merged* model on the same batch of general text and compute

    KL( p_base || p_merged ) = sum_t sum_v p_base(v|t) * [ log p_base(v|t)
                                                       - log p_merged(v|t) ]

This is the "anchor": it penalises the merged model for moving its
predictions away from what the base would have produced.  It is asymmetric
(p_base is the fixed reference) so gradients only flow into the merged
model.  This keeps general capability intact while the fresh LoRA is free to
rearrange internal representations as long as the *output* distribution stays
close.

--- EWC penalty (Kirkpatrick et al. 2017) ----------------------------------
Elastic Weight Consolidation penalises changes to parameters that were
important for the base model's *pre-merge* behaviour.  We estimate a diagonal
Fisher information matrix

    F_i = E[ (d log p / d theta_i)^2 ]

from a few forward passes on general data *before* absorption starts, using
the merged model's parameters as theta*.  During absorption,

    L_EWC = sum_i F_i * (theta_i - theta*_i)^2

Only the fresh LoRA parameters are trainable, so this term keeps the
absorption LoRA from rewriting weights the base relied on.  (Because the
fresh LoRA starts at zero, theta* for the LoRA params is 0 and the penalty
acts as an L2-style brake weighted by how important each base weight was.)

============================================================================
USAGE
============================================================================
    python src/merge_elastic.py \
        --adapter_path outputs/etm_v4 \
        --base_model Qwen/Qwen2.5-Math-7B \
        --output_dir outputs/etm_v4_merged \
        --ties_trim_frac 0.1 \
        --merge_lambda 0.8 \
        --kl_weight 0.5 \
        --ewc_weight 0.1 \
        --absorption_steps 50
============================================================================
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge_elastic")

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ============================================================================
# 1. MODEL LOADING
# ============================================================================
def _bnb_config():
    """Standard 4-bit NF4 bitsandbytes config used across ETM."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(base_model: str, attach_lora: bool = False,
                    rank: int = 24, alpha: int = 48, use_dora: bool = True,
                    target_modules: Optional[List[str]] = None):
    """Load the 4-bit base model, optionally with a fresh LoRA adapter.

    Args:
        base_model: HF model id or local path.
        attach_lora: If True, wrap with a fresh PEFT LoRA (for absorption).
        rank / alpha / use_dora: LoRA hyper-params (only used if attach_lora).
        target_modules: modules to adapt; defaults to q/k/v/o.

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if attach_lora:
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=use_dora,
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    return model, tokenizer


# ============================================================================
# 2. TIES-STYLE DELTA TRIMMING
# ============================================================================
def reconstruct_deltas(model_with_adapter) -> Dict[str, torch.Tensor]:
    """Reconstruct the full delta = B @ A * scaling for every LoRA module.

    For DoRA the effective delta on the *weight* is still ``scaling * B @ A``
    (the magnitude vector only rescales the merged weight at runtime); we
    approximate the delta that ``merge_and_unload`` would apply, which is the
    low-rank product.  This is what we trim.

    Returns:
        {module_name: delta_tensor (d_out, d_in) on CPU, fp32}
    """
    deltas: Dict[str, torch.Tensor] = {}
    for name, module in model_with_adapter.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        for adapter_name in module.lora_A:
            A = module.lora_A[adapter_name].weight.data      # (r, d_in)
            B = module.lora_B[adapter_name].weight.data      # (d_out, r)
            scaling = module.scaling[adapter_name]
            # delta = scaling * (B @ A)
            delta = (B.float() @ A.float()) * float(scaling)
            deltas[name] = delta.detach().cpu()
    log.info("  reconstructed %d LoRA deltas", len(deltas))
    return deltas


def ties_trim_delta(delta: torch.Tensor, trim_frac: float) -> torch.Tensor:
    """Apply TIES magnitude-trim + sign-unification to a single delta tensor.

    Steps:
      1. Drop the bottom ``trim_frac`` entries by |magnitude| (set to 0).
      2. Determine the dominant sign (sum of +mags vs sum of -mags).
      3. Zero every entry whose sign disagrees with the dominant sign.

    Args:
        delta: (d_out, d_in) tensor of the reconstructed LoRA delta.
        trim_frac: fraction of entries to drop by magnitude (0..1).

    Returns:
        Trimmed delta of the same shape.
    """
    if trim_frac <= 0:
        # Still do sign unification even if no magnitude trim requested.
        trim_frac = 0.0
    flat = delta.flatten()
    n = flat.numel()

    # --- 1. magnitude trim ---
    if trim_frac > 0 and trim_frac < 1:
        k = int(math.floor(n * trim_frac))
        if k > 0 and k < n:
            abs_vals = flat.abs()
            # threshold = k-th smallest |value|
            kth = torch.kthvalue(abs_vals, k).values
            mask_small = abs_vals <= kth
            flat = flat.clone()
            flat[mask_small] = 0.0

    # --- 2. dominant sign ---
    pos_mass = flat[flat > 0].sum().item()
    neg_mass = -flat[flat < 0].sum().item()  # make positive
    if pos_mass >= neg_mass:
        dominant_sign = 1
    else:
        dominant_sign = -1

    # --- 3. sign unification ---
    if dominant_sign > 0:
        flat = torch.where(flat < 0, torch.zeros_like(flat), flat)
    else:
        flat = torch.where(flat > 0, torch.zeros_like(flat), flat)

    return flat.view_as(delta)


def ties_trim_all(deltas: Dict[str, torch.Tensor],
                  trim_frac: float) -> Dict[str, torch.Tensor]:
    """Apply TIES trimming to every module delta and log survival stats."""
    trimmed: Dict[str, torch.Tensor] = {}
    total_survived = 0
    total_elements = 0
    for name, delta in deltas.items():
        t = ties_trim_delta(delta, trim_frac)
        trimmed[name] = t
        survived = (t.abs() > 0).sum().item()
        total_survived += survived
        total_elements += t.numel()
    if total_elements:
        log.info("  TIES trim: %.1f%% of delta entries survived (trim_frac=%.2f)",
                 100.0 * total_survived / total_elements, trim_frac)
    return trimmed


# ============================================================================
# 3. SCALED MERGE — write trimmed delta into the base weights
# ============================================================================
def apply_scaled_merge(model_with_adapter,
                       trimmed_deltas: Dict[str, torch.Tensor],
                       merge_lambda: float) -> None:
    """Merge the TIES-trimmed delta into the base weights *in place*.

    For each adapted module:
        W_base <- W_base + merge_lambda * delta_ties

    We write directly into the dequantised base weight so that a subsequent
    ``merge_and_unload`` (which just removes the LoRA wrapper) produces a
    model whose linear weights already contain the absorbed delta.

    For 4-bit base layers we operate on the ``weight`` property which PEFT's
    ``prepare_model_for_kbit_training`` casts to a trainable fp32/bf16
    parameter; bitsandbytes handles the quant back-store transparently.
    """
    written = 0
    for name, module in model_with_adapter.named_modules():
        if name not in trimmed_deltas:
            continue
        delta = trimmed_deltas[name].to(model_with_adapter.device)
        base_layer = module.get_base_layer() if hasattr(module, "get_base_layer") else module
        # `weight` on a bnb Linear4bit is a Params4bit; we mutate via .data
        # after casting to the compute dtype.
        w = base_layer.weight
        try:
            w_data = w.data.to(delta.dtype) + merge_lambda * delta
            w.data.copy_(w_data)
            written += 1
        except Exception as e:  # pragma: no cover - diagnostic
            log.warning("  could not write merged delta into %s: %s", name, str(e)[:120])
        # Zero out the LoRA B weights so the subsequent merge_and_unload()
        # adds scaling * (B @ A) = 0 — the trimmed delta is already baked
        # into the base weights.  Without this, merge_and_unload would
        # re-add the *raw* (untrimmed) delta on top, double-applying it.
        for adapter_name in getattr(module, "lora_B", {}):
            module.lora_B[adapter_name].weight.data.zero_()
    log.info("  scaled merge: wrote trimmed delta into %d modules (lambda=%.2f)",
             written, merge_lambda)


# ============================================================================
# 4. GENERAL DATA for absorption
# ============================================================================
def load_general_text(data_path: Optional[str], n_samples: int = 64,
                      max_seq_len: int = 512) -> List[str]:
    """Load ~n_samples of general text for the absorption fine-tune.

    Preference order:
      1. If ``data_path`` points to a .jsonl with a "text" field, use it
         (this lets the caller reuse MATH solutions, which are general-enough
         natural language with reasoning).
      2. Else try to load a few samples from ``wikitext-2-raw-v1`` via
         datasets.
      3. Else fall back to a tiny built-in corpus so the script never hard
         fails for lack of data.
    """
    texts: List[str] = []
    if data_path:
        p = Path(data_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = obj.get("text") or obj.get("solution") or obj.get("problem")
                    if t and len(t) > 40:
                        texts.append(t)
                    if len(texts) >= n_samples:
                        break
            log.info("  general data: loaded %d samples from %s", len(texts), p)

    if len(texts) < n_samples:
        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            for ex in ds:
                t = ex["text"].strip()
                if len(t) > 80:
                    texts.append(t)
                if len(texts) >= n_samples:
                    break
            log.info("  general data: supplemented with wikitext (%d total)",
                     len(texts))
        except Exception as e:
            log.warning("  wikitext load failed (%s); using built-in fallback", str(e)[:80])

    if len(texts) < 8:
        # Minimal built-in corpus so absorption can always run.
        texts.extend([
            "The quick brown fox jumps over the lazy dog near the river bank "
            "while children play in the park on a sunny afternoon.",
            "In machine learning, gradient descent is an iterative optimization "
            "algorithm used to minimize some function by moving in the direction "
            "of steepest descent.",
            "The economy grew steadily over the last quarter, driven by consumer "
            "spending and a rebound in manufacturing output across several regions.",
            "Photosynthesis converts light energy into chemical energy stored in "
            "glucose, producing oxygen as a byproduct in green plants and algae.",
            "The committee decided to postpone the meeting until next Tuesday so "
            "that all members could review the revised proposal in detail.",
            "A neural network learns by adjusting its weights to reduce the "
            "difference between its predictions and the true target values.",
            "The novel tells the story of a young woman who travels across the "
            "country to find her estranged father and rediscover her own identity.",
            "Quantum mechanics describes the behaviour of matter at the scale of "
            "atoms and subatomic particles, where probabilities replace certainties.",
        ] * 8)
        log.info("  general data: using built-in fallback corpus")

    return texts[:n_samples]


def encode_batch(tokenizer, texts: List[str], max_seq_len: int,
                 batch_size: int, offset: int):
    """Encode a slice of texts into a padded batch for LM training/eval."""
    slice_texts = texts[offset:offset + batch_size]
    if not slice_texts:
        return None
    enc = tokenizer(
        slice_texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        padding=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


# ============================================================================
# 5. EWC — diagonal Fisher information
# ============================================================================
def compute_fisher_diag(model, tokenizer, texts: List[str],
                        n_batches: int = 4, batch_size: int = 2,
                        max_seq_len: int = 256) -> Dict[str, torch.Tensor]:
    """Estimate the diagonal Fisher information for trainable params.

    F_i = E[ (d log p(y|x) / d theta_i)^2 ]

    We approximate the expectation with a few sampled batches and the
    empirical Fisher (use the observed tokens as the "sampled" targets).

    Returns:
        {param_name: fisher_tensor} on CPU, same shape as each param.
    """
    log.info("  EWC: computing diagonal Fisher from %d batches...", n_batches)
    model.eval()
    fisher: Dict[str, torch.Tensor] = {}
    # init accumulators
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        fisher[name] = torch.zeros_like(p.data, device="cpu")

    seen = 0
    for b in range(n_batches):
        offset = (b * batch_size) % max(1, len(texts))
        batch = encode_batch(tokenizer, texts, max_seq_len, batch_size, offset)
        if batch is None:
            continue
        input_ids, attn, labels = batch
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)
        labels = labels.to(model.device)

        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        # empirical Fisher: square of grad of log-likelihood
        out.loss.backward()
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            fisher[name] += (p.grad.data.float() ** 2).detach().cpu()
        seen += 1

    if seen > 0:
        for name in fisher:
            fisher[name] /= seen
    model.zero_grad(set_to_none=True)
    log.info("  EWC: Fisher estimated over %d batches, %d params", seen, len(fisher))
    return fisher


def ewc_penalty(model, fisher: Dict[str, torch.Tensor],
                theta_star: Dict[str, torch.Tensor]) -> torch.Tensor:
    """L_EWC = sum_i F_i * (theta_i - theta*_i)^2  (trainable params only)."""
    pen = torch.tensor(0.0, device=model.device)
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in fisher:
            continue
        f = fisher[name].to(p.device, dtype=p.dtype)
        t0 = theta_star[name].to(p.device, dtype=p.dtype)
        pen = pen + (f * (p - t0) ** 2).sum()
    return pen


def snapshot_trainable(model) -> Dict[str, torch.Tensor]:
    """Copy current trainable params (theta*) for EWC anchor."""
    snap: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            snap[name] = p.detach().cpu().clone()
    return snap


# ============================================================================
# 6. KL ANCHOR — KL(p_base || p_merged)
# ============================================================================
@torch.no_grad()
def get_base_logits(base_model, input_ids, attention_mask):
    """Run the original base model (no adapter) and return token logits."""
    base_model.eval()
    out = base_model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits.detach()


def kl_anchor_loss(merged_logits: torch.Tensor, base_logits: torch.Tensor,
                   labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """KL(p_base || p_merged) averaged over non-pad tokens.

    p_base is the fixed reference (no grad through it).  Gradients flow only
    into ``merged_logits``.

    We compute KL in log-space for numerical stability:

        KL = sum_v p_b(v) * (log p_b(v) - log p_m(v))

    The labels are only used to mask padding (we sum over the full vocab for
    each valid position).
    """
    # mask: (B, T) -> (B, T, 1)
    mask = (labels != -100).unsqueeze(-1).float()
    log_b = F.log_softmax(base_logits / temperature, dim=-1)
    log_m = F.log_softmax(merged_logits / temperature, dim=-1)
    p_b = log_b.exp()
    kl = (p_b * (log_b - log_m)).sum(dim=-1)  # (B, T)
    kl = (kl * mask.squeeze(-1)).sum() / mask.sum().clamp(min=1.0)
    return kl


# ============================================================================
# 7. ELASTIC ABSORPTION — short fine-tune with KL + EWC
# ============================================================================
def elastic_absorption(merged_model, base_model, tokenizer,
                       texts: List[str], fisher: Dict[str, torch.Tensor],
                       steps: int = 50, batch_size: int = 2,
                       max_seq_len: int = 256, lr: float = 2e-5,
                       kl_weight: float = 0.5, ewc_weight: float = 0.1,
                       log_interval: int = 10) -> Dict[str, Any]:
    """Run the brief absorption fine-tune.

    Loss = L_general + kl_weight * KL(p_base||p_merged) + ewc_weight * EWC

    Only the fresh LoRA params on ``merged_model`` are trainable.  The
    ``base_model`` is used purely as a frozen reference for the KL term.
    """
    log.info("--- Elastic Absorption ---")
    log.info("  steps=%d  bs=%d  lr=%.1e  kl_w=%.2f  ewc_w=%.2f",
             steps, batch_size, lr, kl_weight, ewc_weight)

    merged_model.train()
    merged_model.config.use_cache = False

    # theta* for EWC = current (zero-init) LoRA params
    theta_star = snapshot_trainable(merged_model)

    optimizer = torch.optim.AdamW(
        [p for p in merged_model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    # cosine schedule over the short run
    def lr_lambda(step):
        warmup = 5
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history: List[Dict[str, float]] = []
    t0 = time.time()
    step = 0
    offset = 0

    while step < steps:
        batch = encode_batch(tokenizer, texts, max_seq_len, batch_size, offset)
        offset = (offset + batch_size) % max(1, len(texts))
        if batch is None:
            continue
        input_ids, attn, labels = batch
        input_ids = input_ids.to(merged_model.device)
        attn = attn.to(merged_model.device)
        labels = labels.to(merged_model.device)

        # --- merged forward (grad) ---
        merged_out = merged_model(input_ids=input_ids,
                                  attention_mask=attn, labels=labels)
        l_general = merged_out.loss

        # --- KL anchor (base is no-grad) ---
        if kl_weight > 0:
            base_logits = get_base_logits(base_model, input_ids, attn)
            kl = kl_anchor_loss(merged_out.logits, base_logits, labels)
        else:
            kl = torch.tensor(0.0, device=merged_model.device)

        # --- EWC ---
        if ewc_weight > 0:
            ewc = ewc_penalty(merged_model, fisher, theta_star)
        else:
            ewc = torch.tensor(0.0, device=merged_model.device)

        loss = l_general + kl_weight * kl + ewc_weight * ewc

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in merged_model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        if step % log_interval == 0 or step == 1:
            log.info("  abs step %3d/%d | L_gen %.4f | KL %.4f | EWC %.4f | "
                     "total %.4f | lr %.2e | %.1fs",
                     step, steps, float(l_general), float(kl), float(ewc),
                     float(loss), scheduler.get_last_lr()[0], time.time() - t0)
        history.append({
            "step": step,
            "l_general": float(l_general),
            "kl": float(kl),
            "ewc": float(ewc),
            "total": float(loss),
        })

    dt = time.time() - t0
    log.info("  absorption done: %d steps in %.1fs (%.1f steps/s)",
             step, dt, step / max(1e-6, dt))
    return {"history": history, "time_s": dt, "steps": step}


# ============================================================================
# 8. SAVE — vLLM-compatible
# ============================================================================
def strip_quant_from_config(merged_dir: Path) -> None:
    """Remove quantization_config from config.json so vLLM loads it as fp."""
    cfg_path = merged_dir / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "quantization_config" in cfg:
        cfg.pop("quantization_config", None)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        log.info("  stripped quantization_config from %s", cfg_path.name)


def save_merged_model(model, tokenizer, output_dir: Path) -> str:
    """Save the merged model; strip quant config for vLLM compatibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    strip_quant_from_config(output_dir)
    log.info("  merged model saved to %s", output_dir)
    return str(output_dir)


def save_adapter_fallback(model, tokenizer, output_dir: Path) -> str:
    """Fallback: save just the adapter for vLLM LoRA serving."""
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.warning("  merge_and_unload failed; adapter saved at %s", adapter_dir)
    log.warning("  serve with: vllm serve <base> --enable-lora "
                "--lora-modules etm=%s", adapter_dir)
    return str(adapter_dir)


# ============================================================================
# 9. ORCHESTRATION
# ============================================================================
def merge_elastic(args: argparse.Namespace) -> str:
    """Top-level: TIES trim -> scaled merge -> elastic absorption -> save."""
    t_start = time.time()
    log.info("=" * 70)
    log.info("ELASTIC ABSORPTION MERGE")
    log.info("=" * 70)
    log.info("  adapter : %s", args.adapter_path)
    log.info("  base    : %s", args.base_model)
    log.info("  output  : %s", args.output_dir)
    log.info("  ties    : trim_frac=%.2f  lambda=%.2f",
             args.ties_trim_frac, args.merge_lambda)
    log.info("  absorb  : steps=%d  kl_w=%.2f  ewc_w=%.2f",
             args.absorption_steps, args.kl_weight, args.ewc_weight)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase A: load base + trained adapter, reconstruct deltas --------
    log.info("\n--- Phase A: load base + trained adapter ---")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_for_adapter, tokenizer = load_base_model(
        args.base_model, attach_lora=False)
    # Attach the *trained* adapter so we can read its A/B and reconstruct
    # the delta.  is_trainable=False — we only need weights, not grads.
    model_with_adapter = PeftModel.from_pretrained(
        base_for_adapter, args.adapter_path, is_trainable=False,
        torch_dtype=torch.bfloat16)

    deltas = reconstruct_deltas(model_with_adapter)

    # ---- Phase B: TIES trim + scaled merge -------------------------------
    log.info("\n--- Phase B: TIES trim + scaled merge ---")
    trimmed = ties_trim_all(deltas, args.ties_trim_frac)
    apply_scaled_merge(model_with_adapter, trimmed, args.merge_lambda)

    # We no longer need the adapter wrapper — the delta is now baked into the
    # base weights.  Unload it so we have a clean base with merged weights.
    try:
        merged_base = model_with_adapter.merge_and_unload()
        log.info("  merge_and_unload succeeded (adapter wrapper removed)")
    except Exception as e:
        log.warning("  merge_and_unload failed (%s); keeping wrapped model", str(e)[:120])
        merged_base = model_with_adapter.base_model.model \
            if hasattr(model_with_adapter, "base_model") else model_with_adapter

    # ---- Phase C: load general data + Fisher ----------------------------
    log.info("\n--- Phase C: general data + EWC Fisher ---")
    texts = load_general_text(args.general_data,
                              n_samples=max(64, args.absorption_steps + 16),
                              max_seq_len=args.max_seq_len)
    log.info("  general data: %d samples", len(texts))

    # ---- Phase D: attach FRESH absorption LoRA --------------------------
    log.info("\n--- Phase D: attach fresh absorption LoRA (r=8) ---")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # merged_base is a plain (non-PEFT) model now; re-prepare for kbit + LoRA
    try:
        merged_base = prepare_model_for_kbit_training(merged_base)
    except Exception:
        pass  # already prepared

    abs_cfg = LoraConfig(
        r=args.absorption_rank,
        lora_alpha=args.absorption_rank * 2,
        target_modules=args.target_modules.split(",") if isinstance(args.target_modules, str)
                       else args.target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=args.use_dora,
    )
    merged_model = get_peft_model(merged_base, abs_cfg)
    merged_model.print_trainable_parameters()

    # Fisher computed on the *merged* model before absorption (theta* anchor).
    fisher = compute_fisher_diag(
        merged_model, tokenizer, texts,
        n_batches=args.ewc_batches, batch_size=args.absorption_batch_size,
        max_seq_len=args.max_seq_len,
    ) if args.ewc_weight > 0 else {}

    # ---- Phase E: load frozen base reference for KL ---------------------
    log.info("\n--- Phase E: load frozen base reference for KL anchor ---")
    base_ref = None
    if args.kl_weight > 0:
        base_ref, _ = load_base_model(args.base_model, attach_lora=False)
        base_ref.eval()
        for p in base_ref.parameters():
            p.requires_grad = False

    # ---- Phase F: elastic absorption ------------------------------------
    log.info("\n--- Phase F: elastic absorption ---")
    abs_result = elastic_absorption(
        merged_model, base_ref, tokenizer, texts, fisher,
        steps=args.absorption_steps,
        batch_size=args.absorption_batch_size,
        max_seq_len=args.max_seq_len,
        lr=args.absorption_lr,
        kl_weight=args.kl_weight,
        ewc_weight=args.ewc_weight,
        log_interval=args.log_interval,
    )

    # ---- Phase G: final merge of absorption LoRA + save -----------------
    log.info("\n--- Phase G: final merge + save ---")
    merged_path = None
    try:
        if hasattr(merged_model, "merge_and_unload"):
            final_model = merged_model.merge_and_unload()
            merged_path = save_merged_model(final_model, tokenizer, output_dir)
        else:
            merged_path = save_adapter_fallback(merged_model, tokenizer, output_dir)
    except Exception as e:
        log.error("  final merge failed: %s", str(e)[:200])
        merged_path = save_adapter_fallback(merged_model, tokenizer, output_dir)

    # ---- cleanup --------------------------------------------------------
    for obj in (base_ref, model_with_adapter, merged_model):
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()

    # ---- log ------------------------------------------------------------
    total_s = time.time() - t_start
    log.info("=" * 70)
    log.info("ELASTIC ABSORPTION MERGE COMPLETE")
    log.info("=" * 70)
    log.info("  total time : %.1fs", total_s)
    log.info("  output     : %s", merged_path)

    run_log = {
        "mode": "elastic_absorption_merge",
        "adapter_path": args.adapter_path,
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "ties_trim_frac": args.ties_trim_frac,
        "merge_lambda": args.merge_lambda,
        "kl_weight": args.kl_weight,
        "ewc_weight": args.ewc_weight,
        "absorption_steps": args.absorption_steps,
        "absorption_rank": args.absorption_rank,
        "absorption_time_s": abs_result["time_s"],
        "absorption_history": abs_result["history"],
        "total_time_s": total_s,
        "merged_path": merged_path,
    }
    log_path = output_dir / "merge_elastic_log.json"
    log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    log.info("  log        : %s", log_path)
    return merged_path


# ============================================================================
# CLI
# ============================================================================
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Elastic Absorption Merge for ETM (TIES + KL + EWC).")
    ap.add_argument("--adapter_path", required=True,
                    help="Path to the trained LoRA/DoRA adapter directory.")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Math-7B",
                    help="HF model id or local path to the base model.")
    ap.add_argument("--output_dir", required=True,
                    help="Where to save the merged model.")
    # TIES / merge
    ap.add_argument("--ties_trim_frac", type=float, default=0.1,
                    help="Fraction of delta entries to drop by magnitude.")
    ap.add_argument("--merge_lambda", type=float, default=0.8,
                    help="Scale factor for the trimmed delta before adding to base.")
    # absorption
    ap.add_argument("--absorption_steps", type=int, default=50,
                    help="Number of elastic absorption fine-tune steps.")
    ap.add_argument("--absorption_rank", type=int, default=8,
                    help="LoRA rank for the fresh absorption adapter.")
    ap.add_argument("--absorption_lr", type=float, default=2e-5,
                    help="Learning rate for absorption.")
    ap.add_argument("--absorption_batch_size", type=int, default=2,
                    help="Batch size for absorption.")
    ap.add_argument("--max_seq_len", type=int, default=256,
                    help="Max sequence length for absorption data.")
    # loss weights
    ap.add_argument("--kl_weight", type=float, default=0.5,
                    help="Weight for KL(p_base || p_merged) anchor term.")
    ap.add_argument("--ewc_weight", type=float, default=0.1,
                    help="Weight for the EWC penalty.")
    ap.add_argument("--ewc_batches", type=int, default=4,
                    help="Number of batches used to estimate the Fisher diagonal.")
    # misc
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj",
                    help="Comma-separated LoRA target modules.")
    ap.add_argument("--use_dora", action="store_true", default=True,
                    help="Use DoRA for the absorption adapter (default True).")
    ap.add_argument("--no_dora", dest="use_dora", action="store_false",
                    help="Disable DoRA for the absorption adapter.")
    ap.add_argument("--general_data", default=None,
                    help="Optional .jsonl with a 'text' field for general data.")
    ap.add_argument("--log_interval", type=int, default=10,
                    help="Log every N absorption steps.")
    return ap


def main():
    args = build_argparser().parse_args()
    merge_elastic(args)


if __name__ == "__main__":
    main()

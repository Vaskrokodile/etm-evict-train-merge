#!/usr/bin/env python
"""ETM v5: Evict-Train-Merge with MSE Loss.

Core mechanism (same as v4): evict the base model, cache (x, grad_h) pairs,
train only the LoRA params against the cached gradient. No breathing.

Key improvements over v4:
  1. MSE loss instead of cosine — controls direction AND magnitude
     L = ||u + η*grad_h||² / N  →  pushes u toward -η*grad_h
  2. Higher rank (64 vs 24) — more capacity to approximate gradient
  3. 8000 steps (vs 1000) — loss hadn't converged in v4
  4. 2000 cached samples (vs 500) — 4x more gradient signal
  5. Cache on CPU, batch to GPU — allows huge cache (393GB RAM)
  6. η calibrated from data — target ||u|| = 0.1 * ||x||
  7. Regular LoRA (no DoRA) — simpler, reliable merge
  8. bf16 merge — no 4-bit weight corruption
  9. Curated data — MATH Level 4-5, formatted to match eval
  10. Alignment monitoring — log cosine sim and ||u||/||grad_h|| during training
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen2.5-Math-7B"


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD: 4-bit base + LoRA (for caching only)
# ═══════════════════════════════════════════════════════════════════════════════

def load_base_with_lora(rank=64, alpha=128):
    """Load the 7B base in 4-bit NF4 and attach LoRA adapters (no DoRA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        use_dora=False,  # Regular LoRA for reliable merge
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE: forward+backward, capture (x, grad_h) per LoRA module, keep on CPU
# ═══════════════════════════════════════════════════════════════════════════════

def cache_features(model, tokenizer, data, max_samples, max_seq_len,
                   max_tokens_per_sample=64):
    """Run forward+backward on training data, cache (x, grad_h) for each LoRA module.

    Cache stays on CPU (we have 393GB RAM). During training, batches are moved
    to GPU on-the-fly.

    Also calibrates η: computes median ||grad_h|| and ||x||, sets η so that
    target ||u|| = 0.1 * ||x|| (meaningful but not destructive delta).

    Returns:
      cache: {name: {'x_list': [tensor], 'g_list': [tensor], 'scaling': float}}
      n_cached: int
      eta: calibrated step size
    """
    cache = defaultdict(lambda: {'x': [], 'grad_h': []})
    lora_info = {}
    handles = []

    # Collect norms for η calibration
    grad_norms = []
    x_norms = []

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            for adapter_name in module.lora_A:
                lora_info[name] = module.scaling[adapter_name]

            def make_hooks(mod_name):
                def fwd_hook(mod, args, kwargs, output):
                    x = args[0] if args else kwargs.get('input', None)
                    if x is not None:
                        seq_len = x.shape[1]
                        n_tok = min(max_tokens_per_sample, seq_len)
                        idx = torch.randperm(seq_len, device=x.device)[:n_tok]
                        x_sub = x[0, idx].detach()
                        cache[mod_name]['x'].append(x_sub.to(torch.bfloat16).cpu())
                        x_norms.append(x_sub.norm().item())

                def bwd_hook(mod, grad_input, grad_output):
                    g = grad_output[0]
                    if g is not None:
                        seq_len = g.shape[1]
                        n_tok = min(max_tokens_per_sample, seq_len)
                        idx = torch.randperm(seq_len, device=g.device)[:n_tok]
                        g_sub = g[0, idx].detach()
                        cache[mod_name]['grad_h'].append(g_sub.to(torch.bfloat16).cpu())
                        grad_norms.append(g_sub.norm().item())

                return fwd_hook, bwd_hook

            fwd_h, bwd_h = make_hooks(name)
            handles.append(module.register_forward_hook(fwd_h, with_kwargs=True))
            handles.append(module.register_full_backward_hook(bwd_h))

    model.train()
    model.config.use_cache = False
    n_cached = 0
    t0 = time.time()

    for i, item in enumerate(data[:max_samples]):
        text = item["text"]
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_seq_len, padding=False)
        input_ids = enc["input_ids"].cuda()
        labels = input_ids.clone()

        try:
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            n_cached += 1
        except Exception as e:
            print(f"  skip sample {i}: {str(e)[:80]}")

        model.zero_grad(set_to_none=True)

        if (i + 1) % 200 == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            elapsed = time.time() - t0
            print(f"  cached {i+1}/{max_samples} | VRAM {vram:.2f} GB | {elapsed:.0f}s")

    for h in handles:
        h.remove()

    t_cache = time.time() - t0

    # Add scaling to cache
    for name in cache:
        cache[name]['scaling'] = lora_info.get(name, 1.0)

    # Calibrate η
    grad_norms.sort()
    x_norms.sort()
    median_grad = grad_norms[len(grad_norms)//2] if grad_norms else 1.0
    median_x = x_norms[len(x_norms)//2] if x_norms else 1.0

    # Target: ||u|| = 0.1 * ||x|| (10% of hidden state magnitude)
    # u ≈ -η * grad_h, so η = target_||u|| / ||grad_h|| = 0.1 * ||x|| / ||grad_h||
    eta = 0.1 * median_x / (median_grad + 1e-8)

    print(f"\n  [Cache] DONE: {n_cached} samples cached in {t_cache:.1f}s")
    print(f"  [Cache] modules: {len(cache)}")
    print(f"  [Cache] median ||x||: {median_x:.4f}")
    print(f"  [Cache] median ||grad_h||: {median_grad:.6f}")
    print(f"  [Cache] calibrated η: {eta:.4f} (target ||u|| = {0.1*median_x:.4f})")

    # Estimate cache memory
    total_bytes = 0
    for name in cache:
        for t in cache[name]['x']:
            total_bytes += t.nelement() * t.element_size()
        for t in cache[name]['grad_h']:
            total_bytes += t.nelement() * t.element_size()
    print(f"  [Cache] CPU memory: {total_bytes / 1e9:.1f} GB")

    return dict(cache), n_cached, eta


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT: pull LoRA A, B params as standalone GPU tensors
# ═══════════════════════════════════════════════════════════════════════════════

def extract_lora_params(model):
    """Extract LoRA A and B weights as standalone GPU tensors."""
    lora_params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            for adapter_name in module.lora_A:
                A = module.lora_A[adapter_name].weight.data.clone().cuda()
                B = module.lora_B[adapter_name].weight.data.clone().cuda()
                lora_params[name] = {'A': A, 'B': B}
    print(f"  [Extract] {len(lora_params)} LoRA modules extracted")
    return lora_params


# ═══════════════════════════════════════════════════════════════════════════════
# EVICT: delete base model, free VRAM
# ═══════════════════════════════════════════════════════════════════════════════

def evict_model(model):
    """Delete model and free all VRAM."""
    vram_before = torch.cuda.memory_allocated() / 1e9
    del model
    gc.collect()
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1e9
    print(f"  [Evict] VRAM: {vram_before:.2f} GB → {vram_after:.2f} GB")
    return vram_before, vram_after


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS: MSE (direction + magnitude)
# ═══════════════════════════════════════════════════════════════════════════════

def mse_loss(u, v, eta=1.0):
    """MSE loss: L = ||u + η*v||² / N

    Pushes u toward -η*v (gradient descent step with step size η).
    Unlike cosine, this controls both direction AND magnitude.

    Args:
      u: (b, tok, d) — the LoRA delta (scaling * x @ A.T @ B.T)
      v: (b, tok, d) — the cached upstream gradient (grad_h = ∂L/∂h)
      eta: target step size (controls magnitude of LoRA delta)

    Returns:
      scalar loss
    """
    diff = u + eta * v
    return (diff * diff).mean()


def cosine_sim(u, v, eps=1e-8):
    """Cosine similarity for monitoring (not used as loss)."""
    u_flat = u.reshape(-1, u.shape[-1])
    v_flat = v.reshape(-1, v.shape[-1])
    dot = (u_flat * v_flat).sum(dim=-1)
    u_norm = u_flat.norm(dim=-1)
    v_norm = v_flat.norm(dim=-1)
    return (dot / (u_norm * v_norm + eps)).mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN: MSE loss against cached gradient, base model evicted
# ═══════════════════════════════════════════════════════════════════════════════


def move_cache_to_gpu(cache):
    """Move entire cache from CPU to GPU. Call after base model eviction."""
    total_bytes = 0
    for name in cache:
        cache[name]['x'] = [t.cuda() for t in cache[name]['x']]
        cache[name]['grad_h'] = [t.cuda() for t in cache[name]['grad_h']]
        for t in cache[name]['x']:
            total_bytes += t.nelement() * t.element_size()
        for t in cache[name]['grad_h']:
            total_bytes += t.nelement() * t.element_size()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  [Cache→GPU] {total_bytes / 1e9:.1f} GB moved, VRAM: {vram:.2f} GB")


def train_mse(lora_params, cache, n_samples, eta,
              lr=1e-4, batch_size=16, max_steps=8000,
              max_tokens=64, log_interval=50, grad_clip=1.0,
              checkpoint_dir=None, checkpoint_interval=2000):
    """Train LoRA A and B with MSE loss against cached gradient.

    The base model does not exist during training. Only the LoRA params
    (A, B) and the cached (x, grad_h) pairs are used.

    Args:
      lora_params: {name: {'A','B','scaling'}} on GPU
      cache: {name: {'x': [cpu tensors], 'grad_h': [cpu tensors], 'scaling': float}}
      n_samples: number of cached samples
      eta: step size for MSE loss (calibrated from data)
      lr, batch_size, max_steps: training hyperparams
      max_tokens: tokens per cached sample
      log_interval: print every N steps
      grad_clip: max grad norm (0=disable)
      checkpoint_dir: if set, save checkpoints every checkpoint_interval steps

    Returns:
      lora_params, losses, t_train, peak_vram
    """
    print(f"\n[ETM:Train] MSE TRAINING — base model does not exist")
    print(f"  lr={lr}, batch_size={batch_size}, max_steps={max_steps}")
    print(f"  loss=MSE (||u + η*grad_h||²), η={eta:.4f}, grad_clip={grad_clip}")
    print(f"  cached samples: {n_samples}, modules: {len(lora_params)}")

    # Set requires_grad
    for name, p in lora_params.items():
        p['A'] = p['A'].detach().requires_grad_(True)
        p['B'] = p['B'].detach().requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p['A'] for p in lora_params.values()] +
        [p['B'] for p in lora_params.values()],
        lr=lr, weight_decay=0.01,
    )

    # Cosine decay with warmup
    warmup = 200
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    losses = []
    cos_sims = []
    u_norms = []
    g_norms = []
    t0 = time.time()

    while step < max_steps:
        # Sample random batch
        indices = torch.randperm(n_samples).tolist()
        batch_idx = indices[:batch_size]

        optimizer.zero_grad()
        total_loss = 0.0
        total_cos = 0.0
        total_u_norm = 0.0
        total_g_norm = 0.0
        n_modules_this_step = 0

        for name, p in lora_params.items():
            if name not in cache:
                continue
            c = cache[name]
            A = p['A']  # (r, d_in)
            B = p['B']  # (d_out, r)
            scaling = c['scaling']

            # Gather batch from CPU cache, move to GPU
            x = torch.stack([c['x'][i] for i in batch_idx]).float()  # (b, tok, d_in) already on GPU
            g = torch.stack([c['grad_h'][i] for i in batch_idx]).float()  # (b, tok, d_out) already on GPU

            # LoRA forward: u = scaling * x @ A.T @ B.T
            z = x @ A.T        # (b, tok, r)
            out = z @ B.T       # (b, tok, d_out)
            u = scaling * out   # the LoRA delta

            # MSE loss: L = ||u + η*g||² / N
            loss = mse_loss(u, g, eta)
            loss.backward()

            total_loss += loss.item()
            total_cos += cosine_sim(u.detach(), g.detach())
            total_u_norm += u.detach().norm().item() / math.sqrt(u.numel())
            total_g_norm += g.detach().norm().item() / math.sqrt(g.numel())
            n_modules_this_step += 1

        # Gradient clipping
        if grad_clip > 0:
            params_to_clip = [p['A'] for p in lora_params.values()] +                              [p['B'] for p in lora_params.values()]
            torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)

        optimizer.step()
        scheduler.step()
        step += 1

        avg_loss = total_loss / max(1, n_modules_this_step)
        avg_cos = total_cos / max(1, n_modules_this_step)
        avg_u = total_u_norm / max(1, n_modules_this_step)
        avg_g = total_g_norm / max(1, n_modules_this_step)
        losses.append(avg_loss)
        cos_sims.append(avg_cos)
        u_norms.append(avg_u)
        g_norms.append(avg_g)

        if step % log_interval == 0 or step == 1:
            vram = torch.cuda.memory_allocated() / 1e9
            elapsed = time.time() - t0
            sps = step / elapsed
            recent_loss = sum(losses[-log_interval:]) / min(log_interval, len(losses))
            recent_cos = sum(cos_sims[-log_interval:]) / min(log_interval, len(cos_sims))
            print(f"  step {step:5d} | loss {recent_loss:.6f} | cos {recent_cos:+.4f} | "
                  f"||u|| {avg_u:.6f} | ||g|| {avg_g:.6f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {sps:.1f} steps/s | VRAM {vram:.2f} GB")

        # Checkpoint
        if checkpoint_dir and step % checkpoint_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"checkpoint_step{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step,
                'lora_params': {k: {'A': v['A'].detach().cpu(), 'B': v['B'].detach().cpu()}
                               for k, v in lora_params.items()},
                'losses': losses[-100:],
                'eta': eta,
            }, str(ckpt_path))
            print(f"  [Checkpoint] saved to {ckpt_path}")

    t_train = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[ETM:Train] DONE: {step} steps in {t_train:.1f}s ({step/t_train:.1f} steps/s)")
    print(f"[ETM:Train] peak VRAM: {peak_vram:.4f} GB")
    print(f"[ETM:Train] final loss: {sum(losses[-10:])/min(10,len(losses)):.6f}")
    print(f"[ETM:Train] final cosine sim: {sum(cos_sims[-10:])/min(10,len(cos_sims)):+.4f}")
    print(f"[ETM:Train] final ||u||/||g|| ratio: {avg_u/max(avg_g,1e-8):.4f}")

    return lora_params, losses, t_train, peak_vram, cos_sims


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE: reload base in bf16, inject trained params, save adapter + merged model
# ═══════════════════════════════════════════════════════════════════════════════

def save_and_merge(lora_params, output_dir, rank=64, alpha=128):
    """Reload base in bf16 (NOT 4-bit), inject trained LoRA, save adapter + merged model.

    The v4 merge used 4-bit base which corrupted weight shapes.
    This uses full bf16 base for correct merge.
    """
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base in bf16 (no quantization)
    print(f"[ETM:Merge] loading base {MODEL_ID} in bf16 (no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Attach LoRA with same config
    lora_config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        use_dora=False,
    )
    model = get_peft_model(model, lora_config)

    # Inject trained weights
    injected = 0
    for name, module in model.named_modules():
        if name in lora_params:
            for adapter_name in module.lora_A:
                base_device = module.lora_A[adapter_name].weight.device
                module.lora_A[adapter_name].weight.data = (
                    lora_params[name]['A'].detach().to(base_device)
                )
                module.lora_B[adapter_name].weight.data = (
                    lora_params[name]['B'].detach().to(base_device)
                )
                injected += 1
    print(f"[ETM:Merge] injected {injected} modules")

    # Save adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[ETM:Merge] adapter saved to {output_dir}")

    # Merge and save full model
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    print(f"[ETM:Merge] merging adapter into base...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    # Verify config
    cfg = json.load(open(merged_dir / "config.json"))
    assert "quantization_config" not in cfg, "quantization_config leaked!"
    print(f"[ETM:Merge] merged model saved to {merged_dir} (dtype={cfg.get('dtype')})")
    return str(merged_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_etm(args):
    print("=" * 70)
    print("ETM v5: MSE LOSS + HIGHER RANK + MORE DATA + MORE STEPS")
    print("=" * 70)
    print(f"  Model:       {MODEL_ID}")
    print(f"  LoRA:        r={args.rank}, alpha={args.alpha} (regular, no DoRA)")
    print(f"  Loss:        MSE (||u + η*grad_h||²)")
    print(f"  Steps:       {args.max_steps}")
    print(f"  Data:        {args.data}")
    print(f"  Cache:       {args.max_samples} samples × {args.max_tokens} tokens")
    print(f"  No breathing (gradient cached once, base evicted)")

    # Load training data
    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"[ETM] loaded {len(data)} training samples")

    # ── Phase 0: Load base + LoRA ──
    print("\n--- Phase 0: Load base + LoRA ---")
    model, tokenizer = load_base_with_lora(args.rank, args.alpha)
    vram_loaded = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after load: {vram_loaded:.2f} GB")

    # ── Phase 1: Cache (x, grad_h) + calibrate η ──
    print("\n--- Phase 1: Cache activations + calibrate η ---")
    cache, n_cached, eta = cache_features(
        model, tokenizer, data, args.max_samples, args.max_seq_len, args.max_tokens,
    )

    # ── Phase 2: Extract LoRA params ──
    print("\n--- Phase 2: Extract LoRA params ---")
    lora_params = extract_lora_params(model)

    # ── Phase 3: EVICT base model ──
    print("\n--- Phase 3: EVICT base model ---")
    vram_before, vram_after = evict_model(model)
    del model

    # ── Phase 3.5: Move cache to GPU (base is gone, VRAM is free) ──
    print("--- Phase 3.5: Move cache to GPU ---")
    move_cache_to_gpu(cache)

    # ── Phase 4: Train (base evicted) ──
    print("\n--- Phase 4: MSE training (base evicted) ---")
    lora_params, losses, t_train, peak_vram, cos_sims = train_mse(
        lora_params, cache, n_cached, eta,
        lr=args.lr, batch_size=args.batch_size, max_steps=args.max_steps,
        max_tokens=args.max_tokens, log_interval=args.log_interval,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.output_dir, checkpoint_interval=args.checkpoint_interval,
    )

    # ── Phase 5: Merge ──
    print("\n--- Phase 5: Merge (bf16) ---")
    merged_path = save_and_merge(lora_params, args.output_dir, args.rank, args.alpha)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("ETM v5 COMPLETE")
    print("=" * 70)
    total_params = sum(p['A'].numel() + p['B'].numel() for p in lora_params.values())
    print(f"  Trainable params:    {total_params/1e6:.1f}M")
    print(f"  Train time:          {t_train:.1f}s ({len(losses)} steps)")
    print(f"  Train speed:         {len(losses)/t_train:.1f} steps/s")
    print(f"  VRAM (base loaded):  {vram_before:.2f} GB")
    print(f"  VRAM (base evicted): {vram_after:.2f} GB")
    print(f"  VRAM (peak train):   {peak_vram:.2f} GB")
    print(f"  Final loss:          {sum(losses[-10:])/min(10,len(losses)):.6f}")
    print(f"  Final cosine sim:    {sum(cos_sims[-10:])/min(10,len(cos_sims)):+.4f}")
    print(f"  η (step size):       {eta:.4f}")
    print(f"  Merged model:        {merged_path}")

    log = {
        "mode": "etm_v5_mse",
        "model": MODEL_ID,
        "trainable_params_m": total_params / 1e6,
        "n_cached": n_cached,
        "train_time_s": t_train,
        "train_steps": len(losses),
        "train_speed_steps_per_s": len(losses) / t_train,
        "vram_base_loaded_gb": vram_before,
        "vram_base_evicted_gb": vram_after,
        "vram_peak_train_gb": peak_vram,
        "final_loss": sum(losses[-10:]) / min(10, len(losses)),
        "final_cosine_sim": sum(cos_sims[-10:]) / min(10, len(cos_sims)),
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "eta": eta,
        "loss_type": "mse",
        "grad_clip": args.grad_clip,
        "merged_model_path": merged_path,
    }
    with open(Path(args.output_dir) / "etm_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Log: {args.output_dir}/etm_log.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ETM v5: MSE loss + higher rank + more data")
    ap.add_argument("--data", default="/root/etm/data/math_train_v5.jsonl")
    ap.add_argument("--output_dir", default="outputs/etm_v5")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--checkpoint_interval", type=int, default=2000)
    args = ap.parse_args()
    run_etm(args)

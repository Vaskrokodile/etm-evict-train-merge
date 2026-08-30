#!/usr/bin/env python
"""
ETM v4: Breathing Cycle + Bounded Cosine Loss.

THE PROBLEM WITH v3
-------------------
v3 caches (x, grad_h) once at the start, then trains LoRA A and B against that
frozen cache for hundreds of steps. The cached grad_h is the *exact* gradient of
the LM loss w.r.t. the LoRA output — but only for the LoRA weights at step 0.
After step 0 the weights change, so grad_h is stale. The surrogate loss

    L = (scaling * (x @ A.T @ B.T) * grad_h).sum() / N

is an *unbounded* dot product. As A and B drift, the dot product can grow
without limit (observed: loss → -1636). The model does not improve.

THE v4 FIX: TWO INNOVATIONS
----------------------------
1. **Breathing cycle** — every K steps (default 20):
     a. Save current A, B to CPU.
     b. Reload the 4-bit base, reattach LoRA with the *current trained* params.
     c. Run forward+backward on a fresh batch → refresh (x, grad_h) cache.
     d. `del model` — base ceases to exist again.
     e. Continue training against the fresh cache.
   The model "breathes": expand (load base, refresh), contract (evict, train
   15M in isolation). This keeps grad_h within ~K steps of stale instead of
   ~max_steps, so the surrogate stays a good local approximation.

2. **Bounded cosine-similarity loss** — replace the unbounded dot product with

       L = -cos(scaled_out, grad_h)   ∈ [-1, 1]

   where scaled_out = scaling * (x @ A.T @ B.T) and grad_h is the cached
   upstream gradient. Minimizing -cos pushes the LoRA update to *align* with
   the direction the loss wants to move, regardless of magnitude. This is
   bounded and cannot diverge. We also:
     - clip grad norm to 1.0
     - normalize by token count
     - add a tiny epsilon to the denominator for numerical stability

GRADIENT MATH (cosine loss)
---------------------------
Let u = scaling * (x @ A.T @ B.T)   (b, tok, d_out)   [the LoRA delta]
Let v = grad_h                      (b, tok, d_out)   [cached upstream grad]

    cos = <u, v> / (||u|| * ||v||)
    L   = -cos

    dL/du = -(v / ||v||) / ||u||  +  <u,v> * u / (||u||^3 * ||v||)
          = -(1/||u||) * [ v_hat - cos * u_hat ]   where _hat = normalized

    dL/dB = sum over batch,tok of dL/du @ (scaling * (x @ A.T))   [via z = x@A.T]
    dL/dA = sum over batch,tok of (scaling * (dL/du @ B)) outer x

In practice we let autograd handle it — u is a function of A, B and we just
call L.backward(). The matmuls are tiny (r=24), so this is blazing fast.

PIPELINE
--------
  Phase 0: Load base + DoRA, initial cache.
  Phase 1: Extract A, B. Evict base.
  Phase 2: Breathing training loop:
             for step in range(max_steps):
               if step % breath_interval == 0 and step > 0:
                 BREATHE: save A,B → reload base → refresh cache → evict
               train one step against current cache (cosine loss)
  Phase 3: Merge (reload base, inject, save adapter, attempt merge).
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


# ──────────────────────────────────────────────────────────────────────────────
# LOAD: 4-bit base + DoRA
# ──────────────────────────────────────────────────────────────────────────────

def load_base_with_lora(rank=24, alpha=48, use_dora=True):
    """Load the 7B base in 4-bit NF4 and attach DoRA adapters.

    Returns (model, tokenizer). The model has ~15.1M trainable params.
    """
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
        use_dora=use_dora,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# CACHE: forward+backward, capture (x, grad_h) per LoRA module, keep on CPU
# ──────────────────────────────────────────────────────────────────────────────

def cache_features(model, tokenizer, data, max_samples, max_seq_len,
                   max_tokens_per_sample=64, lora_params_to_inject=None):
    """Run forward+backward on training data, cache (x, grad_h) for each LoRA module.

    If lora_params_to_inject is provided, we first inject those A/B weights into
    the model so the cached gradient reflects the *current* trained state (this
    is the key to the breathing cycle).

    Cache is stored on CPU during caching (to avoid VRAM blowup), then moved to
    GPU after the base is evicted.

    Args:
      model: PEFT model with LoRA modules.
      tokenizer: tokenizer.
      data: list of {"text": ...}.
      max_samples: how many samples to cache from.
      max_seq_len: truncation length.
      max_tokens_per_sample: random token positions to subsample per sample.
      lora_params_to_inject: optional {name: {'A','B','scaling'}} to inject
                             before caching (breathing refresh).

    Returns:
      cache: {name: {'x_list': [tensor], 'g_list': [tensor], 'scaling': float}}
      n_cached: int
    """
    # Optionally inject current trained params before caching
    if lora_params_to_inject is not None:
        injected = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if name in lora_params_to_inject:
                    for adapter_name in module.lora_A:
                        base_device = module.lora_A[adapter_name].weight.device
                        module.lora_A[adapter_name].weight.data = (
                            lora_params_to_inject[name]['A'].detach().to(base_device)
                        )
                        module.lora_B[adapter_name].weight.data = (
                            lora_params_to_inject[name]['B'].detach().to(base_device)
                        )
                        injected += 1
        print(f"  [Cache] injected current trained params into {injected} modules")

    cache = defaultdict(lambda: {'x': [], 'grad_h': []})
    lora_info = {}
    handles = []

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
                        cache[mod_name]['x'].append(x[0, idx].detach().cpu())

                def bwd_hook(mod, grad_input, grad_output):
                    g = grad_output[0]
                    if g is not None:
                        seq_len = g.shape[1]
                        n_tok = min(max_tokens_per_sample, seq_len)
                        idx = torch.randperm(seq_len, device=g.device)[:n_tok]
                        cache[mod_name]['grad_h'].append(g[0, idx].detach().cpu())

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

        if (i + 1) % 50 == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"  cached {i+1}/{max_samples} | VRAM {vram:.2f} GB")

    for h in handles:
        h.remove()

    t_cache = time.time() - t0
    print(f"  [Cache] cached {n_cached} samples in {t_cache:.0f}s")

    # Build final cache as lists (NO stacking — too slow for 112 modules)
    final_cache = {}
    total_bytes = 0
    for name in cache:
        if not cache[name]['x'] or not cache[name]['grad_h']:
            continue
        n_tok = max_tokens_per_sample
        x_list = [t[:n_tok] for t in cache[name]['x']]
        g_list = [t[:n_tok] for t in cache[name]['grad_h']]
        n = min(len(x_list), len(g_list))
        final_cache[name] = {
            'x_list': x_list[:n],
            'g_list': g_list[:n],
            'scaling': lora_info.get(name, 1.0),
        }
        total_bytes += sum(t.numel() * t.element_size() for t in x_list[:n])
        total_bytes += sum(t.numel() * t.element_size() for t in g_list[:n])

    print(f"  [Cache] {len(final_cache)} modules, {total_bytes/1e9:.2f} GB on CPU")
    return final_cache, n_cached


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACT: pull A and B as standalone GPU tensors
# ──────────────────────────────────────────────────────────────────────────────

def extract_lora_params(model):
    """Extract LoRA A and B weights as standalone GPU tensors.

    Returns {name: {'A': (r, d_in), 'B': (d_out, r), 'scaling': float}}.
    """
    params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            for adapter_name in module.lora_A:
                A = module.lora_A[adapter_name].weight.data.clone().cuda()
                B = module.lora_B[adapter_name].weight.data.clone().cuda()
                params[name] = {'A': A, 'B': B, 'scaling': module.scaling[adapter_name]}
    total = sum(p['A'].numel() + p['B'].numel() for p in params.values())
    print(f"[ETM:Extract] {len(params)} modules, {total/1e6:.1f}M params")
    return params


# ──────────────────────────────────────────────────────────────────────────────
# EVICT: del model. It ceases to exist in all memory.
# ──────────────────────────────────────────────────────────────────────────────

def evict_model(model):
    """Delete the model from all memory. No CPU offload. It ceases to exist."""
    vram_before = torch.cuda.memory_allocated() / 1e9
    print(f"[ETM:Evict] VRAM before: {vram_before:.2f} GB")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    vram_after = torch.cuda.memory_allocated() / 1e9
    print(f"[ETM:Evict] VRAM after: {vram_after:.2f} GB")
    print(f"[ETM:Evict] MODEL DELETED. Only LoRA params + cache remain.")
    return vram_before, vram_after


# ──────────────────────────────────────────────────────────────────────────────
# CACHE MANAGEMENT: move cache CPU↔GPU, free GPU cache
# ──────────────────────────────────────────────────────────────────────────────

def move_cache_to_gpu(cache):
    """Move all cached (x, grad_h) lists from CPU to GPU."""
    for name in cache:
        cache[name]['x_list'] = [t.cuda() for t in cache[name]['x_list']]
        cache[name]['g_list'] = [t.cuda() for t in cache[name]['g_list']]
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  [Cache→GPU] VRAM: {vram:.2f} GB")


def free_cache_from_gpu(cache):
    """Move all cached (x, grad_h) lists back to CPU and free GPU."""
    for name in cache:
        cache[name]['x_list'] = [t.cpu() for t in cache[name]['x_list']]
        cache[name]['g_list'] = [t.cpu() for t in cache[name]['g_list']]
    gc.collect()
    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  [Cache→CPU] VRAM: {vram:.2f} GB")


# ──────────────────────────────────────────────────────────────────────────────
# BREATHE: reload base, refresh cache, evict
# ──────────────────────────────────────────────────────────────────────────────

def breathe(lora_params, tokenizer, data, rank, alpha, use_dora,
            max_samples, max_seq_len, max_tokens, old_cache):
    """One breathing cycle: reload base, refresh (x, grad_h), evict.

    Steps:
      1. Free old cache from GPU (we need VRAM for base reload).
      2. Reload 4-bit base + DoRA.
      3. Inject current trained A, B into the model.
      4. Forward+backward on fresh batch → new (x, grad_h) cache.
      5. Extract nothing (we already have A, B as standalone tensors).
      6. del model — base ceases to exist.
      7. Move new cache to GPU.

    Returns the refreshed cache (on GPU) and n_cached.
    """
    print("\n  ┌─ BREATH ─────────────────────────────────────────────")
    # 1. Free old cache GPU memory
    free_cache_from_gpu(old_cache)
    del old_cache
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Reload base
    print("  [Breath] reloading 4-bit base + DoRA...")
    model, _ = load_base_with_lora(rank, alpha, use_dora)
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  [Breath] base loaded, VRAM: {vram:.2f} GB")

    # 3+4. Inject current params and cache
    print("  [Breath] refreshing cache with current trained params...")
    new_cache, n_cached = cache_features(
        model, tokenizer, data, max_samples, max_seq_len, max_tokens,
        lora_params_to_inject=lora_params,
    )

    # 5. Evict base
    print("  [Breath] evicting base model...")
    evict_model(model)
    del model

    # 6. Move new cache to GPU
    move_cache_to_gpu(new_cache)
    print("  └─ BREATH DONE ────────────────────────────────────────\n")
    return new_cache, n_cached


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN: breathing cycle with bounded cosine-similarity loss
# ──────────────────────────────────────────────────────────────────────────────

def cosine_loss(u, v, eps=1e-8):
    """Bounded cosine-similarity loss.

    L = cos(u, v) = <u,v> / (||u|| * ||v|| + eps)

    Bounded in [-1, 1]. Minimizing L pushes u to ANTI-align with v (the cached
    upstream gradient ∂L/∂h). Since grad_h points toward increasing loss,
    anti-aligning the LoRA delta with grad_h means delta ≈ -α·grad_h, which
    DECREASES the original loss. This is the correct gradient descent direction.

    Args:
      u: (b, tok, d) — the LoRA delta (scaling * x @ A.T @ B.T)
      v: (b, tok, d) — the cached upstream gradient (grad_h = ∂L/∂h)
      eps: numerical stability

    Returns:
      scalar loss in [-1, 1].
    """
    # Flatten over batch+tokens for cosine, keep dtype
    u_flat = u.reshape(-1, u.shape[-1])
    v_flat = v.reshape(-1, v.shape[-1])
    dot = (u_flat * v_flat).sum(dim=-1)                    # (N,)
    u_norm = u_flat.norm(dim=-1)                           # (N,)
    v_norm = v_flat.norm(dim=-1)                           # (N,)
    cos = dot / (u_norm * v_norm + eps)
    return cos.mean()


def train_breathing(lora_params, cache, n_samples, tokenizer, data,
                    rank, alpha, use_dora,
                    lr=1e-4, batch_size=8, max_steps=500,
                    breath_interval=20, max_tokens=64,
                    max_samples=500, max_seq_len=512,
                    log_interval=10, grad_clip=1.0):
    """Train LoRA A and B with breathing cycle + bounded cosine loss.

    Every `breath_interval` steps, the base model is reloaded, the cache is
    refreshed with the current trained params, and the base is evicted again.
    Between breaths, training runs against the GPU-resident cache using a
    bounded cosine-similarity loss that cannot diverge.

    Args:
      lora_params: {name: {'A','B','scaling'}} on GPU, requires_grad set here.
      cache: initial cache (on GPU) from the first caching pass.
      n_samples: number of cached samples.
      tokenizer: for breathing (needed by cache_features indirectly).
      data: training data list.
      rank, alpha, use_dora: LoRA config for breathing reloads.
      lr, batch_size, max_steps: training hyperparams.
      breath_interval: K — refresh cache every K steps.
      max_tokens, max_samples, max_seq_len: cache config.
      log_interval: print every N steps.
      grad_clip: max grad norm for clipping.

    Returns:
      lora_params, losses, t_train, peak_vram, n_breaths
    """
    print(f"\n[ETM:Train] BREATHING TRAINING — base model does not exist between breaths")
    print(f"  lr={lr}, batch_size={batch_size}, max_steps={max_steps}")
    print(f"  breath_interval={breath_interval} (refresh cache every K steps)")
    print(f"  loss=cosine_similarity (bounded [-1,1]), grad_clip={grad_clip}")

    # Ensure cache is on GPU
    move_cache_to_gpu(cache)

    # Set requires_grad
    for name, p in lora_params.items():
        p['A'] = p['A'].detach().requires_grad_(True)
        p['B'] = p['B'].detach().requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p['A'] for p in lora_params.values()] +
        [p['B'] for p in lora_params.values()],
        lr=lr, weight_decay=0.01,
    )

    def lr_lambda(step):
        warmup = 20
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    losses = []
    n_breaths = 0
    t0 = time.time()
    current_cache = cache
    current_n = n_samples

    while step < max_steps:
        # ── BREATHE: refresh cache every K steps (not on step 0) ──
        if step > 0 and step % breath_interval == 0:
            print(f"\n[ETM:Train] === BREATH #{n_breaths+1} at step {step} ===")
            current_cache, current_n = breathe(
                lora_params, tokenizer, data, rank, alpha, use_dora,
                max_samples, max_seq_len, max_tokens, current_cache,
            )
            n_breaths += 1

        # ── One training step against current cache ──
        indices = torch.randperm(current_n).tolist()
        batch_idx = indices[:batch_size]
        if not batch_idx:
            batch_idx = [0]

        optimizer.zero_grad()
        total_loss = 0.0
        n_modules_this_step = 0

        for name, p in lora_params.items():
            if name not in current_cache:
                continue
            c = current_cache[name]
            A = p['A']  # (r, d_in)
            B = p['B']  # (d_out, r)
            scaling = c['scaling']

            # Gather batch from list — stack on GPU
            x = torch.stack([c['x_list'][i] for i in batch_idx])  # (b, tok, d_in)
            g = torch.stack([c['g_list'][i] for i in batch_idx])  # (b, tok, d_out)

            # LoRA forward: u = scaling * x @ A.T @ B.T
            z = x @ A.T        # (b, tok, r) — TINY
            out = z @ B.T       # (b, tok, d_out) — TINY
            u = scaling * out   # the LoRA delta

            # Bounded cosine-similarity loss: L = -cos(u, g)
            loss = cosine_loss(u, g)
            loss.backward()
            total_loss += loss.item()
            n_modules_this_step += 1

        # Gradient clipping
        if grad_clip > 0:
            params_to_clip = [p['A'] for p in lora_params.values()] + \
                             [p['B'] for p in lora_params.values()]
            torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)

        optimizer.step()
        scheduler.step()
        step += 1
        losses.append(total_loss / max(1, n_modules_this_step))

        if step % log_interval == 0 or step == 1:
            avg = sum(losses[-log_interval:]) / min(log_interval, len(losses))
            vram = torch.cuda.memory_allocated() / 1e9
            elapsed = time.time() - t0
            sps = step / elapsed
            print(f"  step {step:4d} | loss {avg:+.6f} | lr {scheduler.get_last_lr()[0]:.2e} "
                  f"| VRAM {vram:.4f} GB | {sps:.1f} steps/s | breaths {n_breaths}")

    t_train = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[ETM:Train] DONE: {step} steps in {t_train:.1f}s ({step/t_train:.1f} steps/s)")
    print(f"[ETM:Train] breaths taken: {n_breaths}")
    print(f"[ETM:Train] peak VRAM: {peak_vram:.4f} GB")
    print(f"[ETM:Train] final loss: {sum(losses[-10:])/min(10,len(losses)):+.6f}")

    # Free cache from GPU before merge
    free_cache_from_gpu(current_cache)

    return lora_params, losses, t_train, peak_vram, n_breaths


# ──────────────────────────────────────────────────────────────────────────────
# MERGE: reload base, inject trained params, save adapter, attempt merge
# ──────────────────────────────────────────────────────────────────────────────

def save_adapter(lora_params, output_dir, rank=24, alpha=48, use_dora=True):
    """Reload base, inject trained LoRA weights, save adapter + attempt merge.

    Handles DoRA + 4-bit: moves injected params to the module's device before
    assignment. Falls back to adapter-only save if merge fails.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ETM:Merge] reloading base to inject trained params...")
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
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", use_dora=use_dora,
    )
    model = get_peft_model(model, lora_config)

    # Inject trained weights — move to the module's device
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

    # Merge and save full model — try merge, fallback to adapter-only
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    try:
        if hasattr(model, 'merge_and_unload'):
            # For DoRA + 4-bit, we need to move params to the right device
            # before merge. merge_and_unload handles the math.
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))
            # Fix config — remove quantization_config for the merged model
            cfg_path = merged_dir / "config.json"
            if cfg_path.exists():
                cfg = json.load(open(cfg_path))
                cfg.pop('quantization_config', None)
                json.dump(cfg, open(cfg_path, 'w'), indent=2)
            print(f"[ETM:Merge] merged model saved to {merged_dir}")
            return str(merged_dir)
    except Exception as e:
        print(f"[ETM:Merge] merge failed ({str(e)[:120]})")
        print(f"[ETM:Merge] adapter saved at {output_dir}")
        print(f"[ETM:Merge] use vLLM with --enable-lora to serve the adapter")
    return str(output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# VRAM LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def log_vram(label):
    vram = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[VRAM] {label}: allocated={vram:.2f} GB, reserved={reserved:.2f} GB")
    return vram


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_etm(args):
    print("=" * 70)
    print("ETM v4: BREATHING CYCLE + BOUNDED COSINE LOSS")
    print("=" * 70)
    print(f"  Model:       {MODEL_ID}")
    print(f"  LoRA:        DoRA r={args.rank}, alpha={args.alpha}")
    print(f"  Breathing:   every {args.breath_interval} steps")
    print(f"  Loss:        cosine similarity (bounded [-1,1])")
    print(f"  Grad clip:   {args.grad_clip}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Data:        {args.data}")

    # Load training data
    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"[ETM] loaded {len(data)} training samples")

    # ── Phase 0: Load base + LoRA ──
    print("\n--- Phase 0: Load base + LoRA ---")
    model, tokenizer = load_base_with_lora(args.rank, args.alpha, args.use_dora)
    vram_loaded = log_vram("after load")

    # ── Phase 1: Initial cache ──
    print("\n--- Phase 1: Cache activations (initial) ---")
    cache, n_cached = cache_features(
        model, tokenizer, data, args.max_samples, args.max_seq_len, args.max_tokens,
    )

    # ── Phase 2: Extract LoRA params ──
    print("\n--- Phase 2: Extract LoRA params ---")
    lora_params = extract_lora_params(model)

    # ── Phase 3: EVICT base model ──
    print("\n--- Phase 3: EVICT base model ---")
    vram_before, vram_after = evict_model(model)
    del model

    # ── Phase 4: Breathing training ──
    print("\n--- Phase 4: Breathing training (base evicted between breaths) ---")
    lora_params, losses, t_train, peak_vram, n_breaths = train_breathing(
        lora_params, cache, n_cached, tokenizer, data,
        rank=args.rank, alpha=args.alpha, use_dora=args.use_dora,
        lr=args.lr, batch_size=args.batch_size, max_steps=args.max_steps,
        breath_interval=args.breath_interval, max_tokens=args.max_tokens,
        max_samples=args.max_samples, max_seq_len=args.max_seq_len,
        log_interval=args.log_interval, grad_clip=args.grad_clip,
    )

    # ── Phase 5: Merge ──
    print("\n--- Phase 5: Merge ---")
    merged_path = save_adapter(lora_params, args.output_dir, args.rank, args.alpha, args.use_dora)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("ETM v4 COMPLETE")
    print("=" * 70)
    total_params = sum(p['A'].numel() + p['B'].numel() for p in lora_params.values())
    print(f"  Trainable params:    {total_params/1e6:.1f}M")
    print(f"  Breaths taken:       {n_breaths}")
    print(f"  Breath interval:     every {args.breath_interval} steps")
    print(f"  Train time:          {t_train:.1f}s ({len(losses)} steps)")
    print(f"  Train speed:         {len(losses)/t_train:.1f} steps/s")
    print(f"  VRAM (base loaded):  {vram_before:.2f} GB")
    print(f"  VRAM (base evicted): {vram_after:.2f} GB")
    print(f"  VRAM (peak train):   {peak_vram:.2f} GB")
    print(f"  Final loss:          {sum(losses[-10:])/min(10,len(losses)):+.6f}")
    print(f"  Merged model:        {merged_path}")

    log = {
        "mode": "etm_v4_breathing_cosine",
        "model": MODEL_ID,
        "trainable_params_m": total_params / 1e6,
        "n_cached": n_cached,
        "n_breaths": n_breaths,
        "breath_interval": args.breath_interval,
        "train_time_s": t_train,
        "train_steps": len(losses),
        "train_speed_steps_per_s": len(losses) / t_train,
        "vram_base_loaded_gb": vram_before,
        "vram_base_evicted_gb": vram_after,
        "vram_peak_train_gb": peak_vram,
        "final_loss": sum(losses[-10:]) / min(10, len(losses)),
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "loss_type": "cosine_similarity",
        "grad_clip": args.grad_clip,
        "merged_model_path": merged_path,
    }
    with open(Path(args.output_dir) / "etm_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Log: {args.output_dir}/etm_log.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ETM v4: Breathing cycle + bounded cosine loss")
    ap.add_argument("--data", default="/root/etm/data/math_train_cot.jsonl")
    ap.add_argument("--output_dir", default="outputs/etm_v4")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--alpha", type=int, default=48)
    ap.add_argument("--use_dora", action="store_true", default=True)
    ap.add_argument("--no_dora", dest="use_dora", action="store_false")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=64,
                    help="tokens to subsample per sample for cache")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--breath_interval", type=int, default=20,
                    help="K: refresh cache every K steps")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="max grad norm for clipping (0=disable)")
    ap.add_argument("--log_interval", type=int, default=10)
    args = ap.parse_args()
    run_etm(args)

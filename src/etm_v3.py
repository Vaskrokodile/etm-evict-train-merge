#!/usr/bin/env python
"""
ETM v3: Cache on GPU, evict, train 15M in VRAM. No disk, no CPU round-trips.

THE PIPELINE:
  1. Load 4-bit base + DoRA r=24 (15.4M params) — ~12GB VRAM
  2. One forward+backward pass → cache (x, grad_h) ON GPU for each LoRA module
  3. Extract A and B as standalone GPU tensors
  4. del model — BASE IS GONE. VRAM drops to ~7GB (cache) + 0.06GB (adapters)
  5. Train A and B against GPU-resident cached features — blazing fast
  6. Merge: reload base, inject, save

Cache size math (500 samples, 512 tokens, 112 modules):
  per module: 500 × 512 × dim × 2 bytes × 2 tensors (x + grad_h)
  q_proj: 500×512×3584×2×2 = 3.7GB
  k_proj: 500×512×512×2×2  = 0.5GB
  v_proj: same = 0.5GB
  o_proj: same as q = 3.7GB
  per layer: ~8.4GB... × 28 layers = way too much

  FIX: only cache the LAST FEW LAYERS' worth, or subsample.
  Better: cache fewer samples (200) and shorter seqs (256).
  200×256×3584×2×2 = 1.9GB per q_proj module × 28 = 53GB for q alone. Still too much.

  REAL FIX: We don't need all 28 layers × 4 modules. We cache per-module
  but only keep the cache for the modules we're training — and we subsample
  tokens. Cache 200 samples × 64 tokens (random positions) × dim.
  200×64×3584×2×2 = 0.18GB per q_proj × 28 = 5.1GB for all q. Total ~7GB. Fits.

  EVEN BETTER: cache the LOW-RANK projection A·x (r=24) instead of full x.
  Then cache = 200×64×24×2 = tiny. grad_h still needs full dim but we can
  project it too: B·grad_h gives (200×64×24). Total cache ~negligible.

  We use the low-rank cache: store z=Ax and w=B·grad_h, both (n, tok, r).
"""
import argparse
import gc
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen2.5-Math-7B"


def load_base_with_lora(rank=24, alpha=48, use_dora=True):
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
# CACHE: forward+backward, capture (x, grad_h) ON GPU, project to low-rank
# ──────────────────────────────────────────────────────────────────────────────

def cache_on_gpu(model, tokenizer, data, max_samples, max_seq_len, max_tokens_per_sample=64):
    """One forward+backward pass. Cache x and grad_h for each LoRA module ON GPU.

    To keep cache small, we subsample max_tokens_per_sample random token positions
    from each sequence. This gives us representative features without storing
    every token.

    Returns:
      cache: {module_name: {'x': (n, tok, d_in), 'grad_h': (n, tok, d_out), 'scaling': float}}
      n_cached: int
    """
    cache = defaultdict(lambda: {'x': [], 'grad_h': []})
    lora_info = {}  # module_name -> scaling
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
                        idx = torch.randperm(seq_len)[:n_tok]
                        # Store on CPU to avoid VRAM blowup during caching
                        cache[mod_name]['x'].append(x[0, idx].detach().cpu())

                def bwd_hook(mod, grad_input, grad_output):
                    g = grad_output[0]
                    if g is not None:
                        seq_len = g.shape[1]
                        n_tok = min(max_tokens_per_sample, seq_len)
                        idx = torch.randperm(seq_len)[:n_tok]
                        cache[mod_name]['grad_h'].append(g[0, idx].detach().cpu())

                return fwd_hook, bwd_hook

            fwd_h, bwd_h = make_hooks(name)
            handles.append(module.register_forward_hook(fwd_h, with_kwargs=True))
            handles.append(module.register_full_backward_hook(bwd_h))

    # Run forward+backward
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
    print(f"[ETM:Cache] cached {n_cached} samples in {t_cache:.0f}s")

    # NO STACKING — keep cache as lists, gather during training (fast)
    final_cache = {}
    total_cache_bytes = 0
    for name in cache:
        if not cache[name]['x'] or not cache[name]['grad_h']:
            continue
        # Trim each tensor to max_tokens and move to CPU (already on CPU)
        n_tok = max_tokens_per_sample
        x_list = [t[:n_tok] for t in cache[name]['x']]
        g_list = [t[:n_tok] for t in cache[name]['grad_h']]
        n = min(len(x_list), len(g_list))
        final_cache[name] = {
            'x_list': x_list[:n],   # list of (tok, d_in) on CPU
            'g_list': g_list[:n],   # list of (tok, d_out) on CPU
            'scaling': lora_info[name],
        }
        total_cache_bytes += sum(t.numel() * t.element_size() for t in x_list[:n])
        total_cache_bytes += sum(t.numel() * t.element_size() for t in g_list[:n])

    print(f"[ETM:Cache] {len(final_cache)} modules cached (on CPU, will move to GPU after eviction)")
    print(f"[ETM:Cache] total cache: {total_cache_bytes/1e9:.2f} GB")
    for name in list(final_cache.keys())[:4]:
        c = final_cache[name]
        print(f"  {name}: n_samples={len(c['x_list'])} x0={tuple(c['x_list'][0].shape)} scaling={c['scaling']:.4f}")

    return final_cache, n_cached, t_cache


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACT: pull A and B as standalone GPU tensors
# ──────────────────────────────────────────────────────────────────────────────

def extract_lora_params(model):
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
    vram_before = torch.cuda.memory_allocated() / 1e9
    print(f"[ETM:Evict] VRAM before: {vram_before:.2f} GB")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    vram_after = torch.cuda.memory_allocated() / 1e9
    print(f"[ETM:Evict] VRAM after: {vram_after:.2f} GB")
    print(f"[ETM:Evict] MODEL DELETED. Only LoRA params + GPU cache remain.")
    return vram_before, vram_after


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN: train A and B against GPU-resident cached features. NO BASE.
# ──────────────────────────────────────────────────────────────────────────────

def train_isolated(lora_params, cache, n_samples,
                   lr=1e-4, epochs=3, batch_size=8, max_steps=300, log_interval=10):
    """Train LoRA A and B against cached (x, grad_h) features on GPU.

    surrogate_loss = (scaling * (x @ A.T @ B.T) * grad_h).sum() / N

    Gradients:
      ∂L/∂B = scaling * grad_h.T @ (A @ x)    — exact ∂L/∂B
      ∂L/∂A = scaling * (grad_h @ B).T @ x    — exact ∂L/∂A

    Each step: tiny matmuls on 15M params. No 7B forward. Blazing fast.
    """
    print(f"\n[ETM:Train] ISOLATED TRAINING — base model does not exist")
    print(f"  lr={lr}, epochs={epochs}, batch_size={batch_size}, max_steps={max_steps}")

    # Move cache to GPU now that base is evicted
    print("[ETM:Train] moving cache to GPU...")
    for name in cache:
        cache[name]['x_list'] = [t.cuda() for t in cache[name]['x_list']]
        cache[name]['g_list'] = [t.cuda() for t in cache[name]['g_list']]
    cache_vram = torch.cuda.memory_allocated() / 1e9
    print(f"[ETM:Train] cache on GPU, VRAM: {cache_vram:.2f} GB")

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
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    losses = []
    t0 = time.time()

    for epoch in range(epochs):
        print(f"\n[ETM:Train] Epoch {epoch+1}/{epochs}")
        indices = torch.randperm(n_samples).tolist()

        for batch_start in range(0, n_samples, batch_size):
            if step >= max_steps:
                break
            batch_idx = indices[batch_start:batch_start + batch_size]
            if not batch_idx:
                continue

            optimizer.zero_grad()
            total_loss = 0.0

            for name, p in lora_params.items():
                if name not in cache:
                    continue
                c = cache[name]
                A = p['A']  # (r, d_in)
                B = p['B']  # (d_out, r)
                scaling = c['scaling']

                # Gather batch from list — stack on GPU
                x = torch.stack([c['x_list'][i] for i in batch_idx])  # (b, tok, d_in)
                g = torch.stack([c['g_list'][i] for i in batch_idx])  # (b, tok, d_out)

                # LoRA forward: out = scaling * x @ A.T @ B.T
                z = x @ A.T       # (b, tok, r) — TINY
                out = z @ B.T      # (b, tok, d_out) — TINY
                scaled = scaling * out

                # Surrogate loss
                loss = (scaled * g).sum() / len(batch_idx)
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(total_loss)

            if step % log_interval == 0:
                avg = sum(losses[-log_interval:]) / log_interval
                vram = torch.cuda.memory_allocated() / 1e9
                elapsed = time.time() - t0
                sps = step / elapsed
                print(f"  step {step:4d} | loss {avg:.6f} | lr {scheduler.get_last_lr()[0]:.2e} "
                      f"| VRAM {vram:.4f} GB | {sps:.1f} steps/s")

            if step >= max_steps:
                break

    t_train = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[ETM:Train] DONE: {step} steps in {t_train:.1f}s ({step/t_train:.1f} steps/s)")
    print(f"[ETM:Train] peak VRAM: {peak_vram:.4f} GB")
    print(f"[ETM:Train] final loss: {sum(losses[-10:])/min(10,len(losses)):.6f}")
    return lora_params, losses, t_train, peak_vram


# ──────────────────────────────────────────────────────────────────────────────
# MERGE: reload base, inject trained params, save
# ──────────────────────────────────────────────────────────────────────────────

def save_adapter(lora_params, output_dir, rank=24, alpha=48):
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
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", use_dora=True,
    )
    model = get_peft_model(model, lora_config)

    # Inject trained weights — move to the module's device
    injected = 0
    for name, module in model.named_modules():
        if name in lora_params:
            for adapter_name in module.lora_A:
                # Get the device of the base weight
                base_device = module.base_layer.weight.device if hasattr(module, 'base_layer') else 'cuda'
                module.lora_A[adapter_name].weight.data = lora_params[name]['A'].detach().to(base_device)
                module.lora_B[adapter_name].weight.data = lora_params[name]['B'].detach().to(base_device)
                injected += 1
    print(f"[ETM:Merge] injected {injected} modules")

    # Save adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Merge and save full model — try merge, fallback to adapter-only
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    try:
        if hasattr(model, 'merge_and_unload'):
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))
            # Fix config
            cfg_path = merged_dir / "config.json"
            cfg = json.load(open(cfg_path))
            cfg.pop('quantization_config', None)
            json.dump(cfg, open(cfg_path, 'w'), indent=2)
            print(f"[ETM:Merge] merged model saved to {merged_dir}")
            return str(merged_dir)
    except Exception as e:
        print(f"[ETM:Merge] merge failed ({str(e)[:100]}), adapter saved at {output_dir}")
        print(f"[ETM:Merge] use vLLM with --enable-lora to serve the adapter")
    return str(output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_etm(args):
    print("=" * 70)
    print("ETM v3: TRUE EVICTION — cache on GPU, del model, train 15M in VRAM")
    print("=" * 70)

    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"[ETM] loaded {len(data)} training samples")

    # Phase 0: Load
    print("\n--- Phase 0: Load base + LoRA ---")
    model, tokenizer = load_base_with_lora(args.rank, args.alpha, args.use_dora)
    vram_loaded = torch.cuda.memory_allocated() / 1e9
    print(f"[ETM] VRAM after load: {vram_loaded:.2f} GB")

    # Phase 1: Cache on GPU
    print("\n--- Phase 1: Cache activations on GPU ---")
    cache, n_cached, t_cache = cache_on_gpu(
        model, tokenizer, data, args.max_samples, args.max_seq_len, args.max_tokens
    )

    # Phase 2: Extract
    print("\n--- Phase 2: Extract LoRA params ---")
    lora_params = extract_lora_params(model)

    # Phase 3: EVICT
    print("\n--- Phase 3: EVICT base model ---")
    vram_before, vram_after = evict_model(model)
    del model

    # Phase 4: Train in isolation
    print("\n--- Phase 4: Train in isolation (NO BASE) ---")
    lora_params, losses, t_train, peak_vram = train_isolated(
        lora_params, cache, n_cached,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        max_steps=args.max_steps, log_interval=args.log_interval,
    )

    # Free cache before merge (need VRAM for base reload)
    del cache
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 5: Merge
    print("\n--- Phase 5: Merge ---")
    merged_path = save_adapter(lora_params, args.output_dir, args.rank, args.alpha)

    # Summary
    print("\n" + "=" * 70)
    print("ETM v3 COMPLETE")
    print("=" * 70)
    total_params = sum(p['A'].numel()+p['B'].numel() for p in lora_params.values())
    print(f"  Trainable params:    {total_params/1e6:.1f}M")
    print(f"  Cache time:          {t_cache:.0f}s")
    print(f"  Train time:          {t_train:.1f}s ({len(losses)} steps)")
    print(f"  Train speed:         {len(losses)/t_train:.1f} steps/s")
    print(f"  VRAM (base loaded):  {vram_before:.2f} GB")
    print(f"  VRAM (base evicted): {vram_after:.2f} GB")
    print(f"  VRAM (peak train):   {peak_vram:.2f} GB")
    print(f"  VRAM reduction:      {vram_before/peak_vram:.0f}x")
    print(f"  Merged model:        {merged_path}")

    log = {
        "mode": "etm_v3_true_eviction_gpu_cache",
        "trainable_params_m": total_params/1e6,
        "n_cached": n_cached,
        "cache_time_s": t_cache,
        "train_time_s": t_train,
        "train_steps": len(losses),
        "train_speed_steps_per_s": len(losses)/t_train,
        "vram_base_loaded_gb": vram_before,
        "vram_base_evicted_gb": vram_after,
        "vram_peak_train_gb": peak_vram,
        "vram_reduction_x": vram_before / peak_vram,
        "final_loss": sum(losses[-10:])/min(10,len(losses)),
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
        "merged_model_path": merged_path,
    }
    with open(Path(args.output_dir) / "etm_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Log: {args.output_dir}/etm_log.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ETM v3: True eviction, GPU cache")
    ap.add_argument("--data", default="data/math_train_cot.jsonl")
    ap.add_argument("--output_dir", default="outputs/etm_v3")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--alpha", type=int, default=48)
    ap.add_argument("--use_dora", action="store_true", default=True)
    ap.add_argument("--no_dora", dest="use_dora", action="store_false")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=64, help="tokens to subsample per sample for cache")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--log_interval", type=int, default=10)
    args = ap.parse_args()
    run_etm(args)

#!/usr/bin/env python
"""ETM v5b: MSE loss with gentle η + periodic breathing.

Key fixes from v5:
  1. η is 10x smaller (target ||u|| = 0.01 * ||x||, not 0.1)
  2. Breathing every 500 steps (refreshes stale gradient)
  3. Only 3000 steps (avoid over-training against stale gradient)
  4. Keep MSE loss (controls direction + magnitude)
"""
import argparse, gc, json, math, time
from pathlib import Path
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

MODEL_ID = "Qwen/Qwen2.5-Math-7B"

def load_base_with_lora(rank=64, alpha=128):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(r=rank, lora_alpha=alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", use_dora=False)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def cache_features(model, tokenizer, data, max_samples, max_seq_len, max_tokens=64, lora_params_to_inject=None):
    if lora_params_to_inject is not None:
        injected = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if name in lora_params_to_inject:
                    for adapter_name in module.lora_A:
                        dev = module.lora_A[adapter_name].weight.device
                        module.lora_A[adapter_name].weight.data = lora_params_to_inject[name]['A'].detach().to(dev)
                        module.lora_B[adapter_name].weight.data = lora_params_to_inject[name]['B'].detach().to(dev)
                        injected += 1
        print(f"  [Cache] injected {injected} modules")

    cache = defaultdict(lambda: {'x': [], 'grad_h': []})
    lora_info = {}
    handles = []
    grad_norms, x_norms = [], []

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            for adapter_name in module.lora_A:
                lora_info[name] = module.scaling[adapter_name]
            def make_hooks(mod_name):
                def fwd_hook(mod, args, kwargs, output):
                    x = args[0] if args else kwargs.get('input', None)
                    if x is not None:
                        seq_len = x.shape[1]
                        n_tok = min(max_tokens, seq_len)
                        idx = torch.randperm(seq_len, device=x.device)[:n_tok]
                        x_sub = x[0, idx].detach()
                        cache[mod_name]['x'].append(x_sub.to(torch.bfloat16).cpu())
                        x_norms.append(x_sub.norm().item())
                def bwd_hook(mod, grad_input, grad_output):
                    g = grad_output[0]
                    if g is not None:
                        seq_len = g.shape[1]
                        n_tok = min(max_tokens, seq_len)
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
        enc = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=max_seq_len, padding=False)
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
            print(f"  cached {i+1}/{max_samples} | VRAM {vram:.2f} GB | {time.time()-t0:.0f}s")
    for h in handles: h.remove()
    t_cache = time.time() - t0
    for name in cache:
        cache[name]['scaling'] = lora_info.get(name, 1.0)
    grad_norms.sort(); x_norms.sort()
    median_grad = grad_norms[len(grad_norms)//2] if grad_norms else 1.0
    median_x = x_norms[len(x_norms)//2] if x_norms else 1.0
    # GENTLE η: target ||u|| = 0.01 * ||x|| (1% perturbation, not 10%)
    eta = 0.01 * median_x / (median_grad + 1e-8)
    print(f"\n  [Cache] DONE: {n_cached} samples in {t_cache:.1f}s")
    print(f"  [Cache] median ||x||={median_x:.4f}, ||grad_h||={median_grad:.6f}")
    print(f"  [Cache] η={eta:.4f} (GENTLE: target ||u||={0.01*median_x:.4f})")
    total_bytes = sum(t.nelement()*t.element_size() for name in cache for t in cache[name]['x']+cache[name]['grad_h'])
    print(f"  [Cache] CPU memory: {total_bytes/1e9:.1f} GB")
    return dict(cache), n_cached, eta

def extract_lora_params(model):
    lora_params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            for adapter_name in module.lora_A:
                A = module.lora_A[adapter_name].weight.data.clone().cuda()
                B = module.lora_B[adapter_name].weight.data.clone().cuda()
                lora_params[name] = {'A': A, 'B': B}
    print(f"  [Extract] {len(lora_params)} modules")
    return lora_params

def evict_model(model):
    vram_before = torch.cuda.memory_allocated() / 1e9
    del model; gc.collect(); torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1e9
    print(f"  [Evict] VRAM: {vram_before:.2f} → {vram_after:.2f} GB")
    return vram_before, vram_after

def move_cache_to_gpu(cache):
    total_bytes = 0
    for name in cache:
        cache[name]['x'] = [t.cuda() for t in cache[name]['x']]
        cache[name]['grad_h'] = [t.cuda() for t in cache[name]['grad_h']]
        for t in cache[name]['x']: total_bytes += t.nelement()*t.element_size()
        for t in cache[name]['grad_h']: total_bytes += t.nelement()*t.element_size()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  [Cache→GPU] {total_bytes/1e9:.1f} GB, VRAM: {vram:.2f} GB")

def free_cache_from_gpu(cache):
    for name in cache:
        cache[name]['x'] = [t.cpu() for t in cache[name]['x']]
        cache[name]['grad_h'] = [t.cpu() for t in cache[name]['grad_h']]
    gc.collect(); torch.cuda.empty_cache()
    print(f"  [Cache→CPU] VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

def mse_loss(u, v, eta=1.0):
    diff = u + eta * v
    return (diff * diff).mean()

def cosine_sim(u, v, eps=1e-8):
    u_flat = u.reshape(-1, u.shape[-1])
    v_flat = v.reshape(-1, v.shape[-1])
    dot = (u_flat * v_flat).sum(dim=-1)
    return (dot / (u_flat.norm(dim=-1) * v_flat.norm(dim=-1) + eps)).mean().item()

def breathe(lora_params, tokenizer, data, rank, alpha, max_samples, max_seq_len, max_tokens, old_cache):
    print("\n  ┌─ BREATH ─────────────────────────────")
    free_cache_from_gpu(old_cache)
    del old_cache; gc.collect(); torch.cuda.empty_cache()
    print("  [Breath] reloading 4-bit base...")
    model, _ = load_base_with_lora(rank, alpha)
    print("  [Breath] refreshing cache...")
    new_cache, n_cached, eta = cache_features(model, tokenizer, data, max_samples, max_seq_len, max_tokens, lora_params_to_inject=lora_params)
    print("  [Breath] evicting base...")
    evict_model(model)
    del model
    move_cache_to_gpu(new_cache)
    print("  └─ BREATH DONE ────────────────────────\n")
    return new_cache, n_cached, eta

def train(lora_params, cache, n_samples, eta, tokenizer, data, rank, alpha,
          lr=1e-4, batch_size=16, max_steps=3000, breath_interval=500,
          max_tokens=64, max_samples=500, max_seq_len=512,
          log_interval=50, grad_clip=1.0, checkpoint_dir=None, checkpoint_interval=1000):
    print(f"\n[ETM:Train] MSE + BREATHING")
    print(f"  lr={lr}, batch={batch_size}, steps={max_steps}, breath every {breath_interval}")
    print(f"  η={eta:.4f}, grad_clip={grad_clip}")
    move_cache_to_gpu(cache)
    for name, p in lora_params.items():
        p['A'] = p['A'].detach().requires_grad_(True)
        p['B'] = p['B'].detach().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [p['A'] for p in lora_params.values()] + [p['B'] for p in lora_params.values()],
        lr=lr, weight_decay=0.01)
    warmup = 100
    def lr_lambda(step):
        if step < warmup: return step / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step = 0; losses = []; cos_sims = []; n_breaths = 0; t0 = time.time()
    current_cache = cache; current_n = n_samples; current_eta = eta
    while step < max_steps:
        if step > 0 and step % breath_interval == 0:
            print(f"\n[ETM:Train] === BREATH #{n_breaths+1} at step {step} ===")
            current_cache, current_n, current_eta = breathe(
                lora_params, tokenizer, data, rank, alpha,
                max_samples, max_seq_len, max_tokens, current_cache)
            n_breaths += 1
        indices = torch.randperm(current_n).tolist()
        batch_idx = indices[:batch_size]
        optimizer.zero_grad()
        total_loss = 0.0; total_cos = 0.0; total_u = 0.0; total_g = 0.0; n_mod = 0
        for name, p in lora_params.items():
            if name not in current_cache: continue
            c = current_cache[name]
            A = p['A']; B = p['B']; scaling = c['scaling']
            x = torch.stack([c['x'][i] for i in batch_idx]).float()
            g = torch.stack([c['grad_h'][i] for i in batch_idx]).float()
            z = x @ A.T; out = z @ B.T; u = scaling * out
            loss = mse_loss(u, g, current_eta)
            loss.backward()
            total_loss += loss.item()
            total_cos += cosine_sim(u.detach(), g.detach())
            total_u += u.detach().norm().item() / math.sqrt(u.numel())
            total_g += g.detach().norm().item() / math.sqrt(g.numel())
            n_mod += 1
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p['A'] for p in lora_params.values()] + [p['B'] for p in lora_params.values()], grad_clip)
        optimizer.step(); scheduler.step(); step += 1
        avg_loss = total_loss/max(1,n_mod); avg_cos = total_cos/max(1,n_mod)
        avg_u = total_u/max(1,n_mod); avg_g = total_g/max(1,n_mod)
        losses.append(avg_loss); cos_sims.append(avg_cos)
        if step % log_interval == 0 or step == 1:
            vram = torch.cuda.memory_allocated() / 1e9
            sps = step / (time.time() - t0)
            rl = sum(losses[-log_interval:])/min(log_interval, len(losses))
            rc = sum(cos_sims[-log_interval:])/min(log_interval, len(cos_sims))
            print(f"  step {step:5d} | loss {rl:.6f} | cos {rc:+.4f} | ||u|| {avg_u:.6f} | ||g|| {avg_g:.6f} | lr {scheduler.get_last_lr()[0]:.2e} | {sps:.1f} steps/s | breaths {n_breaths}")
        if checkpoint_dir and step % checkpoint_interval == 0:
            ckpt = Path(checkpoint_dir) / f"checkpoint_step{step}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'step': step, 'lora_params': {k: {'A': v['A'].detach().cpu(), 'B': v['B'].detach().cpu()} for k, v in lora_params.items()}, 'eta': current_eta}, str(ckpt))
            print(f"  [Checkpoint] saved to {ckpt}")
    t_train = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[ETM:Train] DONE: {step} steps in {t_train:.1f}s ({step/t_train:.1f} steps/s)")
    print(f"[ETM:Train] breaths: {n_breaths}, peak VRAM: {peak_vram:.2f} GB")
    print(f"[ETM:Train] final loss: {sum(losses[-10:])/min(10,len(losses)):.6f}, cos: {sum(cos_sims[-10:])/min(10,len(cos_sims)):+.4f}")
    free_cache_from_gpu(current_cache)
    return lora_params, losses, t_train, peak_vram, cos_sims, n_breaths

def save_and_merge(lora_params, output_dir, rank=64, alpha=128):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ETM:Merge] loading base in bf16...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    lora_config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", use_dora=False)
    model = get_peft_model(model, lora_config)
    injected = 0
    for name, module in model.named_modules():
        if name in lora_params:
            for adapter_name in module.lora_A:
                dev = module.lora_A[adapter_name].weight.device
                module.lora_A[adapter_name].weight.data = lora_params[name]['A'].detach().to(dev)
                module.lora_B[adapter_name].weight.data = lora_params[name]['B'].detach().to(dev)
                injected += 1
    print(f"[ETM:Merge] injected {injected} modules")
    model.save_pretrained(str(output_dir)); tokenizer.save_pretrained(str(output_dir))
    merged_dir = output_dir / "merged"; merged_dir.mkdir(exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    cfg = json.load(open(merged_dir / "config.json"))
    assert "quantization_config" not in cfg
    print(f"[ETM:Merge] DONE -> {merged_dir}")
    return str(merged_dir)

def run_etm(args):
    print("=" * 70)
    print("ETM v5b: GENTLE MSE + PERIODIC BREATHING")
    print("=" * 70)
    print(f"  LoRA: r={args.rank}, alpha={args.alpha} | η=gentle (1% of ||x||)")
    print(f"  Steps: {args.max_steps} | Breath every {args.breath_interval}")
    data = []
    with open(args.data) as f:
        for line in f: data.append(json.loads(line))
    print(f"[ETM] loaded {len(data)} samples")
    print("\n--- Phase 0: Load base + LoRA ---")
    model, tokenizer = load_base_with_lora(args.rank, args.alpha)
    print("\n--- Phase 1: Cache ---")
    cache, n_cached, eta = cache_features(model, tokenizer, data, args.max_samples, args.max_seq_len, args.max_tokens)
    print("\n--- Phase 2: Extract ---")
    lora_params = extract_lora_params(model)
    print("\n--- Phase 3: Evict ---")
    vram_before, vram_after = evict_model(model)
    del model
    print("\n--- Phase 3.5: Cache to GPU ---")
    move_cache_to_gpu(cache)
    print("\n--- Phase 4: Train ---")
    lora_params, losses, t_train, peak_vram, cos_sims, n_breaths = train(
        lora_params, cache, n_cached, eta, tokenizer, data, args.rank, args.alpha,
        lr=args.lr, batch_size=args.batch_size, max_steps=args.max_steps,
        breath_interval=args.breath_interval, max_tokens=args.max_tokens,
        max_samples=args.max_samples, max_seq_len=args.max_seq_len,
        log_interval=args.log_interval, grad_clip=args.grad_clip,
        checkpoint_dir=args.output_dir, checkpoint_interval=args.checkpoint_interval)
    # Free GPU before merge
    print("\n--- Phase 5: Merge ---")
    merged_path = save_and_merge(lora_params, args.output_dir, args.rank, args.alpha)
    print("\n" + "=" * 70)
    print("ETM v5b COMPLETE")
    print("=" * 70)
    total_params = sum(p['A'].numel() + p['B'].numel() for p in lora_params.values())
    print(f"  Params: {total_params/1e6:.1f}M | Steps: {len(losses)} | Breaths: {n_breaths}")
    print(f"  Time: {t_train:.1f}s | Final loss: {sum(losses[-10:])/min(10,len(losses)):.6f}")
    print(f"  Final cos: {sum(cos_sims[-10:])/min(10,len(cos_sims)):+.4f}")
    print(f"  Merged: {merged_path}")
    log = {"mode": "etm_v5b", "model": MODEL_ID, "params_m": total_params/1e6,
           "steps": len(losses), "breaths": n_breaths, "eta": eta,
           "final_loss": sum(losses[-10:])/min(10,len(losses)),
           "final_cos": sum(cos_sims[-10:])/min(10,len(cos_sims)),
           "merged": merged_path}
    with open(Path(args.output_dir)/"etm_log.json", "w") as f: json.dump(log, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/etm/data/math_train_v5.jsonl")
    ap.add_argument("--output_dir", default="outputs/etm_v5b")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--breath_interval", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--checkpoint_interval", type=int, default=1000)
    args = ap.parse_args()
    run_etm(args)

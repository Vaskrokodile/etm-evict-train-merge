#!/usr/bin/env python
"""Merge ETM v5 checkpoint into Qwen2.5-Math-7B in bf16."""
import torch, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

CKPT = "/root/etm/outputs/etm_v5/checkpoint_step2000.pt"
BASE = "Qwen/Qwen2.5-Math-7B"
OUT = "/root/etm/outputs/etm_v5/merged_step2000"
RANK = 64
ALPHA = 128

print(f"[merge] loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu")
lora_params = ckpt["lora_params"]
print(f"[merge] checkpoint step: {ckpt['step']}, modules: {len(lora_params)}")

print(f"[merge] loading base {BASE} in bf16...")
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

lora_config = LoraConfig(
    r=RANK, lora_alpha=ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", use_dora=False,
)
model = get_peft_model(model, lora_config)

print("[merge] injecting trained weights...")
injected = 0
for name, module in model.named_modules():
    if name in lora_params:
        for adapter_name in module.lora_A:
            dev = module.lora_A[adapter_name].weight.device
            module.lora_A[adapter_name].weight.data = lora_params[name]["A"].to(dev)
            module.lora_B[adapter_name].weight.data = lora_params[name]["B"].to(dev)
            injected += 1
print(f"[merge] injected {injected} modules")

# Save adapter
adapter_dir = "/root/etm/outputs/etm_v5"
model.save_pretrained(adapter_dir)
tok.save_pretrained(adapter_dir)
print(f"[merge] adapter saved to {adapter_dir}")

# Merge
print("[merge] merging...")
merged = model.merge_and_unload()
Path(OUT).mkdir(parents=True, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)

cfg = json.load(open(f"{OUT}/config.json"))
assert "quantization_config" not in cfg
print(f"[merge] DONE -> {OUT} (dtype={cfg.get('dtype')})")

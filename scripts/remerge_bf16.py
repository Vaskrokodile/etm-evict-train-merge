#!/usr/bin/env python
"""Re-merge the DoRA adapter into Qwen2.5-Math-7B loaded in full bf16.

The original merge used a 4-bit NF4 base, which produced malformed weight
shapes (start(0)+length(18944) exceeds dimension size(1)). This script
loads the base in full precision, applies the saved adapter, and merges
correctly.
"""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-Math-7B"
ADAPTER = "/root/etm/outputs/etm_v4_fixed"
OUT = "/root/etm/outputs/etm_v4_fixed/merged_bf16"

print(f"[remerge] loading base {BASE} in bf16 (no quantization)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

print(f"[remerge] applying DoRA adapter from {ADAPTER}...")
model = PeftModel.from_pretrained(model, ADAPTER)

print("[remerge] merging adapter into base weights...")
model = model.merge_and_unload()

Path(OUT).mkdir(parents=True, exist_ok=True)
print(f"[remerge] saving merged model to {OUT}...")
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)

# Verify config has no quantization_config
import json
cfg = json.load(open(f"{OUT}/config.json"))
assert "quantization_config" not in cfg, "quantization_config leaked into merged config!"
print(f"[remerge] config dtype={cfg.get('dtype')}, no quantization_config. OK")
print(f"[remerge] DONE -> {OUT}")

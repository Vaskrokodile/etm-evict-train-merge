import torch, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

CKPT = "/root/etm/outputs/etm_v5b/checkpoint_step1000.pt"
BASE = "Qwen/Qwen2.5-Math-7B"
OUT = "/root/etm/outputs/etm_v5b/merged"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
lora_params = ckpt["lora_params"]
print(f"step: {ckpt['step']}, modules: {len(lora_params)}")

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
cfg = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", use_dora=False)
model = get_peft_model(model, cfg)
for name, module in model.named_modules():
    if name in lora_params:
        for ad in module.lora_A:
            dev = module.lora_A[ad].weight.device
            module.lora_A[ad].weight.data = lora_params[name]["A"].to(dev)
            module.lora_B[ad].weight.data = lora_params[name]["B"].to(dev)
merged = model.merge_and_unload()
Path(OUT).mkdir(parents=True, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
c = json.load(open(f"{OUT}/config.json"))
assert "quantization_config" not in c
print(f"DONE -> {OUT}")

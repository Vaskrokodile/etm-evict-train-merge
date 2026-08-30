#!/bin/bash
cd /root/etm
python -m vllm.entrypoints.openai.api_server   --model /root/etm/outputs/etm_v4_fixed/merged_bf16   --served-model-name trained   --dtype bfloat16 --max-model-len 4096   --gpu-memory-utilization 0.9   --host 0.0.0.0 --port 8000   > /root/etm/vllm_merged.log 2>&1

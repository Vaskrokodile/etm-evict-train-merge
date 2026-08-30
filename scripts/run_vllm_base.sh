#!/bin/bash
cd /root/etm
python -m vllm.entrypoints.openai.api_server   --model Qwen/Qwen2.5-Math-7B   --served-model-name base   --dtype bfloat16 --max-model-len 4096   --gpu-memory-utilization 0.9   --host 0.0.0.0 --port 8001   > /root/etm/vllm_base.log 2>&1

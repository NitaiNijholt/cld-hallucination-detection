#!/usr/bin/env bash
# Deploy CLD judge model to a RunPod GPU instance.
#
# Prerequisites:
#   1. runpodctl installed:  pip install runpod
#   2. API key set:          export RUNPOD_API_KEY=<key>
#   3. Adapters accessible via a public URL or uploaded to RunPod volume
#
# Usage:
#   ./deploy_runpod.sh              # defaults to 14B PoC
#   MODEL=qwen3-32b ./deploy_runpod.sh
#   MODEL=qwen3-235b ./deploy_runpod.sh
set -euo pipefail

MODEL="${MODEL:-qwen3-14b}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
RUNPOD_GPU="${RUNPOD_GPU:-}"

case "$MODEL" in
    qwen3-14b)
        BASE_MODEL="Qwen/Qwen3-14B"
        TP=1
        QUANT=""
        : "${RUNPOD_GPU:=NVIDIA A100 80GB PCIe}"
        : "${ADAPTER_DIR:=finetuning/runs/canonical_shared/62a7584-shared-edge-v1/qwen3_14b_judge_shared_lit_verdict_only/lora_adapter}"
        ;;
    qwen3-32b)
        BASE_MODEL="Qwen/Qwen3-32B"
        TP=1
        QUANT=""
        : "${RUNPOD_GPU:=NVIDIA A100 80GB PCIe}"
        : "${ADAPTER_DIR:=finetuning/runs/canonical_shared/62a7584-shared-edge-v1/qwen3_32b_judge_shared_lit_verdict_only/lora_adapter}"
        ;;
    qwen3-235b)
        BASE_MODEL="Qwen/Qwen3-235B-A22B"
        TP=2
        QUANT="awq"
        : "${RUNPOD_GPU:=NVIDIA H100 80GB HBM3}"
        : "${ADAPTER_DIR:=finetuning/runs/canonical_shared/62a7584-shared-edge-v1/qwen3_235b_judge_shared_lit_verdict_only_s1500_seed42_trainonly/lora_adapter}"
        ;;
    *)
        echo "Unknown model: $MODEL (valid: qwen3-14b, qwen3-32b, qwen3-235b)"
        exit 1
        ;;
esac

cat <<EOF
====================================
  CLD Judge — RunPod Deployment
====================================
  Model     : $BASE_MODEL
  TP Size   : $TP
  Quant     : ${QUANT:-none (FP16)}
  GPU Type  : $RUNPOD_GPU
  Adapter   : $ADAPTER_DIR
====================================

To deploy manually on RunPod:

1. Create a GPU Pod:
   - Image:  vllm/vllm-openai:latest
   - GPU:    $RUNPOD_GPU  $([ "$TP" -gt 1 ] && echo "(x$TP)")
   - Volume: /workspace (50GB+, for HF cache)
   - Expose: port 8000

2. SSH into the pod, then run:

   # Upload adapter from local machine (run from your local terminal):
   rsync -avz --progress $ADAPTER_DIR/ root@<POD_IP>:/workspace/adapter/

   # On the pod — start vLLM:
   python -m vllm.entrypoints.openai.api_server \\
       --model $BASE_MODEL \\
       --host 0.0.0.0 --port 8000 \\
       --tensor-parallel-size $TP \\
       --max-model-len 2048 \\
       --gpu-memory-utilization 0.90 \\
       --enable-lora \\
       --lora-modules "cld_judge=/workspace/adapter" \\
       --max-loras 1 --max-lora-rank 32$([ -n "$QUANT" ] && echo " \\\\
       --quantization $QUANT")

3. Test from local machine:

   curl http://<POD_IP>:8000/health

   curl -X POST http://<POD_IP>:8000/v1/chat/completions \\
       -H "Content-Type: application/json" \\
       -d '{
         "model": "cld_judge",
         "messages": [
           {"role": "system", "content": "You are a causal diagram expert evaluating the quality of a causal explanation."},
           {"role": "user", "content": "SOURCE: Depression\\nTARGET: Social withdrawal\\nRELATIONSHIP: causes\\nEXPLANATION: Depression leads to reduced motivation and energy, causing individuals to withdraw from social activities.\\n\\nAssess whether the causal reasoning is logically sound based on:\\n- Plausibility: is there a clear causal mechanism?\\n- Temporality: does the cause precede the effect?\\n- Strength: is the relationship substantial?\\n- Coherence: is the reasoning internally consistent?\\n\\nRespond with VERDICT only."}
         ],
         "temperature": 0.0,
         "max_tokens": 512
       }'

EOF

# CLD Judge — Model Serving

Serve finetuned CLD hallucination detection models via vLLM with an OpenAI-compatible API.

## Available Models

| Preset | Base Model | TP | GPU Requirement | Adapter Size |
|---|---|---|---|---|
| `qwen3-14b` | `Qwen/Qwen3-14B` | 1 | 1x A100-80GB | 506M |
| `qwen3-32b` | `Qwen/Qwen3-32B` | 1 | 1x A100-80GB | 1.1G |
| `qwen3-235b` | `Qwen/Qwen3-235B-A22B` | 2 | 2x H100-80GB (AWQ) | 25G |

## Quick Start (Docker)

```bash
# Build the image (from finetuning/ directory)
cd finetuning
docker build -f serving/Dockerfile -t cld-judge .

# Serve Qwen3-14B with LoRA adapter
docker run --gpus 1 \
    -e CLD_PRESET=qwen3-14b \
    -e CLD_LORA_PATH=/adapters/lora_adapter \
    -v $(pwd)/../runs/canonical_shared/62a7584-shared-edge-v1/qwen3_14b_judge_shared_lit_verdict_only/lora_adapter:/adapters/lora_adapter:ro \
    -v hf_cache:/root/.cache/huggingface \
    -p 8000:8000 \
    cld-judge
```

## Quick Start (Direct vLLM)

```bash
pip install vllm

python -m finetuning.src.serving.serve_judge_vllm \
    --preset qwen3-14b \
    --lora_path finetuning/runs/canonical_shared/62a7584-shared-edge-v1/qwen3_14b_judge_shared_lit_verdict_only/lora_adapter
```

## Environment Variables

All args can be set via `CLD_*` env vars:

| Variable | Default | Description |
|---|---|---|
| `CLD_PRESET` | | Model preset (qwen3-14b, qwen3-32b, qwen3-235b) |
| `CLD_BASE_MODEL` | | HF model id (overrides preset) |
| `CLD_LORA_PATH` | | Path to LoRA adapter directory |
| `CLD_LORA_NAME` | `cld_judge` | Adapter name for API requests |
| `CLD_TENSOR_PARALLEL_SIZE` | from preset | Number of GPUs for TP |
| `CLD_MAX_MODEL_LEN` | `2048` | Max sequence length |
| `CLD_QUANTIZATION` | | `awq` or `gptq` |
| `CLD_PORT` | `8000` | Server port |

## Testing the Server

```bash
# Health check
curl http://localhost:8000/health

# Inference
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "cld_judge",
      "messages": [
        {"role": "system", "content": "You are a causal diagram expert evaluating the quality of a causal explanation."},
        {"role": "user", "content": "SOURCE: Depression\nTARGET: Social withdrawal\nRELATIONSHIP: causes\nEXPLANATION: Depression leads to reduced motivation and energy, causing individuals to withdraw from social activities.\n\nAssess whether the causal reasoning is logically sound.\nRespond with VERDICT only."}
      ],
      "temperature": 0.0,
      "max_tokens": 512
    }'
```

## Parity Test

After deploying, validate that serving output matches evaluation results:

```bash
python -m finetuning.src.serving.parity_test \
    --server_url http://localhost:8000 \
    --model_name cld_judge \
    --test_data finetuning/data/canonical_shared/62a7584-shared-edge-v1/verdict_only/shared_test_lit.xlsx \
    --n_samples 20
```

## AWQ Quantization (for 235B)

The 235B MoE model needs AWQ quantization to fit on 2x H100. Run on a GPU instance:

```bash
pip install autoawq

python -m finetuning.src.serving.quantize_awq \
    --model_path Qwen/Qwen3-235B-A22B \
    --output_path /workspace/qwen3_235b_awq \
    --calib_samples 128
```

Then serve the AWQ model with the LoRA adapter:

```bash
python -m finetuning.src.serving.serve_judge_vllm \
    --base_model /workspace/qwen3_235b_awq \
    --lora_path /workspace/adapter \
    --tensor_parallel_size 2 \
    --quantization awq
```

## Cloud Deployment (RunPod)

```bash
# Print deployment instructions for each model
./deploy_runpod.sh                      # 14B (default)
MODEL=qwen3-32b ./deploy_runpod.sh     # 32B
MODEL=qwen3-235b ./deploy_runpod.sh    # 235B
```

## docker-compose (Multi-model)

```bash
# Serve a specific model
docker compose up cld-judge-14b

# Or all models (requires multiple GPUs)
docker compose up
```

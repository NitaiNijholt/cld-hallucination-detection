# vLLM Serving for CLD Judge

Serve finetuned CLD judge models via vLLM's OpenAI-compatible API.

## Quick start

```bash
# With LoRA adapter
python -m finetuning.src.serving.serve_judge_vllm \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --lora_path finetuning/runs/mistral7b_judge_gtsynth/lora_adapter \
    --host 0.0.0.0 --port 8000

# Merged model only (no LoRA)
python -m finetuning.src.serving.serve_judge_vllm \
    --base_model finetuning/runs/mistral7b_judge_gtsynth/merged_fp16
```

## SLURM (Snellius)

```bash
# Default: Mistral-7B + LoRA from mistral7b_judge_gtsynth
sbatch finetuning/jobs/serve_judge_vllm.job

# Custom model/adapter via env
LORA_PATH=finetuning/runs/qwen25_7b_judge_gtsynth/lora_adapter \
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
sbatch finetuning/jobs/serve_judge_vllm.job
```

## OpenAI client example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# With LoRA: use adapter name as model
response = client.chat.completions.create(
    model="cld_judge",
    messages=[{"role": "user", "content": "Given the citation: ... Does it support the claim? ..."}],
)
print(response.choices[0].message.content)
```

## LoRA adapter paths

When `--lora_path` points to a directory with `adapter_config.json`, vLLM loads it with `--enable-lora --lora-modules cld_judge=<path>`. Request the LoRA by using `model="cld_judge"` in the API. When serving a merged model (no `--lora_path`), use `model="mistralai/Mistral-7B-Instruct-v0.2"` or the merged model ID.

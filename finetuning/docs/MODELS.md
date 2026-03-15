# Alternative Base Models for CLD Judge Finetuning

Reference for newer, more promising models than Mistral-7B-Instruct-v0.2.
All are open-weight, instruction-tuned, and compatible with QLoRA.

---

## Recommended (7B–14B, single H100)

| Model | HF ID | Params | Context | Notes |
|-------|-------|--------|---------|-------|
| **Qwen 2.5 7B Instruct** | `Qwen/Qwen2.5-7B-Instruct` | 7B | 128K | Strong benchmarks, Apache 2.0 |
| **Qwen 2.5 14B Instruct** | `Qwen/Qwen2.5-14B-Instruct` | 14B | 128K | Best 14B-class on many benchmarks |
| **Gemma 3 12B IT** | `google/gemma-3-12b-it` | 12B | 128K | Multimodal, 140+ languages (March 2025) |
| **Llama 3.1 8B Instruct** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | 128K | Solid baseline, needs HF token |
| **Nemotron Mini 4B Instruct** | `nvidia/Nemotron-Mini-4B-Instruct` | 4B | 8K | Fast, compact; pruned from Nemotron-4 15B |

---

## Larger (32B+, multi-GPU or high VRAM)

| Model | HF ID | Params | Context | Notes |
|-------|-------|--------|---------|-------|
| **Qwen 2.5 32B Instruct** | `Qwen/Qwen2.5-32B-Instruct` | 32B | 128K | Strong, ~18GB in 4-bit |
| **Gemma 3 27B IT** | `google/gemma-3-27b-it` | 27B | 128K | Larger Gemma 3 variant |
| **Qwen 3 14B** | `Qwen/Qwen3-14B` | 14B | 40K | Reasoning/thinking mode (April 2025) |
| **Qwen 3 32B** | `Qwen/Qwen3-32B` | 32B | 40K | Same, larger |

---

## Not Open-Weight

| Model | Status |
|-------|--------|
| **GPT-4o-mini / GPT-20B** | API-only, no open weights. Use Qwen/Gemma in 14B–20B class instead. |
| **Nemotron 4 340B** | Open, but needs 8× H100; use Nemotron-Mini-4B for single-GPU. |

---

## Config Presets

```bash
# Local
python -m finetuning.src.training.train_judge --config-name qwen25_7b

# Local smoke (quick sanity check)
python -m finetuning.src.training.train_judge --config-name smoke

# Snellius + custom model (override base_model and output_dir)
python -m finetuning.src.training.train_judge --config-name snellius \
    base_model=Qwen/Qwen2.5-7B-Instruct \
    output_dir=finetuning/runs/qwen25_7b_judge_gtsynth
```

Available: `qwen25_7b`, `qwen25_14b`, `qwen25_32b`, `gemma3_12b`, `nemotron_mini_4b`, `llama31_8b`

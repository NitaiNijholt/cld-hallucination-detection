# CLD Judge Finetuning

QLoRA finetuning of Mistral-7B-Instruct-v0.2 on the CLD correctness judge task.

**Research question:** Does LoRA finetuning on GT Synth data close the GT Synth → GT Lit transfer gap identified in the thesis?

## Project structure

```
finetuning/
├── configs/
│   ├── default.yaml           # Base config
│   ├── smoke.yaml             # Quick local runs (64 examples, 1 epoch)
│   └── snellius.yaml          # HPC paths
├── src/
│   ├── data/
│   │   └── prepare_judge_data.py
│   ├── training/
│   │   └── train_judge.py
│   └── evaluation/
│       └── evaluate_judge.py
├── tests/
├── jobs/
│   └── train_judge.job
├── pyproject.toml
└── README.md
```

## Setup

```bash
# From repo root
pip install -e finetuning/[dev]
# or: uv pip install -e finetuning/[dev]
```

## Test batch (smoke pipeline)

Small dataset for quick validation (~5 train, 3 val, 1 eval):

```bash
# Regenerate from fixtures
python -m finetuning.src.data.prepare_judge_data \
  --data-root finetuning/tests/fixtures/final_runs \
  --output-dir finetuning/data/test_batch \
  --val-fraction 0.3 --min-file-bytes 0

# Train (local or Snellius)
python -m finetuning.src.training.train_judge --config-name test_batch
sbatch finetuning/jobs/train_smoke.job   # 20 min on H100

# Evaluate
python -m finetuning.src.evaluation.evaluate_judge \
  --finetuned_model finetuning/runs/test_batch_smoke/lora_adapter \
  --val_path finetuning/data/test_batch/judge_val_synth.xlsx \
  --lit_path finetuning/data/test_batch/judge_eval_gtlit.xlsx
```

## Workflow

### 1. Prepare data

```bash
python -m finetuning.src.data.prepare_judge_data
# Outputs: finetuning/src/judge_train.xlsx, judge_val_synth.xlsx, judge_eval_gtlit.xlsx

# With custom paths (e.g. CI or fixtures):
python -m finetuning.src.data.prepare_judge_data \
    --data-root /path/to/final_runs \
    --output-dir /tmp/out \
    --min-file-bytes 0
```

### 2. Train

**Local (smoke test):**
```bash
python -m finetuning.src.training.train_judge --config-name smoke
```

**Local (full run):**
```bash
python -m finetuning.src.training.train_judge
# Hydra overrides: epochs=5 lora_rank=16
```

**Snellius (SLURM):**
```bash
sbatch finetuning/jobs/train_judge.job
```

### 3. Evaluate

```bash
python -m finetuning.src.evaluation.evaluate_judge \
    --finetuned_model finetuning/runs/mistral7b_judge_gtsynth/lora_adapter \
    --eval_base
```

## Config (Hydra)

- **default.yaml**: Full hyperparameters (lora_rank=32, epochs=3, etc.)
- **smoke.yaml**: Overrides for quick sanity check (smoke_test=true, epochs=1)
- **snellius.yaml**: Paths for HPC

CLI overrides: `python -m finetuning.src.training.train_judge epochs=5 lora_rank=8`

## Deploying to Snellius

Copy `finetuning/.env` (W&B API key) to Snellius before training:

```bash
scp finetuning/.env snellius:~/cld-hallucination-detection/finetuning/.env
```

Or with explicit host: `scp finetuning/.env nnijholt@snellius.surf.nl:~/cld-hallucination-detection/finetuning/.env`

## Experiment tracking (W&B)

Set `wandb_project` in config or `WANDB_PROJECT` env to enable:

```bash
export WANDB_PROJECT=cld-judge-finetuning
python -m finetuning.src.training.train_judge
# Or in config: wandb_project: cld-judge-finetuning
```

Optional: `wandb_run_name`, `wandb_group` in config. Disable in CI: `WANDB_DISABLED=true`

## Model versioning (MLflow)

After training, models are logged to MLflow (experiment + registry). Default: local `mlruns/` in repo root.

- **Local:** `mlflow_tracking_uri: mlruns` (default)
- **Remote:** set `MLFLOW_TRACKING_URI` or override in config

See [finetuning/docs/MLFLOW.md](docs/MLFLOW.md) for loading from registry and remote setup.

## Inference (vLLM)

- **Evaluation:** `--inference_backend vllm` for faster batch eval (e.g. `evaluate_judge --inference_backend vllm`)
- **Serving:** vLLM OpenAI-compatible API for production. See [finetuning/docs/SERVING.md](docs/SERVING.md)

## Logging

- Step progress: every 10 steps to stdout and `output_dir/train.log`
- Config: `logging_steps: 10` in config
- SLURM jobs: `PYTHONUNBUFFERED=1` for immediate output

## Reproducibility

- Seeds: `seed=42` in config; `torch.manual_seed`, `np.random.seed` set at startup
- `run_metadata.txt` in output dir: git SHA, full config

## Testing

```bash
# From repo root (install finetuning deps for full coverage)
pip install -e finetuning/[dev]
pytest finetuning/tests/ -v

# Lightweight tests run without full ML deps (data prep, eval utils, config)
```

## Expected results

| Model | GT Synth F1 | GT Synth AUC | GT Lit F1 | GT Lit AUC | Cost/1k |
|---|---|---|---|---|---|
| GPT-4.1 Mechanistic (thesis) | 0.74 | 0.86 | 0.35 | 0.60 | ~$0.30 |
| Mistral-7B base (zero-shot) | baseline | baseline | baseline | baseline | ~$0.001 |
| Mistral-7B QLoRA (GT Synth) | target | target | **key result** | **key result** | ~$0.001 |

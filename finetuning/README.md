# CLD Judge Finetuning — Snellius

QLoRA finetuning of Mistral-7B-Instruct-v0.2 on the CLD correctness judge task.

**Research question:** Does LoRA finetuning on GT Synth data close the GT Synth → GT Lit transfer gap identified in the thesis?

## Files

| File | Purpose |
|---|---|
| `src/prepare_judge_data.py` | Load GT Synth/GT Lit xlsx, deduplicate by edge identity, output train/val/eval splits |
| `src/train_judge.py` | QLoRA training (r=32, 4-bit NF4) on GT Synth only |
| `src/evaluate_judge.py` | Inference + F1/AUC on GT Synth val (in-dist) and GT Lit (out-of-dist) |
| `jobs/train_judge.job` | SLURM job: 1× H100, 3h wall time |

## Workflow

```bash
# 1. Prepare data (run locally, needs cld-hallucination-detection final_runs/)
python finetuning/src/prepare_judge_data.py
# outputs: finetuning/src/judge_train.xlsx, judge_val_synth.xlsx, judge_eval_gtlit.xlsx

# 2. Upload to Snellius
scp finetuning/src/judge_*.xlsx <user>@snellius.surf.nl:~/Thesis-LLM-CLD/src/

# 3. On Snellius: clone Gijs's env repo, activate env, submit job
git clone https://github.com/abcdev123/Thesis-LLM-CLD ~/Thesis-LLM-CLD
cd ~/Thesis-LLM-CLD
cp <uploaded scripts> src/
sbatch jobs/train_judge.job

# 4. After training: run evaluation
python finetuning/src/evaluate_judge.py \
    --finetuned_model Mistral-7B-Instruct-v0.2_qlora_cld_judge_gtsynth/lora_adapter \
    --eval_base
```

## Expected results table

| Model | GT Synth F1 | GT Synth AUC | GT Lit F1 | GT Lit AUC | Cost/1k |
|---|---|---|---|---|---|
| GPT-4.1 Mechanistic (thesis) | 0.74 | 0.86 | 0.35 | 0.60 | ~$0.30 |
| Mistral-7B base (zero-shot) | baseline | baseline | baseline | baseline | ~$0.001 |
| Mistral-7B QLoRA (GT Synth) | target | target | **key result** | **key result** | ~$0.001 |

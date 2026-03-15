# Run Inventory — Snellius

Checklist of steps to execute training and evaluation runs on Snellius.

---

## Prerequisites (one-time)

- [ ] SURF Usage Agreement accepted at https://portal.cua.surf.nl
- [ ] SSH key registered at https://portal.cua.surf.nl/user/keys
- [ ] Repo cloned on Snellius: `~/cld-hallucination-detection` (branch `snellius-finetuning`)
- [ ] Venv created and deps installed: `uv venv venv --python 3.11` + `uv pip install -e finetuning/` (see AGENTS.md; load Python module first)
- [ ] Data prepared: `python -m finetuning.src.data.prepare_judge_data` → outputs `judge_train.xlsx`, `judge_val_synth.xlsx`, `judge_eval_gtlit.xlsx` in `finetuning/src/`

---

## Current state (as of last check)

| Asset | Status |
|-------|--------|
| Full finetune (mistral7b_judge_gtsynth) | Done (job 20704840) |
| Model | `finetuning/runs/mistral7b_judge_gtsynth/lora_adapter` |
| Eval job config | batch_size=16, time=56h |

---

## Steps to run evaluation

1. **Commit and push** (from local):
   ```bash
   git add finetuning/jobs/eval_judge.job
   git commit -m "Eval job: batch_size 16, time 56h"
   git push origin snellius-finetuning
   ```

2. **On Snellius** — pull and submit:
   ```bash
   ssh snellius
   cd ~/cld-hallucination-detection
   git pull
   source venv/bin/activate
   sbatch finetuning/jobs/eval_judge.job
   ```

3. **Monitor**:
   ```bash
   squeue -u nnijholt
   tail -f ~/cld-hallucination-detection/finetuning/logs/cld_judge_eval_<JOBID>.out
   tail -f ~/cld-hallucination-detection/finetuning/logs/cld_judge_eval_<JOBID>.err
   ```

4. **Outputs** (when done):
   - `results/evaluation_results.json`
   - `results/confusion_matrices.png`
   - `results/predictions_*.xlsx`

---

## Full workflow (if starting from scratch)

### 1. Data prep (on Snellius)

```bash
cd ~/cld-hallucination-detection
source venv/bin/activate
python -m finetuning.src.data.prepare_judge_data
# Requires: final_runs/ with xlsx files
# Outputs: finetuning/src/judge_train.xlsx, judge_val_synth.xlsx, judge_eval_gtlit.xlsx
```

### 2. Train

```bash
# Smoke (20 min) — validates pipeline
sbatch finetuning/jobs/train_smoke.job

# Full run (~6.5 h)
sbatch finetuning/jobs/train_judge.job
```

### 3. Evaluate

```bash
sbatch finetuning/jobs/eval_judge.job
# Uses: finetuning/runs/mistral7b_judge_gtsynth/lora_adapter
# Time: 56h, batch_size 16
```

---

## Job summary

| Job | Script | Time | Purpose |
|-----|--------|------|---------|
| train_smoke | train_smoke.job | 20 min | Pipeline smoke test (test_batch) |
| train_judge | train_judge.job | 24 h | Full finetune (snellius config) |
| eval_judge | eval_judge.job | 56 h | Eval on GT Synth Val + GT Lit |

---

## SBU estimate

| Job | Est. SBU |
|-----|----------|
| train_smoke | ~64 |
| train_judge | ~4,600 |
| eval_judge | ~10,800 |

(1× H100 ≈ 192 SBU/hr)

# Test batch

Small dataset for pipeline smoke testing. Regenerated from `finetuning/tests/fixtures/final_runs/`.

**Regenerate:**
```bash
python -m finetuning.src.data.prepare_judge_data \
  --data-root finetuning/tests/fixtures/final_runs \
  --output-dir finetuning/data/test_batch \
  --val-fraction 0.3 \
  --min-file-bytes 0
```

**Run smoke training:**
```bash
# Local
python -m finetuning.src.training.train_judge --config-name test_batch

# Snellius
sbatch finetuning/jobs/train_smoke.job
```

**Evaluate:**
```bash
python -m finetuning.src.evaluation.evaluate_judge \
  --finetuned_model finetuning/runs/test_batch_smoke/lora_adapter \
  --val_path finetuning/data/test_batch/judge_val_synth.xlsx \
  --lit_path finetuning/data/test_batch/judge_eval_gtlit.xlsx
```

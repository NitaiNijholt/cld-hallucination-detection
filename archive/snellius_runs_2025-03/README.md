# Snellius backup (2025-03-15)

Local backup of run data from Snellius before updating eval infra.

## Contents

### logs/
All SLURM job logs (.out, .err):
- **cld_judge_finetune_*** — full training runs (20704840 = successful)
- **cld_judge_smoke_*** — smoke tests (20704764 = successful)
- **cld_judge_eval_*** — eval runs (20714184 = timed out, 20726231 = partial GT Synth Val)
- **eval_20713981** — classification eval

### runs/mistral7b_judge_gtsynth/
- **lora_adapter/** — LoRA weights (324M), use for eval
- **run_metadata.txt** — git SHA, config
- **train.log** — training step log
- **loss_curve.png** — loss plot

### runs/test_batch_smoke/
Not fetched (4.7G). Fetch manually if needed.

### lora_adapter
Not in git (324M). Stored locally in snellius_backup/runs/mistral7b_judge_gtsynth/

### results/
Empty on Snellius (eval runs didn't complete).

## Key metrics (from logs)

- **GT Synth Val** (eval 20726231): F1=0.738, AUC=0.837, accuracy=0.85
- **Full finetune** (20704840): train_loss=0.2378, best checkpoint 406

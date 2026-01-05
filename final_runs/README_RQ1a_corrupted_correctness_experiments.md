# How to Rerun RQ1a Correctness Experiments

## 📍 Location of Scripts

All scripts are located in:
```
/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/
```

## 🚀 Quick Start: Rerun All Experiments

```bash
cd /home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments

# Run all 3 CLDs × 3 runs = 9 experiments
nohup python run_rq1a_corruption_multirun.py \
  --runs 3 \
  --clds depressive social_norms emergency_department \
  > logs/rq1a_correctness_multirun_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor progress
tail -f logs/rq1a_correctness_multirun_*.log
```

## 📂 Output Directory

Results are saved to:
```
/home/nitai/code/causalix.ai/final_runs/RQ1a_corruption_detection_correctness/
```

Structure:
```
RQ1a_corruption_detection_correctness/
├── depressive/
│   ├── run_1/
│   ├── run_2/
│   └── run_3/
├── social_norms/
│   ├── run_1/
│   ├── run_2/
│   └── run_3/
└── emergency_department/
    ├── run_1/
    ├── run_2/
    └── run_3/
```

## 🔧 Scripts Used

### 1. **Multirunner** (Main Controller)
```bash
run_rq1a_corruption_multirun.py
```
- Orchestrates multiple corruption+judging runs
- Reuses base sessions for consistency
- Skips completed runs (safe to restart)

### 2. **Pipeline Script** (Called by multirunner)
```bash
run_corruption_detection_pipeline.sh
```
- Runs single corruption detection experiment
- Calls corruption → judging → analysis

### 3. **Supporting Scripts**
```bash
run_rq1_phase1_single_corrupt.py        # Applies corruption
run_rq1_judge_all_3_prompts.py          # Judges with 3 prompts
analyze_prompt_variants_corrupted.py     # Analyzes results
```

## 🎯 Prompt Variants (Auto-Discovered)

Scripts automatically discover prompts from:
```
data_science/parameter_tuning_experiments/alternative_prompts/
```

Current prompts:
- `prompts_correctness_baseline.yaml` (250 tokens)
- `prompts_correctness_mechanistic_v2.yaml` (250 tokens)
- `prompts_correctness_cot.yaml` (5000 tokens)

## 📋 Key Parameters (Saved in Metadata)

### Reproducibility Parameters:
- **Corruption seed**: Run number (1, 2, 3, ...)
- **Judge seed**: 42 (hardcoded)
- **Corruption temperature**: 0.7
- **Judge temperature**: 0.3
- **Corruption rate**: 30%

### Models:
- **Corruptor**: gpt-4.1
- **Judge**: gpt-4.1

### Metadata Files:
Each run saves:
- `corruption_metadata.json` - Corruption parameters
- `judging_metadata_<timestamp>.json` - Judging parameters
- Session info, timestamps, system info

## 🔍 Run Single Experiment

```bash
cd /home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments

# Run single CLD with specific run number
bash run_corruption_detection_pipeline.sh depressive 1 [base_session_id]

# Base session IDs (optional, will be auto-selected):
# - depressive: 49d62c01-e85c-4392-bb6d-eb89f53136a8
# - social_norms: 278d89e0-7980-4cd5-baf3-d5ed4ee27a13
# - emergency_department: a7d55436-d8c9-409b-a591-78f837a44951
```

## 📊 Analysis

After experiments complete, results include:
- Individual prompt judgments (Excel files)
- Aggregated analysis (metrics, agreement, confusion matrices)
- README with experimental details

## 🆕 Recent Changes (v2.0)

1. ✅ **Auto-discovery of prompts** - No need to hardcode prompt lists
2. ✅ **Enhanced metadata** - Full reproducibility parameters saved
3. ✅ **CoT prompts**: 5000 max tokens (was 250)
4. ✅ **Overwrite protection** - Refuses to overwrite existing runs

## 💾 Backup Note

This directory (`RQ1a_corruption_detection_correctness_backup_oldprompts`) contains experiments run with old prompt configurations. New experiments use the updated auto-discovery system.


Run enhanced analysis with:

/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/analyze_rq1a_aggregate_enhanced.py

---

**Last Updated**: November 2025
**Script Version**: 2.0 (auto-discovery)

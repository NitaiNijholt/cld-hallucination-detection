# OpenAI API Cost Analysis

This directory contains scripts to calculate OpenAI API costs from experiment files.

## Overview

The analysis extracts token usage from the `LLM Usage Stats` sheet in experiment Excel files and calculates costs based on OpenAI API pricing (December 2024).

## Scripts

### 1. `calculate_all_costs.py` (Recommended)

Complete cost analysis including:
- **RQ1a Judge experiments**: Actual costs from LLM Usage Stats
- **RQ1b Corrector experiments**: Estimated costs (assumes correction + rejudging = 2× correctness tokens/edge)

### 2. `calculate_costs_by_experiment.py`

RQ1a costs only, broken down by experiment type.

### 3. `calculate_experiment_costs.py`

Legacy script - costs aggregated by judge type.

## Data Location

Experiment data is located in:
```
/home/nitai/code/causalix.ai/final_runs/
```

This directory contains:
- **RQ1a_corruption_detection_correctness_final/**: Correctness judging on corrupted CLDs
- **RQ1a_corruption_detection_citation_gpt5mini/**: Citation judging on corrupted CLDs (GPT-5-mini)
- **RQ1a_ground_truth_correctness/**: Correctness judging on ground truth CLDs
- **RQ1a_ground_truth_citation/**: Citation judging on ground truth CLDs

Each experiment file (`judged_*.xlsx`) contains:
- **LLM Usage Stats sheet**: Token counts (prompt_tokens, completion_tokens), inference calls
- **All Edges sheet**: Actual CLD edges being validated (used for per-edge metrics)

## Reproduction Steps

### 1. Install Dependencies

```bash
pip install pandas openpyxl numpy
```

### 2. Run the Cost Analysis (Per Experiment Type)

```bash
cd /home/nitai/code/causalix.ai/final_runs/cost_analysis

# Uses final_runs/ as default data directory
python calculate_costs_by_experiment.py --output-dir .

# Or specify explicitly:
python calculate_costs_by_experiment.py \
    --data-dir /home/nitai/code/causalix.ai/final_runs \
    --output-dir .
```

### 3. Output Files

The script generates:
- `cost_by_experiment_report_*.txt`: Human-readable cost report
- `cost_by_experiment.csv`: Summary table
- `cost_by_experiment_table.tex`: LaTeX table for thesis
- `cost_by_experiment_results.json`: Detailed JSON results

## Pricing Used

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| GPT-4.1 | $2.00 | $8.00 |
| GPT-5-mini | $0.15 | $0.60 |
| GPT-4o-mini | $0.15 | $0.60 |
| text-embedding-3-small | $0.02 | N/A |

Source: https://platform.openai.com/docs/pricing (December 2024)

## Expected Results

Based on the experiment files:

### RQ1a Judge Experiments (Actual)

| Experiment | Model | Files | Edges | Tokens/Edge | Cost/Edge (¢) | Total ($) |
|------------|-------|-------|-------|-------------|---------------|-----------|
| Corruption Citation | GPT-5-mini | 28 | 13,800 | 43,313 | 0.73 | 100.14 |
| Ground Truth Citation | GPT-4.1 | 43 | 20,143 | 46,053 | 9.65 | 1,944.70 |
| Corruption Correctness | GPT-4.1 | 30 | 14,220 | 481 | 0.16 | 23.41 |
| Human Validation | GPT-4.1 | 3 | 260 | 42,639 | 8.70 | 22.61 |
| **RQ1a Subtotal** | | | | | | **$2,090.87** |

### RQ1b Corrector Experiments (Estimated)

| Experiment | Model | Files | Edges | Tokens/Edge | Cost/Edge (¢) | Total ($) |
|------------|-------|-------|-------|-------------|---------------|-----------|
| Synth Baseline | GPT-4.1 | 14 | 5,275 | 993 | 0.33 | 17.41 |
| Synth CoT | GPT-4.1 | 14 | 5,077 | 993 | 0.33 | 16.75 |
| Synth Mechanistic | GPT-4.1 | 19 | 8,084 | 993 | 0.33 | 26.68 |
| GT Baseline | GPT-4.1 | 18 | 9,058 | 993 | 0.33 | 29.89 |
| GT CoT | GPT-4.1 | 9 | 5,609 | 993 | 0.33 | 18.51 |
| GT Mechanistic | GPT-4.1 | 13 | 5,169 | 993 | 0.33 | 17.06 |
| **RQ1b Subtotal** | | | | | | **$126.30** |

### Grand Total: **$2,217.16**

Note: Actual spend (~$5,000) includes development runs, debugging, generation experiments, and exploratory work.

### Important Notes

- **RQ1a costs**: Actual values from `LLM Usage Stats` sheet
- **RQ1b costs**: Estimated assuming correction + rejudging = 2× correctness tokens/edge (~497 tok/edge × 2)
- **Edges** = rows in "All Edges" sheet (actual CLD edges)

## Key Insights

1. **Citation judging dominates costs** (96% of total) due to large context windows (~45k tokens/edge from retrieved citations)
2. **GPT-5-mini reduces citation costs by 92.5%** ($3,280 → $246 estimated)
3. **Correctness judging is efficient** (~482 tokens/edge, $0.16¢/edge)

## File Structure

```
cost_analysis/
├── README.md                           # This file
├── calculate_experiment_costs.py       # Main analysis script
├── cost_report_*.txt                   # Generated reports
├── cost_summary.csv                    # Summary table
└── cost_analysis_results.json          # Detailed JSON results
```

## Author

Causalix.ai - December 2024








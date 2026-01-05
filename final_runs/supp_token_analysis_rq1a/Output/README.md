# Token and API Call Analysis for RQ1a Judge Experiments

This folder contains the reproducible analysis of output token counts and API call statistics for all RQ1a judge experiments.

## Experiments Covered

- `RQ1a_corruption_detection_correctness_final` (GPT-4.1 correctness judge)
- `RQ1a_corruption_detection_citation_gpt5mini` (GPT-5-mini citation judge)
- `RQ1a_ground_truth_correctness` (GPT-4.1 correctness judge)
- `RQ1a_ground_truth_citation` (GPT-4.1 citation judge)
- `RQ1_human_validation_citation_judge` (GPT-4.1 citation judge, human validation)

## Files

| File | Description |
|------|-------------|
| `analyze_output_tokens_tiktoken.py` | Main analysis script using GPT-4 tokenizer (tiktoken) |
| `output_token_analysis_*.txt` | Human-readable analysis report |
| `output_token_analysis_*.json` | Machine-readable data for further analysis |
| `latex_table_output_tokens.tex` | LaTeX table for thesis appendix |

## Reproduction

```bash
cd /home/nitai/code/causalix.ai
python final_runs/token_call_analysis_RQ1a_judge/analyze_output_tokens_tiktoken.py
```

## Key Findings

- **Total messages analyzed:** 70,281
- **Total API calls:** 189,736
- **Overall success rate:** 96.0%
- **No truncation detected:** All outputs below max_tokens limits

## Data Source

Excel files from: `/home/nitai/code/causalix.ai/RQ1_excel_snapshot_20251215/final_runs/`

Generated: 2025-12-17








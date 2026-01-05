# RQ1a Corruption Detection - Depressive symptoms in response to a stressor - Run 2

**Date**: $(date +"%B %d, %Y")
**CLD**: Depressive symptoms in response to a stressor
**Experiment Type**: Judge Performance on Corrupted CLD
**Corruption Rate**: 30%

## Overview

This run tests the ability of 3 validated judge prompts to detect synthetic corruptions in the Depressive symptoms in response to a stressor CLD.

## Experimental Design

### Step 1: Corruption (Phase 1)
- **Base CLD**: Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx
- **Corruption Method**: Synthetic corruptions using `run_rq1_phase1_single_corrupt.py`
- **Corruption Rate**: 30%
- **Corruption Types**: 
  - Motivation corruption (spurious explanations)
  - Structural corruption (new edges, polarity flips)

### Step 2: Judging with 3 Prompts
- **Script**: `run_rq1_judge_citations_all_3_prompts.py`
- **Judge Model**: GPT-5-mini
- **Temperature**: 0.0
- **Approach**: Citation-based judging
- **Parallelization**: 10 workers

## Prompts Tested
1. **Depressive_symptoms_in_response_to_a_stressor_baseline_20251204_041329**
   - File: `judged_citation_Depressive_symptoms_in_response_to_a_stressor_baseline_20251204_041329_20251204_042510.xlsx`

2. **Depressive_symptoms_in_response_to_a_stressor_cot_20251204_042559**
   - File: `judged_citation_Depressive_symptoms_in_response_to_a_stressor_cot_20251204_042559_20251204_043917.xlsx`

3. **Depressive_symptoms_in_response_to_a_stressor_mechanistic_20251204_043957**
   - File: `judged_citation_Depressive_symptoms_in_response_to_a_stressor_mechanistic_20251204_043957_20251204_045356.xlsx`

## Results

See `analysis/prompt_variants_analysis.xlsx` for detailed metrics including:
- Judge performance (Precision, Recall, F1, Accuracy)
- Confusion matrices
- Inter-prompt agreement rates
- Verdict distributions

## Files

### Corrupted CLD (Input)
- `CORRUPTED_depressive.xlsx` - Ground truth corrupted CLD

### Judged CLDs (Output)
- `judged_*_baseline_*.xlsx` - Baseline prompt results
- `judged_*_mechanistic_*.xlsx` - Mechanistic prompt results
- `judged_*_sce_*.xlsx` - SCE prompt results

### Metadata
- `judging_metadata_*.json` - Experimental configuration
- `*.json` - Individual prompt metadata

### Analysis
- `analysis/prompt_variants_analysis.xlsx` - Detailed metrics
- `analysis/prompt_variants_comparison.png` - Visualizations

### Logs
- `corruption_*.log` - Corruption phase log
- `judging_*.log` - Judging phase log
- `analysis_*.log` - Analysis phase log

## Scripts Used

1. `run_rq1_phase1_single_corrupt.py` - Corruption
2. `run_rq1_judge_all_3_prompts.py` - Judging
3. `analyze_prompt_variants_corrupted.py` - Analysis
4. `run_corruption_detection_pipeline.sh` - Full pipeline wrapper

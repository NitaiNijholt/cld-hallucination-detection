# RQ1a Corruption Detection - Social norms and obesity prevalence - Run 3

**Date**: $(date +"%B %d, %Y")
**CLD**: Social norms and obesity prevalence
**Experiment Type**: Judge Performance on Corrupted CLD
**Corruption Rate**: 30%

## Overview

This run tests the ability of 3 validated judge prompts to detect synthetic corruptions in the Social norms and obesity prevalence CLD.

## Experimental Design

### Step 1: Corruption (Phase 1)
- **Base CLD**: Social_norms_and_obesity_prevalence.xlsx
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
1. **Social_norms_and_obesity_prevalence_baseline_20251204_040024**
   - File: `judged_citation_Social_norms_and_obesity_prevalence_baseline_20251204_040024_20251204_040738.xlsx`

2. **Social_norms_and_obesity_prevalence_mechanistic_20251204_041543**
   - File: `judged_citation_Social_norms_and_obesity_prevalence_mechanistic_20251204_041543_20251204_042151.xlsx`

3. **Social_norms_and_obesity_prevalence_cot_20251204_040811**
   - File: `judged_citation_Social_norms_and_obesity_prevalence_cot_20251204_040811_20251204_041507.xlsx`

## Results

See `analysis/prompt_variants_analysis.xlsx` for detailed metrics including:
- Judge performance (Precision, Recall, F1, Accuracy)
- Confusion matrices
- Inter-prompt agreement rates
- Verdict distributions

## Files

### Corrupted CLD (Input)
- `CORRUPTED_social_norms.xlsx` - Ground truth corrupted CLD

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

# RQ2 Hallucination Detection - Data Sources

## Overview

The RQ2 analysis uses **63 deduplicated experiment files** from the comprehensive meta-analysis pipeline (Nov 26, 2025).

## Files Analyzed

**Total:** 63 files  
**Source:** `rq2_valid_files_deduplicated_20251126_022337.json`

### By Experiment Type:
- **Citation**: 36 files (ground truth based on citation evidence)
- **Correctness**: 27 files (ground truth based on expert judgment)

### By CLD (Causal Loop Diagram):
- **Depressive symptoms**: 21 files
- **Emergency department**: 21 files  
- **Social norms**: 21 files

### By Prompt Type:
- **Baseline**: 18 files
- **CoT (Chain of Thought)**: 18 files
- **Mechanistic**: 9 files
- **Mechanistic Lit**: 9 files
- **Mechanistic Original**: 9 files

## Analysis Scripts

### Main Script:
**`data_science/rq2_master_report_v2_enhanced.py`**
- Generates comprehensive master report with all phases
- Creates TABLE_1_GRAND_AGGREGATE.xlsx and TABLE_2_OVERFITTING_STORY.xlsx
- Combines results from Phases 1-6

### Pipeline Phases:

1. **Phase 1**: Individual file analysis (`rq2_simple_batch_analyzer.py`)
   - Processes each of 63 files independently
   - Computes correlations, AUCs, t-tests

2. **Phase 2**: Per-experiment aggregates (`rq2_aggregate_by_experiment.py`)
   - Aggregates citation and correctness separately

3. **Phase 3**: Grand aggregate (`rq2_grand_aggregate.py`)
   - Meta-analysis across all 63 files
   - Produces TABLE_1 data

4. **Phase 4**: RFE Feature Selection (`rq2_phase4_rfe_multiclassifier.py`)
   - Uses 36 files (only those with Gen Cosine Similarity)
   - 16,507 edges total

5. **Phase 5**: Ensemble Comparison (`rq2_phase5_ensemble_with_rfe_features.py`)
   - Same 36 files as Phase 4
   - Produces partial TABLE_2 data

6. **Phase 6**: Cross-Domain Validation (`rq2_phase6_cross_cld_with_best.py`)
   - Leave-one-CLD-out validation
   - Produces final TABLE_2 data

## Data Locations

### Input Files:
- **File list**: `parameter_tuning_experiments/rq2_analyses/rq2_valid_files_deduplicated_20251126_022337.json`
- **Raw data**: `final_runs/RQ1a_ground_truth_citation/` and `final_runs/RQ1a_ground_truth_correctness/`

### Output Files (Copied to final_runs/RQ2_hallucination_detection/):
1. **TABLE_1_GRAND_AGGREGATE.xlsx** - Single metric performance (63 files)
2. **TABLE_2_OVERFITTING_STORY.xlsx** - Ensemble classifier cross-phase analysis (36 files)
3. **aggregate_auc_scores.png** - Forest plot (63 files)
4. **ensemble_roc_curves.png** - ROC curves (36 files)
5. **rigorous_feature_selection_plots.png** - RFE analysis (36 files)
6. **feature_distributions.png** - Metric distributions

### Original Analysis Directories:
- **Comprehensive report**: `parameter_tuning_experiments/rq2_analyses/rq2_master_report_comprehensive_20251126_040703/`
- **Phase 1-6 results**: `parameter_tuning_experiments/rq2_analyses/rq2_batch_20251126_022338/` through `rq2_phase6_cross_cld_20251126_025535/`

## Important Notes

### Data Reduction (Phases 4-6):
- **63 files → 36 files** for ensemble analysis
- **Reason**: Gen Cosine Similarity missing in 27 files (mostly correctness experiments)
- **Impact**: Phases 4-6 primarily use citation experiment data
- TABLE_1 uses all 63 files; TABLE_2 uses 36 files

### CI Metrics Analyzed:
All metrics from **generator model** (not judge):
1. **Gen Perplexity** - Model uncertainty about generated edge
2. **Gen Min Prob** - Minimum token probability
3. **Gen Max Window Entropy** - Maximum entropy across token windows
4. **Gen Cosine Similarity** - Similarity to retrieved evidence

### Hallucination Definition:
```
Hallucination = (Classification == 'FP') OR (Classification == 'FN')
```
- **FP (False Positive)**: Edge generated but shouldn't exist
- **FN (False Negative)**: Edge should exist but wasn't generated

## Reproducibility

To reproduce the analysis:

```bash
cd /home/nitai/code/causalix.ai
source .venv/bin/activate

# Run complete RQ2 pipeline (generates all 63 file analyses + master report)
bash data_science/rq2_run_complete_pipeline_v2.sh
```

This will regenerate:
- All individual file analyses
- Phase aggregates
- Grand aggregate (TABLE_1)
- Ensemble analyses (TABLE_2)
- Master report

## Key Findings

### TABLE_1 (All 63 Files):
- **Best Metric**: Gen Cosine Similarity (AUC = 0.677)
- **Most Consistent**: Gen Max Window Entropy (42.9% significant files)

### TABLE_2 (36 Files with Complete Data):
- **Best In-Distribution**: Random Forest (AUC = 0.987)
- **Best Cross-Domain**: Neural Network (AUC = 0.670)
- **Most Stable**: Logistic Regression (drop = 0.043)

---

**Generated**: 2025-12-10  
**Last Updated**: 2025-12-10













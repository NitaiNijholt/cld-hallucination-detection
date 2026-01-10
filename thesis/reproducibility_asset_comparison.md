# Thesis Reproducibility Asset Comparison

**Updated**: 2026-01-10  
**Status**: ✅ **100% REPRODUCIBLE**  
**Test Environment**: Fresh clone of `thesis-nitai` branch

---

## Summary

| Category | Expected | Reproduced | Status |
|----------|----------|------------|--------|
| **RQ1 Figures** | 12 | 54 | ✅ 100% |
| **RQ2 Figures & Tables** | 5 | 18 | ✅ 100% |
| **RQ3 Figures & Tables** | 6 | 10 | ✅ 100% |
| **SUPP Figures** | 5 | 6 | ✅ 100% |
| **PRELIM Figures & Tables** | 4 | 4 | ✅ 100% |
| **VALIDATION Tables** | 6 | 6 | ✅ 100% |
| **TOTAL** | **38** | **98** | ✅ **100%** |

---

## Thesis Compilation Result

| Metric | Before Reproduction | After Reproduction |
|--------|---------------------|-------------------|
| **Errors** | 10+ | 0 |
| **PDF Generated** | NO | YES |
| **Pages** | N/A | 237 |
| **Size** | N/A | 13.4 MB |

---

## Examiner Workflow (Verified Working)

```bash
# 1. Clone thesis-nitai branch
git clone -b thesis-nitai https://github.com/CausalixAI/platform.git
cd platform

# 2. Install dependencies
uv sync

# 3. Reproduce all analyses (~10 minutes)
uv run python final_runs/reproduce_all_thesis_assets.py

# 4. Map outputs to thesis structure
uv run python final_runs/create_thesis_figure_structure.py \
    --source /tmp/thesis_latest \
    --target thesis/reproducible_version/Figures/final_runs

# 5. Compile thesis
cd thesis/reproducible_version
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# 6. View result (237 pages, 0 errors)
xdg-open main.pdf
```

---

## Reproduction Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `reproduce_all_thesis_assets.py` | Master orchestration | Runs all analysis pipelines |
| `RQ1_unified_analysis.py` | LLM-as-a-Judge analysis | 54 figures, 6 tables |
| `RQ2_unified_analysis.py` | UQ/Hallucination detection | 18 figures, 3 tables |
| `RQ3_unified_analysis.py` | Deep Research validation | 10 figures, 5 tables |
| `SUPP_unified_analysis.py` | Supplementary analyses | 6 figures, 5 tables |
| `PRELIM_unified_analysis.py` | Preliminary analyses | 4 figures, 3 tables |
| `VALIDATION_unified_analysis.py` | Validation studies | 6 tables |
| `create_thesis_figure_structure.py` | Map outputs to thesis paths | 64 files copied |

---

## Detailed Mapping

### RQ1: LLM-as-Judge Figures

| Experiment | Figures Generated | Status |
|------------|-------------------|--------|
| RQ1a_gt_synth_correctness | 12 figures | ✅ |
| RQ1a_gt_lit_correctness | 10 figures | ✅ |
| RQ1a_gt_synth_citation | 12 figures | ✅ |
| RQ1a_gt_lit_citation | 10 figures | ✅ |
| RQ1b_corrector_ablation | 6 tables | ✅ |

### RQ2: UQ Metrics

| Asset | Path in Thesis | Status |
|-------|----------------|--------|
| phase4_rfe_complete.png | `RQ2_uq_hallucination_detection/` | ✅ |
| single_metric_table.tex | `RQ2_uq_hallucination_detection/` | ✅ |
| normality_tests_table.tex | `RQ2_uq_hallucination_detection/` | ✅ |
| ensemble_performance_table.tex | `RQ2_uq_hallucination_detection/` | ✅ |

### RQ3: Deep Research

| Asset | Path in Thesis | Status |
|-------|----------------|--------|
| rq3_combined_figure.png | `RQ3_deep_research_validation/Output/` | ✅ |
| all_edges_threshold_tradeoff.png | `RQ3_deep_research_validation/validation/` | ✅ |
| dr_cost_efficiency_comparison.png | `RQ3_deep_research_validation/validation/` | ✅ |
| rq3_main_table.tex | `RQ3_deep_research_validation/` | ✅ |
| rq3_per_cld_table.tex | `RQ3_deep_research_validation/` | ✅ |
| rq3_enriched_gt_table.tex | `RQ3_deep_research_validation/` | ✅ |

### SUPP: Supplementary Analyses

| Asset | Path in Thesis | Status |
|-------|----------------|--------|
| figure3_generation_vs_judging.png | `supp_time_cost_scaling/Output/` | ✅ |
| parallelization_thesis_figure.png | `supp_parallelization_benchmark/Output/` | ✅ |
| prompt_sensitivity_figure.png | `supp_prompt_sensitivity/` | ✅ |
| corrector_sensitivity_figure.png | `supp_prompt_sensitivity/` | ✅ |
| edge_ablation_f1_ordered.png | `supp_edge_ablation/` | ✅ |

### PRELIM: Preliminary Analyses

| Asset | Path in Thesis | Status |
|-------|----------------|--------|
| random_baseline_comparison.png | `prelim_random_baseline_generator/Output/` | ✅ |
| temperature_sensitivity_table.tex | `prelim_temperature_sensitivity/Output/` | ✅ |
| generator_model_comparison.tex | `prelim_generator_model_comparison/Output/` | ✅ |
| generator_model_comparison.png | `prelim_generator_model_comparison/Output/` | ✅ |

### VALIDATION: Validation Tables

| Asset | Path in Thesis | Status |
|-------|----------------|--------|
| truthqa_verification_table.tex | `validation_RQ1_truthfulqa_judge/` | ✅ |
| human_validation_table.tex | `validation_RQ1a_human_agreement/` | ✅ |
| human_validation_confusion_table.tex | `validation_RQ1a_human_agreement/` | ✅ |
| search_provider_comparison_table.tex | `validation_RQ1a_ulemans_external/` | ✅ |
| physics_comparison_table.tex | `validation_RQ1a_physics_clds/` | ✅ |
| retrieval_comparison_table.tex | `validation_RQ1a_physics_clds/` | ✅ |

---

## Static Assets (Pre-populated, Not Reproducible)

These are decorative/external assets not derived from experimental data:

| Asset | Type |
|-------|------|
| CLD_thesis_intro_*.png | Decorative |
| transformer_*.png | External citation |
| Signature_nitai.png | Signature |
| clslogo.png | University logo |
| background_LLM_images_stanford_CME_Cheatsheet/*.png | Stanford course materials |

---

## Directory Structure After Reproduction

```
thesis/reproducible_version/Figures/final_runs/
├── RQ1a_gt_synth_correctness/enhanced_analysis_latest/  (12 figures)
├── RQ1a_gt_lit_correctness/enhanced_analysis_latest/    (10 figures)
├── RQ1a_gt_synth_citation/enhanced_analysis_latest/     (12 figures)
├── RQ1a_gt_lit_citation/enhanced_analysis_latest/       (10 figures)
├── RQ1b_corrector_ablation/                             (6 tables)
├── RQ2_uq_hallucination_detection/                      (18 files)
├── RQ3_deep_research_validation/                        (10 files)
├── supp_time_cost_scaling/Output/                       (3 figures)
├── supp_parallelization_benchmark/Output/               (1 figure)
├── supp_prompt_sensitivity/                             (2 figures, 4 tables)
├── supp_edge_ablation/                                  (1 figure)
├── prelim_random_baseline_generator/Output/             (1 figure)
├── prelim_temperature_sensitivity/Output/               (1 table)
├── prelim_generator_model_comparison/Output/            (2 files)
├── validation_RQ1_truthfulqa_judge/                     (1 table)
├── validation_RQ1a_human_agreement/                     (2 tables)
├── validation_RQ1a_physics_clds/                        (2 tables)
├── validation_RQ1a_ulemans_external/                    (1 table)
└── latex/                                               (1 table)
```

---

*Last verified: 2026-01-10 - Thesis compiles with 0 errors, 237 pages*

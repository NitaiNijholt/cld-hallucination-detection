# Final Runs Directory Map

This directory contains all **final** experimental outputs and **reproducible analysis pipelines** used by the thesis.

The primary design goal is: **a stranger can reproduce every Results table/figure by running a small number of entry points**.

## Quick Start: Full Reproducibility

To reproduce **all** thesis figures and tables with a single command:

```bash
# From project root
uv run python final_runs/reproduce_all_thesis_assets.py --output-dir /tmp/thesis_repro

# Or reproduce to default timestamped directory
uv run python final_runs/reproduce_all_thesis_assets.py
```

This runs all unified analysis scripts and generates a verification report showing coverage.

---

## Unified Analysis Entry Points (Single-command pipelines)

| Script | Purpose |
|--------|---------|
| `reproduce_all_thesis_assets.py` | **Master script**: runs all 4 unified analyses below, generates verification report. |
| `RQ1_unified_analysis.py` | Reproduces **RQ1 (TruthfulQA verification)**, **RQ1a (LLM-as-a-Judge)**, and **RQ1b (LLM-as-a-Corrector)** analyses. |
| `RQ2_unified_analysis.py` | Reproduces the complete **RQ2** pipeline (batch analysis + Phase 4/5/6 ML stages + reporting). |
| `RQ3_unified_analysis.py` | Reproduces the **RQ3 Deep Research** analysis (stats, per-CLD breakdown, enrichment, figures/report). |
| `SUPP_unified_analysis.py` | Reproduces **supplementary** analyses (time/cost scaling, parallelization, prompt sensitivity). |

Each script creates a `*_latest` symlink pointing to the most recent run output for stable referencing.

---

## Directory Structure by Research Question

### RQ1: LLM-as-a-Judge & Corrector

#### RQ1 Preliminaries
- `RQ1_preliminaries/`: One-off preliminary analyses and consolidated runs used before/around main RQ1 execution.

#### RQ1: Judge Capability Verification
- `RQ1_verification_Truth_QA/`: TruthfulQA judge verification.
  - `analysis_script/`: scripts to compute TruthfulQA metrics.
  - `Data_runs/`: result JSONs and artifacts from runs.

#### RQ1a: LLM-as-a-Judge (Hallucination Detection)
- `RQ1a_corruption_detection_correctness_final/`: Correctness judge on **synthetic corruptions**.
- `RQ1a_corruption_detection_citation_gpt5mini/`: Citation judge on **synthetic corruptions** (GPT-5-mini).
- `RQ1a_ground_truth_correctness/`: Correctness judge on **GT Lit**.
- `RQ1a_ground_truth_citation/`: Citation judge on **GT Lit**.
- `RQ1a_ground_truth_correctness_test/`: Small test/validation run used to sanity-check parts of the GT correctness pipeline.

**Common substructure** (inside the above RQ1a folders):
- `{cld}/run_{n}/...`: raw judged Excel workbooks.
- `enhanced_analysis_*/`: analysis outputs generated after the runs (tables/figures/metrics).

#### RQ1a Validation Studies
- `RQ1_human_validation_citation_judge/`: Human-vs-LLM agreement validation input files.
- `RQ1a_ulemans_citation_provider_comparison/`: Search-provider comparison on Ulemans expert CLD.
- `RQ1a_validation_physics_CLDs/`: Physics CLD “sanity check” validation runs.

#### RQ1b: LLM-as-a-Corrector
- `RQ1b_corrector_ablation_analysis/`: Main RQ1b analysis outputs (tables/figures).
- `RQ1b_correction_experiment_final/`: Finalized correction experiment outputs.

Raw RQ1b experiment runs (variant-specific folders):
- `RQ1b_corrector_experiment_corruption_detection_correctness_final_{baseline|cot|mechanistic}*`: Synthetic correction experiments.
- `RQ1b_corrector_experiment_ground_truth_correctness_{baseline|cot|mechanistic}*`: Ground-truth correction experiments.

Concrete folders present in this repo snapshot:
- `RQ1b_corrector_experiment_corruption_detection_correctness_final_baseline`
- `RQ1b_corrector_experiment_corruption_detection_correctness_final_cot`
- `RQ1b_corrector_experiment_corruption_detection_correctness_final_mechanistic`
- `RQ1b_corrector_experiment_corruption_detection_final_cot`
- `RQ1b_corrector_experiment_ground_truth_correctness_baseline`
- `RQ1b_corrector_experiment_ground_truth_correctness_baseline_BACKUP_baseline_judge`
- `RQ1b_corrector_experiment_ground_truth_correctness_baseline_mechanistic_judge`
- `RQ1b_corrector_experiment_ground_truth_correctness_cot`
- `RQ1b_corrector_experiment_ground_truth_correctness_cot_BACKUP_baseline_judge`
- `RQ1b_corrector_experiment_ground_truth_correctness_cot_mechanistic_judge`
- `RQ1b_corrector_experiment_ground_truth_correctness_mechanistic`
- `RQ1b_corrector_experiment_ground_truth_correctness_mechanistic_BACKUP_baseline_judge`
- `RQ1b_corrector_experiment_ground_truth_correctness_mechanistic_mechanistic_judge`

Backups / variants:
- `RQ1b_corrector_ablation_analysis_BACKUP_baseline_judge/`: legacy backup.
- `*_BACKUP_baseline_judge*`, `*_mechanistic_judge*`: runs produced under different judge variants.

Standalone RQ1b artifacts:
- `rq1b_reproducibility_report.txt`: textual reproduction notes/report.
- `RQ1b_mechanistic_judge_ablation_log.txt`: ablation log.

---

### RQ2: Context-Insensitive Hallucination Detection (CI/UQ metrics)

- `RQ2_hallucination_detection/`: Main RQ2 analysis directory.
  - Outputs: plots (`*.png`), tables (`*.tex`/`*.xlsx`/`*.json`).
  - Entry scripts (kept here): `rq2_simple_batch_analyzer.py`, `rq2_master_report_v2_enhanced.py`.

(Phase 4/5/6 implementation scripts live in `data_science/` but are orchestrated by `RQ2_unified_analysis.py`.)

---

### RQ3: Deep Research Validation

- `RQ3_deep_research/`: Main RQ3 directory.
  - `scripts/`: helper scripts (sampling, extrapolation, etc.).
  - `figures/`: RQ3 figures.
  - `data/`, `validation/`: inputs and validation subsets.
  - Analysis entry points inside: `rq3_statistical_test.py`, `rq3_per_cld_detection_rates.py`, `compute_enriched_gt_metrics.py`, `run_rq3_analysis.py`, `generate_figures.py`.

---

## Cross-cutting / Supporting Analyses

### Preliminaries & Benchmarks
- `generator_model_comparison_analysis/`: Generator model selection study (e.g., GPT-4.1 vs alternatives).
- `embedding_model_benchmark/`: Embedding model benchmark used to justify the cosine-similarity embedding choice.
- `random_baseline_generator/`: Structural random baseline generation + summary artifacts.

### Sensitivity & Supplementary Analyses
- `supp_prompt_sensitivity/`: Prompt sensitivity analysis (judge and corrector).
  - `analysis_scripts/`: `sensitivity_analysis_simple.py`, `sensitivity_analysis_corrector.py`.
  - Pre-computed figures: `prompt_sensitivity_figure.png`, `corrector_sensitivity_figure.png`.
- `supp_time_cost_scaling/`: Time and cost scaling analysis.
  - `analysis_scripts/aggregate_time_cost_scaling.py`: generates scaling figures.
- `supp_parallelization_benchmark/`: Parallelization benchmark.
  - `analysis_scripts/analyze_parallelization_results.py`: generates benchmark figures.
- `Sensitivity_analysis/`: Legacy aggregated sensitivity outputs.
- `Sensitivity_analysis_simple/`: Legacy prompt sensitivity artifacts.
- `Sensitivity_analysis_sobol/`: Legacy Sobol-style prompt sensitivity artifacts.

### Temperature Sensitivity
- `temperature_sensitivity_analysis/`: Temperature sensitivity outputs (ANOVA tables, raw CSV, LaTeX table).
- `temperature_sensitivity_assumption_tests.py`: assumption testing script.
- `temperature_sensitivity_assumption_tests.json`: outputs.

### Cost / Scaling / Efficiency
- `cost_analysis/`: Token/cost accounting scripts and generated reports (see `cost_analysis/README.md`).
- `time_and_cost_scaling/`: Scaling analysis artifacts.
- `aggregate_time_cost_scaling.py`: top-level scaling aggregation script.
- `time_cost_scaling_20251217_171722.xlsx`: exported scaling spreadsheet.
- `scaling_analysis_summary.txt`: narrative summary.

### Statistics Utilities
- `multiple_comparisons/`: Multiple-comparisons correction outputs (`*_latest.json`).
- `power_analysis/`: power analysis utilities (`compute_power_analysis.py`).

### Token/Call Diagnostics
- `token_call_analysis_RQ1a_judge/`: Output-token and call diagnostics for RQ1a (see `token_call_analysis_RQ1a_judge/README.md`).

### Global Output Aggregates
- `figures/`: Global figures not tied to a single RQ folder.
- `latex/`: Global LaTeX tables not tied to a single RQ folder.

### Other One-off Artifacts
- `retrieve_effect_sizes.py`: helper to extract effect sizes for reporting.
- `verdict_inconsistencies_report.csv`: QA artifact listing inconsistencies.
- `random_baseline_summary.txt`: random baseline summary.

---

## Metadata / Documentation Files

- `DATA_README.md`: higher-level description of datasets and outputs.
- `ANALYSIS_METADATA.txt`: provenance metadata for scaling analyses.
- `README_RQ1a_corrupted_correctness_experiments.md`: RQ1a corruption/correctness notes.

---

## Legacy / Deprecated

- `deprecated/`: old runs/scripts kept for reference.
- `parallelization_benchmark/`: benchmarking artifacts.
- `test_CLD/`: temporary testing outputs.

**Convention:** folders containing `BACKUP` are not canonical and should not be used as primary sources unless explicitly referenced.








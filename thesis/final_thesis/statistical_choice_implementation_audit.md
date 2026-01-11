# Statistical-choice implementation audit (per table)

Single list: **one bullet per table** (including `\input{...table...}` tables).  
Legend for compliance with your stated rule (Methods Table `tab:uncertainty_rule`):
- **Complies**: repeated evals shown as **mean [min, max]**; CIs reserved for large instance-level N (or Wilson/bootstrap where appropriate)
- **Conflicts**: repeated evals shown as **mean ± SD** or **t/normal-based CI** for very small repeat counts, etc.
- **N/A**: design/config/accounting tables (no uncertainty intended)

- **[Methods] `tab:master` — “Master Experimental Configuration with Hypotheses”**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex`  
  - **Unit / N**: explicitly encodes your nested design (e.g., 3 CLDs × 3 runs × prompts; counts per RQ in table).  
  - **Uncertainty shown**: N/A  
  - **Tests / assumptions**: N/A  
  - **Compliance**: **N/A (design table; but it supports your small-n argument well)**

- **[Methods] `tab:cld_context` — “LLM Context Parameters by CLD (Extracted from Validation Excel Files)”**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex`  
  - **Unit / N**: 3 CLDs; descriptive context.  
  - **Uncertainty shown**: N/A  
  - **Tests / assumptions**: N/A  
  - **Compliance**: **N/A**

- **[Methods] (unlabeled) TruthfulQA judge configuration table (Parameter / Value / Rationale)**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex` (around the TruthfulQA section)  
  - **Unit / N**: config table  
  - **Uncertainty shown**: N/A  
  - **Tests / assumptions**: N/A  
  - **Compliance**: **N/A**

- **[Methods] (unlabeled) UQ metric definition table (Metric / Formula / Expected)**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex`  
  - **Unit / N**: definition table  
  - **Uncertainty shown**: N/A  
  - **Tests / assumptions**: N/A  
  - **Compliance**: **N/A**

- **[Methods] `tab:uq_params` — “UQ Metric Implementation Parameters and Rationale”**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex`  
  - **Unit / N**: implementation choices  
  - **Uncertainty shown**: N/A  
  - **Tests / assumptions**: N/A  
  - **Compliance**: **N/A**

- **[Methods] `tab:dr_params` — “Deep Research Experimental Parameters”**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex`  
  - **Unit / N**: experimental settings  
  - **Uncertainty shown**: N/A  
  - **Tests / assumptions**: N/A  
  - **Compliance**: **N/A**

- **[Methods] `tab:dr_dataset` — “Deep Research Validation Dataset”**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex`  
  - **Unit / N**: edges; TP=44, FP=178, FN=63 (total 285); also gives per-CLD breakdown in text.  
  - **Uncertainty shown**: none (dataset definition)  
  - **Tests / assumptions**: N/A  
  - **Compliance**: **N/A**

- **[Methods] `tab:dr_cost` — “Deep Research Computational Cost and Edge Space by CLD”**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex`  
  - **Unit / N**: cost accounting by CLD  
  - **Uncertainty shown**: N/A  
  - **Tests / assumptions**: N/A  
  - **Compliance**: **N/A**

- **[Methods] `tab:uncertainty_rule` — “Uncertainty Reporting Rule (Applied Consistently Throughout)”**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex`  
  - **Rule**: repeated evals → **mean [min, max]**; instance-level large independent N → **95% CI (bootstrap/Wilson)**; warns folds aren’t independent.  
  - **Compliance**: **This is the rule; used as the reference for ‘Complies/Conflicts’.**

- **[Results] `tab:generator_model_comparison` — “Generator Model Comparison: Edge Recovery Performance”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: explicitly **1 run** per model×CLD (cost constraint).  
  - **Uncertainty shown**: none  
  - **Tests / assumptions**: none  
  - **Compliance**: **Complies** (no replication → no interval implied)

- **[Results] `tab:random_baseline` — “Random Baseline Comparison for Generator Edge Recovery”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: Random baseline Monte Carlo **n=1000**; LLM **3 runs per CLD**.  
  - **Uncertainty shown**:
    - Random baseline: **95% CI** (Monte Carlo; appropriate)
    - LLM: **mean [min, max]** (n=3 runs)
  - **Tests / assumptions**: none (descriptive comparison)
  - **Compliance**: **Complies**

- **[Methods→Results mismatch affecting `tab:random_baseline`] (not a table; but impacts your “implemented consistently” claim)**  
  - **Source**: `thesis/final_thesis/method_experiments_final_body.tex` (random baseline methodology text)  
  - **Issue**: says “Per-CLD 95% CI uses t-distribution (n=3)” for LLM runs, while Results table/figure uses **mean [min,max]** for the n=3 LLM runs.  
  - **Compliance**: **Conflicts (text contradicts your rule + contradicts the Results table).**

- **[Results] `tab:temperature_sensitivity` — “Generator Temperature Sensitivity: Edge F1 Score by CLD”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: per temperature: 3 runs per CLD; also an “Overall” column across 9 runs.  
  - **Uncertainty shown**: **Mean ± Std** (including per-CLD columns where n=3).  
  - **Tests / assumptions**: ANOVA + Shapiro/Levene mentioned in Results text; Kruskal-Wallis mentioned as corroboration.  
  - **Compliance**: **Conflicts** (per-CLD repeated evals with n=3 shown as mean±SD rather than mean[min,max]).

- **[Results] `tab:truthqa_verification` — “TruthfulQA Judge Verification (n=1,634)”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: instances (n=1,634).  
  - **Uncertainty shown**: none  
  - **Tests / assumptions**: none  
  - **Compliance**: **Complies** (large N; descriptive summary)

- **[Results] `tab:ulemans_search_providers` — “Search Provider Comparison on Ulemans Expert CLD (150 edges)”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: edges (n=150).  
  - **Uncertainty shown**: none  
  - **Tests / assumptions**: none  
  - **Compliance**: **Complies** (descriptive infrastructure comparison)

- **[Results] `tab:human_validation` — “Human Validation: LLM Judge vs. Expert Inter-Rater Agreement”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: edges; per dataset and combined (n=72 combined).  
  - **Uncertainty shown**: none  
  - **Tests / assumptions**: correlation significance stars shown; κ interpretation note.  
  - **Compliance**: **Complies**

- **[Results] `tab:human_validation_confusion` — “Confusion Matrix: LLM Judge vs. Human Expert (Combined, n=72)”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: edges (n=72).  
  - **Uncertainty shown**: none  
  - **Tests / assumptions**: none  
  - **Compliance**: **Complies**

- **[Results] `tab:physics-comparison-enhanced` — “Physics CLD Validation: Correctness vs Citation-Based Judging”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: edges (total 31).  
  - **Uncertainty shown**: none  
  - **Tests / assumptions**: none  
  - **Compliance**: **Complies** (descriptive; no CI implied)

- **[Results] `tab:retrieval-comparison` — “Retrieval Success Comparison”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: edges (rates by dataset)  
  - **Uncertainty shown**: none  
  - **Tests / assumptions**: none  
  - **Compliance**: **Complies**

- **[Results] `tab:rq1b_synthetic_correctness` — “Corrector Performance on Synthetically Corrupted Data (Baseline Prompt)”**  
  - **Source**: `final_runs/RQ1b_corrector_ablation_analysis/baseline_table_synthetic.tex` (included via `\input{...}` in `final_results.tex`)  
  - **Unit / N**: 3 runs per CLD; also reports blocked n=9 (CLD,run) for efficacy test.  
  - **Uncertainty shown**: **mean ± std across 3 runs** per CLD; macro avg mean±std across CLDs.  
  - **Tests / assumptions**: **blocked one-sample Wilcoxon** on n=9 blocks.  
  - **Compliance**: **Conflicts** (uncertainty formatting for n=3 repeats).

- **[Results] `tab:rq1b_prompt_ablation_synthetic` — “Corrector Prompt Ablation on Synthetic Corruptions”**  
  - **Source**: `final_runs/RQ1b_corrector_ablation_analysis/ablation_table_synthetic.tex` (via `\input`)  
  - **Unit / N**: 3 runs per CLD per prompt; blocked n=9 for tests.  
  - **Uncertainty shown**: **mean ± std**.  
  - **Tests / assumptions**: **Friedman blocked + Wilcoxon post-hoc + Bonferroni**.  
  - **Compliance**: **Conflicts** (uncertainty formatting).

- **[Results] `tab:rq1b_groundtruth_correctness` — “Corrector Performance on Ground Truth Data … (Baseline Prompt)”**  
  - **Source**: `final_runs/RQ1b_corrector_ablation_analysis/baseline_table_groundtruth.tex` (via `\input`)  
  - **Unit / N**: 3 runs per CLD; blocked n=9 efficacy test.  
  - **Uncertainty shown**: **mean ± std**.  
  - **Tests / assumptions**: **blocked one-sample Wilcoxon**.  
  - **Compliance**: **Conflicts** (uncertainty formatting).

- **[Results] `tab:rq1b_prompt_ablation_groundtruth` — “Corrector Prompt Ablation on Ground Truth”**  
  - **Source**: `final_runs/RQ1b_corrector_ablation_analysis/ablation_table_groundtruth.tex` (via `\input`)  
  - **Unit / N**: 3 runs per CLD per prompt; blocked n=9 for tests.  
  - **Uncertainty shown**: **mean ± std**.  
  - **Tests / assumptions**: **Friedman blocked + Wilcoxon post-hoc + Bonferroni**.  
  - **Compliance**: **Conflicts** (uncertainty formatting).

- **[Results] `tab:rq2_single_metrics` — “Single UQ Metric Performance for Hallucination Detection (Meta-Analysis)”**  
  - **Source**: `thesis/final_thesis/final_results.tex` (inline table)  
  - **Unit / N**: edges (31,704 overall; cosine defined on 16,492); also N files (63 / 35).  
  - **Uncertainty shown**: **95% CIs** for mean AUC and Pearson r across files/edges (as written).  
  - **Tests / assumptions**: Mann–Whitney U (non-param) noted; bootstrap CIs (10k) noted; CLT justification for Pearson r at large n.  
  - **Compliance**: **Complies (not a “3 repeats” situation; inferential model is stated).**

- **[Results] `tab:rq2_normality_tests` — “Shapiro-Wilk Normality Tests for UQ Metric Distributions”**  
  - **Source**: `final_runs/RQ2_hallucination_detection/normality_tests_table.tex` (via `\input`)  
  - **Unit / N**: edges (with subsampling noted due to Shapiro limit).  
  - **Uncertainty shown**: N/A (assumption diagnostics table).  
  - **Tests / assumptions**: Shapiro–Wilk; supports non-param decisions.  
  - **Compliance**: **N/A / Supports your justification**

- **[Results] `tab:rq2_ensemble_performance` — “Ensemble Classifier Performance Across Validation Phases”**  
  - **Source**: `thesis/final_thesis/final_results.tex`  
  - **Unit / N**: CV folds (n=5), CLDs (n=3).  
  - **Uncertainty shown**: **mean [min, max]** across folds/CLDs (explicitly justified in table note).  
  - **Tests / assumptions**: descriptive; caveat about fold dependence is stated.  
  - **Compliance**: **Complies**

- **[Results] `tab:rq3_results` — “Deep Research Detection Rate and Discrimination Analysis (n=285 edges)”**  
  - **Source**: `final_runs/RQ3_deep_research/rq3_main_table.tex` (via `\input`)  
  - **Unit / N**: edges (n=285; TP/FP/FN counts shown).  
  - **Uncertainty shown**: not Wilson rate CIs in-table, but **OR 95% CI** and **Cohen’s h 95% CI** are in notes.  
  - **Tests / assumptions**: Fisher exact; Bonferroni across 3 comparisons.  
  - **Compliance**: **Complies** (inferential uncertainty + exact test; not relying on normality)

- **[Results] `tab:rq3_per_cld` — “Per-CLD Pairwise Discrimination Tests (Fisher’s Exact, Bonferroni m=3)”**  
  - **Source**: `final_runs/RQ3_deep_research/rq3_per_cld_table.tex` (via `\input`)  
  - **Unit / N**: per-CLD exact tests  
  - **Uncertainty shown**: N/A (test summary table)  
  - **Tests / assumptions**: Fisher exact + Bonferroni  
  - **Compliance**: **N/A / Complies**

- **[Results] `tab:enriched_gt` — “LLM Generation Metrics: Two Enrichment Scenarios (Aggregate and Per-CLD)”**  
  - **Source**: `final_runs/RQ3_deep_research/rq3_enriched_gt_table.tex` (via `\input`)  
  - **Unit / N**: scenario accounting based on DR detection flags; per-CLD rows.  
  - **Uncertainty shown**: none (presented explicitly as scenarios/upper-bound style).  
  - **Tests / assumptions**: scenario assumptions spelled out in surrounding text.  
  - **Compliance**: **Complies (exploratory scenario; no small-n CI claims)**

- **[Results] `tab:rq3_verdict_distribution` — “Human Validation: Evidence Verdict Distribution by Edge Classification (n=50)”**  
  - **Source**: `thesis/final_thesis/generated/rq3_verdict_distribution_table.tex` (via `\input`)  
  - **Unit / N**: human-labeled edges (n=50).  
  - **Uncertainty shown**: none (counts/percentages).  
  - **Tests / assumptions**: none.  
  - **Compliance**: **Complies**

- **[Results] `tab:rq3_applicability_distribution` — “Evidence Applicability Score Distribution … (n=50)”**  
  - **Source**: `thesis/final_thesis/generated/rq3_applicability_distribution_table.tex` (via `\input`)  
  - **Unit / N**: applicability scores by class (TP n=9, FP n=32, FN n=9).  
  - **Uncertainty shown**: **Mean ± SD** (plus binned counts).  
  - **Tests / assumptions**: none.  
  - **Compliance**: **Partial** (descriptive is fine, but it’s not following the “mean[min,max] for repeats” pattern you claim to apply consistently).

- **[Results] `tab:applicability_rubric` — “Applicability Score Rubric for Human Validation”**  
  - **Source**: `thesis/final_thesis/final_results.tex` (inline table)  
  - **Unit / N**: rubric definition  
  - **Uncertainty shown**: N/A  
  - **Tests / assumptions**: N/A  
  - **Compliance**: **N/A**

- **[Results] `tab:enriched_gt_validated` — “LLM Generation Metrics with Human-Validated Ground Truth Enrichment (Aggregate, extrapolated)”**  
  - **Source**: `thesis/final_thesis/generated/rq3_enrichment_validation_table.tex` (via `\input`)  
  - **Unit / N**: extrapolation from human validation; shows point + conservative scenarios.  
  - **Uncertainty shown**: no explicit CI; provides conservative bounds via Wilson lower bound logic elsewhere in text.  
  - **Tests / assumptions**: explicitly labeled extrapolation.  
  - **Compliance**: **Complies (exploratory; not relying on unverifiable parametric assumptions)**

- **[Results] `tab:rq3_smart_trigger` — “Smart-trigger Deep Research enrichment estimates …”**  
  - **Source**: `thesis/final_thesis/generated/rq3_smart_trigger_table.tex` (via `\input`)  
  - **Unit / N**: extrapolated enrichment; trigger policy described in Results text.  
  - **Uncertainty shown**: **F1 (95% CI)** for enriched/system rows.  
  - **Tests / assumptions**: CI is tied to the human-calibration step; Results text says it propagates Wilson bounds.  
  - **Compliance**: **Complies** (inferential interval is explicitly model-based; you also provide conservative estimate)

- **[Results] `tab:effect_size_summary` — “Effect Size Summary: Hypothesis Operationalization, Statistical Tests, and Conclusions”**  
  - **Source**: `thesis/final_thesis/generated/effect_size_summary_table.tex` (via `\input`)  
  - **Unit / N**: summary across RQs; includes n where relevant (e.g., 9 blocks, n=72, n=285).  
  - **Uncertainty shown**: none (test+effect summary).  
  - **Tests / assumptions**: explicitly enumerated (Friedman/Wilcoxon/Fisher/etc).  
  - **Compliance**: **Complies**

- **[Results] `tab:scaling_summary` — “Computational Efficiency Summary by Experiment”**  
  - **Source**: `final_runs/latex/table21_scaling_summary.tex` (via `\input`)  
  - **Unit / N**: accounting table (files, edges, tokens/edge, cost).  
  - **Uncertainty shown**: none.  
  - **Tests / assumptions**: none.  
  - **Compliance**: **N/A**

- **[Results] `tab:parallelization` — “Parallelization Benchmark: Wall-Clock Time and Speedup by Judge Type”**  
  - **Source**: `thesis/final_thesis/final_results.tex` (inline table)  
  - **Unit / N**: 3 runs per worker config (n=3).  
  - **Uncertainty shown**: **mean ± SD** (n=3).  
  - **Tests / assumptions**: none.  
  - **Compliance**: **Conflicts** (small repeated evals shown as mean±SD rather than mean[min,max]).


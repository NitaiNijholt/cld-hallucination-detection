## Appendix purpose map (purpose + one-sentence takeaway per section)

Note: Labels are the appendix `\label{...}` tags to make cross-referencing easy from Methods/Results.

### Function-to-Algorithm Mapping (`sec:function_mapping`)
- **Purpose**: Provide a complete trace from result directories → scripts → core implementation functions, to verify the algorithmic descriptions.
- **One-sentence takeaway**: The mapping makes every reported result reproducible and directly auditable against the exact code paths used in Appendix `sec:algorithms`.
- **Appendix reference**: Appendix § “Function-to-Algorithm Mapping” (`sec:function_mapping`) (see also mapping table `tab:function_mapping`).

### Reproducibility (`sec:reproducibility`)
- **Purpose**: Centralize everything needed to replicate the experiments (data, code, environment, and commands).
- **One-sentence takeaway**: The appendix specifies end-to-end reproduction inputs, tooling, and commands so every RQ1–RQ3 run can be recreated from the repository.
- **Appendix reference**: Appendix § “Reproducibility” (`sec:reproducibility`).

#### Data Availability (`sec:data_availability`)
- **Purpose**: Point to the ground-truth datasets and where all experiment outputs live.
- **One-sentence takeaway**: Ground-truth inputs and all final outputs are organized in fixed, named directories so analyses can be rerun from the same artifacts.
- **Appendix reference**: Appendix § Reproducibility (`sec:reproducibility`) → “Data Availability” (`sec:data_availability`).

##### Ground Truth CLD Datasets
- **Purpose**: Identify the exact dataset files used as ground truth for evaluation.
- **One-sentence takeaway**: The three Excel datasets contain edges, polarity, motivations, citations, and generation-time UQ metrics and live under `data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/`.
- **Appendix reference**: Appendix § Reproducibility (`sec:reproducibility`) → Data Availability (`sec:data_availability`) → “Ground Truth CLD Datasets” (table `tab:gt_files`).

##### Final Experiment Results
- **Purpose**: Define where all final experiment results are stored and how they are structured.
- **One-sentence takeaway**: All results are stored under `final_runs/` with a consistent per-RQ/per-CLD/per-run directory layout.
- **Appendix reference**: Appendix § Reproducibility (`sec:reproducibility`) → Data Availability (`sec:data_availability`) → “Final Experiment Results” (directory map block under `final_runs/`).

#### Code Availability (`sec:code_availability`)
- **Purpose**: List the execution and analysis scripts needed to reproduce each experiment and analysis step.
- **One-sentence takeaway**: All primary execution scripts, analysis scripts, and prompt configuration files are enumerated with their repository paths.
- **Appendix reference**: Appendix § Reproducibility (`sec:reproducibility`) → “Code Availability” (`sec:code_availability`) (see tables `tab:exec_scripts`, `tab:analysis_scripts`, `tab:prompt_files`).

##### Software Environment (`sec:software_env`)
- **Purpose**: Specify the hardware/software environment and the exact dependency management approach.
- **One-sentence takeaway**: The full environment is reproducible via `uv` and the repository lockfile, with key dependencies and citations recorded for traceability.
- **Appendix reference**: Appendix § Reproducibility (`sec:reproducibility`) → “Software Environment” (`sec:software_env`) (see tables `tab:system_env`, `tab:software_citations`).

##### Execution Commands (`sec:execution_commands`)
- **Purpose**: Provide exact commands to rerun each experiment and validation pipeline.
- **One-sentence takeaway**: Each RQ pipeline (and supporting analyses like scaling, sensitivity, and benchmarks) is reproducible via explicit shell commands and documented output locations.
- **Appendix reference**: Appendix § Reproducibility (`sec:reproducibility`) → “Execution Commands” (`sec:execution_commands`).

###### RQ1 Judge Verification (TruthfulQA) (`sec:truthqa_verification`)
- **Purpose**: Sanity-check judge behavior on a known hallucination benchmark before CLD-specific evaluation.
- **One-sentence takeaway**: TruthfulQA verification validates the evaluation pipeline’s baseline behavior without implying CLD failures are purely domain-driven.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “RQ1 Judge Verification (TruthfulQA)” (`sec:truthqa_verification`) (config table `tab:truthqa_config`).

###### RQ1a Ground Truth Experiments
- **Purpose**: Show commands to run correctness- and citation-based ground truth judge evaluations.
- **One-sentence takeaway**: Ground-truth judging is rerunnable via `run_rq1a_ground_truth_multirun.py` for both correctness and citation approaches with fixed output dirs.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “RQ1a Ground Truth Experiments” (immediately after `sec:truthqa_verification`; no local `\label`).

###### RQ1a Corruption Detection Experiments
- **Purpose**: Show commands to run corruption detection experiments for both judge types.
- **One-sentence takeaway**: Corruption detection is reproducible via the multi-run drivers for correctness and citation judging with standardized output layouts.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “RQ1a Corruption Detection Experiments” (no local `\label`).

###### RQ1b Corrector Experiments
- **Purpose**: Show commands to run the correction pipeline over judged outputs.
- **One-sentence takeaway**: The corrector pipeline can be rerun by pointing to the ground-truth judging outputs and writing corrected results to a dedicated output directory.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “RQ1b Corrector Experiments” (no local `\label`).

###### RQ1a Validation Studies (`sec:rq1a_validation`)
- **Purpose**: Provide external reliability checks for the citation judge and agreement with a human rater.
- **One-sentence takeaway**: Judge reliability is validated via (i) an external expert-constructed CLD study and (ii) a human–LLM agreement analysis with standard IRR metrics.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “RQ1a Validation Studies” (`sec:rq1a_validation`) (see tables `tab:ulemans_config`, `tab:human_validation_config`).

###### RQ2 Hallucination Detection Analysis
- **Purpose**: Provide commands to compute UQ-metrics analyses and train classifiers for hallucination detection.
- **One-sentence takeaway**: RQ2 is reproducible via the master report script and classifier training script over the `final_runs` artifact set.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “RQ2 Hallucination Detection Analysis” (no local `\label`).

###### RQ3 Deep Research Experiments (`sec:rq3_execution`)
- **Purpose**: Document how to run the multi-agent Deep Research pipeline and the unified analysis + validation workflow.
- **One-sentence takeaway**: RQ3 is reproducible as a 3-stage process (Deep Research execution → unified analysis → human validation/extrapolation) with fully specified configs and outputs.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “RQ3 Deep Research Experiments” (`sec:rq3_execution`) (config table `tab:rq3_config`).

###### RQ3 Human Validation: Evidence Applicability Distribution (`tab:rq3_applicability_distribution`)
- **Purpose**: Summarize how “evidence applicability” scores differ across TP/FP/FN in the human validation sample.
- **One-sentence takeaway**: TP edges show the highest mean applicability and lowest variance, consistent with more literature-grounded relationships than FP/FN.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands → RQ3 (`sec:rq3_execution`) (table `tab:rq3_applicability_distribution`).

###### Generator Temperature Sensitivity Analysis
- **Purpose**: Validate the generator temperature choice by testing how temperature affects edge-recovery F1.
- **One-sentence takeaway**: Temperature is treated as a tested design choice by running a controlled ANOVA over multiple temperatures and seeds on the validation CLDs.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Generator Temperature Sensitivity Analysis” (config table `tab:temp_sensitivity_config`; no local `\label`).

###### Generator Model Comparison (`sec:generator_comparison`)
- **Purpose**: Select the generator model for subsequent experiments via a head-to-head edge-recovery comparison.
- **One-sentence takeaway**: The chosen generator is the model with the best aggregate ground-truth edge F1 under comparable settings (and logprobs availability).
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Generator Model Comparison” (`sec:generator_comparison`) (config table `tab:generator_comparison_config`).

###### Random Baseline Generator (`sec:random_baseline`)
- **Purpose**: Establish a structural baseline using random graphs matched to CLD sizes.
- **One-sentence takeaway**: Erdős–Rényi baselines quantify expected edge F1 under randomness for context against learned generation performance.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Random Baseline Generator” (`sec:random_baseline`) (config table `tab:random_baseline_config`).

###### Prompt Sensitivity Analysis
- **Purpose**: Quantify how prompt changes map to performance changes for judges and correctors.
- **One-sentence takeaway**: Prompt sensitivity is measured via semantic prompt distance vs. F1 change, with nonparametric and repeated-measures testing across experiments.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Prompt Sensitivity Analysis” (see also Supplementary Sensitivity Analysis `app:sensitivity_analysis`).

###### Parallelization Benchmark
- **Purpose**: Measure wall-clock speedups from running the pipeline with multiple workers.
- **One-sentence takeaway**: Parallel execution yields measurable speedups over sequential runs, characterized across multiple worker counts and repeated trials.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Parallelization Benchmark” (config table `tab:parallel_benchmark_config`; no local `\label`).

###### Time and Cost Scaling Analysis (`sec:time_cost_scaling`)
- **Purpose**: Aggregate runtime and token/cost metrics to characterize computational scaling behavior.
- **One-sentence takeaway**: Judging scales roughly linearly with edges and citation judging is substantially more expensive/slower due to retrieval and longer contexts.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Time and Cost Scaling Analysis” (`sec:time_cost_scaling`) (figures `fig:scaling_analysis`, `fig:config_comparison`; config table `tab:scaling_analysis_config`).

###### Physics CLD Validation (`sec:physics_cld_validation`)
- **Purpose**: Validate judge functionality on physics CLDs with well-established causal relationships.
- **One-sentence takeaway**: Physics-grounded CLDs provide a high-confidence “should-pass” testbed for judge sanity checking across correctness and citation modes.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Physics CLD Validation” (`sec:physics_cld_validation`) (config table `tab:physics_cld_config`).

###### Cost Analysis (`sec:cost_analysis`)
- **Purpose**: Compute API costs from token usage recorded in experiment Excel files.
- **One-sentence takeaway**: Costs are derived directly from logged token usage to produce complete and per-experiment cost breakdowns (including Deep Research).
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Cost Analysis” (`sec:cost_analysis`) (config table `tab:cost_analysis_config`).

###### Power Analysis (`sec:power_analysis`)
- **Purpose**: Estimate statistical power for prompt comparisons given observed effect sizes.
- **One-sentence takeaway**: Power estimates translate empirical effect sizes into required sample sizes and expected detectability for pairwise comparisons.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Power Analysis” (`sec:power_analysis`) (no further labeled artifacts).

###### Embedding Model Benchmark (execution) (`sec:embedding_benchmark_execution`)
- **Purpose**: Benchmark embedding models used for cosine similarity within RQ2.
- **One-sentence takeaway**: MPNet is selected because it is dramatically faster while remaining effective for the downstream similarity/reranking workload.
- **Appendix reference**: Appendix § Reproducibility → Execution Commands (`sec:execution_commands`) → “Embedding Model Benchmark” (`sec:embedding_benchmark_execution`) (config table `tab:embedding_benchmark_config`).

#### Output Artifacts (`sec:output_artifacts`)
- **Purpose**: Document what files each run produces and how to interpret/locate them.
- **One-sentence takeaway**: Each run emits standardized metadata and result files (Excel/JSON/plots) with consistent naming conventions to support automated analysis.
- **Appendix reference**: Appendix § Reproducibility (`sec:reproducibility`) → “Output Artifacts” (`sec:output_artifacts`) (output file table `tab:output_files`).

#### Output Token Verification (`sec:token_verification`)
- **Purpose**: Verify that judge outputs were not truncated by `max_tokens` limits.
- **One-sentence takeaway**: Outputs stayed below token limits (near-limit only for baseline), and lower mech-lit success is attributed to citation retrieval failures rather than truncation.
- **Appendix reference**: Appendix § Reproducibility (`sec:reproducibility`) → “Output Token Verification” (`sec:token_verification`) (analysis table `tab:token_analysis`).

### Prompt Engineering Ablation Study (`sec:ablation_edge`)
- **Purpose**: Systematically compare prompt-engineering strategies for CLD edge generation.
- **One-sentence takeaway**: Prompt design materially affects edge-recovery performance, with Nitai_C best overall and strict citation/RAG constraints sometimes hurting discovery.
- **Appendix reference**: Appendix § “Prompt Engineering Ablation Study” (`sec:ablation_edge`) (see technique matrix `tab:prompt_techniques`, results table `tab:f1_scores`, figure `fig:ablation_f1_ordered`).

#### Experimental Design
- **Purpose**: Define the ablation setup (prompts, CLDs, metrics, fixed hyperparameters).
- **One-sentence takeaway**: Nine prompt variants are evaluated on three CLDs using a consistent generator configuration and F1 as the primary metric.
- **Appendix reference**: Appendix § Prompt Engineering Ablation Study (`sec:ablation_edge`) → “Experimental Design” (no local `\label`).

#### Prompt Engineering Techniques
- **Purpose**: Classify prompt variants by technique (CoT, few-shot, roles, decomposition, RAG).
- **One-sentence takeaway**: Variants differ systematically in scaffolding strategies, enabling attribution of performance differences to prompt techniques.
- **Appendix reference**: Appendix § Prompt Engineering Ablation Study (`sec:ablation_edge`) → “Prompt Engineering Techniques” (technique matrix `tab:prompt_techniques`).

#### Results
- **Purpose**: Present performance outcomes across prompt variants.
- **One-sentence takeaway**: Nitai_C achieves the highest average F1 (0.460) while some variants (e.g., overly restrictive definition-grounded prompts) fail entirely.
- **Appendix reference**: Appendix § Prompt Engineering Ablation Study (`sec:ablation_edge`) → “Results” (figure `fig:ablation_f1_ordered`, table `tab:f1_scores`).

#### Key Findings
- **Purpose**: Summarize the main empirical observations from the ablation.
- **One-sentence takeaway**: CoT-rich prompts tend to improve performance, RAG-style strict citation requirements trade off recall, and CLD 2 is consistently the hardest case.
- **Appendix reference**: Appendix § Prompt Engineering Ablation Study (`sec:ablation_edge`) → “Key Findings” (no local `\label`).

#### Interpretation
- **Purpose**: Provide explanatory hypotheses for why specific prompting strategies succeed or fail.
- **One-sentence takeaway**: Step-by-step reasoning scaffolds can help holistic causal discovery more than heavy decomposition, depending on task structure.
- **Appendix reference**: Appendix § Prompt Engineering Ablation Study (`sec:ablation_edge`) → “Interpretation” (no local `\label`).

#### Limitations
- **Purpose**: State constraints that limit generalization of the ablation results.
- **One-sentence takeaway**: With only three CLDs and limited repeats, findings are indicative rather than definitive and may be model-specific.
- **Appendix reference**: Appendix § Prompt Engineering Ablation Study (`sec:ablation_edge`) → “Limitations” (no local `\label`).

### Node Generation Prompt Ablation (`sec:ablation_node`)
- **Purpose**: Test how sensitive node generation is to prompt variants under a factorial repeated-run design.
- **One-sentence takeaway**: Node generation is robust—no significant performance differences across prompts, suggesting it is an easier and less prompt-sensitive subtask than edge generation.
- **Appendix reference**: Appendix § “Node Generation Prompt Ablation” (`sec:ablation_node`) (node ablation table `tab:node_ablation`).

### Embedding Model Benchmark (`sec:embedding_benchmark`)
- **Purpose**: Justify the embedding model choice used for cosine similarity computations.
- **One-sentence takeaway**: MPNet is preferred due to large speed advantages and strong reranking relevance, despite higher aggregate benchmark scores for larger models.
- **Appendix reference**: Appendix § “Embedding Model Benchmark” (`sec:embedding_benchmark`) (tables `tab:embed_benchmark`, `tab:mteb_comparison`).

### Algorithmic Descriptions (`sec:algorithms`)
- **Purpose**: Provide formalized, implementation-traceable algorithm descriptions of all core system components.
- **One-sentence takeaway**: The system can be audited algorithm-by-algorithm (generation, judging, corruption, correction, UQ metrics, classification, citation fetching, deep research) with explicit code anchors.
- **Appendix reference**: Appendix § “Algorithmic Descriptions” (`sec:algorithms`).

#### Algorithm 0: CLD Generation (`alg:cld_generation`)
- **Purpose**: Describe the two-phase CLD generation pipeline (nodes then edges) and how UQ metrics are captured.
- **One-sentence takeaway**: CLDs are generated by iteratively proposing nodes and then evaluating all ordered variable pairs for edges while logging token-level uncertainty signals.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → Algorithm 0 (`alg:cld_generation`).

#### Algorithm 1: Judge-Correctness (`alg:judge_correctness`)
- **Purpose**: Specify how correctness judging is performed without external evidence.
- **One-sentence takeaway**: The correctness judge scores edge motivations via per-edge prompting and aggregates across judges by mean score and majority vote.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → Algorithm 1 (`alg:judge_correctness`).

#### Algorithm 2: Judge-Citation (`alg:judge_citation`)
- **Purpose**: Specify how citation-grounded judging fetches sources and evaluates support.
- **One-sentence takeaway**: Citation judging scores each cited source, aggregates per-citation scores, and synthesizes an edge-level verdict grounded in retrieved text.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → Algorithm 2 (`alg:judge_citation`).

#### Algorithm 3: Corruption (`alg:corruption`)
- **Purpose**: Describe controlled corruption generation with balanced corruption types.
- **One-sentence takeaway**: Corruptions are stratified and reproducible, injecting spurious edges or altering motivations/polarity while logging full corruption metadata.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → Algorithm 3 (`alg:corruption`).

#### Algorithm 4: Corrector (`alg:corrector`)
- **Purpose**: Describe how flagged edges are repaired via tool-based LLM actions.
- **One-sentence takeaway**: The corrector tries ordered remediation actions per edge and can optionally rejudge to verify post-correction quality.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → Algorithm 4 (`alg:corrector`).

#### Algorithm 5: UQ Metrics Computation (`alg:uq_metrics`)
- **Purpose**: Define uncertainty and similarity metrics computed from logprobs and embeddings.
- **One-sentence takeaway**: Perplexity/min-prob/max-entropy and citation similarity are computed from generation outputs to support downstream hallucination modeling.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → Algorithm 5 (`alg:uq_metrics`).

#### Algorithm 6: Hallucination Classification (`alg:classification`)
- **Purpose**: Define the classifier training/evaluation pipeline over UQ metrics.
- **One-sentence takeaway**: Multiple classifiers are trained with scaling + cross-validation to predict hallucination labels from UQ features.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → Algorithm 6 (`alg:classification`).

#### Algorithm 7: Citation Fetcher (`alg:citation_fetcher`)
- **Purpose**: Specify how citations are fetched and converted to usable text.
- **One-sentence takeaway**: Retrieval uses a primary Jina Reader path with retries/backoff and a PDF-capable fallback to maximize robust citation ingestion.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → Algorithm 7 (`alg:citation_fetcher`).

#### Deep Research Multi-Agent Components (`sec:dr_agents_appendix`)
- **Purpose**: Enumerate the agents, models, and responsibilities in the Deep Research pipeline.
- **One-sentence takeaway**: Deep Research is decomposed into specialized agents (planning, searching, scoring, selecting, judging, causal-evidence detection, verdict synthesis) coordinated by an orchestrator.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → “Deep Research Multi-Agent Components” (`sec:dr_agents_appendix`) (agent table `tab:dr_agents`).

#### Algorithm 8: Deep Research Multi-Agent Pipeline (`alg:deep_research`)
- **Purpose**: Provide a step-by-step algorithm for iterative literature validation of causal claims.
- **One-sentence takeaway**: Deep Research executes parallel search/scoring with smart selection and early stopping, producing a structured verdict plus direct-evidence flags.
- **Appendix reference**: Appendix § Algorithmic Descriptions (`sec:algorithms`) → Algorithm 8 (`alg:deep_research`).

### Judge Score Heatmaps (`sec:appendix_judge_heatmaps`)
- **Purpose**: Provide detailed heatmaps of judge score behavior across tasks, prompts, and CLDs.
- **One-sentence takeaway**: Heatmaps visualize whether judges assign high scores to true relationships and low scores to hallucinations across experimental conditions.
- **Appendix reference**: Appendix § “Judge Score Heatmaps” (`sec:appendix_judge_heatmaps`) (figures `fig:rq1a_gt_correctness_heatmap`, `fig:rq1a_gt_citation_heatmap`, `fig:rq1a_corr_correctness_heatmap`, `fig:rq1a_corr_citation_heatmap`).

### ROC Curves (`sec:appendix_roc_curves`)
- **Purpose**: Visualize discrimination behavior and AUC for RQ1a experiments.
- **One-sentence takeaway**: Because aggregate scores are discrete (0, 0.5, 1), ROC curves have only a few meaningful operating points and primarily serve to contextualize AUC under a fixed 0.5 threshold.
- **Appendix reference**: Appendix § “ROC Curves” (`sec:appendix_roc_curves`) (figures `fig:rq1a_corr_correctness_roc`, `fig:rq1a_corr_citation_roc`, `fig:rq1a_gt_correctness_roc`, `fig:rq1a_gt_citation_roc`).

### RQ1b Corrector Action Counts
- **Purpose**: Provide full action breakdowns (including None and Error) beyond the condensed main-text tables.
- **One-sentence takeaway**: The tables verify totals and expose non-action/error cases so correction outcomes can be audited end-to-end.
- **Appendix reference**: Appendix § “RQ1b Corrector Action Counts” (no section `\label`; see tables `tab:rq1b_action_counts_synthetic`, `tab:rq1b_action_counts_groundtruth`).

### Supplementary Sensitivity Analysis (`app:sensitivity_analysis`) (from `appendix_sensitivity.tex`)
- **Purpose**: Extend prompt sensitivity analyses for judges and correctors with repeated-measures testing and effect sizes.
- **One-sentence takeaway**: Prompt choice strongly affects correctness judging but less consistently affects citation judging, and simpler corrector prompts can outperform more complex ones on synthetic corruptions.
- **Appendix reference**: Appendix § “Supplementary Sensitivity Analysis” (`app:sensitivity_analysis`) (included via `appendix_sensitivity.tex`).

#### Derivation: First-Order Sobol Index for a Categorical Factor (`app:sobol_eta2_derivation`)
- **Purpose**: Provide the short derivation connecting variance decomposition/ANOVA effect sizes to first-order Sobol sensitivity for a single categorical factor.
- **One-sentence takeaway**: \(\eta^2\) corresponds to first-order Sobol sensitivity \(S_1\) under independent sampling, motivating \(\eta^2_p\) as a prompt-sensitivity analogue in the blocked design.
- **Appendix reference**: Appendix § Supplementary Sensitivity Analysis (`app:sensitivity_analysis`) → “Derivation: First-Order Sobol Index…” (`app:sobol_eta2_derivation`).

#### Judge Prompt Sensitivity (`app:judge_sensitivity`)
- **Purpose**: Quantify judge performance stability across prompt variants using prompt-distance and repeated-measures ANOVA.
- **One-sentence takeaway**: Prompt effects are large for correctness judging (explaining most within-subject variance) and weaker for citation judging (often not surviving family-wise correction).
- **Appendix reference**: Appendix § Supplementary Sensitivity Analysis (`app:sensitivity_analysis`) → “Judge Prompt Sensitivity” (`app:judge_sensitivity`) (figure `fig:sensitivity_analysis_app`, table `tab:prompt_sensitivity_app`).

#### Corrector Prompt Sensitivity (`app:corrector_sensitivity`)
- **Purpose**: Measure how corrector prompt complexity impacts post-correction improvements.
- **One-sentence takeaway**: For synthetic corruptions, simpler prompts improve more than mechanistic prompts, while ground-truth correction effects are smaller/marginal.
- **Appendix reference**: Appendix § Supplementary Sensitivity Analysis (`app:sensitivity_analysis`) → “Corrector Prompt Sensitivity” (`app:corrector_sensitivity`) (figure `fig:corrector_sensitivity_app`, table `tab:corrector_sensitivity_app`).

### Experimental Prompts (`sec:appendix_prompts`)
- **Purpose**: Record all prompts verbatim as used in the final experiments.
- **One-sentence takeaway**: The appendix provides the exact prompting artifacts (judges + corrector) so results can be replicated and ablations can be reinterpreted precisely.
- **Appendix reference**: Appendix § “Experimental Prompts” (`sec:appendix_prompts`).

#### Generator Prompt (Nitai\_C) (`sec:generator_prompt_nitai_c`)
- **Purpose**: Provide the exact generator prompt configuration used for node generation and edge generation in the final experiments.
- **One-sentence takeaway**: The Nitai\_C generator prompt specifies the full `generateNodes` and `generateEdges` instructions used to produce CLD variables and relationships.
- **Appendix reference**: Appendix § Experimental Prompts (`sec:appendix_prompts`) → “Generator Prompt (Nitai\_C)” (`sec:generator_prompt_nitai_c`) (listing `lst:generator_nitai_c_yaml`).

#### Correctness Judge Prompts (`sec:correctness_prompts`)
- **Purpose**: Provide the full correctness-judge prompt variants.
- **One-sentence takeaway**: Correctness judging is fully specified for baseline, CoT, and mechanistic (Bradford Hill) variants.
- **Appendix reference**: Appendix § Experimental Prompts (`sec:appendix_prompts`) → “Correctness Judge Prompts” (`sec:correctness_prompts`) (e.g., listing `lst:correctness_baseline`).

#### Citation Judge Prompts (`sec:citation_prompts`)
- **Purpose**: Provide the full citation-judge prompt variants.
- **One-sentence takeaway**: Citation judging is fully specified for baseline, CoT, mechanistic, and mechanistic-lit (literature-grounded) variants.
- **Appendix reference**: Appendix § Experimental Prompts (`sec:appendix_prompts`) → “Citation Judge Prompts” (`sec:citation_prompts`) (e.g., listing `lst:citation_baseline`).

#### Corrector Prompts (`sec:corrector_prompts`)
- **Purpose**: Provide the full corrector prompt variants and tool-selection framing.
- **One-sentence takeaway**: The corrector is fully specified as a tool-using agent with baseline, CoT, and mechanistic variants that govern how it chooses between revising motivations and changing edge types.
- **Appendix reference**: Appendix § Experimental Prompts (`sec:appendix_prompts`) → “Corrector Prompts” (`sec:corrector_prompts`) (e.g., listing `lst:corrector_baseline`).



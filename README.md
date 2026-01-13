# CLD Hallucination Detection - Thesis Reproduction

[![WORCS-inspired](https://img.shields.io/badge/WORCS-inspired-orange)](https://doi.org/10.3233/DS-210031)
[![Analysis reproducible](https://img.shields.io/badge/analysis-reproducible-brightgreen)](#reproduction-in-4-steps)
[![Simulation data available](https://img.shields.io/badge/simulation%20data-available-brightgreen)](#experimental-data-locations)
[![Simulations not reproducible](https://img.shields.io/badge/simulations-not%20reproducible-red)](https://github.com/CausalixAI/platform)

## Computational reproducibility package for the MSc thesis:
### *"Detecting and Mitigating Hallucinations in LLM-Generated Causal Loop Diagrams"*
**by Nitai Nijholt**

<details>
<summary><b>Abstract</b></summary>

Causal loop diagrams (CLDs) can map cause and effect in complex systems but are costly for experts to produce through usual tools such as qualitative interview, literature study and group model building, as edge deliberation cost scales with number of variables squared. LLMs offer a solution by generating initial CLDs, reducing costs, but suffer from hallucinations. LLM-powered text-to-CLD extraction provides grounding but is constrained by source text availability. The alternative, LLM-based CLD generation, loses grounding, increasing the risk of hallucinations.

This thesis tackles this problem by applying established hallucination mitigation methods at test time—through LLM-as-a-judge, LLM-as-a-corrector, and UQ metrics (cosine similarity and logprob-derived measures)—previously untested in CLD generation. We ask: (1) Can LLM-as-a-judge and LLM-as-a-corrector improve CLD F1 versus literature ground truth? (2) Can UQ metrics detect hallucinations? (3) Can adaptive test-time compute, guided by UQ metrics, improve the cost–accuracy trade-off?

Findings show, in our dataset of three health CLDs, LLM-based CLD generation performance exceeds a random baseline. In a controlled synthetic hallucination setting (GT Synth), Correctness Judge performance is high, but it drops in the literature ground-truth setting (GT Lit), with only our causal mechanistic prompt beating random baselines. The Citation Judge, which retrieves supporting articles post hoc, does not reliably exceed chance-level discrimination across domains despite substantial agreement with a human rater (80.6%; κ = 0.618), and high performance on causally unambiguous physics CLDs, suggesting weak discrimination is driven primarily by evidence type and retrieval quality rather than judge malfunction. LLM-as-a-corrector improves judge scores and slightly improves CLD quality vs GT Synth, but not vs GT Lit. UQ metrics (perplexity, min probability, max window entropy, cosine similarity) do not effectively flag hallucinations independently but do as ensemble input to supervised classifiers in-distribution; however, fail to generalise out-of-CLD.

As UQ hallucination flagging and Corrector were ineffective, we pilot a multi-agent deep research (DR) system that finds causal evidence for 62.4% of false positives and 42.9% of false negatives. Preliminary human validation suggests for 66% of these edges causal evidence is applicable; however, limited inter-rater agreement (n=13) means lowerbound extrapolated F1 gains (0.267 → 0.472) should be interpreted as exploratory. Using a mechanistic Correctness Judge to flag hallucinations for adaptive deployment of DR yields an estimated 15% accuracy-per-dollar increase. Results suggest that LLM-based CLD generation and multi-agent validation-driven research can assist experts in CLD creation; future work includes validating the system across additional CLD domains and conducting component-wise ablation studies of DR to find more efficient cost–accuracy trade-offs.

</details>

---

## Quick Start

### At a glance

- **Reproducible here**: All **analysis outputs** (figures + tables) regenerated from experimental data in `final_runs/`
- **Not reproducible here**: Full simulation pipeline lives in private repo: [CausalixAI/platform](https://github.com/CausalixAI/platform)
- **Pre-registration**: [Thesis_proposal.pdf](Thesis_proposal.pdf)

---

## Reproduction Workflow

### Option A: Local Installation

#### Requirements

| Component | Version | Installation |
|-----------|---------|--------------|
| OS | Ubuntu 24.04 / WSL2 / macOS | — |
| Python | 3.10+ | `sudo apt install python3 python3-pip` |
| uv | 0.8+ | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| LaTeX | TeX Live 2023+ | `sudo apt install texlive-full` |

<details>
<summary><b>Tested environment</b></summary>

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 LTS on WSL2 |
| WSL | 6.6.87.2-microsoft-standard-WSL2 |
| Python | 3.12.3 |
| uv | 0.8.3 |
| TeX Live | 2023 |

</details>

### Reproduction in 4 Steps

#### Step 1: Clone & Compile BEFORE Reproduction

```bash
git clone https://github.com/NitaiNijholt/cld-hallucination-detection.git
cd cld-hallucination-detection/thesis/reproducible_version
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Result**: PDF showing:
- Static figures (methodology diagrams, intro images) load normally
- Generated figures show `[NOT YET GENERATED]` placeholder boxes
- Generated tables show `[Table: run reproduce_all_thesis_assets.py]` placeholders

---

#### Step 2: Run Reproduction (~10 minutes)

```bash
cd ../..  # Back to repo root
uv sync
uv run python final_runs/reproduce_all_thesis_assets.py
```

**Result**: All figures and tables freshly generated from experimental data.

---

#### Step 3: Compile AFTER Reproduction

```bash
cd thesis/reproducible_version
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Result**: Full thesis PDF (~271 pages) with all generated figures and tables visible.

---

#### Step 4: Verify Against Submission

Compare the reproduced thesis (`thesis/reproducible_version/main.pdf`) against the submission draft.

**Expected**: All statistical results, figures, and tables should be identical.

---

### Option B: Docker (Full Environment Isolation)

Use Docker if you want complete environment isolation or encounter dependency issues with local installation.

#### Requirements

| Component | Installation |
|-----------|--------------|
| Git | `sudo apt install git` |
| Docker | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) |

#### Build and Run

```bash
git clone https://github.com/NitaiNijholt/cld-hallucination-detection.git
cd cld-hallucination-detection

# Build and run (~15-20 min first time, includes TeX Live)
docker build -t thesis-reproduction .
mkdir -p output
docker run -v $(pwd)/output:/output thesis-reproduction
```

**Output** (in local `output/` folder):
- `thesis_BEFORE.pdf` — compiled with placeholders (before reproduction)
- `thesis_AFTER.pdf` — compiled with generated figures/tables (after reproduction)

---

## What Gets Reproduced

### Figures (generated from experimental data)

| Figure | Label | Description |
|--------|-------|-------------|
| 7.1 | `fig:random_baseline_final` | LLM generator vs random baseline edge F1 |
| 7.2–7.5 | `fig:rq1a_corr_correct_*` | Corruption Correctness Judge performance (4 panels) |
| 7.6–7.9 | `fig:rq1a_gt_correctness_*` | Ground Truth Correctness Judge performance (4 panels) |
| 7.10–7.13 | `fig:rq1a_corr_citation_*` | Corruption Citation Judge performance (4 panels) |
| 7.14–7.17 | `fig:rq1a_gt_citation_*` | Ground Truth Citation Judge performance (4 panels) |
| 7.18 | `fig:rq2_distributions_by_cld` | UQ metric distributions with effect sizes |
| 7.19 | `fig:rq2_rfe` | RFE feature importance for hallucination detection |
| 7.20 | `fig:rq3_combined` | Deep Research validation rates |
| 7.21 | `fig:rq3_human_validation_elbow` | Threshold selection elbow curve |
| 7.22 | `fig:rq3_dr_cost_efficiency` | DR cost-efficiency analysis |
| 7.23 | `fig:gen_vs_judge` | Generation vs judging time comparison |
| 7.24 | `fig:parallelization` | Parallelization benchmark results |
| D.1–D.2 | Appendix | Prompt sensitivity figures (judge + corrector) |

> **Note:** Reproducing cosine similarity distances in prompt sensitivity figures (D.1–D.2) may require custom configuration of the scripts in `final_runs/supp_prompt_sensitivity/analysis_scripts/`, which if not done will have the column show all 0's.

### Tables (generated from experimental data)

| Table | Label | Description |
|-------|-------|-------------|
| 7.1 | `tab:generator_model_comparison` | Generator model comparison (GPT-4.1, Sonar-Pro, Claude) |
| 7.2 | `tab:random_baseline` | Random baseline for generator edge recovery |
| 7.3 | `tab:truthqa_verification` | TruthfulQA Judge verification (n=1,634) |
| 7.4 | `tab:human_validation` | LLM Judge vs Human inter-rater agreement |
| 7.5 | `tab:human_validation_confusion` | Confusion matrix: LLM vs Human (n=72) |
| 7.6 | `tab:physics-comparison-enhanced` | Physics CLD validation results |
| 7.7 | `tab:retrieval-comparison` | Retrieval success comparison |
| 7.8 | `tab:dr_cost` | Deep Research computational cost by CLD |
| 7.9 | `tab:rq3_human_validation_tradeoff` | DR enrichment validation table |
| 7.10 | `tab:parallelization` | Parallelization benchmark summary |
| 7.11 | `tab:scaling_summary` | Computational efficiency summary |
| D.1–D.2 | Appendix | Prompt sensitivity tables (judge + corrector) |

---

## How It Works

1. **Static vs. Generated**: Static figures (intro, methodology) use `\includegraphics` and always display. Generated result figures use a custom `\genfig` command that shows a placeholder if the file doesn't exist.
2. `thesis/reproducible_version/Figures/final_runs/` contains placeholder `.tex` files for tables
3. `thesis/reproducible_version/generated/` contains placeholder LaTeX macros
4. Reproduction scripts analyze raw data in `final_runs/` and generate fresh outputs
5. `create_thesis_figure_structure.py` copies outputs to thesis directories
6. Recompiling after reproduction loads the freshly generated figures/tables

### Dependency Management

Following WORCS principles, we use multiple layers of dependency management to ensure computational reproducibility:

| Layer | Tool | Purpose |
|-------|------|---------|
| **Python packages** | [uv](https://github.com/astral-sh/uv) + `pyproject.toml` | Fast, deterministic package resolution with `uv.lock` |
| **Full environment** | Docker | Complete containerization including TeX Live |
| **LaTeX** | TeX Live 2023+ | Document compilation |

```bash
# Option 1: uv (recommended for development)
uv sync                    # Install exact versions from uv.lock
uv run python script.py    # Run with managed environment

# Option 2: Docker (recommended for verification)
docker build -t thesis .   # Build complete environment
docker run thesis          # Reproduce everything
```

The `uv.lock` file pins exact package versions, ensuring that `uv sync` installs identical dependencies across machines. This is analogous to R's `renv` approach recommended by WORCS, but for Python.

---

## Repository Structure

```
cld-hallucination-detection/
├── final_runs/                                    # ALL REPRODUCTION LOGIC + DATA
│   │
│   │── reproduce_all_thesis_assets.py             # ⭐ MASTER ENTRY POINT
│   │── create_thesis_figure_structure.py          #    Copies outputs to thesis
│   │
│   │── RQ1_unified_analysis.py                    # ─┐
│   │── RQ2_unified_analysis.py                    #  │ Unified analysis
│   │── RQ3_unified_analysis.py                    #  │ entry points
│   │── SUPP_unified_analysis.py                   #  │ (called by master)
│   │── PRELIM_unified_analysis.py                 #  │
│   │── VALIDATION_unified_analysis.py             # ─┘
│   │
│   │── analysis_lib/                              # Shared analysis modules
│   │   ├── analyze_rq1a_*.py                      #   RQ1a analyzers
│   │   ├── analyze_temperature_sensitivity.py    #   Temperature analysis
│   │   ├── logit_metrics.py                       #   UQ metric calculations
│   │   └── data/                                  #   Shared input data (121 files)
│   │
│   │── prompts/                                   # LLM prompts used in experiments
│   │   ├── PRELIM_generator/                      #   Generator prompts
│   │   ├── RQ1a_judge/                            #   Judge prompts (baseline, CoT, mechanistic)
│   │   └── RQ1b_corrector/                        #   Corrector prompts
│   │
│   │── RQ1a_gt_synth_correctness/                 # ─┐ RQ1a: Judge experiments
│   │   ├── {depressive,emergency_department,social_norms}/
│   │   │   └── run_{1,2,3}/                       #   │ Raw experimental data
│   │   │       ├── *.json                         #   │   (LLM outputs)
│   │   │       └── judged_*.xlsx                  #   │   (judge results)
│   │   └── enhanced_analysis_latest/              #   │ Generated figures
│   │── RQ1a_gt_synth_citation/                    #   │
│   │── RQ1a_gt_lit_correctness/                   #   │
│   │── RQ1a_gt_lit_citation/                      # ─┘
│   │
│   │── RQ1b_corrector_ablation/                   # RQ1b: Corrector experiments
│   │   ├── Data/                                  #   Raw data (555 files)
│   │   └── analysis_scripts/                      #   Analysis scripts
│   │
│   │── RQ2_uq_hallucination_detection/            # RQ2: UQ metrics
│   │   └── analysis_scripts/                      #   14 analysis scripts
│   │       ├── rq2_paths.py                       #     (shared paths module)
│   │       ├── rq2_data_preparation.py            #     (data loading module)
│   │       └── rq2_*.py                           #     (analysis scripts)
│   │
│   │── RQ3_deep_research_validation/              # RQ3: Deep Research
│   │   ├── Data/                                  #   DR results (3 files)
│   │   ├── Output/                                #   Generated figures
│   │   ├── validation/                            #   Human validation data
│   │   └── analysis_scripts/                      #   12 analysis scripts
│   │
│   │── prelim_*/                                  # Preliminary experiments
│   │   ├── Output/                                #   Generated outputs
│   │   └── analysis_scripts/
│   │
│   │── supp_*/                                    # Supplementary analyses
│   │   ├── Data/                                  #   Raw timing/cost data
│   │   ├── Output/                                #   Generated figures
│   │   └── analysis_scripts/
│   │
│   └── validation_*/                              # Validation experiments
│       ├── Data/                                  #   Validation data
│       └── analysis_scripts/
│
├── placeholders/                                  # PLACEHOLDER CONTENT
│   ├── placeholder_figure.png                     #   Placeholder image for figures
│   └── placeholder_table.tex                      #   Placeholder LaTeX for tables
│
├── thesis/
│   ├── reproducible_version/                      # Clean version for verification
│   │   ├── main.tex                               #   Uses \genfig for generated figures
│   │   ├── Chapters/                              #   Thesis chapters
│   │   ├── Figures/final_runs/                    #   ← Where outputs are copied
│   │   └── generated/                             #   Placeholder LaTeX macros
│
├── output/                                        # Docker output directory (gitignored)
│   ├── thesis_BEFORE.pdf                          #   Thesis compiled before reproduction
│   └── thesis_AFTER.pdf                           #   Thesis compiled after reproduction
│
├── Dockerfile                                     # Container for full reproduction
├── reset_to_placeholders.py                       # Reset generated files to placeholders
├── pyproject.toml                                 # Python dependencies
├── uv.lock                                        # Locked versions
├── NijholtN_MSc_Thesis_CLD_Hallucination_Detection_2026.pdf  # Thesis PDF
├── Thesis_proposal.pdf                            # Pre-registration document
├── CITATION.cff                                   # Citation metadata
├── LICENSE                                        # MIT License
└── README.md
```

### Analysis Scripts

Scripts in `final_runs/`:

| Script | Purpose |
|--------|---------|
| `reproduce_all_thesis_assets.py` | ⭐ **Master entry point** — runs all analyses |
| `create_thesis_figure_structure.py` | Copies outputs to thesis directories |
| `RQ1_unified_analysis.py` | RQ1a/b: LLM-as-a-Judge + Corrector |
| `RQ2_unified_analysis.py` | RQ2: UQ hallucination detection |
| `RQ3_unified_analysis.py` | RQ3: Deep Research validation |
| `SUPP_unified_analysis.py` | Supplementary: time/cost, parallelization, sensitivity |
| `PRELIM_unified_analysis.py` | Preliminary: random baseline, temperature, model comparison |
| `VALIDATION_unified_analysis.py` | Validation: TruthfulQA, human agreement, physics CLDs |

Scripts in repo root:

| Script | Purpose |
|--------|---------|
| `reset_to_placeholders.py` | Reset generated figures/tables to placeholders |

Shared modules in `final_runs/analysis_lib/`:
- `analyze_rq1a_*.py` — RQ1a analyzers
- `logit_metrics.py` — UQ metric calculations
- `data/` — Ground truth CLDs, shared input data

### Experimental Data Locations

All raw experimental data lives in `final_runs/`. Here's exactly where:

| Location | Files | Contents |
|----------|-------|----------|
| **RQ1a: Judge Experiments** |||
| `RQ1a_gt_synth_correctness/{cld}/run_{1,2,3}/` | 152 | LLM outputs (`.json`) + judge results (`judged_*.xlsx`) |
| `RQ1a_gt_synth_citation/{cld}/run_{1,2,3}/` | 193 | Citation judge experimental runs |
| `RQ1a_gt_lit_correctness/{cld}/run_{1,2,3}/` | 90 | Ground-truth correctness runs |
| `RQ1a_gt_lit_citation/{cld}/run_{1,2,3}/` | 130 | Ground-truth citation runs |
| **RQ1b: Corrector Experiments** |||
| `RQ1b_corrector_ablation/Data/` | 555 | Corrector ablation results (per-prompt subdirs) |
| **RQ2: UQ Metrics** |||
| `RQ2_uq_hallucination_detection/*.json` | 5 | Metric results, permutation importance |
| **RQ3: Deep Research** |||
| `RQ3_deep_research_validation/Data/` | 3 | DR results per CLD (`deep_research_results_*.json`) |
| `RQ3_deep_research_validation/validation/` | 12 | Human validation data + threshold analysis |
| **Supplementary** |||
| `supp_time_cost_scaling/Data/` | 26 | Timing measurements |
| `supp_parallelization_benchmark/Data/` | 2 | Parallelization benchmarks |
| **Validation** |||
| `validation_RQ1_truthfulqa_judge/Data/` | 3 | TruthfulQA verification data |
| **Shared** |||
| `analysis_lib/data/ground_truth_clds/` | 3 | Expert-validated ground truth CLDs (YAML) |
| `analysis_lib/data/temperature_sensitivity_*/` | 118 | Temperature sensitivity raw results |
| `prompts/` | 9 | LLM prompts used in experiments (YAML) |

**Note**: `{cld}` = `depressive`, `emergency_department`, or `social_norms`

### Generated Outputs (Regenerated by Scripts)

| Location | Contents |
|----------|----------|
| `*/Output/` | Generated figures and tables |
| `*/enhanced_analysis_latest/` | RQ1a analysis figures |
| `thesis/reproducible_version/Figures/final_runs/` | Copied outputs for LaTeX |

---

## WORCS-inspired practices

This repository draws inspiration from the [Workflow for Open Reproducible Code in Science (WORCS)](https://doi.org/10.3233/DS-210031) (Van Lissa et al., 2021). This is **not** a claim of full WORCS compliance—only an indication of which principles we strive adhere to.

### TOP Guidelines Addressed

WORCS operationalizes open science via the [TOP Guidelines](https://www.cos.io/initiatives/top-guidelines) (Nosek et al., 2015):

| TOP Guideline | Status | Implementation |
|---------------|--------|----------------|
| 1. **Citation** | ✓ | `CITATION.cff` and BibTeX provided |
| 2. **Data sharing** | ✓ | All experimental data in `final_runs/` |
| 3. **Code sharing** | Partial | Analysis scripts shared; platform code private |
| 4. **Materials sharing** | ✓ | Prompts, ground truth CLDs, experimental configs |
| 5. **Design & analysis** | ✓ | Full methodology in thesis + reproducible scripts |
| 6. **Pre-registration** | Partial | `Thesis_proposal.pdf` (not formal pre-registration) |
| 7. **Analysis plan** | Partial | Documented in thesis; scripts serve as executable plan |

### FAIR Principles

| Principle | Implementation |
|-----------|----------------|
| **Findable** | GitHub-indexed, searchable via GitHub search |
| **Accessible** | Public repository, MIT license, Docker support |
| **Interoperable** | Plain-text (`.json`, `.yaml`, `.tex`) and `.xlsx` for experimental data |
| **Reusable** | Documentation, permissive license, reproducible scripts |

### Transparency Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Citation** | ✓ | `CITATION.cff` and BibTeX provided |
| **Data sharing** | ✓ | Simulation data shared; raw platform logs not included |
| **Code sharing** | Partial | Analysis scripts shared; platform code is private ([CausalixAI/platform](https://github.com/CausalixAI/platform)) |
| **Materials** | Partial | Prompts and ground truth CLDs included; full platform materials private |
| **Methods** | ✓ | Full methodology documented in thesis |
| **Pre-registration** | Partial | `Thesis_proposal.pdf` included (not formal pre-registration) |

---

## License

MIT License

---

## Citation

```bibtex
@mastersthesis{nijholt2025hallucination,
  author = {Nijholt, Nitai},
  title = {Detecting and Mitigating Hallucinations in LLM-Generated Causal Loop Diagrams},
  school = {University of Amsterdam},
  year = {2026}
}
```

### WORCS Framework

This repository follows practices from:

> Van Lissa, C. J., Brandmaier, A. M., Brinkman, L., Lamprecht, A., Peikert, A., Struiksma, M. E., & Vreede, B. (2021). WORCS: A Workflow for Open Reproducible Code in Science. *Data Science*, 4(1), 29-49. DOI: [10.3233/DS-210031](https://doi.org/10.3233/DS-210031)

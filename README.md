# CLD Hallucination Detection - Thesis Reproduction Package

**Computational Reproducibility Package** for the MSc thesis:  
*"Hallucination Detection for LLM-Generated Causal Loop Diagrams"*

> **Scope**: This repository reproduces all figures and tables in the Results chapter from included experimental data. Simulation/data-generation requires access to the private platform.

---

## Quick Start (3 Steps)

```bash
# 1. Clone repository
git clone https://github.com/NitaiNijholt/cld-hallucination-detection.git
cd cld-hallucination-detection

# 2. Run reproduction (~10 minutes)
uv sync
uv run python final_runs/reproduce_all_thesis_assets.py

# 3. Compile thesis with generated figures
cd thesis/reproducible_version
sed -i 's/oneside, draft/oneside/' main.tex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Before reproduction**: Thesis compiles with placeholder boxes and "[Table: run reproduction]" text  
**After reproduction**: Full thesis with all 30 generated figures and tables

---

## Research Questions

| RQ | Question | Key Finding |
|----|----------|-------------|
| **RQ1** | How accurately can LLMs judge the correctness of CLD edges? | GPT-4o-mini achieves F1 > 0.85 for correctness, > 0.75 for citation verification |
| **RQ2** | Can generator logit probabilities predict hallucination likelihood? | Ensemble of entropy + mean logprob achieves AUC = 0.72 for hallucination detection |
| **RQ3** | Does deep research improve CLD edge accuracy? | 47% hallucination detection rate with human-validated precision 0.70-0.76 |

---

## What This Reproduces

| Category | Description | Assets |
|----------|-------------|--------|
| **RQ1** | LLM-as-a-Judge Performance | 12 figures |
| **RQ2** | UQ Hallucination Detection | 5 figures/tables |
| **RQ3** | Deep Research Validation | 8 figures/tables |
| **SUPP** | Supplementary Analyses | 5 figures |
| **Total** | All Results-chapter assets | **30 assets** |

---

## Repository Structure

```
cld-hallucination-detection/
├── final_runs/                         # Experimental data + analysis scripts
│   ├── reproduce_all_thesis_assets.py  # Master reproduction script
│   ├── RQ1_unified_analysis.py         # RQ1 analysis pipeline
│   ├── RQ2_unified_analysis.py         # RQ2 analysis pipeline
│   ├── RQ3_unified_analysis.py         # RQ3 analysis pipeline
│   ├── SUPP_unified_analysis.py        # Supplementary analyses
│   ├── PRELIM_unified_analysis.py      # Preliminary analyses
│   ├── create_thesis_figure_structure.py
│   ├── analysis_lib/                   # Self-contained analysis code
│   │   ├── analyze_rq1a_*.py
│   │   ├── logit_metrics.py
│   │   └── data/                       # Ground truth CLDs
│   ├── RQ1a_*/                         # RQ1 experimental data (444 xlsx)
│   ├── RQ2_uq_hallucination_detection/ # RQ2 data
│   ├── RQ3_deep_research_validation/   # RQ3 data
│   └── supp_*/                         # Supplementary data
├── thesis/
│   └── reproducible_version/           # Thesis LaTeX source
├── data_science/                       # Analysis utilities
├── pyproject.toml                      # Python dependencies
└── uv.lock                             # Locked dependency versions
```

---

## Requirements

| Requirement | Installation |
|-------------|--------------|
| **Python 3.10+** | System package manager |
| **uv** package manager | `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **LaTeX** (TeX Live) | `sudo apt install texlive-full` (Ubuntu) |

---

## Unified Analysis Entry Points

| Script | Description | Outputs |
|--------|-------------|---------|
| `RQ1_unified_analysis.py` | Judge capability verification | 12 PNG figures |
| `RQ2_unified_analysis.py` | UQ hallucination detection | 5 figures + 3 tables |
| `RQ3_unified_analysis.py` | Deep research validation | 4 figures + 4 tables |
| `SUPP_unified_analysis.py` | Time/cost, parallelization, sensitivity | 5 figures |
| `PRELIM_unified_analysis.py` | Random baseline, temperature sensitivity | 3 figures + 2 tables |

---

## Data Availability

| Data Type | Location | Records |
|-----------|----------|---------|
| RQ1a Judge Results | `final_runs/RQ1a_*/` | 444 xlsx files |
| RQ2 UQ Metrics | `final_runs/RQ2_uq_hallucination_detection/` | 63 xlsx files |
| RQ3 Deep Research | `final_runs/RQ3_deep_research_validation/` | 285 edges |
| Ground Truth CLDs | `final_runs/analysis_lib/data/` | 3 JSON files |

---

## Verification

After running reproduction:

```bash
# Check reproduction report
cat /tmp/thesis_reproducibility_*/reproducibility_report.json

# Expected output:
# "total_coverage": 100.0
# "all_reproduced": true
```

---

## License

MIT License - See LICENSE file

## Citation

```bibtex
@mastersthesis{nijholt2026hallucination,
  author = {Nijholt, Nitai},
  title = {Hallucination Detection for LLM-Generated Causal Loop Diagrams},
  school = {University of Amsterdam},
  year = {2026},
  type = {MSc Thesis}
}
```

## Related Resources

- **Thesis Proposal**: `docs/Thesis_proposal.pdf` (design document)
- **Private Platform**: Access upon request for full simulation capabilities

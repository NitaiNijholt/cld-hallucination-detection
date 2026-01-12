# CLD Hallucination Detection - Thesis Reproduction

Computational reproducibility package for the MSc thesis:  
*"Hallucination Detection for LLM-Generated Causal Loop Diagrams"*

---

## Quick Start (4 Steps)

### Step 1: Clone and Setup

```bash
git clone https://github.com/NitaiNijholt/cld-hallucination-detection.git
cd cld-hallucination-detection
uv sync
```

---

### Step 2: Compile BEFORE Reproduction (verify empty)

```bash
cd thesis/reproducible_thesis
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Result**: PDF compiles but figures show placeholder boxes or missing file warnings.  
This proves figures are not pre-baked into the package.

---

### Step 3: Run Reproduction (~10 minutes)

```bash
cd ../..
uv run python final_runs/reproduce_all_thesis_assets.py
```

**Result**: All 30 figures and tables generated from experimental data.

---

### Step 4: Compile AFTER Reproduction (verify complete)

```bash
cd thesis/reproducible_thesis
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Result**: Full thesis PDF with all reproduced figures and tables.

---

## What Gets Reproduced

| Category | Assets |
|----------|--------|
| RQ1 - LLM-as-a-Judge | 12 figures |
| RQ2 - UQ Detection | 5 figures/tables |
| RQ3 - Deep Research | 8 figures/tables |
| Supplementary | 5 figures |
| **Total** | **30 assets** |

---

## Requirements

| Requirement | Installation |
|-------------|--------------|
| Python 3.10+ | System package manager |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| LaTeX | `sudo apt install texlive-full` (Ubuntu) |

---

## Repository Structure

```
cld-hallucination-detection/
├── final_runs/                     # Experimental data + analysis scripts
│   ├── reproduce_all_thesis_assets.py   # Master reproduction script
│   ├── RQ1_unified_analysis.py     # RQ1 analysis
│   ├── RQ2_unified_analysis.py     # RQ2 analysis
│   ├── RQ3_unified_analysis.py     # RQ3 analysis
│   ├── RQ1a_gt_*/                  # RQ1 experimental data
│   ├── RQ2_uq_*/                   # RQ2 experimental data
│   └── RQ3_deep_research_*/        # RQ3 experimental data
│
└── thesis/
    ├── reproducible_thesis/        # Clean thesis (53 files, 4MB)
    │   ├── main.tex                # Main document
    │   ├── Chapters/               # All chapters
    │   ├── Appendices/             # Appendices
    │   └── Figures/                # Static figures only
    └── final_runs -> ../final_runs # Symlink to data
```

---

## Verification

After Step 3, check the reproduction report:

```bash
cat /tmp/thesis_reproducibility_*/reproducibility_report.json | grep total_coverage
# Expected: "total_coverage": 100.0
```

---

## License

MIT License

# CLD Hallucination Detection - Thesis Reproduction

Computational reproducibility package for the MSc thesis:  
*"Hallucination Detection for LLM-Generated Causal Loop Diagrams"*

---

## Quick Start (3 Steps)

### Step 1: Clone and Compile BEFORE Reproduction

```bash
git clone https://github.com/NitaiNijholt/cld-hallucination-detection.git
cd cld-hallucination-detection/thesis/reproducible_version
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Result**: PDF compiles with placeholder boxes for figures and "[Table: run reproduction]" text.

---

### Step 2: Run Reproduction (~10 minutes)

```bash
cd ../..
uv sync
uv run python final_runs/reproduce_all_thesis_assets.py
```

**Result**: All 30 figures and tables generated from experimental data.

---

### Step 3: Remove Draft Mode and Recompile

```bash
cd thesis/reproducible_version
sed -i 's/oneside, draft/oneside/' main.tex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Result**: Full thesis PDF with all generated figures and tables.

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

## Verification

After Step 2, check the reproduction report:

```bash
cat /tmp/thesis_reproducibility_*/reproducibility_report.json | grep total_coverage
# Expected: "total_coverage": 100.0
```

---

## License

MIT License


# Reproducible Thesis Version

This is a **fully isolated reproducible version** of the thesis. The `Figures/final_runs/` directory is intentionally empty - all figures and tables referenced can be regenerated from the raw experimental data.

**Key difference from submission:** This version uses `\graphicspath` that only searches local directories, ensuring no figures are accidentally found from parent paths.

## Quick Start (Examiner Workflow)

```bash
# From the project root (/home/nitai/code/causalix.ai)
cd /home/nitai/code/causalix.ai

# 1. Reproduce all analyses from raw data (~5 minutes)
uv run python final_runs/reproduce_all_thesis_assets.py --output-dir /tmp/thesis_repro

# 2. Populate this thesis with reproduced figures and tables
uv run python final_runs/create_thesis_figure_structure.py \
    --source /tmp/thesis_repro \
    --target thesis/final_thesis/reproducible_version/Figures/final_runs

# 3. Compile the thesis
cd thesis/final_thesis/reproducible_version
latexmk -pdf main.tex

# 4. Verify page count matches submission (should be 256 pages)
pdfinfo main.pdf | grep Pages
```

## What Gets Reproduced

| Category | Assets | Description |
|----------|--------|-------------|
| **RQ1** | 12+ figures | LLM-as-a-Judge performance metrics, ROC curves, heatmaps |
| **RQ2** | 4 figures, 3 tables | RFE feature selection, ensemble performance, UQ metrics |
| **RQ3** | 3 figures, 3 tables | Deep Research combined figure, threshold tradeoff, cost efficiency |
| **SUPP** | 5 figures, 4 tables | Parallelization, time/cost scaling, prompt sensitivity |

Total: ~33 reproducible assets + ~17 static tables/figures

## Verification

After reproduction, compare with the original submission:

```bash
# Page count should match
pdfinfo thesis/final_thesis/def_submission_template/main.pdf | grep Pages  # 256
pdfinfo thesis/final_thesis/reproducible_version/main.pdf | grep Pages     # 256

# Figure count
find thesis/final_thesis/reproducible_version/Figures/final_runs -name "*.png" | wc -l
```

## Directory Structure

```
reproducible_version/
├── main.tex                    # Main thesis (isolated graphicspath)
├── REPRODUCIBILITY_README.md   # This file
├── Chapters/                   # Chapter .tex files (uses enhanced_analysis_latest)
├── Appendices/                 # Appendix .tex files
├── background_llms.tex         # Background content (copied locally)
├── background_prompts.tex      
├── final_results.tex           # Results content (uses local paths)
├── Figures/
│   ├── final_runs/            # EMPTY → populated by reproduction scripts
│   ├── *.png/pdf              # Static assets (pre-populated)
│   └── background_*/          # Background diagrams (pre-populated)
└── References.bib
```

## Key Modifications

1. **Isolated graphicspath**: Only searches `./`, `Pictures/`, `Figures/`, `figures/`
2. **Local content files**: `background_llms.tex`, `background_prompts.tex`, `final_results.tex` copied locally
3. **Consistent paths**: All figure paths use `enhanced_analysis_latest` instead of timestamps
4. **No parent references**: All `../` references replaced with local paths

## Notes

- The reproduction scripts require the `uv` package manager and project dependencies
- Static/preliminary figures (random baseline, etc.) are copied from original `final_runs/`
- Compilation requires a full LaTeX distribution with `latexmk`

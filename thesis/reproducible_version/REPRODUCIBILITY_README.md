# Reproducible Thesis Version

This is a **fully isolated reproducible version** of the thesis. The `Figures/final_runs/` directory is intentionally empty - all figures and tables referenced can be regenerated from the raw experimental data.

## Quick Start (Examiner Workflow)

```bash
# From the cloned repository root
cd /path/to/platform

# 1. Install dependencies
uv sync

# 2. Reproduce all analyses from raw data (~10 minutes)
uv run python final_runs/reproduce_all_thesis_assets.py

# 3. Populate thesis with reproduced figures and tables
uv run python final_runs/create_thesis_figure_structure.py \
    --source /tmp/thesis_latest \
    --target thesis/reproducible_version/Figures/final_runs

# 4. Compile the thesis
cd thesis/reproducible_version
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# 5. View the compiled thesis
xdg-open main.pdf  # or open main.pdf on macOS
```

## Expected Results

### BEFORE Running Reproduction

If you try to compile the thesis without running the reproduction scripts first, you will see errors like:

```
! LaTeX Error: File `Figures/final_runs/RQ1a_gt_synth_correctness/enhanced_analysis_latest/rq1a_row1_performance_metrics.png' not found.
```

This is expected - the `Figures/final_runs/` directory is empty.

### AFTER Running Reproduction

After running steps 2-3 above:
- All figures will be generated and copied to `Figures/final_runs/`
- The thesis will compile successfully
- Generated figures should match those in the submitted PDF

## What Gets Reproduced

| Category | Assets | Description |
|----------|--------|-------------|
| **RQ1** | 12+ figures, 4 tables | LLM-as-a-Judge performance metrics, ROC curves, heatmaps |
| **RQ2** | 2 figures, 3 tables | RFE feature selection, ensemble performance, UQ metrics |
| **RQ3** | 3 figures, 5 tables | Deep Research combined figure, threshold tradeoff, cost efficiency |
| **SUPP** | 5 figures, 5 tables | Parallelization, time/cost scaling, prompt sensitivity |
| **PRELIM** | 3 figures, 3 tables | Random baseline, temperature sensitivity, generator comparison |
| **VALIDATION** | 6 tables | TruthfulQA, human validation, external validation |

**Total: ~38 reproducible assets**

## Directory Structure

```
reproducible_version/
├── main.tex                    # Main thesis document
├── Thesis.cls                  # LaTeX class file
├── References.bib              # Bibliography
├── Chapters/                   # Chapter .tex files
│   ├── Results.tex             # References Figures/final_runs/...
│   └── ...
├── Appendices/                 # Appendix .tex files
├── Figures/
│   ├── final_runs/            # EMPTY before reproduction
│   │   ├── .gitkeep           # Placeholder
│   │   └── [populated after reproduction]
│   ├── *.png/pdf              # Static background figures
│   └── background_*/          # Background diagrams
├── Pictures/                   # University logos, signatures
└── generated/                  # Generated inline tables
```

## Key Modifications from Submission

1. **Isolated graphicspath**: Only searches local directories (`./`, `Pictures/`, `Figures/`, `figures/`)
2. **Normalized paths**: All figure paths use `enhanced_analysis_latest` instead of timestamps
3. **No parent references**: All `../` references replaced with local paths
4. **Empty figures**: `Figures/final_runs/` is empty until reproduction is run

## Verification

After reproduction and compilation, compare with the original submission:

```bash
# Count reproduced figures
find thesis/reproducible_version/Figures/final_runs -name "*.png" | wc -l

# Should produce ~25-30 figures
```

## Troubleshooting

### "File not found" errors during compilation
Run the reproduction scripts (steps 2-3) first.

### Missing dependencies
Run `uv sync` to install all required Python packages.

### Compilation errors after reproduction
Ensure you ran `create_thesis_figure_structure.py` to map outputs to correct paths.

## Notes

- The reproduction scripts require Python 3.10+ and the `uv` package manager
- Total reproduction time: ~10 minutes (depending on hardware)
- Compilation requires a full LaTeX distribution (TeX Live recommended)

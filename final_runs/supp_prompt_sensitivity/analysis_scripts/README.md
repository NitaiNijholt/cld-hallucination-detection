# Sensitivity Analysis Scripts

Scripts used to generate the prompt sensitivity analysis for RQ1a (Judge) and RQ1b (Corrector).

## Files

### Primary (Thesis Tables 74 & 75 — RM ANOVA with partial η²_p)

- `sensitivity_analysis_simple_rm_anova.py` - Judge prompt sensitivity (4 experiments × 3–4 prompts)
- `sensitivity_analysis_corrector_rm_anova.py` - Corrector prompt sensitivity (2 experiments × 3 prompts)

### Alternative (Friedman test with Kendall's W)

- `sensitivity_analysis_simple.py` - Judge prompt sensitivity (non-parametric alternative)
- `sensitivity_analysis_corrector.py` - Corrector prompt sensitivity (non-parametric alternative)

## Usage

```bash
cd /home/nitai/code/causalix.ai
source .venv/bin/activate

# Run judge sensitivity (RM ANOVA — produces Table 74)
python final_runs/Sensitivity_analysis_simple/scripts/sensitivity_analysis_simple_rm_anova.py

# Run corrector sensitivity (RM ANOVA — produces Table 75)
python final_runs/Sensitivity_analysis_simple/scripts/sensitivity_analysis_corrector_rm_anova.py
```

## Outputs

- `prompt_sensitivity_figure.png` - Judge sensitivity visualization
- `prompt_sensitivity_table.tex` - Judge LaTeX table (Table 74)
- `corrector_sensitivity_figure.png` - Corrector sensitivity visualization
- `corrector_sensitivity_table.tex` - Corrector LaTeX table (Table 75)
- `*_data.xlsx` - Raw data files

## Methodology (RM ANOVA — Primary)

1. **Design**: Within-subjects (repeated measures)
   - Subjects: (CLD × run) pairs (n = 9)
   - Within-subject factor: Prompt type

2. **Input**: Prompts embedded with MPNet (judge) or Jina v2 (corrector), cosine distance from baseline

3. **Output**: F1 (judges) or F1 Δ (correctors)

4. **Statistics**:
   - **Prompt effect**: Repeated-measures ANOVA
   - **Effect size**: Partial eta-squared η²_p = SS_effect / (SS_effect + SS_error)
   - **Interpretation**: η²_p < 0.01 negligible, 0.01–0.06 small, 0.06–0.14 medium, ≥0.14 large
   - **Multiple comparisons**: Bonferroni correction (RQ1a: 4 tests α_adj=0.0125; RQ1b: 2 tests α_adj=0.025)









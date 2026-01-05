#!/usr/bin/env python3
import pandas as pd

excel_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/exp_20251007_234410_10d61c06/combo_1/prompts_Nitai_C/Social_norms_and_obesity_prevalence/run1_ddce5c21/results_exp_20251007_234410_10d61c06_param_combo_nr_1_prompts_Nitai_C_run1_Social_norms_and_obesity_prevalence_20251008_001425.xlsx"

# Check what sheets exist
xl = pd.ExcelFile(excel_file)
print("📄 Available sheets:")
for sheet in xl.sheet_names:
    print(f"   - {sheet}")

print()

# Check columns in All Edges
df_all = pd.read_excel(excel_file, sheet_name="All Edges")
print(f"📊 'All Edges' sheet: {len(df_all)} rows, {len(df_all.columns)} columns")
print("Columns:")
for col in df_all.columns:
    print(f"   - {col}")

print()

# Check Session Graph sheet
df_session = pd.read_excel(excel_file, sheet_name="Session Graph")
print(f"📊 'Session Graph' sheet: {len(df_session)} rows, {len(df_session.columns)} columns")
print("Columns (first 20):")
for col in list(df_session.columns)[:20]:
    print(f"   - {col}")

# Check if CI metrics are in Session Graph
ci_cols = [c for c in df_session.columns if 'perplexity' in c.lower() or 'cosine' in c.lower() or 'min_prob' in c.lower()]
if ci_cols:
    print()
    print("🔍 Found CI-related columns in Session Graph:")
    for col in ci_cols:
        print(f"   - {col}")
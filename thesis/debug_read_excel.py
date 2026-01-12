
import pandas as pd
import sys

file_path = "final_runs/RQ1a_gt_lit_citation/depressive/run_1/judged_Depressive_symptoms_in_response_to_a_stressor_cot_20251117_003704_20251117_004626.xlsx"

try:
    df = pd.read_excel(file_path, sheet_name='LLM Usage Stats')
    print(df.to_string())
except Exception as e:
    print(f"Error reading Excel: {e}")

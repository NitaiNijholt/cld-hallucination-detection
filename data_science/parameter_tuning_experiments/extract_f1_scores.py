import pandas as pd
import numpy as np

# Read the Excel file
df = pd.read_excel('/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results_all_prompts_15_05_2025/results_exp_20250520_013716_b11b9d2.xlsx')

results = []
current_prompt = None

for idx, row in df.iterrows():
    # Check if this is a prompt name row
    if pd.notna(row.iloc[0]) and str(row.iloc[0]).startswith('Prompt:'):
        current_prompt = str(row.iloc[0]).replace('Prompt:', '').strip()
    
    # Check if this is the F1 Score row
    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip() == 'F1 Score':
        if current_prompt:
            # Extract F1 scores for all 3 CLDs
            cld1_f1 = row.iloc[1] if pd.notna(row.iloc[1]) else None
            cld2_f1 = row.iloc[3] if pd.notna(row.iloc[3]) else None
            cld3_f1 = row.iloc[5] if pd.notna(row.iloc[5]) else None
            
            # Calculate average
            f1_scores = [x for x in [cld1_f1, cld2_f1, cld3_f1] if x is not None]
            avg_f1 = np.mean(f1_scores) if f1_scores else None
            
            results.append({
                'Prompt': current_prompt,
                'CLD_1_F1': cld1_f1,
                'CLD_2_F1': cld2_f1,
                'CLD_3_F1': cld3_f1,
                'Average_F1': avg_f1
            })

# Create DataFrame and sort by average F1
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Average_F1', ascending=False)

print("F1 Scores by Prompt Variant:")
print("=" * 80)
print(results_df.to_string(index=False))
print("\n" + "=" * 80)
print(f"\nBest performing prompt: {results_df.iloc[0]['Prompt']} (F1 = {results_df.iloc[0]['Average_F1']:.4f})")
print(f"Worst performing prompt: {results_df.iloc[-1]['Prompt']} (F1 = {results_df.iloc[-1]['Average_F1']:.4f})")
print(f"Range: {results_df.iloc[0]['Average_F1'] - results_df.iloc[-1]['Average_F1']:.4f}")

# Save to CSV
results_df.to_csv('/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/f1_scores_summary.csv', index=False)
print("\nResults saved to: f1_scores_summary.csv")

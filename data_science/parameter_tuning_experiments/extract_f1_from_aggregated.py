import pandas as pd
import numpy as np

# Read the aggregated Excel file
df = pd.read_excel('/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results_all_prompts_15_05_2025/results_exp_20250520_013716_b11b9d2.xlsx')

results = []
current_prompt = None
current_f1_scores = {}

for idx, row in df.iterrows():
    # Check if this is a prompt name row
    first_col = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
    
    if first_col.startswith('Prompt:'):
        # New prompt section
        if current_prompt and current_f1_scores:
            # Save previous prompt results
            results.append(current_f1_scores)
        
        current_prompt = first_col.replace('Prompt:', '').strip()
        current_f1_scores = {'Prompt': current_prompt}
    
    # Check if this is the F1 Score row
    elif first_col == 'F1 Score' and current_prompt:
        # Extract F1 scores for all 3 CLDs
        cld1_f1 = float(row.iloc[1]) if pd.notna(row.iloc[1]) else None
        cld2_f1 = float(row.iloc[3]) if pd.notna(row.iloc[3]) else None
        cld3_f1 = float(row.iloc[5]) if pd.notna(row.iloc[5]) else None
        
        current_f1_scores['CLD1_F1'] = cld1_f1
        current_f1_scores['CLD2_F1'] = cld2_f1
        current_f1_scores['CLD3_F1'] = cld3_f1
        
        # Calculate average
        f1_scores = [x for x in [cld1_f1, cld2_f1, cld3_f1] if x is not None]
        current_f1_scores['Average_F1'] = np.mean(f1_scores) if f1_scores else None

# Add last prompt
if current_prompt and current_f1_scores:
    results.append(current_f1_scores)

# Create DataFrame and sort by average F1
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Average_F1', ascending=False)

print("=" * 100)
print("F1 Scores by Prompt Variant (from aggregated file):")
print("=" * 100)
print(results_df.to_string(index=False))
print("\n" + "=" * 100)

if not results_df.empty:
    best = results_df.iloc[0]
    worst = results_df.iloc[-1]
    print(f"\nBest performing prompt: {best['Prompt']} (F1 = {best['Average_F1']:.4f})")
    print(f"Worst performing prompt: {worst['Prompt']} (F1 = {worst['Average_F1']:.4f})")
    print(f"Range: {best['Average_F1'] - worst['Average_F1']:.4f}")
    print(f"Mean F1 across all prompts: {results_df['Average_F1'].mean():.4f}")
    print(f"Std F1 across all prompts: {results_df['Average_F1'].std():.4f}")

# Save to CSV
results_df.to_csv('/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/f1_scores_from_aggregated.csv', index=False)
print("\nResults saved to: f1_scores_from_aggregated.csv")

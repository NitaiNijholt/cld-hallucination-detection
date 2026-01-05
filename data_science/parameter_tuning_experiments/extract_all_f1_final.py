import pandas as pd
import numpy as np
import os

# Read the mapping file
mapping_df = pd.read_csv('/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results_all_prompts_15_05_2025/param_combo_mapping_exp_20250520_013716_b11b9d2d.csv')

results_dir = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results_all_prompts_15_05_2025/'

# Extract unique prompt names and CLDs
prompts = mapping_df['prompt'].unique()
clds = mapping_df['cld_prefix'].unique()

print(f"Found {len(prompts)} prompts: {list(prompts)}")
print(f"Found {len(clds)} CLDs")
print("\nExtracting F1 scores...")

results = []

for prompt in prompts:
    prompt_data = {'Prompt': prompt}
    f1_scores = []
    
    for idx, cld in enumerate(clds, 1):
        # Get the filename for this prompt-CLD combination
        row = mapping_df[(mapping_df['prompt'] == prompt) & (mapping_df['cld_prefix'] == cld)]
        
        if not row.empty:
            filename = row.iloc[0]['excel_filename']
            filepath = os.path.join(results_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    # Read the "All Edges Summary" sheet (index 1)
                    df = pd.read_excel(filepath, sheet_name='All Edges Summary', header=None)
                    
                    # Find the F1 Score row
                    f1_row = df[df.iloc[:, 0] == 'F1 Score']
                    
                    if not f1_row.empty:
                        f1_value = float(f1_row.iloc[0, 1])
                        prompt_data[f'CLD{idx}_F1'] = f1_value
                        f1_scores.append(f1_value)
                        print(f"  {prompt} - CLD{idx}: F1 = {f1_value:.4f}")
                    else:
                        print(f"  {prompt} - CLD{idx}: F1 Score not found")
                        prompt_data[f'CLD{idx}_F1'] = None
                        
                except Exception as e:
                    print(f"  Error reading {filename}: {e}")
                    prompt_data[f'CLD{idx}_F1'] = None
            else:
                print(f"  File not found: {filename}")
                prompt_data[f'CLD{idx}_F1'] = None
        else:
            prompt_data[f'CLD{idx}_F1'] = None
    
    # Calculate average
    if f1_scores:
        prompt_data['Average_F1'] = np.mean(f1_scores)
        print(f"  → Average F1 for {prompt}: {prompt_data['Average_F1']:.4f}\n")
    else:
        prompt_data['Average_F1'] = None
    
    results.append(prompt_data)

# Create DataFrame and sort by average F1
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Average_F1', ascending=False)

# Clean up prompt names for display
results_df['Prompt_Display'] = results_df['Prompt'].str.replace('prompts_', '').str.replace('_formatfixed', '')

print("\n" + "=" * 100)
print("F1 Scores by Prompt Variant (sorted by average F1):")
print("=" * 100)
print(results_df[['Prompt_Display', 'CLD1_F1', 'CLD2_F1', 'CLD3_F1', 'Average_F1']].to_string(index=False))
print("\n" + "=" * 100)

if not results_df.empty and not results_df['Average_F1'].isna().all():
    best = results_df.iloc[0]
    worst = results_df.iloc[-1]
    print(f"\nBest performing: {best['Prompt_Display']} (F1 = {best['Average_F1']:.4f})")
    print(f"Worst performing: {worst['Prompt_Display']} (F1 = {worst['Average_F1']:.4f})")
    print(f"Range: {best['Average_F1'] - worst['Average_F1']:.4f}")
    print(f"Mean F1 across all prompts: {results_df['Average_F1'].mean():.4f}")
    print(f"Std F1 across all prompts: {results_df['Average_F1'].std():.4f}")

# Save to CSV
output_file = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/prompt_f1_scores_complete.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

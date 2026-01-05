import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# F1 scores data
data = {
    'Prompt': ['Nitai_C', 'Nitai_A', 'Nitai_B', 'current', 'Rick_B', 'Nitai_E', 'cillian_B', 'Rick_A', 'cillian_C'],
    'CLD1_F1': [0.5909, 0.5897, 0.5882, 0.5882, 0.4444, 0.4375, 0.5152, 0.3804, 0.0000],
    'CLD2_F1': [0.2042, 0.2176, 0.0918, 0.2390, 0.1857, 0.2075, 0.1524, 0.1455, 0.0000],
    'CLD3_F1': [0.5833, 0.4000, 0.4545, 0.2857, 0.4375, 0.3846, 0.2857, 0.2687, 0.0000],
    'Average_F1': [0.4595, 0.4025, 0.3782, 0.3710, 0.3559, 0.3432, 0.3177, 0.2648, 0.0000]
}

df = pd.DataFrame(data)

# Create figure with appropriate size for thesis
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by Average F1 for better visualization
df = df.sort_values('Average_F1', ascending=True)

# Create bar chart
bars = ax.barh(df['Prompt'], df['Average_F1'], color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.8)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, df['Average_F1'])):
    ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{value:.3f}', 
            va='center', ha='left', fontsize=10, fontweight='bold')

# Styling
ax.set_xlabel('Average F1 Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Prompt Variant', fontsize=12, fontweight='bold')
ax.set_title('Prompt Engineering Ablation Study: F1 Score Performance\nAcross 3 Causal Loop Diagrams', 
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis limits with some padding
ax.set_xlim(0, max(df['Average_F1']) * 1.15)

# Add grid for readability
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add mean line
mean_f1 = df['Average_F1'].mean()
ax.axvline(mean_f1, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean F1: {mean_f1:.3f}')
ax.legend(loc='lower right', fontsize=10)

# Tight layout
plt.tight_layout()

# Save figure
output_path = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ablation_study_f1_barchart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Bar chart saved to: {output_path}")

# Also save as PDF for thesis
pdf_path = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ablation_study_f1_barchart.pdf'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"PDF version saved to: {pdf_path}")

plt.close()

# Create a second figure showing F1 scores for each CLD separately
fig, ax = plt.subplots(figsize=(12, 7))

# Sort by average F1
df_sorted = df.sort_values('Average_F1', ascending=False)

x = np.arange(len(df_sorted))
width = 0.25

# Create grouped bars
bars1 = ax.bar(x - width, df_sorted['CLD1_F1'], width, label='CLD 1: Depressive Symptoms', 
               color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x, df_sorted['CLD2_F1'], width, label='CLD 2: Social Norms & Obesity', 
               color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.8)
bars3 = ax.bar(x + width, df_sorted['CLD3_F1'], width, label='CLD 3: Emergency Dept. Visits', 
               color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.8)

# Styling
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Prompt Variant', fontsize=12, fontweight='bold')
ax.set_title('F1 Score Performance by Prompt Variant and CLD', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df_sorted['Prompt'], rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_ylim(0, max(df_sorted['CLD1_F1'].max(), df_sorted['CLD3_F1'].max()) * 1.1)

plt.tight_layout()

# Save detailed figure
detailed_path = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ablation_study_f1_by_cld.png'
plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
print(f"Detailed chart saved to: {detailed_path}")

detailed_pdf = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ablation_study_f1_by_cld.pdf'
plt.savefig(detailed_pdf, dpi=300, bbox_inches='tight', format='pdf')
print(f"Detailed PDF saved to: {detailed_pdf}")

plt.close()

print("\nVisualization complete!")

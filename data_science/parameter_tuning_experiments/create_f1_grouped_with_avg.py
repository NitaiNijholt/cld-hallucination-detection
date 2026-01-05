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

# Sort by Average F1 in ASCENDING order (low to high)
df = df.sort_values('Average_F1', ascending=True)

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(df))
width = 0.2  # Width of bars

# Create grouped bars - 4 bars per prompt
bars1 = ax.bar(x - 1.5*width, df['CLD1_F1'], width, 
               label='CLD 1: Depressive Symptoms', 
               color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x - 0.5*width, df['CLD2_F1'], width, 
               label='CLD 2: Social Norms & Obesity', 
               color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.8)
bars3 = ax.bar(x + 0.5*width, df['CLD3_F1'], width, 
               label='CLD 3: Emergency Dept. Visits', 
               color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.8)
bars4 = ax.bar(x + 1.5*width, df['Average_F1'], width, 
               label='Average F1 (across 3 CLDs)', 
               color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.2)

# Add value labels on average bars only (to avoid clutter)
for i, (bar, value) in enumerate(zip(bars4, df['Average_F1'])):
    if value > 0:
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, 
                f'{value:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Styling
ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Prompt Variant (ordered by Average F1, low to high)', fontsize=13, fontweight='bold')
ax.set_title('F1 Score Performance: Individual CLDs and Average\nPrompts Ordered by Average F1 Performance', 
             fontsize=14, fontweight='bold', pad=20)

ax.set_xticks(x)
ax.set_xticklabels(df['Prompt'], rotation=45, ha='right', fontsize=11)

# Add legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

# Add grid
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Set y-axis limit
ax.set_ylim(0, max(df['CLD1_F1'].max(), df['CLD3_F1'].max(), df['Average_F1'].max()) * 1.15)

# Add mean line for average F1
mean_f1 = df['Average_F1'].mean()
ax.axhline(mean_f1, color='purple', linestyle='--', linewidth=2, alpha=0.6, 
           label=f'Overall Mean: {mean_f1:.3f}')
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

plt.tight_layout()

# Save figure
output_path = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ablation_study_grouped_with_avg.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Grouped bar chart with average saved to: {output_path}")

# Also save as PDF for thesis
pdf_path = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ablation_study_grouped_with_avg.pdf'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"PDF version saved to: {pdf_path}")

plt.close()

# Print ordered ranking
print("\n" + "=" * 90)
print("PROMPT RANKING (Low to High Average F1) with Individual CLD Scores:")
print("=" * 90)
print(f"{'Rank':<5} {'Prompt':<15} {'CLD1':>8} {'CLD2':>8} {'CLD3':>8} {'Average':>10}")
print("-" * 90)
for i, (idx, row) in enumerate(df.iterrows(), 1):
    print(f"{i:<5} {row['Prompt']:<15} {row['CLD1_F1']:>8.3f} {row['CLD2_F1']:>8.3f} "
          f"{row['CLD3_F1']:>8.3f} {row['Average_F1']:>10.3f}")
print("=" * 90)

print(f"\nPerformance Statistics:")
print(f"  Overall Mean F1:   {mean_f1:.4f}")
print(f"  Overall Median F1: {df['Average_F1'].median():.4f}")
print(f"  Overall Std Dev:   {df['Average_F1'].std():.4f}")
print(f"  Overall Range:     {df['Average_F1'].max() - df['Average_F1'].min():.4f}")

print("\nVisualization complete!")

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

# Create figure with appropriate size for thesis
fig, ax = plt.subplots(figsize=(12, 7))

# Create horizontal bar chart
bars = ax.barh(df['Prompt'], df['Average_F1'], color='steelblue', alpha=0.8, 
               edgecolor='black', linewidth=1.0, height=0.6)

# Color-code bars by performance
colors = []
for val in df['Average_F1']:
    if val >= 0.45:
        colors.append('#2ecc71')  # Green for top performers
    elif val >= 0.35:
        colors.append('#3498db')  # Blue for medium performers
    elif val >= 0.25:
        colors.append('#f39c12')  # Orange for low performers
    else:
        colors.append('#e74c3c')  # Red for very low performers

for bar, color in zip(bars, colors):
    bar.set_color(color)
    bar.set_alpha(0.8)

# Add value labels on bars with better formatting
for i, (bar, value) in enumerate(zip(bars, df['Average_F1'])):
    if value > 0:
        ax.text(value + 0.015, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', 
                va='center', ha='left', fontsize=11, fontweight='bold')
    else:
        ax.text(0.015, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', 
                va='center', ha='left', fontsize=11, fontweight='bold')

# Styling
ax.set_xlabel('Average F1 Score (across 3 CLDs)', fontsize=13, fontweight='bold')
ax.set_ylabel('Prompt Variant', fontsize=13, fontweight='bold')
ax.set_title('Prompt Engineering Ablation Study: Average F1 Score Performance\nOrdered from Lowest to Highest Performance', 
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis limits with some padding
ax.set_xlim(0, max(df['Average_F1']) * 1.2)

# Add grid for readability
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Add mean line
mean_f1 = df['Average_F1'].mean()
ax.axvline(mean_f1, color='red', linestyle='--', linewidth=2.5, alpha=0.7, 
           label=f'Mean: {mean_f1:.3f}')

# Add median line
median_f1 = df['Average_F1'].median()
ax.axvline(median_f1, color='purple', linestyle=':', linewidth=2.5, alpha=0.7,
           label=f'Median: {median_f1:.3f}')

# Enhanced legend
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

# Make y-axis labels larger
ax.tick_params(axis='y', labelsize=11)
ax.tick_params(axis='x', labelsize=10)

# Tight layout
plt.tight_layout()

# Save figure
output_path = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ablation_study_f1_ordered.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Ordered bar chart saved to: {output_path}")

# Also save as PDF for thesis
pdf_path = '/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ablation_study_f1_ordered.pdf'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"PDF version saved to: {pdf_path}")

plt.close()

# Print ordered ranking
print("\n" + "=" * 80)
print("PROMPT RANKING (Low to High Average F1):")
print("=" * 80)
for i, (idx, row) in enumerate(df.iterrows(), 1):
    print(f"{i:2d}. {row['Prompt']:15s} | Avg F1: {row['Average_F1']:.4f} | "
          f"CLD1: {row['CLD1_F1']:.3f} | CLD2: {row['CLD2_F1']:.3f} | CLD3: {row['CLD3_F1']:.3f}")
print("=" * 80)

print(f"\nPerformance Statistics:")
print(f"  Mean F1:   {mean_f1:.4f}")
print(f"  Median F1: {median_f1:.4f}")
print(f"  Std Dev:   {df['Average_F1'].std():.4f}")
print(f"  Range:     {df['Average_F1'].max() - df['Average_F1'].min():.4f}")
print(f"  Best:      {df.iloc[-1]['Prompt']} ({df['Average_F1'].max():.4f})")
print(f"  Worst:     {df.iloc[0]['Prompt']} ({df['Average_F1'].min():.4f})")

print("\nVisualization complete!")

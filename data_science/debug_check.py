import pandas as pd
import traceback

try:
    df = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/corrected_672a19cd_20251105_155600.xlsx")
    print("Successfully loaded file")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns)[:10])
    
    print("\nRelationship type column:")
    print("Type:", type(df['relationship_type'].iloc[0]))
    print("First 5 values:")
    for i in range(min(5, len(df))):
        print(f"  {i}: {repr(df['relationship_type'].iloc[i])}")
    
except Exception as e:
    print("ERROR:", str(e))
    traceback.print_exc()

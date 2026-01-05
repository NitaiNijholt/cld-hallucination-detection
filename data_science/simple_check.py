import pandas as pd
import sys

try:
    df = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/corrected_672a19cd_20251105_155600.xlsx")
    
    print("Total edges: " + str(len(df)))
    
    pos_count = (df['Relationship Type'] == 'POSITIVE').sum()
    neg_count = (df['Relationship Type'] == 'NEGATIVE').sum()
    none_count = (df['Relationship Type'] == 'NONE').sum()
    
    print("POSITIVE: " + str(pos_count))
    print("NEGATIVE: " + str(neg_count))
    print("NONE: " + str(none_count))
    
    if 'Classification' in df.columns:
        tp = (df['Classification'] == 'TP').sum()
        fp = (df['Classification'] == 'FP').sum()
        fn = (df['Classification'] == 'FN').sum()
        
        print("TP: " + str(tp))
        print("FP: " + str(fp))
        print("FN: " + str(fn))
        
        if (tp + fp) > 0 and (tp + fn) > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            print("F1: " + str(f1))

except Exception as e:
    print("Error: " + str(e), file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)

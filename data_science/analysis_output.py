import pandas as pd
import sys

output_file = "analysis_results.txt"
with open(output_file, 'w') as f:
    try:
        # Read all files
        corrupted = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/corrupted_b46ac92d_20251105_141144.xlsx")
        no_correction = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/judged_no_correction_aaa235c9_20251105_141523.xlsx")
        corrected = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/corrected_672a19cd_20251105_155600.xlsx")
        ground_truth = pd.read_excel("parameter_tuning_experiments/ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx")
        
        f.write("=" * 80 + "\n")
        f.write("PHASE COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        # Ground truth
        gt_causal = ground_truth[ground_truth['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
        f.write("1. GROUND TRUTH:\n")
        f.write(f"   Total causal edges: {len(gt_causal)}\n")
        f.write(f"   POSITIVE: {(ground_truth['Relationship Type'] == 'POSITIVE').sum()}\n")
        f.write(f"   NEGATIVE: {(ground_truth['Relationship Type'] == 'NEGATIVE').sum()}\n\n")
        
        # Corrupted
        corr_causal = corrupted[corrupted['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
        f.write("2. CORRUPTED (Phase 1):\n")
        f.write(f"   Total causal edges: {len(corr_causal)}\n")
        if 'Classification' in corrupted.columns:
            tp = (corrupted['Classification'] == 'TP').sum()
            fp = (corrupted['Classification'] == 'FP').sum()
            fn = (corrupted['Classification'] == 'FN').sum()
            f.write(f"   TP: {tp}, FP: {fp}, FN: {fn}\n\n")
        
        # No correction
        no_corr_causal = no_correction[no_correction['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
        f.write("3. JUDGED WITHOUT CORRECTION (Phase 2a):\n")
        f.write(f"   Total causal edges: {len(no_corr_causal)}\n")
        if 'Classification' in no_correction.columns:
            tp = (no_correction['Classification'] == 'TP').sum()
            fp = (no_correction['Classification'] == 'FP').sum()
            fn = (no_correction['Classification'] == 'FN').sum()
            f.write(f"   TP: {tp}, FP: {fp}, FN: {fn}\n")
            if tp + fp > 0 and tp + fn > 0:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                f1 = 2 * prec * rec / (prec + rec)
                f.write(f"   F1: {f1:.3f}\n\n")
        
        # Corrected
        corrected_causal = corrected[corrected['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
        f.write("4. CORRECTED (Phase 2b):\n")
        f.write(f"   Total causal edges: {len(corrected_causal)}\n")
        if 'Classification' in corrected.columns:
            tp = (corrected['Classification'] == 'TP').sum()
            fp = (corrected['Classification'] == 'FP').sum()
            fn = (corrected['Classification'] == 'FN').sum()
            f.write(f"   TP: {tp}, FP: {fp}, FN: {fn}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("GROUND TRUTH CAUSAL EDGES:\n")
        f.write("=" * 80 + "\n")
        for _, row in gt_causal.iterrows():
            f.write(f"{row['Source']} -> {row['Target']} [{row['Relationship Type']}]\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CORRECTED CAUSAL EDGES:\n")
        f.write("=" * 80 + "\n")
        for _, row in corrected_causal.iterrows():
            classif = row.get('Classification', 'N/A')
            f.write(f"{row['Source']} -> {row['Target']} [{row['Relationship Type']}] ({classif})\n")
        
        print(f"Analysis written to {output_file}")
        
    except Exception as e:
        f.write(f"ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc(file=f)
        print(f"Error occurred, check {output_file}", file=sys.stderr)

# Also print to console
with open(output_file, 'r') as f:
    print(f.read())

import pandas as pd

# Read all three phases
corrupted = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/corrupted_b46ac92d_20251105_141144.xlsx")
no_correction = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/judged_no_correction_aaa235c9_20251105_141523.xlsx")
corrected = pd.read_excel("parameter_tuning_experiments/results/rq1_corruption_experiment_20251105_140909/corrected_672a19cd_20251105_155600.xlsx")
ground_truth = pd.read_excel("parameter_tuning_experiments/ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx")

print("=" * 80)
print("COMPARISON: GROUND TRUTH vs CORRUPTED vs NO CORRECTION vs CORRECTED")
print("=" * 80)

# Ground truth
gt_causal = ground_truth[ground_truth['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
print("\n1. GROUND TRUTH:")
print("   Total causal edges: " + str(len(gt_causal)))
print("   POSITIVE: " + str((ground_truth['Relationship Type'] == 'POSITIVE').sum()))
print("   NEGATIVE: " + str((ground_truth['Relationship Type'] == 'NEGATIVE').sum()))

# Corrupted
corr_causal = corrupted[corrupted['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
print("\n2. CORRUPTED (Phase 1):")
print("   Total causal edges: " + str(len(corr_causal)))
print("   POSITIVE: " + str((corrupted['Relationship Type'] == 'POSITIVE').sum()))
print("   NEGATIVE: " + str((corrupted['Relationship Type'] == 'NEGATIVE').sum()))
if 'Classification' in corrupted.columns:
    tp = (corrupted['Classification'] == 'TP').sum()
    fp = (corrupted['Classification'] == 'FP').sum()
    fn = (corrupted['Classification'] == 'FN').sum()
    print("   TP: " + str(tp) + ", FP: " + str(fp) + ", FN: " + str(fn))

# No correction
no_corr_causal = no_correction[no_correction['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
print("\n3. JUDGED WITHOUT CORRECTION (Phase 2a):")
print("   Total causal edges: " + str(len(no_corr_causal)))
print("   POSITIVE: " + str((no_correction['Relationship Type'] == 'POSITIVE').sum()))
print("   NEGATIVE: " + str((no_correction['Relationship Type'] == 'NEGATIVE').sum()))
if 'Classification' in no_correction.columns:
    tp = (no_correction['Classification'] == 'TP').sum()
    fp = (no_correction['Classification'] == 'FP').sum()
    fn = (no_correction['Classification'] == 'FN').sum()
    print("   TP: " + str(tp) + ", FP: " + str(fp) + ", FN: " + str(fn))
    if tp + fp > 0 and tp + fn > 0:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        print("   Precision: " + str(round(prec, 3)) + ", Recall: " + str(round(rec, 3)) + ", F1: " + str(round(f1, 3)))

# Corrected
corrected_causal = corrected[corrected['Relationship Type'].isin(['POSITIVE', 'NEGATIVE'])]
print("\n4. CORRECTED (Phase 2b):")
print("   Total causal edges: " + str(len(corrected_causal)))
print("   POSITIVE: " + str((corrected['Relationship Type'] == 'POSITIVE').sum()))
print("   NEGATIVE: " + str((corrected['Relationship Type'] == 'NEGATIVE').sum()))
if 'Classification' in corrected.columns:
    tp = (corrected['Classification'] == 'TP').sum()
    fp = (corrected['Classification'] == 'FP').sum()
    fn = (corrected['Classification'] == 'FN').sum()
    print("   TP: " + str(tp) + ", FP: " + str(fp) + ", FN: " + str(fn))
    if tp + fp > 0:
        prec = tp / (tp + fp)
        print("   Precision: " + str(round(prec, 3)))
    if tp + fn > 0:
        rec = tp / (tp + fn)
        print("   Recall: " + str(round(rec, 3)))

print("\n" + "=" * 80)
print("SAMPLE CAUSAL EDGES FROM EACH PHASE:")
print("=" * 80)

print("\nGround Truth causal edges:")
print(gt_causal[['Source', 'Target', 'Relationship Type']].to_string(index=False))

print("\n\nCorrected causal edges:")
print(corrected_causal[['Source', 'Target', 'Relationship Type', 'Classification']].to_string(index=False))

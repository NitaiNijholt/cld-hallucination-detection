#!/usr/bin/env python3
"""Verify test edges against social norms ground truth CLD."""

import json

# Ground truth edges from Social_norms_and_obesity_prevalence CLD
GROUND_TRUTH_EDGES = [
    ("Group-level BMI", "Norm BMI"),
    ("Socio-cultural Ideal BMI", "Norm BMI"),
    ("Norm BMI", "Individual Ideal BMI"),
    ("Healthy BMI", "Individual Ideal BMI"),
    ("Individual Ideal BMI", "Total Daily Energy Intake (TDEI)"),
    ("Total Daily Energy Intake (TDEI)", "Body Mass Index (BMI)"),
    ("Body Mass Index (BMI)", "individual ideal BMI"),  # Note: lowercase
    ("Individual Ideal BMI", "Physical Activity Level"),
    ("Physical Activity Level", "Total Daily Energy Expenditure (TDEE)"),
    ("Total Daily Energy Expenditure (TDEE)", "Body Mass Index (BMI)"),
    ("Body Mass Index (BMI)", "Basal Metabolic Rate"),
    ("Basal Metabolic Rate", "Total Daily Energy Expenditure (TDEE)"),
]

# Test edges from test_deep_research_sample.py
TEST_EDGES = {
    "TP": [
        ("Basal Metabolic Rate", "Total Daily Energy Expenditure (TDEE)"),
        ("Total Daily Energy Expenditure (TDEE)", "Body Mass Index (BMI)"),
        ("Total Daily Energy Intake (TDEI)", "Body Mass Index (BMI)"),
        ("Body Mass Index (BMI)", "Weight Norm Misperception"),
    ],
    "FP": [
        ("Socio-cultural Ideal BMI", "Individual Ideal BMI"),
        ("Physical Activity Level", "Basal Metabolic Rate"),
        ("Actual BMI", "Weight Norm Misperception"),
        ("Perceived Weight Status", "Weight Control Efforts"),
    ],
    "TN": [
        ("Individual Ideal BMI", "Basal Metabolic Rate"),
        ("Weight Norm Misperception", "Physical Activity Level"),
    ],
    "FN": [
        ("Body Mass Index (BMI)", "BMI Heritability"),
    ],
}

print("="*100)
print("VERIFICATION: Test Edges vs Social Norms Ground Truth CLD")
print("="*100)

# Create ground truth set for easy lookup
gt_set = set(GROUND_TRUTH_EDGES)

print(f"\n📚 GROUND TRUTH CLD: {len(gt_set)} edges")
print("-"*100)
for i, (src, tgt) in enumerate(GROUND_TRUTH_EDGES, 1):
    print(f"  {i:2}. {src:40} → {tgt}")

all_test_edges = []
for classification, edges in TEST_EDGES.items():
    all_test_edges.extend([(classification, src, tgt) for src, tgt in edges])

print(f"\n🧪 TEST EDGES: {len(all_test_edges)} edges")
print("-"*100)

# Analyze each test edge
in_gt = 0
not_in_gt = 0

for classification, src, tgt in all_test_edges:
    edge = (src, tgt)
    is_in_gt = edge in gt_set
    status = "✅ IN GT" if is_in_gt else "❌ NOT IN GT"
    
    if is_in_gt:
        in_gt += 1
    else:
        not_in_gt += 1
    
    print(f"  [{classification:2}] {src:40} → {tgt:40} {status}")

print(f"\n" + "="*100)
print(f"SUMMARY")
print(f"="*100)
print(f"  In ground truth:     {in_gt:2}/{len(all_test_edges)} ({in_gt/len(all_test_edges)*100:.0f}%)")
print(f"  NOT in ground truth: {not_in_gt:2}/{len(all_test_edges)} ({not_in_gt/len(all_test_edges)*100:.0f}%)")

# Analyze by classification
print(f"\n📊 BREAKDOWN BY CLASSIFICATION:")
print("-"*100)
for classification in ["TP", "FP", "TN", "FN"]:
    edges = TEST_EDGES[classification]
    in_this_class = sum(1 for src, tgt in edges if (src, tgt) in gt_set)
    print(f"  {classification}: {in_this_class}/{len(edges)} in ground truth ({in_this_class/len(edges)*100:.0f}%)")

print(f"\n" + "="*100)
print("KEY FINDINGS:")
print("="*100)

# Find which edges are NOT in ground truth
not_in_gt_edges = [(cls, src, tgt) for cls, src, tgt in all_test_edges if (src, tgt) not in gt_set]
if not_in_gt_edges:
    print(f"\n⚠️  EDGES NOT IN GROUND TRUTH ({len(not_in_gt_edges)}):")
    for cls, src, tgt in not_in_gt_edges:
        print(f"  [{cls}] {src} → {tgt}")
        # Check if variables are in GT
        gt_sources = [s for s, t in GROUND_TRUTH_EDGES]
        gt_targets = [t for s, t in GROUND_TRUTH_EDGES]
        all_gt_vars = set(gt_sources + gt_targets)
        
        src_in_vars = src in all_gt_vars
        tgt_in_vars = tgt in all_gt_vars
        
        if not src_in_vars and not tgt_in_vars:
            print(f"       → Both variables NOT in ground truth CLD")
        elif not src_in_vars:
            print(f"       → Source variable '{src}' NOT in ground truth CLD")
        elif not tgt_in_vars:
            print(f"       → Target variable '{tgt}' NOT in ground truth CLD")
        else:
            print(f"       → Edge direction may be wrong or it's a spurious edge")

print(f"\n✅ EDGES IN GROUND TRUTH ({in_gt}):")
for cls, src, tgt in all_test_edges:
    if (src, tgt) in gt_set:
        print(f"  [{cls}] {src} → {tgt}")


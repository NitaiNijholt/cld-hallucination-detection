#!/usr/bin/env python3
"""Get average aggregate judge scores across all domains."""

import pandas as pd
import json
import glob
import os

print("=" * 80)
print("AVERAGE AGGREGATE JUDGE SCORES")
print("=" * 80)

# 1. Physics CLDs
print("\n1. PHYSICS CLDs (Expert-provided references)")
with open("results/physics_cld_raw_results_20251204_182810.json") as f:
    data = json.load(f)

citation_results = data.get("citation_results", [])
retrieved_scores = []

for cld in citation_results:
    for edge in cld.get("edges", []):
        score = edge.get("aggregate_score")
        verdict = edge.get("judge_verdict", "")
        if verdict != "NO_CITATION" and score is not None:
            retrieved_scores.append(score)

no_cit_count = sum(1 for cld in citation_results for e in cld.get("edges", []) if e.get("judge_verdict") == "NO_CITATION")
total = sum(len(cld.get("edges", [])) for cld in citation_results)

print(f"  Total edges: {total}, NO_CITATION: {no_cit_count}, Retrieved: {total - no_cit_count}")
print(f"  Retrieval rate: {(total - no_cit_count) / total * 100:.1f}%")
if retrieved_scores:
    print(f"  Avg score (retrieved only): {sum(retrieved_scores) / len(retrieved_scores):.3f}")

# 2. Ulemans
print("\n2. ULEMANS ALZHEIMER'S (Expert-provided references)")
xlsx_path = "../RQ1a_ulemans_citation_provider_comparison/citation_fetcher_comparison_20251113_v2.xlsx"
df = pd.read_excel(xlsx_path, sheet_name="OpenAlex", engine='openpyxl')
jina_scores = df["Jina Score"].dropna()
print(f"  Total edges: {len(df)}")
print(f"  Retrieval rate (Jina): 100.0%")
print(f"  Avg score (Jina): {jina_scores.mean():.3f}")

# ContentScraper
cs_verdicts = df["ContentScraper Verdict"]
cs_retrieved = cs_verdicts != "NO_CITATION"
cs_scores = df.loc[cs_retrieved, "ContentScraper Score"].dropna()
print(f"  Retrieval rate (ContentScraper): {cs_retrieved.sum() / len(df) * 100:.1f}%")
if len(cs_scores) > 0:
    print(f"  Avg score (ContentScraper, retrieved): {cs_scores.mean():.3f}")

# 3. Health CLDs
print("\n3. HEALTH CLDs (Automated fetch)")
base_path = "../RQ1a_gt_lit_citation"
clds = ["depressive", "social_norms", "emergency_department"]
cld_names = {"depressive": "Depressive", "social_norms": "Social Norms", "emergency_department": "Emergency Dept"}

for cld in clds:
    cld_path = os.path.join(base_path, cld)
    if not os.path.exists(cld_path):
        continue
    all_scores = []
    total_edges = 0
    no_cit = 0
    for run in ["run_1", "run_2", "run_3"]:
        run_path = os.path.join(cld_path, run)
        if not os.path.exists(run_path):
            continue
        xlsx_files = glob.glob(os.path.join(run_path, "judged_*baseline*.xlsx"))
        for xp in xlsx_files[:1]:
            try:
                df_h = pd.read_excel(xp, sheet_name="All Edges", engine='openpyxl')
                total_edges += len(df_h)
                if "Aggregate Score" in df_h.columns:
                    scores = df_h["Aggregate Score"].dropna()
                    all_scores.extend(scores.tolist())
                verdict_counts = df_h["Judge Verdict"].value_counts()
                no_cit += verdict_counts.get("NO_CITATION", 0) + verdict_counts.get("No citation", 0)
            except Exception as e:
                print(f"  Warning: {e}")
    if total_edges > 0:
        retr_rate = (total_edges - no_cit) / total_edges * 100
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        print(f"  {cld_names[cld]}: {total_edges} edges, {retr_rate:.1f}% retrieval, avg score: {avg_score:.3f}")

print("\n" + "=" * 80)
print("\nSUMMARY TABLE FOR THESIS:")
print("=" * 80)
print(f"{'Domain':<25} {'Citation Source':<18} {'Retrieval':<12} {'Avg Score'}")
print("-" * 80)









#!/usr/bin/env python3
"""
Stratified random sampling of deep research edges for human validation.

Samples edges stratified by:
1. CLD (Causal Loop Diagram)
2. Classification (TP, FP, FN)

Outputs an Excel file with a structured format for human review.
"""

import json
import random
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
import math
from datetime import datetime
from openpyxl.utils import get_column_letter
from collections import Counter

# Configuration
SAMPLE_SIZE = 50
RANDOM_SEED = 42  # For reproducibility
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "validation"

# Optional: join in the *pre-modification* generator narrative (Motivation) and relationship type
# from the RQ1a mechanistic judged XLSX files (All Edges sheet).
# This is what humans originally saw as the causal narrative during edge generation.
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RQ1A_ENHANCED = BASE_DIR / "final_runs/RQ1a_gt_lit_correctness/enhanced_analysis_20251225_233458_e5fcc326/rq1a_ground_truth_enhanced_results.xlsx"

# Canonical CLD names (must match CLD_full in DR results)
CLD_MAP = {
    "depressive": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
    "social_norms": "Social_norms_and_obesity_prevalence",
    "emergency_department": "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet",
}


def load_all_edges(data_dir: Path, direct_causal_only: bool = True) -> list[dict]:
    """Load all edges from all deep research result files.
    
    Args:
        data_dir: Directory containing the JSON files
        direct_causal_only: If True, only include edges where direct_causal_found=True
    """
    all_edges = []
    filtered_count = 0
    
    for json_file in data_dir.glob("deep_research_results_*.json"):
        print(f"Loading: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cld_name = data.get("cld", "Unknown")
        results = data.get("results", [])
        
        for edge in results:
            # Filter for direct_causal_found=True if requested
            if direct_causal_only and not edge.get("direct_causal_found", False):
                filtered_count += 1
                continue
                
            edge["_source_file"] = json_file.name
            edge["_cld_short"] = cld_name.split("_")[0][:20]  # Short name for display
            all_edges.append(edge)
    
    if direct_causal_only:
        print(f"Filtered out {filtered_count} edges without direct causal evidence")
    
    return all_edges


def stratified_sample(edges: list[dict], target_size: int, seed: int) -> list[dict]:
    """
    Perform stratified sampling across CLDs and classifications.
    
    Uses proportional allocation with rounding to ensure representation.
    """
    random.seed(seed)
    
    # Group edges by (CLD, classification)
    strata = defaultdict(list)
    for edge in edges:
        cld = edge.get("CLD", "Unknown")
        classification = edge.get("classification", "Unknown")
        strata[(cld, classification)].append(edge)
    
    total_edges = len(edges)
    sampled = []
    
    # Calculate proportional allocation for each stratum
    allocations = {}
    remaining = target_size
    
    for key, stratum_edges in strata.items():
        # Proportional allocation
        proportion = len(stratum_edges) / total_edges
        allocation = proportion * target_size
        
        # Ensure at least 1 sample from each stratum (if stratum is non-empty)
        allocation = max(1, int(round(allocation)))
        # Don't sample more than available
        allocation = min(allocation, len(stratum_edges))
        allocations[key] = allocation
    
    # Adjust if we over-allocated
    total_allocated = sum(allocations.values())
    while total_allocated > target_size:
        # Reduce from largest allocations first
        max_key = max(allocations.keys(), key=lambda k: allocations[k])
        if allocations[max_key] > 1:
            allocations[max_key] -= 1
            total_allocated -= 1
    
    # Perform sampling
    print("\nStratified allocation:")
    print("-" * 60)
    for (cld, classification), allocation in sorted(allocations.items()):
        stratum_edges = strata[(cld, classification)]
        sample = random.sample(stratum_edges, allocation)
        sampled.extend(sample)
        cld_short = cld[:40] + "..." if len(cld) > 40 else cld
        print(f"  {cld_short:45} | {classification:3} | {allocation:2}/{len(stratum_edges):3}")
    
    print("-" * 60)
    print(f"  Total sampled: {len(sampled)}")
    
    # Shuffle the final sample to randomize presentation order
    random.shuffle(sampled)
    
    return sampled


# Regex pattern for illegal Excel characters (control characters except tab, newline, carriage return)
ILLEGAL_CHARACTERS_RE = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'
)


def safe_str(value, default="") -> str:
    """Safely convert a value to string, handling None and illegal Excel characters."""
    if value is None:
        return default
    text = str(value)
    # Remove illegal Excel characters
    text = ILLEGAL_CHARACTERS_RE.sub('', text)
    return text


def truncate(text: str, max_len: int = 500) -> str:
    """Truncate text to max length with ellipsis."""
    text = safe_str(text)
    return text[:max_len] + "..." if len(text) > max_len else text


def norm_key_part(s: str | None) -> str:
    """Normalize strings for key matching."""
    return "" if s is None else str(s).strip().lower()


def detect_cld_from_path(file_path: str) -> str:
    """Detect CLD from a file path (same convention used in other RQ3 scripts)."""
    fp_lower = str(file_path).lower()
    if "depressive" in fp_lower:
        return CLD_MAP["depressive"]
    if "social_norms" in fp_lower:
        return CLD_MAP["social_norms"]
    if "emergency" in fp_lower or "older_persons" in fp_lower:
        return CLD_MAP["emergency_department"]
    raise ValueError(f"Unknown CLD in path: {file_path}")


def build_preclaim_lookup() -> tuple[dict[tuple[str, str, str], str], dict[tuple[str, str, str], str]]:
    """
    Build a lookup for pre-modification narratives from RQ1a mechanistic judged files.

    Returns:
      - preclaim_lookup[(cld_full, source, target)] -> Motivation text (best-effort)
      - rel_lookup[(cld_full, source, target)] -> Relationship Type (POSITIVE/NEGATIVE/NONE)

    Notes:
      - We aggregate across the 9 mechanistic judged XLSX files and choose the most
        common Motivation/Relationship Type per edge (robust to minor run differences).
      - If files are missing, returns empty lookups (sampling still works).
    """
    if not RQ1A_ENHANCED.exists():
        print(f"[WARN] RQ1A_ENHANCED not found: {RQ1A_ENHANCED} (skipping pre-claim lookup)")
        return {}, {}

    try:
        df_raw = pd.read_excel(RQ1A_ENHANCED, sheet_name="Raw Data", engine='openpyxl')
    except Exception as e:
        print(f"[WARN] Failed to read RQ1A_ENHANCED Raw Data: {e} (skipping pre-claim lookup)")
        return {}, {}

    mech = df_raw[df_raw["prompt"].astype(str).str.lower() == "mechanistic"]
    file_paths = [Path(p) for p in mech["file_path"].dropna().unique().tolist()]
    print(f"Pre-claim lookup: found {len(file_paths)} mechanistic judged files")

    pre_bucket: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    rel_bucket: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for fp in file_paths:
        if not fp.exists():
            print(f"[WARN] Missing judged XLSX: {fp}")
            continue
        try:
            df_edges = pd.read_excel(fp, sheet_name="All Edges", engine='openpyxl')
        except Exception as e:
            print(f"[WARN] Failed reading All Edges from {fp}: {e}")
            continue

        try:
            cld_full = detect_cld_from_path(str(fp))
        except Exception as e:
            print(f"[WARN] {e} (skipping file: {fp})")
            continue

        for _, row in df_edges.iterrows():
            src = norm_key_part(row.get("Source"))
            tgt = norm_key_part(row.get("Target"))
            if not src or not tgt:
                continue
            key = (norm_key_part(cld_full), src, tgt)

            # Prefer Original Motivation when present (e.g., after correction workflows),
            # fall back to Motivation.
            mot = row.get("Original Motivation")
            if pd.isna(mot) or not str(mot).strip():
                mot = row.get("Motivation")
            rel = row.get("Relationship Type")

            if pd.notna(mot) and str(mot).strip():
                pre_bucket[key].append(str(mot))
            if pd.notna(rel) and str(rel).strip():
                rel_bucket[key].append(str(rel).strip())

    pre_lookup: dict[tuple[str, str, str], str] = {}
    rel_lookup: dict[tuple[str, str, str], str] = {}

    for k, vals in pre_bucket.items():
        pre_lookup[k] = Counter(vals).most_common(1)[0][0]
    for k, vals in rel_bucket.items():
        rel_lookup[k] = Counter(vals).most_common(1)[0][0]

    print(f"Pre-claim lookup built: {len(pre_lookup)} keys with narratives")
    return pre_lookup, rel_lookup


def extract_post_claim(edge: dict) -> str:
    """
    Extract the best available *post-modification* natural-language causal claim text.

    In the RQ3 Deep Research outputs, this is typically stored as `modified_claim`.
    """
    for key in ("modified_claim", "claim", "explanation"):
        val = edge.get(key)
        if val:
            return safe_str(val)

    hist = edge.get("iteration_history") or []
    if isinstance(hist, list) and hist:
        last = hist[-1] if isinstance(hist[-1], dict) else None
        if last and last.get("claim"):
            return safe_str(last.get("claim"))

    return ""


def create_validation_excel(sampled_edges: list[dict], output_path: Path):
    """Create an Excel file with structured columns for human validation."""

    # Optional lookup to show the original generator narrative (pre-modification claim)
    pre_lookup, rel_lookup = build_preclaim_lookup()

    rows = []
    for idx, edge in enumerate(sampled_edges, 1):
        # Create a readable edge label
        source = safe_str(edge.get("source", ""))
        target = safe_str(edge.get("target", ""))
        edge_label = f"{source} → {target}"
        
        # Get CLD short name
        cld_full = safe_str(edge.get("CLD", "Unknown"))
        cld_parts = cld_full.replace("_", " ").split()
        cld_short = " ".join(cld_parts[:3]) if len(cld_parts) > 3 else " ".join(cld_parts)
        
        # Evidence summary
        evidence_summary = safe_str(edge.get("strongest_causal_evidence_summary", ""))
        passage = safe_str(edge.get("strongest_causal_evidence_passage", ""))
        url = safe_str(edge.get("strongest_causal_evidence_url", ""))

        # Pre/post claim text
        cld_full_key = norm_key_part(edge.get("CLD", ""))
        src_key = norm_key_part(edge.get("source", ""))
        tgt_key = norm_key_part(edge.get("target", ""))
        join_key = (cld_full_key, src_key, tgt_key)
        pre_mod_claim = pre_lookup.get(join_key, "")
        relationship_type = rel_lookup.get(join_key, "")

        post_claim = extract_post_claim(edge)
        causal_narrative = truncate(post_claim if post_claim.strip() else pre_mod_claim, 600)
        
        row = {
            # Validation columns (to be filled by human)
            "ID": idx,
            "human_verdict": "",  # To be filled: supported/partially_supported/unsupported/invalid
            "quote_match": "",    # To be filled: Yes/No/Partial
            "notes": "",          # Free text notes
            
            # Edge identification (exact JSON field names)
            "CLD": cld_short,
            "classification": safe_str(edge.get("classification", "")),
            "source": source,
            "target": target,

            # Pre/post causal narrative shown to the human (explicit + easy to find)
            "relationship_type": safe_str(relationship_type),
            "pre_modification_claim": truncate(pre_mod_claim, 600),
            "causal_narrative": causal_narrative,  # post if available else pre
            
            # Deep Research verdict (exact JSON field names)
            "verdict": safe_str(edge.get("verdict", "")),
            "confidence": safe_str(edge.get("confidence", "")),
            "direct_causal_found": safe_str(edge.get("direct_causal_found", "")),
            
            # Evidence details (exact JSON field names)
            "strongest_causal_evidence_confidence": safe_str(edge.get("strongest_causal_evidence_confidence", "")),
            "strongest_causal_evidence_study_type": safe_str(edge.get("strongest_causal_evidence_study_type", "")),
            "strongest_causal_evidence_url": url,
            "strongest_causal_evidence_summary": truncate(evidence_summary, 500),
            "strongest_causal_evidence_passage": truncate(passage, 500),
            
            # Additional context (exact JSON field names)
            "judge_reasoning": truncate(edge.get("judge_reasoning", ""), 500),
            "modified_claim": safe_str(edge.get("modified_claim", "")),  # kept for backward compatibility
            
            # Metadata
            "CLD_full": cld_full,
            "_source_file": safe_str(edge.get("_source_file", "")),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Create Excel writer with formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Edge Validation')
        
        # Get worksheet for formatting
        worksheet = writer.sheets['Edge Validation']
        
        # Set column widths by column name (robust to column insertions/reordering)
        widths_by_name = {
            "ID": 5,
            "human_verdict": 18,
            "quote_match": 12,
            "Fully applicable": 14,  # present in post-annotation files, not here
            "notes": 25,
            "CLD": 18,
            "classification": 12,
            "source": 25,
            "target": 25,
            "relationship_type": 14,
            "pre_modification_claim": 70,
            "causal_narrative": 70,
            "verdict": 18,
            "confidence": 10,
            "direct_causal_found": 18,
            "strongest_causal_evidence_confidence": 14,
            "strongest_causal_evidence_study_type": 35,
            "strongest_causal_evidence_url": 50,
            "strongest_causal_evidence_summary": 60,
            "strongest_causal_evidence_passage": 60,
            "judge_reasoning": 60,
            "modified_claim": 60,
            "CLD_full": 40,
            "_source_file": 30,
        }

        for i, col_name in enumerate(df.columns, start=1):
            letter = get_column_letter(i)
            worksheet.column_dimensions[letter].width = widths_by_name.get(col_name, 20)
        
        # Freeze first row and validation columns
        worksheet.freeze_panes = 'E2'
    
    print(f"\nExcel file saved to: {output_path}")
    

def create_summary_stats(edges: list[dict], sampled: list[dict]) -> pd.DataFrame:
    """Create summary statistics of the sampling."""
    
    # Count edges by stratum
    full_counts = defaultdict(lambda: defaultdict(int))
    sample_counts = defaultdict(lambda: defaultdict(int))
    
    for edge in edges:
        cld = edge.get("CLD", "Unknown")
        classification = edge.get("classification", "Unknown")
        full_counts[cld][classification] += 1
    
    for edge in sampled:
        cld = edge.get("CLD", "Unknown")
        classification = edge.get("classification", "Unknown")
        sample_counts[cld][classification] += 1
    
    rows = []
    for cld in sorted(full_counts.keys()):
        for classification in ['TP', 'FP', 'FN']:
            if full_counts[cld][classification] > 0:
                cld_short = cld[:40] + "..." if len(cld) > 40 else cld
                rows.append({
                    'CLD': cld_short,
                    'Classification': classification,
                    'Total Edges': full_counts[cld][classification],
                    'Sampled': sample_counts[cld][classification],
                    'Sample %': f"{100 * sample_counts[cld][classification] / full_counts[cld][classification]:.1f}%"
                })
    
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Deep Research Edge Sampling for Human Validation")
    print("=" * 60)
    
    # Load all edges (only those with direct_causal_found=True)
    print(f"\nLoading edges from: {DATA_DIR}")
    print("Filter: direct_causal_found=True only")
    all_edges = load_all_edges(DATA_DIR, direct_causal_only=True)
    print(f"Total edges loaded: {len(all_edges)}")
    
    # Perform stratified sampling
    print(f"\nPerforming stratified sampling (target: {SAMPLE_SIZE} edges)")
    sampled_edges = stratified_sample(all_edges, SAMPLE_SIZE, RANDOM_SEED)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"edge_validation_sample_{timestamp}.xlsx"
    
    # Create Excel file
    create_validation_excel(sampled_edges, output_file)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Sampling Summary")
    print("=" * 60)
    summary_df = create_summary_stats(all_edges, sampled_edges)
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Validation Instructions")
    print("=" * 60)
    print("""
For each edge, please fill in:

1. Human Verdict: Your assessment of the causal claim
   - supported: Evidence strongly supports the causal claim
   - partially_supported: Some evidence, but with caveats
   - unsupported: Evidence does not support the claim
   - invalid: The edge/claim itself is unclear or invalid

2. Quote Match: Does the key quote support the verdict?
   - Yes: Quote clearly supports the DR verdict
   - No: Quote does not support the DR verdict
   - Partial: Quote partially supports the verdict

3. Notes: Any additional observations or concerns

The causal narrative to validate is provided in the 'causal_narrative' column
(also kept as 'modified_claim' for backward compatibility).
""")
    
    print(f"\nOutput file: {output_file}")


if __name__ == "__main__":
    main()


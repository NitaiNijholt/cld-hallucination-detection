#!/usr/bin/env python3
"""
Convert Alzheimer's expert CLD from supplementary Excel to standard format.

This script:
1. Parses the References sheet to create author(year) -> full citation mapping
2. Parses the Connections sheet to extract edges and their support citations
3. Converts to standard format (3-sheet Excel + companion JSONs)
4. Creates a citation hints JSON for the refiller agent

Output files:
- Alzheimers_disease_expert_validated.xlsx (3 sheets: Variable_definitions, Variable_links, Context)
- Alzheimers_disease_vars_data.json (variable list)
- Alzheimers_disease_edges_data.json (edge list)
- Alzheimers_disease_citation_hints.json (citation mapping for refiller)
"""

import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Paths
BASE_DIR = Path(__file__).parent / "ground_truth_clds_for_experiments"
INPUT_FILE = BASE_DIR / "alzheimers_supplementary.xlsx"
OUTPUT_EXCEL = BASE_DIR / "Alzheimers_disease_expert_validated.xlsx"
OUTPUT_VARS_JSON = BASE_DIR / "Alzheimers_disease_vars_data.json"
OUTPUT_EDGES_JSON = BASE_DIR / "Alzheimers_disease_edges_data.json"
OUTPUT_CITATION_HINTS = BASE_DIR / "Alzheimers_disease_citation_hints.json"


def parse_references_sheet(excel_path: str) -> Dict[str, str]:
    """
    Parse the References sheet to create author(year) -> full citation mapping.
    
    Returns:
        Dict mapping "Author (year)" to full citation text
    """
    print("\n" + "="*80)
    print("STEP 1: Parsing References Sheet")
    print("="*80)
    
    df_ref = pd.read_excel(excel_path, sheet_name='References')
    
    # References are in the first column
    references = df_ref.iloc[:, 0].dropna().tolist()
    
    print(f"Found {len(references)} references")
    
    # Build mapping from author(year) to full citation
    ref_map = {}
    
    for ref in references:
        ref = str(ref).strip()
        if not ref:
            continue
        
        # Extract first author last name and year
        # Pattern: "LastName, FirstName... (YYYY). Title..."
        match = re.search(r'^([^,]+),.*?\((\d{4})\)', ref)
        if match:
            author_last = match.group(1).strip()
            # Handle special characters in author names (e.g., "Åkerstedt")
            # Extract just the last name (first word before comma)
            author_last = author_last.split()[0] if ' ' in author_last else author_last
            year = match.group(2)
            key = f"{author_last} ({year})"
            ref_map[key] = ref
            print(f"  Mapped: {key}")
        else:
            print(f"  WARNING: Could not parse reference: {ref[:80]}...")
    
    print(f"\nSuccessfully mapped {len(ref_map)} references")
    return ref_map


def parse_connections_sheet(excel_path: str, ref_map: Dict[str, str]) -> Tuple[List[Dict], Dict[Tuple[str, str], List[str]]]:
    """
    Parse the Connections sheet to extract edges and build citation hints map.
    
    Returns:
        Tuple of (edges_list, citation_hints_map)
        - edges_list: List of edge dictionaries
        - citation_hints_map: Dict mapping (source, target) to list of full citations
    """
    print("\n" + "="*80)
    print("STEP 2: Parsing Connections Sheet")
    print("="*80)
    
    df_conn = pd.read_excel(excel_path, sheet_name='Connections', header=1)
    
    # Drop rows where Origin is NaN (these are section headers)
    df_conn = df_conn.dropna(subset=['Origin'])
    
    print(f"Found {len(df_conn)} connections")
    
    edges = []
    citation_hints = {}
    unmapped_citations = set()
    
    for idx, row in df_conn.iterrows():
        source = str(row['Origin']).strip()
        target = str(row['Destination']).strip()
        polarity = str(row['Polarity']).strip()
        support = str(row['Support']).strip() if pd.notna(row['Support']) else ""
        
        # Convert polarity to lowercase (positive/negative)
        polarity_map = {'+': 'positive', '-': 'negative'}
        polarity_clean = polarity_map.get(polarity, polarity.lower())
        
        # Add edge
        edges.append({
            "source": source,
            "target": target,
            "polarity": polarity_clean
        })
        
        # Parse support citations (format: "Author (year); Author (year); ...")
        if support:
            # Split by semicolon
            citation_keys = [c.strip() for c in support.split(';') if c.strip()]
            
            # Look up full citations
            full_citations = []
            for key in citation_keys:
                if key in ref_map:
                    full_citations.append(ref_map[key])
                else:
                    # Try to match with slightly different formatting
                    # Sometimes there might be extra spaces, etc.
                    found = False
                    for ref_key in ref_map.keys():
                        # Simple fuzzy match: same author and year
                        if key.replace(' ', '').lower() == ref_key.replace(' ', '').lower():
                            full_citations.append(ref_map[ref_key])
                            found = True
                            break
                    
                    if not found:
                        unmapped_citations.add(key)
                        print(f"  WARNING: Citation not found in references: {key}")
            
            if full_citations:
                citation_hints[(source, target)] = full_citations
    
    print(f"\nProcessed {len(edges)} edges")
    print(f"Created citation hints for {len(citation_hints)} edges")
    
    if unmapped_citations:
        print(f"\n⚠️  WARNING: {len(unmapped_citations)} citation keys could not be mapped to references:")
        for key in sorted(unmapped_citations):
            print(f"    - {key}")
    
    return edges, citation_hints


def extract_variables(edges: List[Dict]) -> List[Dict]:
    """
    Extract unique variables from edges.
    
    Returns:
        List of variable dictionaries
    """
    print("\n" + "="*80)
    print("STEP 3: Extracting Variables")
    print("="*80)
    
    # Collect unique variable names
    var_names = set()
    for edge in edges:
        var_names.add(edge['source'])
        var_names.add(edge['target'])
    
    var_names = sorted(var_names)
    
    print(f"Found {len(var_names)} unique variables")
    
    # Create variable list (with empty definitions for now)
    variables = []
    for idx, name in enumerate(var_names):
        variables.append({
            "index": idx,
            "name": name,
            "definition": ""  # Empty definition, can be filled later
        })
    
    return variables


def create_standard_excel(variables: List[Dict], edges: List[Dict], output_path: str):
    """
    Create standard 3-sheet Excel file.
    
    Sheets:
    - Variable_definitions: Variable, Definition
    - Variable_links: Source, Target, Polarity
    - Context: Target, Temporal Scale, Spatial Scale
    """
    print("\n" + "="*80)
    print("STEP 4: Creating Standard Excel File")
    print("="*80)
    
    # Sheet 1: Variable_definitions
    df_vars = pd.DataFrame([
        {"Variable": v["name"], "Definition": v["definition"]}
        for v in variables
    ])
    
    # Sheet 2: Variable_links
    df_links = pd.DataFrame([
        {"Source": e["source"], "Target": e["target"], "Polarity": e["polarity"]}
        for e in edges
    ])
    
    # Sheet 3: Context
    # For Alzheimer's, we'll use general context information
    df_context = pd.DataFrame([
        {"Target": "Spatial Scale", "Context": "Older adults, community-dwelling elderly"},
        {"Target": "Temporal Scale", "Context": "Chronic, aging-related, multi-year processes"},
        {"Target": "Target Variable", "Context": "Alzheimer's disease risk and progression"}
    ])
    
    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_vars.to_excel(writer, sheet_name='Variable_definitions', index=False)
        df_links.to_excel(writer, sheet_name='Variable_links', index=False)
        df_context.to_excel(writer, sheet_name='Context', index=False)
    
    print(f"✅ Created Excel file: {output_path}")
    print(f"   - Variable_definitions: {len(df_vars)} variables")
    print(f"   - Variable_links: {len(df_links)} edges")
    print(f"   - Context: {len(df_context)} metadata rows")


def save_json_files(variables: List[Dict], edges: List[Dict], citation_hints: Dict[Tuple[str, str], List[str]],
                     vars_path: str, edges_path: str, hints_path: str):
    """
    Save companion JSON files.
    """
    print("\n" + "="*80)
    print("STEP 5: Saving JSON Files")
    print("="*80)
    
    # Save variables JSON
    with open(vars_path, 'w', encoding='utf-8') as f:
        json.dump(variables, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved variables: {vars_path} ({len(variables)} variables)")
    
    # Save edges JSON
    # Convert to format matching validation files
    edges_json = {
        "edges": [
            {
                "source": e["source"],
                "target": e["target"],
                "type": e["polarity"].upper()  # POSITIVE or NEGATIVE
            }
            for e in edges
        ]
    }
    with open(edges_path, 'w', encoding='utf-8') as f:
        json.dump(edges_json, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved edges: {edges_path} ({len(edges)} edges)")
    
    # Save citation hints JSON
    # Convert tuple keys to strings for JSON serialization
    hints_json = {
        f"{src} -> {tgt}": citations
        for (src, tgt), citations in citation_hints.items()
    }
    with open(hints_path, 'w', encoding='utf-8') as f:
        json.dump(hints_json, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved citation hints: {hints_path} ({len(citation_hints)} edges with citations)")


def main():
    print("\n" + "="*80)
    print("ALZHEIMER'S CLD CONVERSION TO STANDARD FORMAT")
    print("="*80)
    print(f"\nInput: {INPUT_FILE}")
    print(f"Output: {OUTPUT_EXCEL}")
    print(f"        {OUTPUT_VARS_JSON}")
    print(f"        {OUTPUT_EDGES_JSON}")
    print(f"        {OUTPUT_CITATION_HINTS}")
    
    # Step 1: Parse references
    ref_map = parse_references_sheet(INPUT_FILE)
    
    # Step 2: Parse connections and build citation hints
    edges, citation_hints = parse_connections_sheet(INPUT_FILE, ref_map)
    
    # Step 3: Extract variables
    variables = extract_variables(edges)
    
    # Step 4: Create standard Excel
    create_standard_excel(variables, edges, OUTPUT_EXCEL)
    
    # Step 5: Save JSON files
    save_json_files(variables, edges, citation_hints, OUTPUT_VARS_JSON, OUTPUT_EDGES_JSON, OUTPUT_CITATION_HINTS)
    
    # Summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Variables: {len(variables)}")
    print(f"   Edges: {len(edges)}")
    print(f"   Edges with citations: {len(citation_hints)}")
    print(f"   Total citation references: {sum(len(cits) for cits in citation_hints.values())}")
    
    # Check polarity distribution
    polarity_dist = {}
    for edge in edges:
        pol = edge['polarity']
        polarity_dist[pol] = polarity_dist.get(pol, 0) + 1
    print(f"\n   Polarity distribution:")
    for pol, count in polarity_dist.items():
        print(f"     {pol}: {count}")
    
    print("\n✅ All files created successfully!")
    print("\n📝 Next steps:")
    print("   1. Load CLD into Neo4j using run_discovery_experiment()")
    print("   2. Run fill_in_missing_citations_with_hints() with the citation hints")
    print("   3. Export enhanced CLD with web URLs for judge validation")


if __name__ == "__main__":
    main()




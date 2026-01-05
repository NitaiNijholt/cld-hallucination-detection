#!/usr/bin/env python3
"""
Map CLD Edges to Supporting References

This script reads all edges from the 3 ground truth CLDs and generates
search queries to find supporting references. The references can then be
used for citation-based judging similar to the Ulemans CLD approach.

Usage:
    python map_cld_edges_to_references.py
"""

import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime

# Paths to ground truth CLDs
GT_CLD_DIR = Path(__file__).parent / "ground_truth_CLDs"

CLD_FILES = {
    "Social_norms_and_obesity_prevalence": GT_CLD_DIR / "Social_norms_and_obesity_prevalence.xlsx",
    "Depressive_symptoms_in_response_to_a_stressor": GT_CLD_DIR / "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
    "Emergency_department": GT_CLD_DIR / "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx"
}

# Source paper DOIs for each CLD
SOURCE_PAPERS = {
    "Social_norms_and_obesity_prevalence": {
        "doi": "10.1111/obr.13044",
        "first_author": "Crielaard",
        "year": 2020,
        "title": "Social norms and obesity prevalence: From cohort to system dynamics models"
    },
    "Depressive_symptoms_in_response_to_a_stressor": {
        "doi": "10.1038/s44260-024-00017-9",
        "first_author": "Uleman",
        "year": 2024,
        "title": "Triangulation for causal loop diagrams"
    },
    "Emergency_department": {
        "doi": "10.1007/s41999-023-00816-8",
        "first_author": "Smeekes",
        "year": 2023,
        "title": "A causal loop diagram of older persons' emergency department visits"
    }
}


def clean_variable_name(var_name: str) -> str:
    """Remove questionnaire abbreviations from variable names."""
    # Remove content in parentheses like (PSS), (CERQ-R), etc.
    cleaned = re.sub(r'\s*\([^)]+\)', '', var_name).strip()
    return cleaned


def generate_pubmed_query(source: str, target: str) -> str:
    """Generate a PubMed search query for an edge."""
    source_clean = clean_variable_name(source)
    target_clean = clean_variable_name(target)
    
    # Build query components
    terms = []
    
    # Add source terms
    source_parts = source_clean.lower().split()
    if len(source_parts) > 1:
        terms.append(f'"{source_clean}"[Title/Abstract]')
    else:
        terms.append(f'{source_clean}[Title/Abstract]')
    
    # Add target terms
    target_parts = target_clean.lower().split()
    if len(target_parts) > 1:
        terms.append(f'"{target_clean}"[Title/Abstract]')
    else:
        terms.append(f'{target_clean}[Title/Abstract]')
    
    # Add causal relationship terms
    terms.append('(cause[Title/Abstract] OR effect[Title/Abstract] OR relationship[Title/Abstract] OR influence[Title/Abstract])')
    
    return ' AND '.join(terms)


def generate_semantic_scholar_query(source: str, target: str) -> str:
    """Generate a Semantic Scholar search query for an edge."""
    source_clean = clean_variable_name(source)
    target_clean = clean_variable_name(target)
    return f'{source_clean} {target_clean} causal relationship'


def generate_brave_query(source: str, target: str) -> str:
    """Generate a Brave search query for an edge."""
    source_clean = clean_variable_name(source)
    target_clean = clean_variable_name(target)
    return f'"{source_clean}" AND "{target_clean}" causal relationship scientific study'


def read_cld_edges(cld_name: str, excel_path: Path) -> list:
    """Read edges from a CLD Excel file."""
    df = pd.read_excel(excel_path, sheet_name='Variable_links')
    
    edges = []
    for _, row in df.iterrows():
        source = row['Source']
        target = row['Target']
        polarity = row.get('Polarity', 'unknown')
        
        edge = {
            "cld": cld_name,
            "source": source,
            "target": target,
            "polarity": polarity,
            "source_clean": clean_variable_name(source),
            "target_clean": clean_variable_name(target),
            "search_queries": {
                "pubmed": generate_pubmed_query(source, target),
                "semantic_scholar": generate_semantic_scholar_query(source, target),
                "brave": generate_brave_query(source, target)
            },
            "source_paper": SOURCE_PAPERS.get(cld_name, {})
        }
        edges.append(edge)
    
    return edges


def map_all_cld_edges():
    """Map all edges from all CLDs to search queries."""
    all_edges = []
    
    for cld_name, excel_path in CLD_FILES.items():
        if not excel_path.exists():
            print(f"WARNING: File not found: {excel_path}")
            continue
            
        print(f"\nProcessing {cld_name}...")
        edges = read_cld_edges(cld_name, excel_path)
        all_edges.extend(edges)
        print(f"  Found {len(edges)} edges")
    
    return all_edges


def save_edge_mappings(edges: list, output_path: str = None):
    """Save edge mappings to JSON file."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / f"cld_edges_with_search_queries_{timestamp}.json"
    
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "total_edges": len(edges),
        "edges_by_cld": {},
        "edges": edges
    }
    
    # Count by CLD
    for edge in edges:
        cld = edge["cld"]
        if cld not in output_data["edges_by_cld"]:
            output_data["edges_by_cld"][cld] = 0
        output_data["edges_by_cld"][cld] += 1
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEdge mappings saved to: {output_path}")
    return output_path


def print_summary(edges: list):
    """Print a summary of the edge mappings."""
    print("\n" + "=" * 80)
    print("EDGE MAPPING SUMMARY")
    print("=" * 80)
    
    # Group by CLD
    by_cld = {}
    for edge in edges:
        cld = edge["cld"]
        if cld not in by_cld:
            by_cld[cld] = []
        by_cld[cld].append(edge)
    
    for cld, cld_edges in by_cld.items():
        print(f"\n{cld}:")
        print(f"  Total edges: {len(cld_edges)}")
        print(f"  Source paper DOI: {SOURCE_PAPERS.get(cld, {}).get('doi', 'N/A')}")
        print(f"\n  Sample edges with queries:")
        for edge in cld_edges[:3]:
            print(f"    {edge['source']} -> {edge['target']}")
            print(f"      Brave: {edge['search_queries']['brave'][:80]}...")


def main():
    print("=" * 80)
    print("MAPPING CLD EDGES TO REFERENCE SEARCH QUERIES")
    print("=" * 80)
    
    # Map all edges
    all_edges = map_all_cld_edges()
    
    # Print summary
    print_summary(all_edges)
    
    # Save to JSON
    output_path = save_edge_mappings(all_edges)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Use the search queries to find references for each edge via:
   - PubMed API
   - Semantic Scholar API
   - Brave Search API (as in Ulemans methodology)

2. For each edge, the source paper DOI provides primary evidence

3. Additional references can be fetched using the generated search queries

4. Create Neo4j sessions with edges and their citations for judge testing
""")
    
    return all_edges, output_path


if __name__ == "__main__":
    edges, output_path = main()







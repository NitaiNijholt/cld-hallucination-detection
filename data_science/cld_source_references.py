#!/usr/bin/env python3
"""
CLD Source References for Thesis Motivation

This file documents the source papers for the 3 CLDs used in the thesis experiments,
as selected by Cillian Hourican. These CLDs were chosen because they have published
references backing their causal relationships.

Papers verified on 2024-12-18.
"""

import json
from pathlib import Path

# ============================================================================
# SOURCE PAPER CITATIONS
# ============================================================================

CLD_SOURCE_PAPERS = {
    "Social_norms_and_obesity_prevalence": {
        "full_citation": (
            "Crielaard, L., Dutta, P., Quax, R., Nicolaou, M., Merabet, N., "
            "Stronks, K., & Sloot, P. M. A. (2020). Social norms and obesity "
            "prevalence: From cohort to system dynamics models. Obesity Reviews, "
            "21(9), e13044."
        ),
        "doi": "10.1111/obr.13044",
        "pmcid": "PMC7507199",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7507199/",
        "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/32307880/",
        "year": 2020,
        "journal": "Obesity Reviews",
        "notes": (
            "Uses HELIUS cohort data (Dutch, Moroccan, South-Asian Surinamese groups). "
            "CLD contains 10 variables, 12 edges. Key variables: Individual Ideal BMI, "
            "Group-level BMI, Norm BMI, Socio-cultural Ideal BMI, BMI, TDEE, TDEI, etc."
        ),
        "authors": ["Crielaard L", "Dutta P", "Quax R", "Nicolaou M", "Merabet N", "Stronks K", "Sloot PMA"],
        "cld_stats": {
            "variables": 10,
            "edges": 12
        }
    },
    
    "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults": {
        "full_citation": (
            "Uleman, J. F., Luijten, M., Abdo, W. F., Vyrastekova, J., Gerhardus, A., "
            "Runge, J., Rod, N. H., & Verhagen, M. (2024). Triangulation for causal "
            "loop diagrams: constructing biopsychosocial models using group model "
            "building, literature review, and causal discovery. npj Complexity, 1(1), 1-12."
        ),
        "doi": "10.1038/s44260-024-00017-9",
        "url": "https://www.nature.com/articles/s44260-024-00017-9",
        "year": 2024,
        "journal": "npj Complexity",
        "notes": (
            "Uses NESDA data (Netherlands Study of Depression and Anxiety). "
            "Triangulation approach combining GMB + Literature + Causal Discovery. "
            "CLD contains 14 variables, 35 edges. Variables include specific questionnaire "
            "abbreviations: PSS (Perceived Stress Scale), CERQ-R (Cognitive Emotion "
            "Regulation Questionnaire - Rumination), FFMQ (Five Facet Mindfulness "
            "Questionnaire), MSPSS (Multidimensional Scale of Perceived Social Support), "
            "UCLS (UCLA Loneliness Scale), IDS-SR (Inventory of Depressive Symptomatology), "
            "PSQI (Pittsburgh Sleep Quality Index), SBQ (Sedentary Behavior Questionnaire), "
            "SMQ (Smoking Questionnaire), PGG (Public Goods Game for prosocial behavior)."
        ),
        "authors": ["Uleman JF", "Luijten M", "Abdo WF", "Vyrastekova J", "Gerhardus A", "Runge J", "Rod NH", "Verhagen M"],
        "cld_stats": {
            "variables": 14,
            "edges": 35
        }
    },
    
    "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet": {
        "full_citation": (
            "Smeekes, O. S., Willems, H. C., Blomberg, I., & Buurman, B. M. (2023). "
            "A causal loop diagram of older persons' emergency department visits and "
            "interactions of its contributing factors: a group model building approach. "
            "European Geriatric Medicine, 14(4), 829-837."
        ),
        "doi": "10.1007/s41999-023-00816-8",
        "pmcid": "PMC10447269",
        "pubmed": "37391681",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10447269/",
        "link_springer": "https://link.springer.com/article/10.1007/s41999-023-00816-8",
        "year": 2023,
        "journal": "European Geriatric Medicine",
        "notes": (
            "Amsterdam-based study using Group Model Building (GMB) with interdisciplinary "
            "team of 9 experts across 6 online sessions. Part of DOLCE VITA project. "
            "CLD contains 33 variables, 66 edges, 18 feedback loops. "
            "Four primary factors: Acute events, Frailty, Functioning of healthcare "
            "professionals, Availability of alternatives to ED."
        ),
        "authors": ["Smeekes OS", "Willems HC", "Blomberg I", "Buurman BM"],
        "cld_stats": {
            "variables": 33,
            "edges": 66
        }
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_paper_info(cld_name: str) -> dict:
    """Get source paper information for a CLD."""
    return CLD_SOURCE_PAPERS.get(cld_name, {})

def generate_search_query_for_edge(source: str, target: str, cld_name: str) -> str:
    """
    Generate a search query to find references supporting an edge.
    
    Args:
        source: Source variable name
        target: Target variable name  
        cld_name: Name of the CLD
        
    Returns:
        Search query string for PubMed/Semantic Scholar
    """
    # Clean variable names (remove questionnaire abbreviations in parentheses)
    import re
    source_clean = re.sub(r'\s*\([^)]+\)', '', source).strip()
    target_clean = re.sub(r'\s*\([^)]+\)', '', target).strip()
    
    # Build query
    query = f'"{source_clean}" AND "{target_clean}" AND (causal OR causes OR effect OR relationship)'
    
    return query

def save_references_json(output_path: str = None):
    """Save the reference information to a JSON file."""
    if output_path is None:
        output_path = Path(__file__).parent / "cld_source_references.json"
    
    with open(output_path, 'w') as f:
        json.dump(CLD_SOURCE_PAPERS, f, indent=2)
    
    print(f"References saved to: {output_path}")
    return output_path

def print_all_citations():
    """Print all source paper citations in a formatted way."""
    print("=" * 80)
    print("CLD SOURCE PAPER CITATIONS")
    print("=" * 80)
    
    for cld_name, info in CLD_SOURCE_PAPERS.items():
        print(f"\n{cld_name}:")
        print("-" * 60)
        print(f"Citation: {info['full_citation']}")
        print(f"DOI: {info['doi']}")
        print(f"URL: {info['url']}")
        print(f"Variables: {info['cld_stats']['variables']}, Edges: {info['cld_stats']['edges']}")
        print()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print_all_citations()
    
    # Save to JSON
    json_path = save_references_json()
    
    print("\n" + "=" * 80)
    print("SEARCH QUERY EXAMPLES")
    print("=" * 80)
    
    # Example search queries for a few edges
    example_edges = [
        ("Group-level BMI", "Norm BMI", "Social_norms_and_obesity_prevalence"),
        ("Perceived stress (PSS)", "Depressive symptoms (IDS-SR)", "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults"),
        ("Frailty", "Acute care demand leading to an ED visit", "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet"),
    ]
    
    for source, target, cld in example_edges:
        query = generate_search_query_for_edge(source, target, cld)
        print(f"\nEdge: {source} -> {target}")
        print(f"Query: {query}")







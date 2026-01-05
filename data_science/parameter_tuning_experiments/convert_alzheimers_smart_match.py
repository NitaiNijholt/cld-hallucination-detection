#!/usr/bin/env python3
"""
Smart conversion of Alzheimer's CLD citations using fuzzy matching.

This script:
1. Extracts full citations from References sheet
2. Uses fuzzy matching to map short citations (e.g., "Barth (2000)") to full citations
3. Creates comprehensive citation hints with both short and full text
"""

import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

# Paths
BASE_DIR = Path(__file__).parent / "ground_truth_clds_for_experiments"
INPUT_FILE = BASE_DIR / "alzheimers_supplementary.xlsx"
OUTPUT_CITATION_HINTS = BASE_DIR / "Alzheimers_disease_citation_hints.json"


def extract_author_year_from_full_citation(citation: str) -> Tuple[str, str]:
    """
    Extract first author last name and year from full citation.
    
    Examples:
        "Barth J, Schneider S, von Känel R. ... 2010;72(3):229-38." -> ("Barth", "2010")
        "Åkerstedt, T., ... (2012). Predicting sleep..." -> ("Åkerstedt", "2012")
    """
    # Try format 1: Author(s). (YEAR). Title...
    match = re.search(r'^([A-Za-zÀ-ÿ\s&\'-]+?)[,\s]+.*?\((\d{4}[a-z]?)\)', citation)
    if match:
        author = match.group(1).strip().split()[0]  # First word = first author last name
        year = match.group(2).strip()
        return (author, year)
    
    # Try format 2: Author LastName, ... Title. Journal. YEAR;vol
    match = re.search(r'^([A-Za-zÀ-ÿ\'-]+)', citation)
    if match:
        author = match.group(1).strip()
        # Find year
        year_match = re.search(r'[\s\.](\d{4})[;\.,]', citation)
        if year_match:
            return (author, year_match.group(1))
    
    return (None, None)


def similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(short_citation: str, full_citations: List[str]) -> Tuple[str, float]:
    """
    Find the best matching full citation for a short citation.
    
    Args:
        short_citation: e.g., "Barth (2000)"
        full_citations: List of full citation texts
        
    Returns:
        Tuple of (best_match_citation, confidence_score)
    """
    # Extract author and year from short citation
    match = re.match(r'^([A-Za-zÀ-ÿ\s&\'-]+?)\s*\((\d{4}[a-z]?)\)', short_citation.strip())
    if not match:
        return (None, 0.0)
    
    short_author = match.group(1).strip().lower()
    short_year = match.group(2).strip()
    
    best_match = None
    best_score = 0.0
    
    for full_cite in full_citations:
        full_author, full_year = extract_author_year_from_full_citation(full_cite)
        
        if not full_author or not full_year:
            continue
        
        # Calculate match score
        author_sim = similarity(short_author, full_author.lower())
        
        # Year match (with 1-year tolerance for typos/versioning)
        year_match = 0.0
        try:
            if abs(int(short_year[:4]) - int(full_year[:4])) <= 1:
                year_match = 1.0
            elif abs(int(short_year[:4]) - int(full_year[:4])) <= 2:
                year_match = 0.5
        except:
            pass
        
        # Combined score (weighted)
        score = (author_sim * 0.6) + (year_match * 0.4)
        
        if score > best_score:
            best_score = score
            best_match = full_cite
    
    return (best_match, best_score)


def main():
    print("\n" + "="*80)
    print("SMART CITATION MATCHING FOR ALZHEIMER'S CLD")
    print("="*80)
    
    # Step 1: Load all full citations from References sheet
    print("\n📚 Loading full citations from References sheet...")
    df_ref = pd.read_excel(INPUT_FILE, sheet_name='References', header=0)
    full_citations = df_ref['References'].dropna().tolist()
    print(f"   Found {len(full_citations)} full citations")
    
    # Step 2: Load edges with short citations from Connections sheet
    print("\n🔗 Loading edges from Connections sheet...")
    df_conn = pd.read_excel(INPUT_FILE, sheet_name='Connections', header=1)
    df_conn = df_conn.dropna(subset=['Origin'])
    print(f"   Found {len(df_conn)} edges")
    
    # Step 3: Build citation hints with fuzzy matching
    print("\n🧠 Matching short citations to full citations...")
    citation_hints = {}
    match_stats = {"high_confidence": 0, "medium_confidence": 0, "low_confidence": 0, "no_match": 0}
    
    for idx, row in df_conn.iterrows():
        source = str(row['Origin']).strip()
        target = str(row['Destination']).strip()
        support = str(row['Support']).strip() if pd.notna(row['Support']) else ""
        
        edge_key = f"{source} -> {target}"
        
        if not support or support == 'nan':
            continue
        
        # Split on semicolons to get individual short citations
        short_citations = [c.strip() for c in support.split(';') if c.strip()]
        
        # Match each short citation to full citation
        matched_citations = []
        for short_cite in short_citations:
            full_cite, confidence = find_best_match(short_cite, full_citations)
            
            if confidence >= 0.8:
                match_stats["high_confidence"] += 1
            elif confidence >= 0.6:
                match_stats["medium_confidence"] += 1
            elif confidence >= 0.4:
                match_stats["low_confidence"] += 1
            else:
                match_stats["no_match"] += 1
            
            # Store as dict with both short and full
            matched_citations.append({
                "short": short_cite,
                "full": full_cite if full_cite else short_cite,
                "confidence": round(confidence, 2)
            })
        
        citation_hints[edge_key] = matched_citations
    
    print(f"\n   Processed {len(citation_hints)} edges with citations")
    print(f"   Total citation mappings: {sum(len(v) for v in citation_hints.values())}")
    print(f"\n   Match quality:")
    print(f"      High confidence (≥80%): {match_stats['high_confidence']}")
    print(f"      Medium confidence (60-80%): {match_stats['medium_confidence']}")
    print(f"      Low confidence (40-60%): {match_stats['low_confidence']}")
    print(f"      No match (<40%): {match_stats['no_match']}")
    
    # Step 4: Save citation hints
    print(f"\n💾 Saving citation hints...")
    with open(OUTPUT_CITATION_HINTS, 'w') as f:
        json.dump(citation_hints, f, indent=2)
    
    print(f"   ✅ Saved to {OUTPUT_CITATION_HINTS}")
    
    # Step 5: Show examples
    print(f"\n📋 Sample matched citations:")
    for edge_key in list(citation_hints.keys())[:3]:
        print(f"\n   {edge_key}:")
        for cite_info in citation_hints[edge_key][:2]:
            print(f"      Short: {cite_info['short']}")
            print(f"      Full:  {cite_info['full'][:80]}...")
            print(f"      Confidence: {cite_info['confidence']}")
    
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()




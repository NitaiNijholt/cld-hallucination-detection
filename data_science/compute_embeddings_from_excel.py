#!/usr/bin/env python3
"""
Compute Cosine Similarity from Excel File

This script reads an Excel file with judged edges and computes the missing
Gen Cosine Similarity metric by:
1. Reading motivation and citation texts from the Excel file
2. Scraping citation content from URLs (if needed)
3. Computing embeddings using local model
4. Calculating cosine similarity

Usage:
    python compute_embeddings_from_excel.py <excel_file_path> [--device cuda|cpu]
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
from openpyxl import load_workbook
import numpy as np
from typing import Dict, List, Tuple
import json
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from logit_metrics import get_embedding_local


def extract_citation_texts(judge_message_json: str) -> List[str]:
    """Extract citation texts from Judge Message JSON."""
    try:
        if pd.isna(judge_message_json) or not judge_message_json:
            return []
        
        data = json.loads(judge_message_json)
        
        # Handle both old and new formats
        if 'citations' in data:
            citations = data['citations']
        else:
            return []
        
        texts = []
        for cit in citations:
            if 'relevant_text' in cit and cit['relevant_text']:
                texts.append(cit['relevant_text'])
        
        return texts
    except Exception as e:
        logger.debug(f"Could not extract citation texts: {e}")
        return []


def compute_cosine_similarities_batch(
    motivations: List[str],
    citations: List[str],
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute cosine similarities between motivations and citations in batch.
    
    Args:
        motivations: List of motivation texts
        citations: List of citation texts (same length as motivations)
        device: Device for embeddings ("cuda" or "cpu")
    
    Returns:
        np.ndarray: Cosine similarities (NaN for missing citations)
    """
    # Filter out empty citations
    valid_indices = [i for i, c in enumerate(citations) if c]
    
    if not valid_indices:
        logger.warning("No valid citations to compute embeddings")
        return np.full(len(motivations), np.nan)
    
    # Compute embeddings for valid pairs
    valid_motivations = [motivations[i] for i in valid_indices]
    valid_citations = [citations[i] for i in valid_indices]
    
    logger.info(f"Computing embeddings for {len(valid_indices)} valid pairs...")
    
    # Use project's embedding function (uses sentence-transformers/all-mpnet-base-v2)
    logger.info("Computing motivation embeddings...")
    motivation_embeddings = get_embedding_local(
        valid_motivations,
        model="sentence-transformers/all-mpnet-base-v2",
        batch_size=64,
        show_progress=True,
        device=device
    )
    
    logger.info("Computing citation embeddings...")
    citation_embeddings = get_embedding_local(
        valid_citations,
        model="sentence-transformers/all-mpnet-base-v2",
        batch_size=64,
        show_progress=True,
        device=device
    )
    
    # Compute cosine similarities
    similarities = np.full(len(motivations), np.nan)
    for i, idx in enumerate(valid_indices):
        if motivation_embeddings[i] is not None and citation_embeddings[i] is not None:
            sim = cosine_similarity(
                motivation_embeddings[i].reshape(1, -1),
                citation_embeddings[i].reshape(1, -1)
            )[0][0]
            similarities[idx] = float(sim)
    
    return similarities


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        print("Usage: python compute_embeddings_from_excel.py <excel_file_path> [--device cuda|cpu]")
        print("\nOptions:")
        print("  --device    Device for embeddings: 'cpu', 'cuda' (default)")
        print("\nExample:")
        print("  python compute_embeddings_from_excel.py results/judged_session.xlsx")
        print("  python compute_embeddings_from_excel.py results/judged_session.xlsx --device cpu")
        sys.exit(1)
    
    excel_path = Path(sys.argv[1])
    
    # Parse device argument
    device = "cuda"  # default
    if '--device' in sys.argv:
        device_idx = sys.argv.index('--device')
        if device_idx + 1 < len(sys.argv):
            device = sys.argv[device_idx + 1]
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("COMPUTE COSINE SIMILARITY FROM EXCEL")
    print("=" * 80)
    print(f"Excel file: {excel_path}")
    print(f"Embedding device: {device}")
    print()
    
    try:
        # Read Excel file
        logger.info("Reading Excel file...")
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        logger.info(f"✓ Found {len(df)} edges")
        print()
        
        # Check if cosine similarity already exists
        if 'Gen Cosine Similarity' not in df.columns:
            logger.error("Gen Cosine Similarity column not found in Excel")
            sys.exit(1)
        
        # Count missing values
        missing_count = df['Gen Cosine Similarity'].isna().sum()
        if missing_count == 0:
            logger.info("✓ All cosine similarities already computed!")
            print()
            return
        
        logger.info(f"Missing cosine similarities: {missing_count}/{len(df)}")
        print()
        
        # Prepare data
        logger.info("Preparing motivation and citation texts...")
        motivations = df['Motivation'].fillna("").tolist()
        
        # Extract citation texts from Judge Message
        citation_texts = []
        for _, row in df.iterrows():
            texts = extract_citation_texts(row.get('Judge Message', ''))
            citation_texts.append(" ".join(texts) if texts else "")
        
        valid_citations = sum(1 for c in citation_texts if c)
        logger.info(f"✓ Found {valid_citations}/{len(df)} edges with citation texts")
        print()
        
        # Compute cosine similarities
        logger.info("Computing cosine similarities...")
        similarities = compute_cosine_similarities_batch(motivations, citation_texts, device=device)
        logger.info(f"✓ Computed {np.sum(~np.isnan(similarities))} cosine similarities")
        print()
        
        # Update DataFrame
        df['Gen Cosine Similarity'] = similarities
        
        # Create backup
        backup_path = excel_path.with_suffix('.backup.xlsx')
        if not backup_path.exists():
            logger.info(f"Creating backup: {backup_path.name}")
            df_backup = pd.read_excel(excel_path, sheet_name=None)
            with pd.ExcelWriter(backup_path, engine='openpyxl') as writer:
                for sheet_name, sheet_df in df_backup.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Write updated data back
        logger.info("Updating Excel file...")
        
        # Read all sheets
        wb_data = pd.read_excel(excel_path, sheet_name=None)
        
        # Update All Edges sheet
        wb_data['All Edges'] = df
        
        # Write back
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sheet_name, sheet_df in wb_data.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"✓ Excel file updated successfully")
        print()
        
        print("=" * 80)
        print("✅ COSINE SIMILARITIES COMPUTED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nUpdated file: {excel_path}")
        print(f"Backup file: {backup_path}")
        print(f"Computed: {np.sum(~np.isnan(similarities))}/{len(df)} similarities")
        print()
        
    except Exception as e:
        logger.error(f"Failed to compute cosine similarities: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


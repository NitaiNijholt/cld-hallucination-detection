#!/usr/bin/env python3
"""
Add CI Metrics to Existing Excel File

This script takes an existing judged Excel file and computes CI (Context-Insensitive) 
metrics and embeddings, adding them to the Excel file.

This is useful when you ran judging with ci_compute_embeddings=False for speed,
and now want to add the expensive metrics afterwards.

Usage:
    python add_ci_metrics_to_excel.py <excel_file_path>
    
Example:
    python add_ci_metrics_to_excel.py parameter_tuning_experiments/results/rq1_base_judging_correctness_20251011_031233/judged_52ea7da0-247f-4faa-8b68-2813d5d5346b_base_correctness.xlsx
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from modules import CausalDiscovery, _compute_edge_level_context_insensitive_metrics


def read_session_id_from_excel(excel_path: Path) -> str:
    """Extract session_id from the Params sheet of an Excel file."""
    try:
        df_params = pd.read_excel(excel_path, sheet_name='Params')
        # Find the row where Parameter == 'session_id'
        session_id_row = df_params[df_params['Parameter'] == 'session_id']
        if session_id_row.empty:
            raise ValueError("session_id not found in Params sheet")
        session_id = session_id_row['Value'].iloc[0]
        return session_id
    except Exception as e:
        logger.error(f"Failed to read session_id from Excel: {e}")
        raise


def compute_ci_metrics_for_session(session_id: str, compute_embeddings: bool = True, embedding_provider: str = "local", fill_citations: bool = True, embedding_device: str = "cuda") -> dict:
    """
    Compute CI metrics for a given session.
    
    Args:
        session_id: Neo4j session ID
        compute_embeddings: Whether to compute expensive embeddings (default: True)
        embedding_provider: "openai" or "local" for embeddings (default: "local")
        fill_citations: Whether to fill missing citations via Brave search (default: True)
        embedding_device: Device for local embeddings: "cpu", "cuda", "cuda:0", etc. (default: "cuda")
    
    Returns:
        Dictionary mapping (source, target) tuples to CI metrics
    """
    logger.info(f"Computing CI metrics for session: {session_id} (provider: {embedding_provider}, device: {embedding_device})")
    
    # Create a minimal CausalDiscovery instance just for CI metrics computation
    # We use dummy values since we only need the graph_db connection
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), "../backend/configs/prompts.yaml")
    discovery = CausalDiscovery(
        target_variable="dummy",
        temporal_scale="dummy",
        spatial_scale="dummy",
        yaml_path=yaml_path
    )
    
    # Fill missing citations if requested
    if fill_citations:
        logger.info("Filling missing citations via Brave search...")
        try:
            # Temporarily set the session_id to the correct one
            old_session_id = discovery.session_id
            discovery.session_id = session_id
            
            discovery.fill_in_missing_citations(
                llm_provider=None,  # Skip LLM for speed
                llm_model=None,
                search_provider="brave",
                citation_fill_parallel=True,
                citation_fill_max_workers=5
            )
            
            # Restore original session_id
            discovery.session_id = old_session_id
            logger.info("✓ Citations filled successfully")
        except Exception as e:
            logger.warning(f"Citation filling failed: {e}. Continuing with CI metrics...")
    
    # Compute CI metrics
    # Using all-mpnet-base-v2: small model (420MB) can handle parallel processing
    ci_metrics = _compute_edge_level_context_insensitive_metrics(
        discovery=discovery,
        session_id=session_id,
        ci_fetch_reuse=True,
        ci_chunk_chars_override=None,
        ci_overlap_ratio=0.2,
        ci_compute_embeddings=compute_embeddings,
        ci_parallel=True,
        ci_max_workers=10,
        embedding_provider=embedding_provider,
        embedding_device=embedding_device
    )
    
    return ci_metrics


def update_excel_with_ci_metrics(excel_path: Path, ci_metrics: dict, backup: bool = True):
    """
    Update the Excel file with CI metrics.
    
    This function works with Excel files that have CI columns but empty/NaN values
    (typically from running with ci_compute_embeddings=False).
    
    Args:
        excel_path: Path to the Excel file
        ci_metrics: Dictionary of CI metrics keyed by (source, target)
        backup: Whether to create a backup of the original file
    """
    logger.info(f"Updating Excel file with CI metrics: {excel_path}")
    
    # Create backup if requested
    if backup:
        backup_path = excel_path.with_suffix('.backup.xlsx')
        if not backup_path.exists():
            import shutil
            shutil.copy2(excel_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
    
    # Load the workbook
    xl = pd.ExcelFile(excel_path)
    
    # Process each sheet that contains edges
    edge_sheets = [name for name in xl.sheet_names if 'Edges' in name or name == 'All Edges']
    
    # Track statistics
    total_sheets = 0
    total_edges = 0
    total_updated = 0
    
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for sheet_name in edge_sheets:
            logger.info(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Count empty CI columns before
            ci_check_cols = ['Cosine Similarity', 'Gen Cosine Similarity']
            before_filled = sum(df[col].notna().sum() for col in ci_check_cols if col in df.columns)
            before_total = len(df) * len([c for c in ci_check_cols if c in df.columns])
            
            total_sheets += 1
            total_edges += len(df)
            
            # Add CI metric columns if they don't exist
            ci_columns = [
                'Perplexity', 'Min Prob', 'Max Window Entropy',
                'Gen Perplexity', 'Gen Min Prob', 'Gen Max Window Entropy',
                'Judge Perplexity', 'Judge Min Prob', 'Judge Max Window Entropy',
                'Cosine Similarity', 'Gen Cosine Similarity', 'Judge Cosine Similarity'
            ]
            
            # Update rows with CI metrics
            updated_count = 0
            for idx, row in df.iterrows():
                # Skip rows without Source/Target columns (e.g., summary sheets)
                if 'Source' not in row or 'Target' not in row:
                    continue
                source = row['Source']
                target = row['Target']
                edge_key = (source, target)
                
                if edge_key in ci_metrics:
                    metrics = ci_metrics[edge_key]
                    
                    # Update DataFrame with metrics
                    if metrics.get('perplexity') is not None:
                        df.at[idx, 'Perplexity'] = metrics['perplexity']
                    if metrics.get('min_prob') is not None:
                        df.at[idx, 'Min Prob'] = metrics['min_prob']
                    if metrics.get('max_window_entropy') is not None:
                        df.at[idx, 'Max Window Entropy'] = metrics['max_window_entropy']
                    
                    if metrics.get('gen_perplexity') is not None:
                        df.at[idx, 'Gen Perplexity'] = metrics['gen_perplexity']
                    if metrics.get('gen_min_prob') is not None:
                        df.at[idx, 'Gen Min Prob'] = metrics['gen_min_prob']
                    if metrics.get('gen_max_window_entropy') is not None:
                        df.at[idx, 'Gen Max Window Entropy'] = metrics['gen_max_window_entropy']
                    
                    if metrics.get('judge_perplexity') is not None:
                        df.at[idx, 'Judge Perplexity'] = metrics['judge_perplexity']
                    if metrics.get('judge_min_prob') is not None:
                        df.at[idx, 'Judge Min Prob'] = metrics['judge_min_prob']
                    if metrics.get('judge_max_window_entropy') is not None:
                        df.at[idx, 'Judge Max Window Entropy'] = metrics['judge_max_window_entropy']
                    
                    if metrics.get('cosine_similarity') is not None:
                        df.at[idx, 'Cosine Similarity'] = metrics['cosine_similarity']
                    if metrics.get('gen_cosine_similarity') is not None:
                        df.at[idx, 'Gen Cosine Similarity'] = metrics['gen_cosine_similarity']
                    if metrics.get('judge_cosine_similarity') is not None:
                        df.at[idx, 'Judge Cosine Similarity'] = metrics['judge_cosine_similarity']
                    
                    updated_count += 1
            
            total_updated += updated_count
            
            # Write updated DataFrame back to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Count filled columns after
            after_filled = sum(df[col].notna().sum() for col in ci_check_cols if col in df.columns)
            newly_filled = after_filled - before_filled
            
            logger.info(f"  Updated {updated_count}/{len(df)} edges in sheet '{sheet_name}'")
            if newly_filled > 0:
                logger.info(f"  ✓ Filled {newly_filled} new CI metric values")
    
    logger.info(f"\n📊 Summary: Updated {total_updated} edges across {total_sheets} sheets")
    return total_updated


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        print("Usage: python add_ci_metrics_to_excel.py <excel_file_path> [--no-embeddings] [--no-citations] [--provider local|openai] [--device cpu|cuda] [--session-id <id>]")
        print("\nOptions:")
        print("  --no-embeddings    Skip expensive embedding computations (faster)")
        print("  --no-citations     Skip citation filling (use if citations already exist)")
        print("  --provider         Embedding provider: 'local' (default) or 'openai'")
        print("  --device           Device for local embeddings: 'cpu', 'cuda' (default), 'cuda:0', etc.")
        print("  --session-id       Manually specify session ID (if Excel lacks Params sheet)")
        print("\nExample:")
        print("  python add_ci_metrics_to_excel.py results/judged_session.xlsx")
        print("  python add_ci_metrics_to_excel.py results/judged_session.xlsx --provider local --device cuda")
        print("  python add_ci_metrics_to_excel.py results/judged_session.xlsx --device cpu")
        print("  python add_ci_metrics_to_excel.py results/judged_session.xlsx --no-citations")
        print("  python add_ci_metrics_to_excel.py results/judged_session.xlsx --session-id abc-123-def")
        sys.exit(1)
    
    excel_path = Path(sys.argv[1])
    compute_embeddings = '--no-embeddings' not in sys.argv
    fill_citations = '--no-citations' not in sys.argv
    
    # Parse provider argument
    embedding_provider = "local"  # default
    if '--provider' in sys.argv:
        provider_idx = sys.argv.index('--provider')
        if provider_idx + 1 < len(sys.argv):
            embedding_provider = sys.argv[provider_idx + 1]
            if embedding_provider not in ["local", "openai"]:
                logger.error(f"Invalid provider: {embedding_provider}. Must be 'local' or 'openai'")
                sys.exit(1)
    
    # Parse device argument
    embedding_device = "cuda"  # default
    if '--device' in sys.argv:
        device_idx = sys.argv.index('--device')
        if device_idx + 1 < len(sys.argv):
            embedding_device = sys.argv[device_idx + 1]
    
    # Parse session-id argument
    manual_session_id = None
    if '--session-id' in sys.argv:
        session_id_idx = sys.argv.index('--session-id')
        if session_id_idx + 1 < len(sys.argv):
            manual_session_id = sys.argv[session_id_idx + 1]
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("ADD CI METRICS TO EXCEL FILE")
    print("=" * 80)
    print(f"Excel file: {excel_path}")
    print(f"Fill citations: {fill_citations}")
    print(f"Compute embeddings: {compute_embeddings}")
    print(f"Embedding provider: {embedding_provider}")
    print(f"Embedding device: {embedding_device}")
    if manual_session_id:
        print(f"Session ID (manual): {manual_session_id}")
    print()
    
    try:
        # Step 1: Read or use provided session_id
        if manual_session_id:
            logger.info(f"Step 1: Using provided session_id: {manual_session_id}")
            session_id = manual_session_id
        else:
            logger.info("Step 1: Reading session_id from Excel...")
            session_id = read_session_id_from_excel(excel_path)
            logger.info(f"✓ Found session_id: {session_id}")
        print()
        
        # Step 2: Compute CI metrics (and fill citations if needed)
        logger.info("Step 2: Computing CI metrics (and filling citations if needed)...")
        ci_metrics = compute_ci_metrics_for_session(session_id, compute_embeddings=compute_embeddings, embedding_provider=embedding_provider, fill_citations=fill_citations, embedding_device=embedding_device)
        logger.info(f"✓ Computed CI metrics for {len(ci_metrics)} edges")
        print()
        
        # Step 3: Update Excel file
        logger.info("Step 3: Updating Excel file...")
        update_excel_with_ci_metrics(excel_path, ci_metrics, backup=True)
        logger.info(f"✓ Excel file updated successfully")
        print()
        
        print("=" * 80)
        print("✅ CI METRICS ADDED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nUpdated file: {excel_path}")
        print(f"Backup file: {excel_path.with_suffix('.backup.xlsx')}")
        print()
        
    except Exception as e:
        logger.error(f"Failed to add CI metrics: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

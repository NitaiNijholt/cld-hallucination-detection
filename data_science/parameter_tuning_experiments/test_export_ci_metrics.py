#!/usr/bin/env python3
"""
Test script to export a test CLD session to Excel and verify generator CI metrics.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.dev", override=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import export_edges_comparison_to_excel
from backend.lm_core.causal_discovery import CausalDiscovery

def main():
    # Test session ID from seed_1
    session_id = "6efe8a11-920e-41c1-83c6-e1c4bbfb03e0"
    
    # Validation files for test CLD
    script_dir = Path(__file__).parent
    validation_dir = script_dir / "test_CLD"
    validation_vars = str(validation_dir / "CLD_test_vars_data.json")
    validation_edges = str(validation_dir / "CLD_test_edges_data.json")
    
    # Output directory
    output_dir = script_dir / "test_output"
    output_dir.mkdir(exist_ok=True)
    output_file = str(output_dir / f"test_cld_session_{session_id[:8]}_ci_metrics.xlsx")
    
    print(f"\n{'='*80}")
    print("EXPORTING TEST CLD SESSION TO EXCEL")
    print(f"{'='*80}")
    print(f"Session ID: {session_id}")
    print(f"Output file: {output_file}")
    print(f"Validation vars: {validation_vars}")
    print(f"Validation edges: {validation_edges}")
    print()
    
    # Create CausalDiscovery instance
    discovery = CausalDiscovery()
    discovery.session_id = session_id
    
    # Export to Excel with CI metrics
    print("Exporting to Excel with CI metrics...")
    result = export_edges_comparison_to_excel(
        discovery=discovery,
        session_ids=session_id,
        validation_vars_json_path=validation_vars,
        validation_edges_json_path=validation_edges,
        output_filename=output_file,
        embedding_enable=True,  # Enable CI metrics
        ci_compute_embeddings=True,  # Compute embeddings
        embedding_provider="local",  # Use local embeddings
        embedding_device="cuda",  # Use GPU
        ci_parallel=True,  # Parallel processing
        ci_max_workers=10
    )
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"Excel file: {result}")
    print()
    print("To verify generator CI metrics:")
    print(f"  1. Open the Excel file: {result}")
    print("  2. Check the 'All Edges' sheet")
    print("  3. Look for columns: Gen Perplexity, Gen Min Prob, Gen Mean Token Prob, etc.")
    print("  4. Verify these columns have non-null values for edges with citations")
    print()

if __name__ == "__main__":
    main()

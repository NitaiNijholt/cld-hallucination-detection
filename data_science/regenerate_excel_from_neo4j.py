#!/usr/bin/env python3
"""
Regenerate Excel file from existing Neo4j data without rerunning experiments.
This saves API costs by just re-exporting the data that's already been computed.
"""

import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from modules import CausalDiscovery
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def regenerate_excel_for_session(session_id: str, output_excel_path: str, 
                                  validation_vars_path: str = None,
                                  validation_edges_path: str = None):
    """
    Regenerate the Excel export for a given session_id without rerunning the experiment.
    
    Args:
        session_id: The Neo4j session ID
        output_excel_path: Path where to save the Excel file
        validation_vars_path: Path to validation variables JSON (optional)
        validation_edges_path: Path to validation edges JSON (optional)
    """
    print(f"🔄 Regenerating Excel for session: {session_id}")
    print(f"📂 Output file: {output_excel_path}")
    
    # Create a minimal CausalDiscovery object just to access Neo4j
    disco = CausalDiscovery(
        generator_config={"provider": "openai", "model": "gpt-4o"},  # Dummy, won't be used
        db_name="neo4j"
    )
    
    # Export to Excel
    try:
        disco.export_edges_comparison_to_excel(
            session_id=session_id,
            output_file=output_excel_path,
            validation_vars_path=validation_vars_path,
            validation_edges_path=validation_edges_path
        )
        print(f"✅ Excel file regenerated successfully!")
        print(f"📊 Location: {output_excel_path}")
        
        # Quick sanity check
        df = pd.read_excel(output_excel_path, sheet_name='All Edges')
        print(f"\n📈 Quick Stats:")
        print(f"   Total edges: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        if 'Relationship Type' in df.columns:
            print(f"   Relationship types: {df['Relationship Type'].value_counts().to_dict()}")
        if 'Classification' in df.columns:
            print(f"   Classifications: {df['Classification'].value_counts().to_dict()}")
            
    except Exception as e:
        logger.error(f"❌ Failed to regenerate Excel: {e}")
        logger.exception("Full traceback:")
        return False
    
    return True


if __name__ == "__main__":
    # Find the most recent Test CLD experiment
    import glob
    
    results_dir = Path("/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results")
    
    # Look for the most recent test CLD Excel file
    test_cld_files = list(results_dir.glob("results_exp_*_CLD_test*.xlsx"))
    
    if not test_cld_files:
        print("❌ No Test CLD Excel files found!")
        print("Please provide session_id manually.")
        sys.exit(1)
    
    # Sort by modification time, get most recent
    most_recent = max(test_cld_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Found most recent Test CLD file: {most_recent.name}")
    
    # Try to extract session ID from the file
    # The file typically has metadata or we can search for it
    try:
        df_session = pd.read_excel(most_recent, sheet_name='Session Graph')
        print(f"✅ File contains {len(df_session)} edges in Session Graph")
    except Exception as e:
        print(f"⚠️  Could not read Session Graph: {e}")
    
    # We need to get the session_id from the experiment metadata
    # Let's search for the param combo mapping that corresponds to this file
    exp_id = most_recent.stem.split('_')[1:6]  # Extract exp_YYYYMMDD_HHMMSS_hash
    exp_id_str = '_'.join(exp_id)
    
    print(f"\n🔍 Looking for experiment: {exp_id_str}")
    
    # Find the metadata CSV
    metadata_file = results_dir / f"param_combo_mapping_{exp_id_str}.csv"
    
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        print("Please provide session_id and paths manually:")
        print("python regenerate_excel_from_neo4j.py <session_id> <output_excel> [validation_vars] [validation_edges]")
        sys.exit(1)
    
    # Read metadata to get session_id
    metadata = pd.read_csv(metadata_file)
    print(f"✅ Found metadata with {len(metadata)} runs")
    
    if len(metadata) == 0:
        print("❌ Metadata is empty!")
        sys.exit(1)
    
    # Get the first run (or you can filter by CLD name)
    test_cld_rows = metadata[metadata['cld_name'] == 'test']
    
    if len(test_cld_rows) == 0:
        print("❌ No 'test' CLD found in metadata!")
        print("Available CLDs:", metadata['cld_name'].unique())
        sys.exit(1)
    
    first_test_row = test_cld_rows.iloc[0]
    session_id = first_test_row['session_id']
    
    print(f"\n🆔 Session ID: {session_id}")
    
    # Get validation file paths from the same CLD
    cld_name = first_test_row['cld_name']
    validation_vars_path = f"/home/nitai/code/causalix.ai/data_science/causal_loop_diagrams/Variables_{cld_name}.json"
    validation_edges_path = f"/home/nitai/code/causalix.ai/data_science/causal_loop_diagrams/Edges_{cld_name}.json"
    
    print(f"📂 Validation vars: {validation_vars_path}")
    print(f"📂 Validation edges: {validation_edges_path}")
    
    # Create output path
    output_excel = results_dir / f"results_{exp_id_str}_CLD_{cld_name}_REGENERATED.xlsx"
    
    # Regenerate!
    success = regenerate_excel_for_session(
        session_id=session_id,
        output_excel_path=str(output_excel),
        validation_vars_path=validation_vars_path,
        validation_edges_path=validation_edges_path
    )
    
    if success:
        print(f"\n✅ SUCCESS! Excel regenerated from existing Neo4j data.")
        print(f"📊 File: {output_excel}")
    else:
        print(f"\n❌ FAILED to regenerate Excel.")
        sys.exit(1)

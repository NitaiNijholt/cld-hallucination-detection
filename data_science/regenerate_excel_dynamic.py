#!/usr/bin/env python3
"""
Regenerate Excel from Neo4j using a session ID with dynamic CLD detection.
Usage: 
    python regenerate_excel_dynamic.py <session_id>
    python regenerate_excel_dynamic.py <session_id> --cld <cld_name>
    python regenerate_excel_dynamic.py <session_id> --list-clds
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from neo4j import GraphDatabase

# Enable more verbose logging to see where it hangs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, str(Path(__file__).parent))

from modules import export_edges_comparison_to_excel, CausalDiscovery

# Base path for ground truth CLDs
GROUND_TRUTH_PATH = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments"

# Mapping of CLD names to their file prefixes
CLD_MAPPING = {
    "social_norms": "Social_norms_and_obesity_prevalence",
    "depressive_symptoms": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
    "older_persons": "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet",
}

def list_available_clds():
    """List all available CLDs with JSON files."""
    print("\n📋 Available CLDs:")
    for short_name, file_prefix in CLD_MAPPING.items():
        vars_file = os.path.join(GROUND_TRUTH_PATH, f"{file_prefix}_vars_data.json")
        edges_file = os.path.join(GROUND_TRUTH_PATH, f"{file_prefix}_edges_data.json")
        status = "✅" if os.path.exists(vars_file) and os.path.exists(edges_file) else "❌"
        print(f"  {status} {short_name}: {file_prefix}")
    print()

def get_cld_from_filesystem(session_id):
    """Detect CLD by searching for session JSON files in results directory."""
    results_base = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results"
    
    if not os.path.exists(results_base):
        return None
    
    # Search for JSON files containing this session_id
    import glob
    pattern = os.path.join(results_base, "**", f"*{session_id[:8]}*.json")
    json_files = glob.glob(pattern, recursive=True)
    
    if not json_files:
        # Try searching with full session ID
        pattern = os.path.join(results_base, "**", "*.json")
        all_json_files = glob.glob(pattern, recursive=True)
        
        # Check which files contain this session_id
        for json_file in all_json_files:
            try:
                with open(json_file, 'r') as f:
                    import json
                    data = json.load(f)
                    if data.get('session_id') == session_id:
                        json_files = [json_file]
                        break
            except:
                continue
    
    if json_files:
        # Extract CLD name from directory path
        json_path = json_files[0]
        path_parts = json_path.split(os.sep)
        
        # Look for CLD name in path components
        for part in path_parts:
            for short_name, file_prefix in CLD_MAPPING.items():
                if file_prefix in part:
                    return short_name
    
    return None

def get_cld_from_session(session_id):
    """Detect CLD from session - tries filesystem first, then Neo4j."""
    
    # First try filesystem (more reliable)
    print("  Checking filesystem for session...")
    cld_name = get_cld_from_filesystem(session_id)
    if cld_name:
        return cld_name
    
    # Fallback to Neo4j query
    print("  Querying Neo4j for session metadata...")
    try:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Try to find the excel_path in node properties
            query = """
            MATCH (n)
            WHERE n.session_id = $session_id AND n.excel_path IS NOT NULL
            RETURN n.excel_path as excel_path
            LIMIT 1
            """
            result = session.run(query, session_id=session_id)
            record = result.single()
            
            if record and record["excel_path"]:
                excel_path = record["excel_path"]
                driver.close()
                
                # Extract CLD name from Excel path
                excel_basename = os.path.basename(excel_path)
                excel_name = os.path.splitext(excel_basename)[0]
                
                # Find matching CLD
                for short_name, file_prefix in CLD_MAPPING.items():
                    if file_prefix in excel_name or excel_name in file_prefix:
                        return short_name
                
                print(f"⚠️  Found Excel path but couldn't match to known CLD: {excel_path}")
                return None
        
        driver.close()
    except Exception as e:
        print(f"⚠️  Could not query Neo4j for session metadata: {e}")
    
    return None

def get_cld_paths(cld_name):
    """Get validation file paths for a given CLD name."""
    if cld_name not in CLD_MAPPING:
        print(f"❌ Unknown CLD: {cld_name}")
        print(f"Available CLDs: {', '.join(CLD_MAPPING.keys())}")
        return None, None
    
    file_prefix = CLD_MAPPING[cld_name]
    vars_path = os.path.join(GROUND_TRUTH_PATH, f"{file_prefix}_vars_data.json")
    edges_path = os.path.join(GROUND_TRUTH_PATH, f"{file_prefix}_edges_data.json")
    
    # Check if files exist
    if not os.path.exists(vars_path):
        print(f"❌ Variables file not found: {vars_path}")
        print(f"💡 Generate it with: cd data_loading_scripts && python -c \"from excel_cld_loader_v3 import load_cld_from_excel; load_cld_from_excel('{GROUND_TRUTH_PATH}/{file_prefix}.xlsx', export_json=True)\"")
        return None, None
    
    if not os.path.exists(edges_path):
        print(f"❌ Edges file not found: {edges_path}")
        return None, None
    
    return vars_path, edges_path

def main():
    parser = argparse.ArgumentParser(description="Regenerate Excel from Neo4j session with dynamic CLD detection")
    parser.add_argument("session_id", help="Neo4j session ID")
    parser.add_argument("--cld", help="CLD name (e.g., social_norms, depressive_symptoms, older_persons)")
    parser.add_argument("--list-clds", action="store_true", help="List available CLDs")
    parser.add_argument("--output", help="Output Excel file path (optional)")
    
    args = parser.parse_args()
    
    if args.list_clds:
        list_available_clds()
        return
    
    session_id = args.session_id
    print(f"🆔 Session ID: {session_id}")
    
    # Determine which CLD to use
    cld_name = args.cld
    
    if not cld_name:
        print("🔍 Auto-detecting CLD from session metadata...")
        cld_name = get_cld_from_session(session_id)
        
        if not cld_name:
            print("\n❌ Could not auto-detect CLD. Please specify with --cld")
            list_available_clds()
            sys.exit(1)
        
        print(f"✅ Detected CLD: {cld_name}")
    else:
        print(f"📌 Using specified CLD: {cld_name}")
    
    # Get validation file paths
    validation_vars, validation_edges = get_cld_paths(cld_name)
    
    if not validation_vars or not validation_edges:
        sys.exit(1)
    
    print(f"📁 Variables: {os.path.basename(validation_vars)}")
    print(f"📁 Edges: {os.path.basename(validation_edges)}")
    
    # Set output path
    if args.output:
        output_excel = args.output
    else:
        output_excel = f"/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/session_{session_id[:8]}_{cld_name}_REGENERATED.xlsx"
    
    print(f"📂 Output: {output_excel}")
    
    # Create discovery object
    disco = CausalDiscovery(
        target_variable="dummy",
        temporal_scale="dummy",
        spatial_scale="dummy",
        yaml_path="/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/prompts_Nitai_C.yaml",
        generator_config={"provider": "openai", "model": "gpt-4o"}
    )
    
    # Export
    print("\n📊 Exporting to Excel...")
    print("✅ Context-insensitive metrics ENABLED (embeddings + cosine similarity)")
    print("\n🔍 PROGRESS TRACKING:")
    print("  [1/7] Initializing export...")
    
    import time
    start_time = time.time()
    
    try:
        # Pre-compute node comparison ONCE to avoid redundant LLM calls for each scenario
        print("  [2/7] Computing node comparison (once for all scenarios)...")
        from modules import compare_session_graph_to_validation
        
        node_comparison_start = time.time()
        initial_metrics = compare_session_graph_to_validation(
            discovery=disco,
            session_ids=session_id,
            validation_vars_json_path=validation_vars,
            validation_edges_json_path=validation_edges,
            plot_session_graph=False,
            plot_validation_graph=False,
            only_cited_edges=False,  # Use "All Edges" scenario for initial comparison
            only_consistent_edges=False,
            node_match_mode="llm_batch",
        )
        
        # Extract node comparison data to reuse across all scenarios
        node_comparison_data = {
            "session_nodes": initial_metrics["session_nodes"],
            "validation_nodes": initial_metrics["validation_nodes"],
            "matched_pairs": [(k, v) for k, v in initial_metrics.get("node_mapping", {}).items()],
            "node_precision": initial_metrics.get("node_precision"),
            "node_recall": initial_metrics.get("node_recall"),
            "node_f1": initial_metrics.get("node_f1"),
        }
        
        node_comparison_elapsed = time.time() - node_comparison_start
        print(f"     ✅ Node comparison complete in {node_comparison_elapsed:.1f}s")
        print(f"     Matched {len(node_comparison_data['matched_pairs'])} node pairs")
        
        # Add progress tracking by checking file modification time
        import threading
        def progress_monitor():
            last_log_time = time.time()
            while not stop_monitoring:
                time.sleep(5)  # Check every 5 seconds
                elapsed = time.time() - start_time
                if time.time() - last_log_time >= 5:
                    print(f"  ... Still processing (elapsed: {elapsed:.1f}s)")
                    last_log_time = time.time()
        
        stop_monitoring = False
        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()
        
        print("  [3/7] Generating Excel (reusing node comparison for all scenarios)...")
        export_edges_comparison_to_excel(
            discovery=disco,
            session_ids=session_id,
            validation_vars_json_path=validation_vars,
            validation_edges_json_path=validation_edges,
            output_filename=output_excel,
            embedding_enable=True,  # Enable context-insensitive embeddings (cosine similarity)
            node_comparison_data=node_comparison_data,  # Pass pre-computed node comparison
        )
        
        stop_monitoring = True
        elapsed = time.time() - start_time
        print(f"\n✅ Success! File saved to:")
        print(f"   {output_excel}")
        print(f"⏱️  Total time: {elapsed:.1f}s")
    except KeyboardInterrupt:
        stop_monitoring = True
        print(f"\n\n⚠️  Interrupted by user after {time.time() - start_time:.1f}s")
        print("💡 The process was likely stuck. This helps identify the hanging point.")
        sys.exit(1)
    except Exception as e:
        stop_monitoring = True
        print(f"\n❌ Error during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

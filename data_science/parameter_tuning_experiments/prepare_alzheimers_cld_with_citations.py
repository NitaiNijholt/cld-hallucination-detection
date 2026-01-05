#!/usr/bin/env python3
"""
Orchestration script to prepare Alzheimer's expert CLD with web citations.

This script:
1. Loads the Alzheimer's expert CLD into Neo4j (variables + edges)
2. Reads citation hints from JSON
3. Runs the citation refiller agent to populate web URLs
4. Exports the enhanced CLD with citations for judge validation

Usage:
    python prepare_alzheimers_cld_with_citations.py [--skip-neo4j-load]

Options:
    --skip-neo4j-load: Skip loading CLD into Neo4j (if already loaded)
"""

import sys
import os
import json
import logging
from pathlib import Path
import pandas as pd

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import CausalDiscovery, export_edges_comparison_to_excel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent / "ground_truth_clds_for_experiments"
EXCEL_FILE = BASE_DIR / "Alzheimers_disease_expert_validated.xlsx"
VARS_JSON = BASE_DIR / "Alzheimers_disease_vars_data.json"
EDGES_JSON = BASE_DIR / "Alzheimers_disease_edges_data.json"
CITATION_HINTS_JSON = BASE_DIR / "Alzheimers_disease_citation_hints.json"
OUTPUT_EXCEL = Path(__file__).parent / "results" / "Alzheimers_disease_with_web_citations.xlsx"

# Ensure output directory exists
OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)


def load_cld_into_neo4j(discovery: CausalDiscovery) -> str:
    """
    Load the Alzheimer's CLD variables and edges into Neo4j.
    
    Returns:
        Session ID
    """
    logger.info("="*80)
    logger.info("STEP 1: Loading CLD into Neo4j")
    logger.info("="*80)
    
    # Load variables from JSON
    with open(VARS_JSON, 'r', encoding='utf-8') as f:
        variables_data = json.load(f)
    
    # Load edges from JSON
    with open(EDGES_JSON, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)
    
    edges = edges_data['edges']
    
    logger.info(f"Loaded {len(variables_data)} variables and {len(edges)} edges from JSON files")
    
    # Create variables in Neo4j
    session_id = discovery.session_id
    
    with discovery.graph_db._get_session() as session:
        # Create variable nodes
        create_query = """
        CREATE (n:variable {
            name: $name,
            description: $description,
            session_id: $session_id,
            target: $is_target,
            deleted: false
        })
        """
        
        for var_data in variables_data:
            var_name = var_data["name"]
            var_def = var_data.get("definition", "")
            # First variable as target (or could be specified differently)
            is_target = (var_data["index"] == 0)
            
            session.run(create_query, {
                "name": var_name,
                "description": f"definition: {var_def}" if var_def else "",
                "session_id": session_id,
                "is_target": is_target
            })
            discovery.variables.append(var_name)
        
        logger.info(f"✅ Created {len(variables_data)} variable nodes in Neo4j")
        
        # Create edge relationships
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            rel_type = edge["type"]  # Should be POSITIVE or NEGATIVE
            
            # Ensure relationship type is uppercase
            if rel_type.lower() == 'positive':
                rel_type = 'POSITIVE'
            elif rel_type.lower() == 'negative':
                rel_type = 'NEGATIVE'
            
            edge_query = f"""
            MATCH (s:variable {{name: $source, session_id: $session_id}})
            MATCH (t:variable {{name: $target, session_id: $session_id}})
            CREATE (s)-[r:{rel_type}]->(t)
            SET r.session_id = $session_id,
                r.motivation = 'Expert-validated causal relationship',
                r.expert_validated = true
            """
            
            session.run(edge_query, {
                "source": source,
                "target": target,
                "session_id": session_id
            })
        
        logger.info(f"✅ Created {len(edges)} edge relationships in Neo4j")
    
    logger.info(f"✅ CLD loaded into Neo4j with session_id: {session_id}")
    return session_id


def load_citation_hints() -> dict:
    """
    Load citation hints from JSON and convert to format expected by refiller.
    
    Returns:
        Dictionary mapping (source, target) tuples to lists of full citations
    """
    logger.info("="*80)
    logger.info("STEP 2: Loading Citation Hints")
    logger.info("="*80)
    
    with open(CITATION_HINTS_JSON, 'r', encoding='utf-8') as f:
        hints_json = json.load(f)
    
    logger.info(f"Loaded citation hints for {len(hints_json)} edges")
    
    # Convert from "Source -> Target" string keys to (Source, Target) tuples
    hint_map = {}
    for edge_key, citations in hints_json.items():
        # Parse "Source -> Target" format
        parts = edge_key.split(' -> ')
        if len(parts) == 2:
            source, target = parts
            hint_map[(source, target)] = citations
        else:
            logger.warning(f"Skipping malformed edge key: {edge_key}")
    
    logger.info(f"✅ Converted {len(hint_map)} citation hints")
    
    # Show sample
    sample_edges = list(hint_map.items())[:3]
    logger.info(f"\nSample citation hints:")
    for (src, tgt), cits in sample_edges:
        logger.info(f"  {src} -> {tgt}: {len(cits)} citations")
        for cit in cits[:2]:  # Show first 2 citations
            logger.info(f"    - {cit[:80]}...")
    
    return hint_map


def run_citation_refiller(discovery: CausalDiscovery, hint_map: dict):
    """
    Run the citation refiller agent to populate web URLs.
    """
    logger.info("="*80)
    logger.info("STEP 3: Running Citation Refiller")
    logger.info("="*80)
    
    logger.info(f"Processing {len(hint_map)} edges with bibliographic hints")
    logger.info("Using Semantic Scholar API to find web URLs for academic papers...")
    
    # Call the citation refiller method
    # Using Semantic Scholar (free, academic-focused, best for biomedical papers)
    edges_updated = discovery.fill_in_missing_citations_with_hints(
        hint_map=hint_map,
        search_provider="semantic_scholar",  # Free academic paper search
        max_urls_per_citation=2,  # Keep top 2 URLs per citation for better coverage
        use_trusted_domains=False  # Disable domain filtering for maximum coverage
    )
    
    logger.info(f"✅ Citation refiller completed: {edges_updated} edges updated")
    return edges_updated


def export_enhanced_cld(discovery: CausalDiscovery, session_id: str) -> str:
    """
    Export the enhanced CLD with web citations to Excel.
    
    Returns:
        Path to the exported file
    """
    logger.info("="*80)
    logger.info("STEP 4: Exporting Enhanced CLD")
    logger.info("="*80)
    
    logger.info(f"Exporting to: {OUTPUT_EXCEL.parent}")
    
    # Use the export function from modules
    export_edges_comparison_to_excel(
        discovery=discovery,
        session_ids=session_id,
        validation_vars_json_path=str(VARS_JSON),
        validation_edges_json_path=str(EDGES_JSON),
        output_filename=str(OUTPUT_EXCEL)
    )
    
    logger.info(f"✅ Enhanced CLD exported successfully")
    
    # Find the actual exported file (it has a timestamp suffix)
    import glob
    pattern = str(OUTPUT_EXCEL.parent / "Alzheimers_disease_with_web_citations_*.xlsx")
    matching_files = sorted(glob.glob(pattern), key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    if matching_files:
        actual_file = matching_files[0]
        logger.info(f"   Exported file: {Path(actual_file).name}")
        
        # Verify citations were added
        try:
            df = pd.read_excel(actual_file, sheet_name='All Edges')
            edges_with_citations = df['Citation'].notna().sum()
            
            logger.info(f"\n📊 Export Summary:")
            logger.info(f"   Total edges: {len(df)}")
            logger.info(f"   Edges with citations: {edges_with_citations}")
            logger.info(f"   Citation coverage: {edges_with_citations/len(df)*100:.1f}%")
        except Exception as e:
            logger.warning(f"   Could not verify citations: {e}")
        
        return actual_file
    else:
        logger.warning("   Could not find exported file with timestamp")
        return str(OUTPUT_EXCEL)


def main():
    """
    Main orchestration function.
    """
    print("\n" + "="*80)
    print("ALZHEIMER'S CLD PREPARATION WITH WEB CITATIONS")
    print("="*80)
    
    # Check if we should skip Neo4j loading
    skip_neo4j = '--skip-neo4j-load' in sys.argv
    
    if skip_neo4j:
        logger.info("⚠️  Skipping Neo4j load (--skip-neo4j-load flag detected)")
        logger.info("   Make sure the CLD is already loaded in Neo4j!")
        # TODO: Need to get the session_id from somewhere
        logger.error("❌ Cannot skip Neo4j load without providing session_id")
        logger.error("   This feature is not yet implemented")
        sys.exit(1)
    
    # Initialize CausalDiscovery instance
    logger.info("\n📊 Initializing CausalDiscovery...")
    
    discovery = CausalDiscovery(
        target_variable="Alzheimer's disease risk",
        temporal_scale="Chronic, aging-related, multi-year processes",
        spatial_scale="Older adults, community-dwelling elderly",
        yaml_path=str(Path(__file__).parent / "prompts_Nitai_C.yaml"),
        generator_config={"provider": "openai", "model": "gpt-4o"},
        judge_config={"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"}
    )
    
    logger.info(f"✅ CausalDiscovery initialized with session_id: {discovery.session_id}")
    
    # Step 1: Load CLD into Neo4j
    session_id = load_cld_into_neo4j(discovery)
    
    # Step 2: Load citation hints
    hint_map = load_citation_hints()
    
    # Step 3: Run citation refiller
    edges_updated = run_citation_refiller(discovery, hint_map)
    
    # Step 4: Export enhanced CLD
    actual_output_file = export_enhanced_cld(discovery, session_id)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"\n📝 Summary:")
    print(f"   Session ID: {session_id}")
    print(f"   Variables: {len(discovery.variables)}")
    print(f"   Edges updated with citations: {edges_updated}")
    print(f"   Output file: {actual_output_file}")
    print(f"\n📊 Next steps:")
    print(f"   1. Review the output Excel file to verify citations")
    print(f"   2. Use this CLD for judge validation experiments (Step 4 of thesis narrative)")
    print(f"   3. Run judge on the expert-validated edges with their domain literature citations")
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("❌ Pipeline failed with error:")
        sys.exit(1)


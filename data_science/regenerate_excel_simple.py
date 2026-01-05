#!/usr/bin/env python3
"""
Smart script to regenerate Excel from Neo4j.
Just pass the path to an experiment result directory or Excel file.
"""

import sys
import pandas as pd
import json
import re
from pathlib import Path
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent))

from modules import export_edges_comparison_to_excel, CausalDiscovery

def extract_session_id_from_excel(excel_path):
    """Extract session ID from Excel file metadata."""
    try:
        # Try to read the Excel metadata or look for embedded session ID
        # Session IDs are UUIDs in format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        xls = pd.ExcelFile(excel_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=10)
            for col in df.columns:
                # Look specifically for a column named "Session ID" (exact match)
                if col == 'Session ID' or col == 'session_id':
                    values = df[col].dropna().astype(str)
                    for val in values:
                        # Check if it looks like a UUID
                        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', val.lower()):
                            return val
    except Exception as e:
        print(f"⚠️  Could not extract session ID from Excel: {e}")
    return None

def find_session_id_in_neo4j(cld_name_part):
    """Query Neo4j to find session IDs matching the CLD name."""
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        with driver.session() as session:
            # Get all sessions and find the most recent one matching the CLD
            result = session.run("""
                MATCH (v:variable)
                WHERE v.session_id IS NOT NULL
                RETURN DISTINCT v.session_id AS session_id, max(v.created_at) AS created_at
                ORDER BY created_at DESC
                LIMIT 20
            """)
            sessions = [record["session_id"] for record in result]
            driver.close()
            return sessions
    except Exception as e:
        print(f"⚠️  Could not query Neo4j: {e}")
        return []

def detect_cld_name(input_path):
    """Detect CLD name from path or filename."""
    path_str = str(input_path).lower()
    
    # Check for known CLDs
    if 'social_norms' in path_str or 'social norms' in path_str:
        return 'Social_norms_and_obesity_prevalence'
    elif 'test' in path_str or 'cld_test' in path_str:
        return 'CLD_test'
    elif 'depressive' in path_str:
        return 'Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults'
    
    # Try to extract from filename
    match = re.search(r'run1_(.+?)(?:\.xlsx|$)', path_str)
    if match:
        return match.group(1)
    
    return None

def find_validation_files(cld_name, base_dir):
    """Find validation JSON files for a given CLD."""
    base_path = Path(base_dir)
    
    if not cld_name:
        print("⚠️  Could not detect CLD name, skipping validation files")
        return None, None
    
    # Try exact match first
    vars_file = base_path / f"{cld_name}_vars_data.json"
    edges_file = base_path / f"{cld_name}_edges_data.json"
    
    if vars_file.exists() and edges_file.exists():
        return str(vars_file), str(edges_file)
    
    # Try to find by partial match
    for f in base_path.glob("*_vars_data.json"):
        if cld_name.lower() in f.stem.lower():
            vars_file = f
            edges_file = base_path / f.name.replace('_vars_data.json', '_edges_data.json')
            if edges_file.exists():
                return str(vars_file), str(edges_file)
    
    print(f"⚠️  Could not find validation files for CLD: {cld_name}")
    return None, None

# Parse command line argument
if len(sys.argv) < 2:
    print("Usage: python regenerate_excel_simple.py <path_to_experiment_or_excel>")
    sys.exit(1)

input_path = Path(sys.argv[1])
print(f"📂 Input path: {input_path}")

# Determine what we're dealing with
excel_path = None
session_id = None
cld_name = None

if input_path.is_file() and input_path.suffix == '.xlsx':
    print("📄 Input is an Excel file")
    excel_path = input_path
    # Try to extract session ID from the Excel
    session_id = extract_session_id_from_excel(excel_path)
    # Detect CLD from filename
    cld_name = detect_cld_name(input_path)
    
elif input_path.is_dir():
    print("📁 Input is a directory")
    # Look for Excel files in the directory
    excel_files = list(input_path.glob("*.xlsx"))
    if excel_files:
        excel_path = excel_files[0]
        print(f"   Found Excel: {excel_path.name}")
        session_id = extract_session_id_from_excel(excel_path)
        cld_name = detect_cld_name(excel_path)
    else:
        print("   No Excel files found, will query Neo4j")
        cld_name = detect_cld_name(input_path)
else:
    print(f"❌ Invalid input: {input_path}")
    print("   Please provide a path to an Excel file or experiment directory")
    sys.exit(1)

print(f"🔍 Detected CLD: {cld_name or 'Unknown'}")

# If we couldn't get session ID from Excel, query Neo4j
if not session_id:
    print("🔍 Querying Neo4j for session IDs...")
    all_sessions = find_session_id_in_neo4j(cld_name)
    if all_sessions:
        session_id = all_sessions[0]  # Take the most recent
        print(f"   Found {len(all_sessions)} sessions, using most recent: {session_id}")
    else:
        print("❌ Could not find session ID")
        sys.exit(1)

print(f"🆔 Session ID: {session_id}")

# Find validation files
validation_base = Path(__file__).parent / "parameter_tuning_experiments/ground_truth_clds_for_experiments"
validation_vars, validation_edges = find_validation_files(cld_name, validation_base)

if validation_vars and validation_edges:
    print(f"✅ Found validation files:")
    print(f"   Variables: {Path(validation_vars).name}")
    print(f"   Edges: {Path(validation_edges).name}")
else:
    print("⚠️  No validation files found, will export without validation comparison")

# Determine output filename
if excel_path:
    output_name = excel_path.stem + "_REGENERATED.xlsx"
    output_excel = excel_path.parent / output_name
else:
    output_excel = Path(__file__).parent / "parameter_tuning_experiments/results/REGENERATED.xlsx"

print(f"\n📂 Output file: {output_excel}")

# Create a minimal CausalDiscovery object for Neo4j access
print("\n🔧 Creating discovery object...")
disco = CausalDiscovery(
    target_variable="dummy",
    temporal_scale="dummy",
    spatial_scale="dummy",
    yaml_path=str(Path(__file__).parent / "parameter_tuning_experiments/prompts_Nitai_C.yaml"),
    generator_config={"provider": "openai", "model": "gpt-4o"}
)

# Call the function with correct parameters
print("📊 Exporting to Excel...")
export_edges_comparison_to_excel(
    discovery=disco,
    session_ids=session_id,
    validation_vars_json_path=validation_vars,
    validation_edges_json_path=validation_edges,
    output_filename=str(output_excel)
)

print(f"\n✅ Excel file regenerated successfully!")

# Quick sanity check
df = pd.read_excel(output_excel, sheet_name='All Edges')
print(f"\n📈 Quick Stats:")
print(f"   Total edges: {len(df)}")
print(f"   Columns: {len(df.columns)}")

if 'Relationship Type' in df.columns:
    print(f"\n🔗 Relationship Types:")
    print(df['Relationship Type'].value_counts())
    
if 'Classification' in df.columns:
    print(f"\n📊 Classifications:")
    print(df['Classification'].value_counts())

# Check for NONE relationships
if 'Relationship Type' in df.columns:
    none_edges = df[df['Relationship Type'].str.upper().isin(['NONE', 'NONE', 'NO RELATIONSHIP'])]
    if len(none_edges) > 0:
        print(f"\n⚠️  Found {len(none_edges)} edges with NONE relationship type")
        print(f"   These are shown in the Excel with their relationship type!")

print(f"\n✅ SUCCESS! File saved to: {output_excel}")

#!/usr/bin/env python3
"""
Export a Neo4j CLD session to Excel format.

Usage:
    python export_session_to_excel.py --session-id <session_id> --output <output.xlsx>
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from modules import CausalDiscovery

def main():
    parser = argparse.ArgumentParser(description="Export Neo4j session to Excel")
    parser.add_argument("--session-id", required=True, help="Session ID to export")
    parser.add_argument("--output", required=True, help="Output Excel file path")
    args = parser.parse_args()
    
    try:
        # Initialize CausalDiscovery with the session
        discovery = CausalDiscovery(session_id=args.session_id)
        
        # Export to Excel
        discovery.export_to_excel(args.output)
        
        print(f"✅ Session {args.session_id[:8]}... exported to {args.output}")
        return 0
        
    except Exception as e:
        print(f"❌ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())



#!/usr/bin/env python3
"""Test if APOC is working"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.dev")

try:
    from neo4j_session_cloner import SessionCloner
    cloner = SessionCloner()
    print('✅✅✅ APOC IS WORKING! Session cloning is ready!')
    cloner.close()
except Exception as e:
    print(f'❌ APOC still not working: {e}')
    import traceback
    traceback.print_exc()

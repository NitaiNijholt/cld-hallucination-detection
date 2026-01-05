#!/usr/bin/env python3
"""
Quick fix for Neo4j connection issues in the data science environment.
This script patches the Neo4j connection to use localhost:7687 directly.
"""

import os
import sys
import logging

def fix_neo4j_connection():
    """
    Set up environment variables and patch Neo4j connection for localhost access.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("neo4j_fix")
    
    # Force Neo4j to use localhost
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "bvdGGOJxqQp3Wdz"
    
    logger.info("✅ Neo4j environment variables set to localhost")
    logger.info(f"NEO4J_URI: {os.environ.get('NEO4J_URI')}")
    
    # Patch the Neo4jClient if it's already imported
    try:
        import neo4j
        from backend.db_clients.neo4j_client import Neo4jClient
        
        # Create a fixed connect method
        def fixed_connect(self):
            """Fixed connection method that forces localhost connection."""
            logger.info("=== Using fixed Neo4j connect method ===")
            
            # Force to use localhost credentials
            self._uri = "bolt://localhost:7687"
            self._username = "neo4j"
            self._password = "bvdGGOJxqQp3Wdz"
            
            logger.info(f"Connecting to: {self._uri}")
            
            try:
                self._driver = neo4j.GraphDatabase.driver(
                    self._uri, 
                    auth=(self._username, self._password)
                )
                self._driver.verify_connectivity()
                logger.info("✅ Successfully connected to Neo4j database!")
                return self._driver
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j database: {e}")
                raise
        
        # Apply the patch
        Neo4jClient.connect = fixed_connect
        logger.info("✅ Neo4jClient.connect method patched")
        
    except ImportError as e:
        logger.info(f"Neo4jClient not yet imported, will patch when needed: {e}")
    
    return True

if __name__ == "__main__":
    fix_neo4j_connection()
    print("Neo4j connection fix applied!")
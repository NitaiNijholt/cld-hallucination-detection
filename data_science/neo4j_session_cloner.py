"""
Neo4j Session Cloner for RQ1 Experiments

This utility clones CLD sessions in Neo4j to save on token costs.
Instead of regenerating entire CLDs for different corruption rates,
we generate once and clone multiple times with different corruption.

Usage:
    from neo4j_session_cloner import clone_session
    
    new_session_id = clone_session(
        original_session_id="abc123",
        new_session_id="def456"
    )

Requires:
    - Neo4j with APOC plugin installed
    - neo4j-driver Python package
"""

import os
import uuid
import logging
from typing import Optional
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionCloner:
    """Clone Neo4j sessions for CLD experiments."""
    
    def __init__(self, uri=None, user=None, password=None):
        """Initialize connection to Neo4j.
        
        Reads from environment variables if not provided:
        - NEO4J_URI (default: bolt://localhost:7687)
        - NEO4J_USERNAME (default: neo4j)
        - NEO4J_PASSWORD (default: bvdGGOJxqQp3Wdz)
        """
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_apoc()
    
    def _verify_apoc(self):
        """Verify APOC is installed."""
        with self.driver.session() as session:
            try:
                result = session.run("RETURN apoc.version() as version")
                version = result.single()["version"]
                logger.info(f"✅ APOC version {version} detected")
            except Exception as e:
                logger.error("❌ APOC not installed! Install with: neo4j-admin install apoc")
                raise RuntimeError("APOC required for session cloning") from e
    
    def clone_session(
        self,
        original_session_id: str,
        new_session_id: Optional[str] = None,
        include_judge_data: bool = False
    ) -> str:
        """
        Clone an entire session (all nodes and relationships) with a new session_id.
        
        Args:
            original_session_id: Session ID to clone
            new_session_id: New session ID (generated if None)
            include_judge_data: If True, includes judge verdicts in clone
                               If False, only copies generation data (edges + motivations)
        
        Returns:
            New session ID
        """
        if new_session_id is None:
            new_session_id = str(uuid.uuid4())
        
        logger.info(f"Cloning session {original_session_id[:8]}... → {new_session_id[:8]}...")
        
        with self.driver.session() as session:
            # Step 1: Get all nodes from original session
            nodes_query = """
            MATCH (n:variable)
            WHERE n.session_id = $original_session_id
            RETURN collect(n) as nodes
            """
            result = session.run(nodes_query, {"original_session_id": original_session_id})
            record = result.single()
            
            if not record or not record["nodes"]:
                raise ValueError(f"No nodes found for session {original_session_id}")
            
            node_count = len(record["nodes"])
            logger.info(f"  Found {node_count} nodes")
            
            # Step 2: Get all relationships from original session
            rels_query = """
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $original_session_id 
              AND t.session_id = $original_session_id
            RETURN collect(r) as relationships
            """
            result = session.run(rels_query, {"original_session_id": original_session_id})
            record = result.single()
            rel_count = len(record["relationships"]) if record and record["relationships"] else 0
            logger.info(f"  Found {rel_count} relationships")
            
            # Step 3: Clone using APOC
            # Note: We can't use apoc.refactor.cloneSubgraph directly with session_id change
            # We need to manually copy nodes and relationships with new session_id
            
            # Clone nodes
            clone_nodes_query = """
            MATCH (n:variable)
            WHERE n.session_id = $original_session_id
            WITH n
            CREATE (clone:variable)
            SET clone = properties(n),
                clone.session_id = $new_session_id
            RETURN count(clone) as cloned_nodes
            """
            result = session.run(
                clone_nodes_query,
                {
                    "original_session_id": original_session_id,
                    "new_session_id": new_session_id
                }
            )
            cloned_nodes = result.single()["cloned_nodes"]
            logger.info(f"  ✓ Cloned {cloned_nodes} nodes")
            
            # Clone relationships
            if include_judge_data:
                # Include ALL relationship properties (including judge verdicts)
                clone_rels_query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $original_session_id 
                  AND t.session_id = $original_session_id
                WITH s, r, t, type(r) as rel_type, properties(r) as props
                MATCH (s_clone:variable {name: s.name, session_id: $new_session_id})
                MATCH (t_clone:variable {name: t.name, session_id: $new_session_id})
                CALL apoc.create.relationship(
                    s_clone, 
                    rel_type, 
                    apoc.map.setKey(props, 'session_id', $new_session_id),
                    t_clone
                ) 
                YIELD rel
                RETURN count(rel) as cloned_relationships
                """
            else:
                # Only include generation data, skip judge properties
                clone_rels_query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $original_session_id 
                  AND t.session_id = $original_session_id
                WITH s, r, t, type(r) as rel_type, properties(r) as props
                MATCH (s_clone:variable {name: s.name, session_id: $new_session_id})
                MATCH (t_clone:variable {name: t.name, session_id: $new_session_id})
                WITH s_clone, t_clone, rel_type, props,
                     // Filter out judge properties
                     apoc.map.removeKeys(props, [
                         'judge_verdict', 'judge_message', 'aggregate_score',
                         'aggregate_verdict', 'aggregator_message',
                         'judge_perplexity', 'judge_min_prob', 'judge_max_window_entropy'
                     ]) as filtered_props
                CALL apoc.create.relationship(
                    s_clone, 
                    rel_type, 
                    apoc.map.setKey(filtered_props, 'session_id', $new_session_id),
                    t_clone
                ) 
                YIELD rel
                RETURN count(rel) as cloned_relationships
                """
            
            result = session.run(
                clone_rels_query,
                {
                    "original_session_id": original_session_id,
                    "new_session_id": new_session_id
                }
            )
            cloned_rels = result.single()["cloned_relationships"]
            logger.info(f"  ✓ Cloned {cloned_rels} relationships")
        
        logger.info(f"✅ Session cloned successfully: {new_session_id[:8]}")
        return new_session_id
    
    def verify_clone(self, original_session_id: str, cloned_session_id: str) -> dict:
        """
        Verify that clone has same structure as original.
        
        Returns:
            Dictionary with comparison stats
        """
        with self.driver.session() as session:
            verify_query = """
            // Count nodes and edges in both sessions
            MATCH (n1:variable {session_id: $session1})
            WITH count(n1) as nodes1
            MATCH (n2:variable {session_id: $session2})
            WITH nodes1, count(n2) as nodes2
            MATCH (s1:variable)-[r1]->(t1:variable)
            WHERE s1.session_id = $session1 AND t1.session_id = $session1
            WITH nodes1, nodes2, count(r1) as rels1
            MATCH (s2:variable)-[r2]->(t2:variable)
            WHERE s2.session_id = $session2 AND t2.session_id = $session2
            WITH nodes1, nodes2, rels1, count(r2) as rels2
            RETURN nodes1, nodes2, rels1, rels2,
                   nodes1 = nodes2 as nodes_match,
                   rels1 = rels2 as rels_match
            """
            result = session.run(
                verify_query,
                {"session1": original_session_id, "session2": cloned_session_id}
            )
            record = result.single()
            
            stats = {
                "original_nodes": record["nodes1"],
                "cloned_nodes": record["nodes2"],
                "original_rels": record["rels1"],
                "cloned_rels": record["rels2"],
                "nodes_match": record["nodes_match"],
                "rels_match": record["rels_match"]
            }
            
            if stats["nodes_match"] and stats["rels_match"]:
                logger.info(f"✅ Clone verified: {stats['cloned_nodes']} nodes, {stats['cloned_rels']} rels")
            else:
                logger.warning(f"⚠️ Clone mismatch: {stats}")
            
            return stats
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()


# Convenience functions
def clone_session(
    original_session_id: str,
    new_session_id: Optional[str] = None,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    include_judge_data: bool = False
) -> str:
    """
    Convenience function to clone a session.
    
    Args:
        original_session_id: Session ID to clone
        new_session_id: New session ID (auto-generated if None)
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        include_judge_data: Whether to include judge verdicts
    
    Returns:
        New session ID
    """
    cloner = SessionCloner(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    try:
        new_id = cloner.clone_session(
            original_session_id=original_session_id,
            new_session_id=new_session_id,
            include_judge_data=include_judge_data
        )
        cloner.verify_clone(original_session_id, new_id)
        return new_id
    finally:
        cloner.close()


if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv
    
    # Load environment
    load_dotenv("../.env.dev")
    
    if len(sys.argv) < 2:
        print("Usage: python neo4j_session_cloner.py <original_session_id> [new_session_id]")
        print("Example: python neo4j_session_cloner.py abc123def456")
        sys.exit(1)
    
    original_id = sys.argv[1]
    new_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Get Neo4j credentials from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    print(f"\n{'='*60}")
    print("NEO4J SESSION CLONER")
    print(f"{'='*60}")
    print(f"\nOriginal session: {original_id}")
    print(f"Neo4j URI: {uri}")
    print()
    
    new_session_id = clone_session(
        original_session_id=original_id,
        new_session_id=new_id,
        neo4j_uri=uri,
        neo4j_user=user,
        neo4j_password=password,
        include_judge_data=False
    )
    
    print(f"\n✅ Clone complete!")
    print(f"New session ID: {new_session_id}")
    print(f"{'='*60}\n")

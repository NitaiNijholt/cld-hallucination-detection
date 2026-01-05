import os
import json
import pandas as pd
import datetime
import uuid
import glob
from dotenv import load_dotenv
from pathlib import Path

# Import necessary classes/functions - assuming CausalDiscovery is available
# from existing codebase
import os
import json
import pandas as pd
import datetime
import uuid
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# Import from the modules file
from data_science.modules_03_05_2025 import (
    CausalDiscovery,
    compare_session_graph_to_validation,
    build_networkx_graph,
    extract_networkx_from_session,
    filter_validation_graph_to_session_nodes

)


# Load environment variables
load_dotenv(override=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("prompt_evaluation.log")
    ]
)
logger = logging.getLogger("prompt_evaluator")

class PromptEvaluator:
    def __init__(self, 
                 yaml_dir: str, 
                 ground_truth_dir: str, 
                 results_dir: str, 
                 generator_model: str = None,
                 judge_models: list = None,
                 num_judges: int = 3,
                 corruption_rate: float = 0.0):
        """
        Initialize the PromptEvaluator.
        
        Args:
            yaml_dir: Directory containing YAML prompt files
            ground_truth_dir: Directory containing ground truth CLD Excel files
            results_dir: Directory where results will be saved
            generator_model: Model to use for relationship generation
            judge_models: List of models to use for judging relationships
            num_judges: Number of judges to use
            corruption_rate: Rate at which to corrupt relationships (0.0-1.0)
        """
        self.yaml_dir = Path(yaml_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.results_dir = Path(results_dir)
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set models
        self.generator_model = generator_model or os.environ.get("PERPLEXITY_MODEL", "sonar-pro")
        self.judge_models = judge_models or [os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")] * num_judges
        self.num_judges = num_judges
        self.corruption_rate = corruption_rate
        
        # Load ground truth CLDs from Excel files
        self.ground_truths = self._load_ground_truths_excel()
        
    def _load_ground_truths_excel(self):
        """Load all ground truth CLDs from Excel files in the ground truth directory."""
        ground_truths = {}
        
        # Look for Excel files
        excel_files = list(self.ground_truth_dir.glob("*.xlsx"))
        
        for excel_file in excel_files:
            base_name = excel_file.stem
            
            # Check if the Excel file has the required sheets
            try:
                # Load the spreadsheet to check for required sheets
                xls = pd.ExcelFile(excel_file)
                required_sheets = ["Variable definitions", "Variable_links"]
                
                if all(sheet in xls.sheet_names for sheet in required_sheets):
                    ground_truths[base_name] = {
                        "excel_file": str(excel_file)
                    }
                    logger.info(f"Found valid ground truth file: {excel_file}")
                else:
                    logger.warning(f"Excel file {excel_file} is missing required sheets. Skipping.")
            except Exception as e:
                logger.error(f"Error processing Excel file {excel_file}: {e}")
                
        return ground_truths
    
    def build_networkx_graph_from_excel(self, excel_file, plot_graph=False):
        """
        Build a NetworkX graph from an Excel file containing variables and edges.
        
        Args:
            excel_file: Path to the Excel file
            plot_graph: Whether to visualize the graph
            
        Returns:
            A NetworkX DiGraph representing the CLD
        """
        # Read the Excel sheets
        var_df = pd.read_excel(excel_file, sheet_name="Variable definitions")
        edge_df = pd.read_excel(excel_file, sheet_name="Variable_links")
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes from variable definitions
        for _, row in var_df.iterrows():
            variable = row["Variable"]
            definition = row["Definition"]
            G.add_node(variable, definition=definition)
        
        # Add edges from variable links
        for _, row in edge_df.iterrows():
            source = row["Source"]
            target = row["Target"]
            polarity = row["Polarity"].lower() if "Polarity" in row else "positive"
            
            # Translate polarity to relationship type
            if polarity == "positive":
                relationship = "POSITIVE"
            elif polarity == "negative":
                relationship = "NEGATIVE"
            else:
                relationship = "NONLINEAR"  # Default for other values
            
            G.add_edge(source, target, relationship=relationship)
        
        # Plot the graph if requested
        if plot_graph:
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700)
            nx.draw_networkx_labels(G, pos)
            
            # Draw edges with different colors based on relationship
            edge_colors = []
            for u, v, d in G.edges(data=True):
                if d.get("relationship") == "POSITIVE":
                    edge_colors.append("green")
                elif d.get("relationship") == "NEGATIVE":
                    edge_colors.append("red")
                else:
                    edge_colors.append("blue")
            
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True)
            
            plt.axis("off")
            plt.title("Ground Truth CLD")
            plt.tight_layout()
            plt.show()
        
        return G
    
    def compare_graph_to_validation_excel(self, discovery, session_ids, excel_file, plot_session_graph=False, 
                                     plot_validation_graph=False, only_cited_edges=False, 
                                     only_consistent_edges=False, filter_validation_nodes_to_session_nodes=True):
        """
        Compare a session graph (from Neo4j) with a validation graph (from Excel).
        
        Args:
            discovery: CausalDiscovery object with .graph_db for Neo4j queries
            session_ids: Session ID(s) to use
            excel_file: Path to Excel file with ground truth data
            plot_session_graph: Whether to visualize the session graph
            plot_validation_graph: Whether to visualize the validation graph
            only_cited_edges: If True, only edges with citation data are extracted from Neo4j
            only_consistent_edges: If True, only edges with "consistent" judge verdict are extracted
            filter_validation_nodes_to_session_nodes: Whether to filter validation nodes to match session nodes
            
        Returns:
            Dictionary with metrics
        """
        # 1) Build the session graph from Neo4j
        G_session = extract_networkx_from_session(
            discovery=discovery,
            session_ids=session_ids,
            plot_graph=plot_session_graph,
            only_cited_edges=only_cited_edges,
            only_consistent_edges=only_consistent_edges
        )
        session_edges = set(G_session.edges())
        session_nodes = set(G_session.nodes())

        # 2) Build the validation graph from Excel
        G_validation = self.build_networkx_graph_from_excel(
            excel_file=excel_file,
            plot_graph=plot_validation_graph
        )

        validation_edges = set(G_validation.edges())
        validation_nodes = set(G_validation.nodes())
        
        if filter_validation_nodes_to_session_nodes:
            # Filter validation graph to include only nodes that are also in session graph
            G_filtered_validation = G_validation.copy()
            nodes_to_remove = [node for node in validation_nodes if node not in session_nodes]
            G_filtered_validation.remove_nodes_from(nodes_to_remove)
            
            validation_edges = set(G_filtered_validation.edges())
            validation_nodes = set(G_filtered_validation.nodes())

        # 3) Directly compare edge sets
        #    -------------------------------------
        #    True Positive (TP) = edges in both
        #    False Positive (FP) = edges only in session graph
        #    False Negative (FN) = edges only in validation graph
        #    True Negative (TN) = pairs of nodes that are in neither set
        
        # Intersection of edges: found in both
        overlap_set = session_edges & validation_edges
        tp = len(overlap_set)

        # Edges that session has but validation doesn't
        fp = len(session_edges - validation_edges)

        # Edges that validation has but session doesn't
        fn = len(validation_edges - session_edges)

        # 4) Compute TN:
        #    The universe is all possible directed pairs among the union of *all* nodes
        union_nodes = session_nodes.union(validation_nodes)
        all_possible_pairs = {
            (src, tgt)
            for src in union_nodes
            for tgt in union_nodes
            if src != tgt
        }

        # All edges that appear in session or validation
        combined_edges = session_edges.union(validation_edges)

        # True negatives are the pairs that appear in *neither* graph
        tn = len(all_possible_pairs - combined_edges)

        # 5) Metrics
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        metrics = {
            "session_graph_edges": session_edges,
            "validation_graph_edges": validation_edges,
            "session_nodes": session_nodes,
            "validation_nodes": validation_nodes,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

        return metrics
    
    def get_yaml_files_to_process(self):
        """
        Get list of YAML files that need to be processed (no results yet).
        """
        yaml_files = list(self.yaml_dir.glob("*.yaml"))
        files_to_process = []
        
        for yaml_file in yaml_files:
            result_file = self.results_dir / f"{yaml_file.stem}.csv"
            if not result_file.exists():
                files_to_process.append(yaml_file)
                
        return files_to_process
    
    def run_evaluation(self, target_variable="", temporal_scale="", spatial_scale="", context=""):
        """
        Run the evaluation for all YAML files against all ground truth CLDs.
        """
        yaml_files = self.get_yaml_files_to_process()
        
        for yaml_file in yaml_files:
            print(f"Processing {yaml_file.name}...")
            
            results = []
            
            for gt_name, gt_files in self.ground_truths.items():
                print(f"  Evaluating against {gt_name} ground truth...")
                
                # Try to load context information from the Excel file
                excel_file = gt_files["excel_file"]
                try:
                    context_df = pd.read_excel(excel_file, sheet_name="Context")
                    if not context and not context_df.empty:
                        # Extract context information from the Excel file
                        target_row = context_df[context_df.iloc[:, 0] == "Target"]
                        spatial_row = context_df[context_df.iloc[:, 0] == "Spatial Scale"]
                        temporal_row = context_df[context_df.iloc[:, 0] == "Temporal Scale"]
                        scope_row = context_df[context_df.iloc[:, 0] == "Scope"]
                        
                        target_variable = target_row.iloc[0, 1] if not target_row.empty else target_variable
                        spatial_scale = spatial_row.iloc[0, 1] if not spatial_row.empty else spatial_scale
                        temporal_scale = temporal_row.iloc[0, 1] if not temporal_row.empty else temporal_scale
                        context = scope_row.iloc[0, 1] if not scope_row.empty else context
                except Exception as e:
                    logger.warning(f"Could not load context from Excel file: {e}")
                
                # Run causal discovery with the current YAML file
                metrics = self.evaluate_prompt(
                    yaml_file=str(yaml_file),
                    ground_truth_name=gt_name,
                    excel_file=excel_file,
                    target_variable=target_variable,
                    temporal_scale=temporal_scale,
                    spatial_scale=spatial_scale,
                    context=context
                )
                
                # Add ground truth name to metrics
                metrics["ground_truth"] = gt_name
                results.append(metrics)
            
            # Save results to CSV
            results_df = pd.DataFrame(results)
            output_file = self.results_dir / f"{yaml_file.stem}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"  Results saved to {output_file}")
    
    def evaluate_prompt(self, 
                        yaml_file: str,
                        ground_truth_name: str,
                        excel_file: str,
                        target_variable: str = "",
                        temporal_scale: str = "",
                        spatial_scale: str = "",
                        context: str = ""):
        """
        Evaluate a single YAML prompt file against a ground truth CLD.
        
        Args:
            yaml_file: Path to the YAML prompt file
            ground_truth_name: Name of the ground truth CLD
            excel_file: Path to the Excel file with ground truth data
            target_variable: Target variable for the CLD
            temporal_scale: Temporal scale for the CLD
            spatial_scale: Spatial scale for the CLD
            context: Context for the CLD
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create a new session ID
        session_id = str(uuid.uuid4())
        print(f"  Using session ID: {session_id}")
        
        # Initialize CausalDiscovery with the YAML file
        discovery = CausalDiscovery(
            target_variable=target_variable,
            temporal_scale=temporal_scale,
            spatial_scale=spatial_scale,
            context=context,
            yaml_path=yaml_file,
            dev_mode=False,
            generator_model=self.generator_model,
            corruptor_model=self.generator_model,  # Use same model for corruption
            judge_model=self.judge_models[0]
        )
        
        # Set the session ID
        discovery.set_session_id(session_id)
        
        # Load variables from the Excel file
        var_df = pd.read_excel(excel_file, sheet_name="Variable definitions")
        variables = var_df["Variable"].tolist()
        
        # Add variables to the graph database
        self._add_variables_to_graph(discovery, variables, session_id, target_variable, var_df)
        
        # Generate relationships
        print(f"  Generating relationships...")
        relationships = discovery.discover_relationships(
            parallel=False,
            max_workers=3,
            corruption_rate=self.corruption_rate
        )
        
        relationship_count = len(relationships)
        corrupted_count = discovery.corrupted_count
        print(f"  Discovered {relationship_count} relationships; {corrupted_count} were corrupted/spurious.")
        
        # Judge relationships
        print(f"  Judging relationships...")
        try:
            judged_edges = discovery.judge_all_edges_with_citations_serial(
                judge_models=self.judge_models,
                num_judges=self.num_judges
            )
        except Exception as e:
            print(f"  Error running judgment with citations: {e}")
            print(f"  Falling back to regular judgment...")
            judged_edges = discovery.judge_all_edges_serial(
                judge_models=self.judge_models,
                num_judges=self.num_judges
            )
        
        print(f"  Judged {len(judged_edges)} edges")
        
        # Compare to ground truth
        print(f"  Comparing to ground truth...")
        metrics = self.compare_graph_to_validation_excel(
            discovery=discovery,
            session_ids=session_id,
            excel_file=excel_file,
            plot_session_graph=False,
            plot_validation_graph=False,
            only_cited_edges=False,
            only_consistent_edges=False,
            filter_validation_nodes_to_session_nodes=True
        )
        
        # Create summary metrics
        basic_metrics = {
            "yaml_file": os.path.basename(yaml_file),
            "session_id": session_id,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return basic_metrics
    
    def _add_variables_to_graph(self, discovery, variables, session_id, target_variable, var_df):
        """Add variables to the graph database with definitions from Excel."""
        discovery.variables = []  # Prepare empty, then fill
        
        # Create a dictionary for quick lookup of definitions
        var_to_def = dict(zip(var_df["Variable"], var_df["Definition"]))
        
        with discovery.graph_db._get_session() as session:
            create_query = """
            CREATE (n:variable {
                name: $name,
                definition: $description,
                session_id: $session_id,
                target: $is_target,
                deleted: false
            })
            """
            
            # First add the target variable if it exists
            if target_variable and target_variable in variables:
                definition = var_to_def.get(target_variable, f"Target variable: {target_variable}")
                session.run(create_query, {
                    "name": target_variable,
                    "description": definition,
                    "session_id": session_id,
                    "is_target": True
                })
                discovery.variables.append(target_variable)
            
            # Then add all other variables
            for var in variables:
                if var == target_variable:
                    continue
                
                # Get definition from Excel
                definition = var_to_def.get(var, f"Variable: {var}")
                
                session.run(create_query, {
                    "name": var,
                    "description": definition,
                    "session_id": session_id,
                    "is_target": False
                })
                discovery.variables.append(var)
        
        print(f"  Created {len(discovery.variables)} variables with session ID: {session_id}")


def main():
    # Load environment variables
    load_dotenv(override=True)
    
    # Configuration
    YAML_DIR = "prompts"
    GROUND_TRUTH_DIR = "ground_truths"
    RESULTS_DIR = "results"
    
    # Model configuration
    JUDGE_MODELS = [
        "claude-3-7-sonnet-20250219", 
        "claude-3-7-sonnet-20250219", 
        "claude-3-7-sonnet-20250219"
    ]
    NUM_JUDGES = 3
    
    # Initialize and run evaluator
    evaluator = PromptEvaluator(
        yaml_dir=YAML_DIR,
        ground_truth_dir=GROUND_TRUTH_DIR,
        results_dir=RESULTS_DIR,
        judge_models=JUDGE_MODELS,
        num_judges=NUM_JUDGES
    )
    
    # Run the evaluation (context parameters will be read from Excel)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
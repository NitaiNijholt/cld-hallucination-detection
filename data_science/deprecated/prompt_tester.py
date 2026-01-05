import os
import json
import pandas as pd
import datetime
import uuid
import glob
from dotenv import load_dotenv
from pathlib import Path

# Import necessary classes/functions - assuming CausalDiscovery is available
# from your existing codebase
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

class PromptEvaluator:
    def __init__(self, 
                 yaml_dir: str, 
                 ground_truth_dir: str, 
                 results_dir: str, 
                 generator_model: str = None,
                 judge_models: list = None,
                 num_judges: int = 3):
        """
        Initialize the PromptEvaluator.
        
        Args:
            yaml_dir: Directory containing YAML prompt files
            ground_truth_dir: Directory containing ground truth CLD JSON files
            results_dir: Directory where results will be saved
            generator_model: Model to use for relationship generation
            judge_models: List of models to use for judging relationships
            num_judges: Number of judges to use
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
        
        # Load ground truth CLDs
        self.ground_truths = self._load_ground_truths()
        
    def _load_ground_truths(self):
        """Load all ground truth CLDs from the ground truth directory."""
        ground_truths = {}
        
        # Look for pairs of JSON files (vars and edges)
        var_files = list(self.ground_truth_dir.glob("*_vars_data.json"))
        
        for var_file in var_files:
            base_name = var_file.stem.replace("_vars_data", "")
            edge_file = self.ground_truth_dir / f"{base_name}_edges_data.json"
            
            if edge_file.exists():
                ground_truths[base_name] = {
                    "vars_file": str(var_file),
                    "edges_file": str(edge_file)
                }
                
        return ground_truths
    
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
    
    def run_evaluation(self, target_variable, temporal_scale, spatial_scale, context):
        """
        Run the evaluation for all YAML files against all ground truth CLDs.
        """
        yaml_files = self.get_yaml_files_to_process()
        
        for yaml_file in yaml_files:
            print(f"Processing {yaml_file.name}...")
            
            results = []
            
            for gt_name, gt_files in self.ground_truths.items():
                print(f"  Evaluating against {gt_name} ground truth...")
                
                # Run causal discovery with the current YAML file
                metrics = self.evaluate_prompt(
                    yaml_file=str(yaml_file),
                    ground_truth_name=gt_name,
                    vars_json_path=gt_files["vars_file"],
                    edges_json_path=gt_files["edges_file"],
                    target_variable=target_variable,
                    temporal_scale=temporal_scale,
                    spatial_scale=spatial_scale,
                    context=context
                )
                
                # Add ground truth name to metrics
                metrics["ground_truth"] = gt_name
                results.append(metrics)
            
            # Save results to CSV
            if results:
                result_df = pd.DataFrame(results)
                result_file = self.results_dir / f"{yaml_file.stem}.csv"
                result_df.to_csv(result_file, index=False)
                print(f"  Results saved to {result_file}")
    
    def evaluate_prompt(self, yaml_file, ground_truth_name, vars_json_path, 
                        edges_json_path, target_variable, temporal_scale, 
                        spatial_scale, context):
        """
        Evaluate a single YAML prompt file against a single ground truth CLD.
        
        Returns:
            dict: Metrics from the comparison
        """
        # 1. Load the variables from the ground truth
        with open(vars_json_path, "r") as f:
            data = json.load(f)
            
        variables = data.get("unique_variables", [])
        if not variables:
            raise ValueError(f"No 'unique_variables' found in {vars_json_path}")
        
        # 2. Initialize CausalDiscovery with the YAML file
        discovery = CausalDiscovery(
            target_variable=target_variable,
            temporal_scale=temporal_scale,
            spatial_scale=spatial_scale,
            context=context,
            yaml_path=yaml_file,
            dev_mode=False,
            generator_model=self.generator_model,
            corruptor_model=self.generator_model,  # Using same model
            judge_model=self.judge_models[0]  # Using first judge model
        )
        
        # 3. Create a new session
        session_id = discovery.session_id
        print(f"  Using session ID: {session_id}")
        
        # 4. Add variables to the graph database
        self._add_variables_to_graph(discovery, variables, session_id, target_variable)
        
        # 5. Discover relationships
        relationships = discovery.discover_relationships(
            parallel=False,
            max_workers=3,
            corruption_rate=0.0  # No corruption for evaluation
        )
        
        print(f"  Discovered {len(relationships)} relationships")
        
        # 6. Judge the relationships
        try:
            # Try with citations first
            judged_edges = discovery.judge_all_edges_with_citations_serial(
                judge_models=self.judge_models,
                num_judges=self.num_judges
            )
        except Exception as e:
            print(f"  Judging with citations failed: {e}")
            # Fallback to judging without citations
            judged_edges = discovery.judge_all_edges_serial(
                judge_models=self.judge_models,
                num_judges=self.num_judges
            )
        
        print(f"  Judged {len(judged_edges)} edges")
        
        # 7. Compare to ground truth
        metrics = compare_session_graph_to_validation(
            discovery=discovery,
            session_ids=session_id,
            validation_vars_json_path=vars_json_path,
            validation_edges_json_path=edges_json_path,
            plot_session_graph=False,
            plot_validation_graph=False,
            only_cited_edges=False,
            only_consistent_edges=False,
            filter_validation_nodes_to_session_nodes=True
        )
        
        # 8. Add prompt file info to metrics
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
    
    def _add_variables_to_graph(self, discovery, variables, session_id, target_variable):
        """Add variables to the graph database."""
        discovery.variables = []  # Prepare empty, then fill
        
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
            if target_variable:
                session.run(create_query, {
                    "name": target_variable,
                    "description": f"Target variable: {target_variable}",
                    "session_id": session_id,
                    "is_target": True
                })
                discovery.variables.append(target_variable)
            
            # Then add all other variables
            for var in variables:
                if var == target_variable:
                    continue
                
                # Select appropriate category based on variable name
                category = self._determine_variable_category(var)
                
                session.run(create_query, {
                    "name": var,
                    "description": f"{category}: {var}",
                    "session_id": session_id,
                    "is_target": False
                })
                discovery.variables.append(var)
        
        print(f"  Created {len(discovery.variables)} variables with session ID: {session_id}")
    
    def _determine_variable_category(self, var):
        """Assign a category to a variable based on its name."""
        if var in ["Anxiety", "Depression", "Sleep"]:
            return "Sum Score"
        elif var in ["Pain", "Hearing", "General_Health", "Memory_Complaints"]:
            return "Symptom"
        elif var in ["MAP", "Grip", "BMI", "WHR", "Physical_Activity", "Alcohol_Use"]:
            return "Sign (objective measure)"
        elif var in ["Respiratory_Problems", "Heart_Disease", "Arterial_Disease", "Diabetes", 
                    "CVA", "Osteoarthritis", "Rheumatoid_Arthritis", "Cancer"]:
            return "Chronic Disease"
        else:
            return "ADL functional limitation"


def main():
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
    
    # Dataset parameters (customize as needed)
    target_variable = ""  # Empty string for no specific target
    temporal_scale = "Approximately 3-year intervals and contemporaneous relations"
    spatial_scale = "At the individual level aggregated to the population-level"
    context = "How do symptoms, signs, (chronic) diseases, and limitations in activities of daily living interact dynamically over time to drive multimorbidity and functional decline in older adults?"
    
    # Initialize and run evaluator
    evaluator = PromptEvaluator(
        yaml_dir=YAML_DIR,
        ground_truth_dir=GROUND_TRUTH_DIR,
        results_dir=RESULTS_DIR,
        judge_models=JUDGE_MODELS,
        num_judges=NUM_JUDGES
    )
    
    evaluator.run_evaluation(
        target_variable=target_variable,
        temporal_scale=temporal_scale,
        spatial_scale=spatial_scale,
        context=context
    )


if __name__ == "__main__":
    main()
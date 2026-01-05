# Cell 1: Import libraries and set up logging



# Near the top of your file, before the import that's failing

import os
import sys
# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import json
import random
import time
import uuid
import logging

from typing import Dict, List, Tuple, Optional, Any, Union, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import datetime
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)
import tqdm as tqdm
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from datetime import datetime
import logging
import sys
import traceback
import asyncio
from pydantic import BaseModel, Field
from agents import Agent, Runner, trace
import datetime
import uuid
from dotenv import load_dotenv
from itertools import permutations
import traceback
import os
import neo4j
import logging
import importlib
from backend.db_clients.neo4j_client import Neo4jClient
from data_loading_scripts.excel_cld_loader_v3 import load_cld_from_excel
import tempfile
import re
import glob
import statistics
import openpyxl
# from backend.agents.equivalence_two_agent import Equivalency_Checker_agent, OutputEquivalent, InputTwoEquivalent




# Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler("causal_discovery.log")
#     ]
# )


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("causal_discovery.log")
    ]
)
logger = logging.getLogger("causal_discovery")

# # Set app loggers to show ERROR level
# app_loggers = [
#     'backend',         # Main application logs
#     'routes',          # Routes.py logs
#     'prompt_tree',     # PromptTree logs
#     'neo4j',           # Neo4j logs
#     '__main__'         # Jupyter notebook logs
# ]

# for logger_name in app_loggers:
#     logging.getLogger(logger_name).setLevel(logging.DEBUG)

# # Make sure root logger shows everything
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.DEBUG)

# # Add a StreamHandler to ensure output goes to stdout
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# root_logger.addHandler(handler)

# print("Logging configured for maximum visibility")
# logger = logging.getLogger("causal_discovery")

# Fix for notebook environment - use current working directory
notebook_path = os.getcwd()
parent_path = os.path.dirname(notebook_path)
sys.path.append(parent_path)

# Import from existing codebase
from backend.citation_content_extractor import CitationProcessor
from backend.db_clients.neo4j_client import Neo4jClient
from backend.lm_clients.perplexity_client import PerplexityClient
from backend.prompt_tree import PromptTree
from backend.lm_clients.claude_client import ClaudeClient
from backend.lm_clients.openai_client_not_working import OpenAIClient
from backend.citation_content_extractor import CitationProcessor

from dotenv import load_dotenv

load_dotenv("../.env.dev", override=True)

# Now check if the API key is set
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"OpenAI API key is set: {api_key[:8]}...{api_key[-4:]}")
else:
    print("WARNING: OPENAI_API_KEY is not set in the environment!")

def mean_and_95ci_bounds(values):
    """Return mean and the actual lower/upper bounds of the 95% CI"""
    if not values:
        return 0.0, 0.0, 0.0
    values_array = np.array(values)
    mu = np.mean(values_array)
    # Calculate actual 2.5th and 97.5th percentiles
    lower = np.percentile(values_array, 2.5)
    upper = np.percentile(values_array, 97.5)
    return mu, lower, upper

def safe_bool(x):
    return True if x is True else False



import os
import json
import random
import time
import uuid
import logging
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import datetime
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np

class CausalDiscovery:
    """
    Class for discovering causal relationships between variables using LLM.
    """
    def __init__(
                self,
                target_variable,
                temporal_scale,
                spatial_scale,
                yaml_path,
                dev_mode=False,

                # Model names
                generator_model=None,
                corruptor_model=None,
                judge_model=None,
                context=None,

                # Sampling parameters for each role
                generator_temperature=0.7,
                generator_top_p=1.0,
                corruptor_temperature=0.7,
                corruptor_top_p=1.0,
                judge_temperature=0.7,
                judge_top_p=1.0
            ):
                """
                Initialize the causal discovery algorithm.

                Args:
                    target_variable: The target variable for causal discovery
                    temporal_scale: Temporal scale for analysis
                    spatial_scale: Spatial scale for analysis
                    yaml_path: Path to the prompt template YAML file
                    dev_mode: Whether to run in development mode

                    generator_model: Model name for the generator LLM (provider inferred from name)
                    corruptor_model: Model name for the corruptor LLM (provider inferred from name)
                    judge_model: Model name for the judge LLM (defaults to Claude if not specified)

                    generator_temperature: Temperature for the generator LLM
                    generator_top_p: Top-p for the generator LLM
                    corruptor_temperature: Temperature for the corruptor LLM
                    corruptor_top_p: Top-p for the corruptor LLM
                    judge_temperature: Temperature for the judge LLM
                    judge_top_p: Top-p for the judge LLM
                """
                # 1) Connect to Neo4j
                self.graph_db = Neo4jClient(
                    uri=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    logger=logging.getLogger('app.db.neo4j')
                )

                # 2) Load environment variables for each provider
                perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY", 'None')
                perplexity_api_url = os.environ.get("PERPLEXITY_API_URL", 'None')
                perplexity_default_model = os.environ.get("PERPLEXITY_MODEL", 'None')

                claude_api_key = os.environ.get("CLAUDE_API_KEY", 'None')
                claude_api_url = os.environ.get("CLAUDE_API_URL", 'None')
                claude_default_model = os.environ.get("CLAUDE_MODEL", 'None')

                openai_api_key = os.environ.get("OPENAI_API_KEY", 'None')
                openai_api_url = os.environ.get("OPENAI_API_URL", 'None')
                openai_default_model = os.environ.get("OPENAI_MODEL", 'None')

                # Track inference times/call counts
                self.inference_times = {
                    "generator": [],
                    "corruptor": [],
                    "judge": [],
                    "total": 0.0
                }
                self.api_call_counts = {
                    "generator": 0,
                    "corruptor": 0,
                    "judge": 0,
                    "total": 0
                }

                
                print(f"generator_model CausalDiscovery: {generator_model}")

                # 3) Resolve model names (default/fallback logic)
                #    If no generator_model, default to the perplexity_default_model
                if not generator_model:
                    generator_model = perplexity_default_model
                    self.generator_provider = "perplexity"
                else:
                    # Infer the provider from the model name
                    gen_lower = generator_model.lower()
                    if "claude" in gen_lower:
                        self.generator_provider = "claude"
                    elif "gpt" in gen_lower or "openai" in gen_lower:
                        self.generator_provider = "openai"
                    else:
                        self.generator_provider = "perplexity"

                if not corruptor_model:
                    corruptor_model = generator_model
                    self.corruptor_provider = self.generator_provider
                else:
                    corr_lower = corruptor_model.lower()
                    if "claude" in corr_lower:
                        self.corruptor_provider = "claude"
                    elif "gpt" in corr_lower or "openai" in corr_lower:
                        self.corruptor_provider = "openai"
                    else:
                        self.corruptor_provider = "perplexity"

                if not judge_model:
                    # If no judge model, default to claude_default_model
                    judge_model = claude_default_model

                # 4) Instantiate the generator LLM client
                if self.generator_provider == "perplexity":
                    self.nucleus = PerplexityClient(
                        api_key=perplexity_api_key,
                        api_url=perplexity_api_url,
                        model=generator_model,
                        dev_mode=dev_mode,
                        logger=logging.getLogger('app.lm.perplexity'),
                        temperature=generator_temperature,  # pass new param
                        top_p=generator_top_p               # pass new param
                    )
                    logger.info(f"Using Perplexity as generator LLM with model: {generator_model}")
                elif self.generator_provider == "openai":
                    self.nucleus = OpenAIClient(
                        api_key=openai_api_key,
                        api_url=openai_api_url,
                        model=generator_model,
                        dev_mode=dev_mode,
                        logger=logging.getLogger('app.llm.openai'),
                        temperature=generator_temperature,
                        top_p=generator_top_p,
                        enable_websearch=True
                    )
                    logger.info(f"Using OpenAI as generator LLM with model: {generator_model}")
                else:  # claude
                    self.nucleus = ClaudeClient(
                        api_key=claude_api_key,
                        api_url=claude_api_url,
                        model=generator_model,
                        dev_mode=dev_mode,
                        logger=logging.getLogger('app.llm.claude'),
                        temperature=generator_temperature,
                        top_p=generator_top_p
                    )
                    logger.info(f"Using Claude as generator LLM with model: {generator_model}")


                print(f"corruptor_model CausalDiscovery: {generator_model}")

                # 5) Instantiate the corruptor LLM
                #    If corruptor_model == generator_model, we reuse self.nucleus
                if corruptor_model != generator_model:
                    if self.corruptor_provider == "perplexity":
                        self.corruptor_llm = PerplexityClient(
                            api_key=perplexity_api_key,
                            api_url=perplexity_api_url,
                            model=corruptor_model,
                            dev_mode=dev_mode,
                            logger=logging.getLogger('app.lm.perplexity.corruptor'),
                            temperature=corruptor_temperature,
                            top_p=corruptor_top_p
                        )
                        logger.info(f"Using Perplexity as corruptor LLM with model: {corruptor_model}")
                    elif self.corruptor_provider == "openai":
                        self.corruptor_llm = OpenAIClient(
                            api_key=openai_api_key,
                            api_url=openai_api_url,
                            model=corruptor_model,
                            dev_mode=dev_mode,
                            logger=logging.getLogger('app.llm.openai.corruptor'),
                            temperature=corruptor_temperature,
                            top_p=corruptor_top_p
                        )
                        logger.info(f"Using OpenAI as corruptor LLM with model: {corruptor_model}")
                    else:  # claude
                        self.corruptor_llm = ClaudeClient(
                            api_key=claude_api_key,
                            api_url=claude_api_url,
                            model=corruptor_model,
                            dev_mode=dev_mode,
                            logger=logging.getLogger('app.llm.claude.corruptor'),
                            temperature=corruptor_temperature,
                            top_p=corruptor_top_p
                        )
                        logger.info(f"Using Claude as corruptor LLM with model: {corruptor_model}")
                else:
                    self.corruptor_llm = self.nucleus
                    logger.info(f"Using same model for generator and corruptor: {corruptor_model}")

                # 6) Instantiate the judge LLM
                #    We always create a fresh client because it may be a different model
                #    (or at least with different temperature).
                #    We'll assume "judge" = "claude" if the name includes "claude"
                j_lower = judge_model.lower()
                if "claude" in j_lower:
                    self.judge_llm = ClaudeClient(
                        api_key=claude_api_key,
                        api_url=claude_api_url,
                        model=judge_model,
                        dev_mode=dev_mode,
                        logger=logging.getLogger('app.llm.claude.judge'),
                        temperature=judge_temperature,
                        top_p=judge_top_p
                    )
                    logger.info(f"Using Claude as judge LLM with model: {judge_model}")
                elif "gpt" in j_lower or "openai" in j_lower:
                    self.judge_llm = OpenAIClient(
                        api_key=openai_api_key,
                        api_url=openai_api_url,
                        model=judge_model,
                        dev_mode=dev_mode,
                        logger=logging.getLogger('app.llm.openai.judge'),
                        temperature=judge_temperature,
                        top_p=judge_top_p
                    )
                    logger.info(f"Using OpenAI as judge LLM with model: {judge_model}")
                else:
                    # Default to Perplexity if neither "claude" nor "gpt/openai" in name
                    self.judge_llm = PerplexityClient(
                        api_key=perplexity_api_key,
                        api_url=perplexity_api_url,
                        model=judge_model,
                        dev_mode=dev_mode,
                        logger=logging.getLogger('app.llm.perplexity.judge'),
                        temperature=judge_temperature,
                        top_p=judge_top_p
                    )
                    logger.info(f"Using Perplexity as judge LLM with model: {judge_model}")

                # 7) Initialize PromptTree
                self.tree = PromptTree(yaml_file=yaml_path, logger=logger)

                # Store key parameters
                self.target_variable = target_variable
                self.temporal_scale = temporal_scale
                self.spatial_scale = spatial_scale
                self.context = context

                # Store sampling parameters for reference (optional)
                self.generator_temperature = generator_temperature
                self.generator_top_p = generator_top_p
                self.corruptor_temperature = corruptor_temperature
                self.corruptor_top_p = corruptor_top_p
                self.judge_temperature = judge_temperature
                self.judge_top_p = judge_top_p

                # Session & variables
                self.session_id = str(uuid.uuid4())
                self.variables = []
                self.processed_pairs = set()

                # Set the session ID in Neo4j
                self.graph_db.set_session_id(self.session_id)

                logger.info(f"Initialized CausalDiscovery with session_id={self.session_id}")
                logger.info(f"Target: {self.target_variable}, Temporal: {self.temporal_scale}, Spatial: {self.spatial_scale}")
                logger.info(f"Generator temperature={generator_temperature}, top_p={generator_top_p}")
                logger.info(f"Corruptor temperature={corruptor_temperature}, top_p={corruptor_top_p}")
                logger.info(f"Judge temperature={judge_temperature}, top_p={judge_top_p}")
        
    def generate_variables(self, num_variables: int = 5) -> List[str]:
        """Generate variables related to the target variable."""
        logger.info(f"Generating {num_variables} variables related to {self.target_variable}")
        
        # Start with the target variable
        self.variables = []
        
        # Create the target node in the graph
        try:
            self.graph_db.create_node(
                node_label="variable",
                node_properties={
                    "name": self.target_variable,
                    "definition": "Pain in the lower back region",
                    "citations": [],
                    "target": True,
                    "deleted": False,
                    "session_id": self.session_id
                }
            )
            logger.info(f"Created target node: {self.target_variable}")
            self.variables.append(self.target_variable)
        except Exception as e:
            logger.error(f"Error creating target node: {e}")
            raise
        
        # Generate additional variables using PromptTree
        variable_count = 1  # We already have the target variable
        
        while variable_count < num_variables:
            try:
                # ===== 1) "variableGenerator" expects: target, temporal, spatial, context  =====
                sys_prompt, usr_prompt, usr_prompt_extension  = self.tree.build_prompt(
                    id="variableGenerator",
                    variable_dict={
                        "target": self.target_variable,
                        "temporal": self.temporal_scale,
                        "spatial": self.spatial_scale,
                        "context": self.context
                    }
                )

                # Append excluded variables at the end of the user prompt
                excluded_str = ", ".join(self.variables)
                full_usr_prompt = f"{usr_prompt}\n{usr_prompt_extension} {excluded_str}"    
                response = self.nucleus.send_message(
                    sys_prompt=sys_prompt,
                    usr_prompt=full_usr_prompt,
                    stream=False
                )
            
                # Parse the response
                data = self.nucleus.parse_static_response(response)
                content = data['content']
                citations = data.get('citations', [])
                
                # Process the variable response
                name, definition, citations = self.tree.read_variable_response(content, citations)
                
                # Skip if already in variables
                if name in self.variables:
                    logger.warning(f"Variable {name} already exists, skipping")
                    continue
                
                # Create node in the graph
                self.graph_db.create_node(
                    node_label="variable",
                    node_properties={
                        "name": name,
                        "definition": definition,
                        "citations": citations,
                        "target": False,
                        "deleted": False,
                        "session_id": self.session_id
                    }
                )
                
                logger.info(f"Created variable node: {name}")
                self.variables.append(name)
                variable_count += 1
                
                # Sleep to avoid hitting API rate limits
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error generating variable {variable_count+1}: {e}")
                # Retry with a different variable instead of failing completely
                continue
        
        logger.info(f"Finished generating variables. Total: {len(self.variables)}")
        logger.info(f"Generated {len(self.variables)} variables: {self.variables}")
        return self.variables

    # def generate_variable_pairs_old(self) -> List[Tuple[str, str]]:
    #     """Generate all pairs of variables for relationship discovery."""
    #     pairs = []
    #     for i, source in enumerate(self.variables):
    #         for target in self.variables[i+1:]:
    #             # analyze all possible pairs
    #             pairs.append((source, target))
        
    #     logger.info(f"Generated {len(pairs)} variable pairs")
    #     return pairs

    def generate_variable_pairs(self,
        *,
        ordered:       bool = True,
        include_self:  bool = False,
    ) -> List[Tuple[str, str]]:
        """
        Return a list of (source, target) variable pairs.

        Parameters
        ----------
        variables : iterable of str
            The variable names.
        ordered : bool, default True
            • True  – produce ordered pairs:  (A,B) and (B,A) are distinct.  
            • False – produce unordered pairs: only one of (A,B)/(B,A) kept.

        include_self : bool, default False
            • True  – include self-loops (A,A), (B,B), …  
            • False – skip self-loops.

        Returns
        -------
        List[Tuple[str, str]]
        """
        vars_list = list(self.variables)          # ensure we can index

        pairs: List[Tuple[str, str]] = []

        if ordered:
            # O(N²) enumeration of every (source, target)
            for source in vars_list:
                for target in vars_list:
                    if not include_self and source == target:
                        continue
                    pairs.append((source, target))
        else:
            # Unordered: walk only the upper triangle
            for i, source in enumerate(vars_list):
                for target in vars_list[i + (0 if include_self else 1):]:
                    pairs.append((source, target))

        return pairs
        
    def process_variable_pair(self, pair: Tuple[str, str], corruption_rate: float = 0, existing_variables: List[str] = None) -> Optional[Tuple[str, str, str, str, str]]:
        """
        Process a single pair of variables to determine relationship.
        
        Args:
            pair: Tuple of (source, target) variable names
            corruption_rate: Rate at which to generate spurious motivations (0-1)
            existing_variables: List of variable names to check for mediation
        
        Returns:
            A 5-tuple: (source, target, relationship_type, original_motivation, spurious_motivation)
            or None if no relationship found or already processed.
        """
        source, target = pair
        
        # If existing_variables is None, fetch them from database
        # Fetch all existing variables in the CLD
        all_variables = self.get_variables()
        existing_variables = [var["name"] for var in all_variables]
        logger.debug(f"Fetched {len(existing_variables)} variables for mediation check")
        
        # Log potential mediators
        if existing_variables:
            logger.info(f"\n{'='*40}")
            logger.info(f"MEDIATOR CHECK: Evaluating {source} → {target}")
            logger.info(f"Considering {len(existing_variables)} potential mediators: {existing_variables}")
            logger.info(f"{'='*40}\n")

        try:
            # Fetch variable definitions from the database
            source_node = self.graph_db.fetch_nodes(
                node_label="variable",
                filter_property="name",
                filter_value=source,
                return_fields=["definition"]
            )
            source_def = source_node[0]["definition"] if source_node and "definition" in source_node[0] else ""
            
            target_node = self.graph_db.fetch_nodes(
                node_label="variable",
                filter_property="name",
                filter_value=target,
                return_fields=["definition"]
            )
            target_def = target_node[0]["definition"] if target_node and "definition" in target_node[0] else ""
            
            # 1) Build the prompt for 'generateEdges' with complete information
            sys_prompt, usr_prompt, _ = self.tree.build_prompt(
                id="generateEdges",
                variable_dict={
                    'var1': source, 
                    'var2': target, 
                    'def1': source_def,
                    'def2': target_def,
                    'temporal': self.temporal_scale,
                    'spatial': self.spatial_scale,
                    'variables': existing_variables
                }
            )

            # Send request to the LLM
            logger.info(f"Analyzing relationship between {source} and {target}")
            response = self.nucleus.send_message(
                sys_prompt=sys_prompt,
                usr_prompt=usr_prompt,
                stream=False,
            )



            # Parse the response
            data = self.nucleus.parse_static_response(response)
            logger.info(f"Returned data: {data}")
            content = data['content']
            citations = data.get('citations', [])
            logger.info(f"Citations: {citations}")
            print('citations', citations)

            # Process the relationship
            relationship, motivation, citations = self.tree.read_relationships(content, citations)

            if relationship != "NONE":
                # Possibly generate a spurious motivation
                # 2) "generateSpuriousMotivation" => uses: source, target, relationship, motivation
                spurious_motivation = None
                if random.random() < corruption_rate:
                    spurious_motivation = self.generate_spurious_motivation(
                        source, target, relationship, motivation
                    )
                    logger.info(f"Added spurious motivation for {source}-[{relationship}]->{target}")

                # Use the spurious motivation for the DB update
                edge_motivation = spurious_motivation if spurious_motivation else motivation

                # Create relationship in the graph
                # We'll store both the "motivation" (or spurious) and "citations"
                self.graph_db.create_relationships(
                    node1_name=source,
                    node2_name=target,
                    node_label="variable",
                    edge_properties=[edge_motivation, citations],
                    relationship_type=relationship
                )

                logger.info(f"Created relationship: {source}-[{relationship}]->{target}")

                # Return the entire set
                return (source, target, relationship, motivation, spurious_motivation)

            else:
                logger.debug(f"No relationship found between {source} and {target}")
                return None

        except Exception as e:
            logger.error(f"Error processing pair {source}->{target}: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            return None

    
    def discover_relationships(
        self,
        parallel: bool = False,
        max_workers: int = 3,
        corruption_rate: float = 0
    ) -> List[Tuple[str, str, str]]:
        """
        Discover causal relationships between variables with optional spurious motivations.
        
        Returns:
            A list of (source, target, relationship_type).
        """
        logger.info("Discovering causal relationships between variables")
        
        # Generate all pairs of variables
        pairs = self.generate_variable_pairs()

        all_variables = self.get_variables()

        existing_variables = [var["name"] for var in all_variables]
        
        relationships = []
        corrupted_relationships = {}
        corrupted_count = 0
        
        # Process pairs in parallel or sequentially
        if parallel and len(pairs) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_pair = {
                    executor.submit(self.process_variable_pair, pair, corruption_rate, existing_variables): pair
                    for pair in pairs
                }
                # Process results as they complete
                for future in as_completed(future_to_pair):
                    result = future.result()
                    if result:
                        (src, tgt, rel_type, orig_motivation, spur_motivation) = result
                        relationships.append((src, tgt, rel_type))
                        
                        # Track corrupted relationships
                        if spur_motivation:
                            corrupted_count += 1
                            edge_id = f"{src}-{rel_type}->{tgt}"
                            corrupted_relationships[edge_id] = {
                                "source": src,
                                "target": tgt,
                                "type": rel_type,
                                "original_motivation": orig_motivation,
                                "spurious_motivation": spur_motivation
                            }
        else:
            # Process sequentially
            for pair in pairs:
                result = self.process_variable_pair(pair, corruption_rate, existing_variables)
                if result:
                    (src, tgt, rel_type, orig_motivation, spur_motivation) = result
                    relationships.append((src, tgt, rel_type))
                    
                    # Track corrupted relationships
                    if spur_motivation:
                        corrupted_count += 1
                        edge_id = f"{src}-{rel_type}->{tgt}"
                        corrupted_relationships[edge_id] = {
                            "source": src,
                            "target": tgt,
                            "type": rel_type,
                            "original_motivation": orig_motivation,
                            "spurious_motivation": spur_motivation
                        }
                    
                # Sleep to avoid hitting API rate limits
                time.sleep(1)
        
        
        logger.info(f"Discovered {len(existing_variables)} variables existing")
        logger.info(f"Discovered {len(relationships)} relationships, {corrupted_count} with spurious motivations")
        
        # Save the corrupted relationships metadata if needed
        if corrupted_relationships:
            metadata_file = f"corrupted_relationships_{self.session_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(corrupted_relationships, f, indent=2)
            logger.info(f"Saved corrupted relationships metadata to {metadata_file}")
        
        # Store stats
        self.corrupted_count = corrupted_count
        
        return relationships
    
    def create_graph_representation(
        self,
        variables: List[str],
        relationships: List[Tuple[str, str, str]]
    ) -> Dict:
        """Create a graph representation that can be easily visualized."""
        nodes = []
        edges = []
        
        # Add nodes
        for variable in variables:
            nodes.append({
                "id": variable,
                "target": (variable == self.target_variable)
            })
        
        # Add edges
        for source, target, rel_type in relationships:
            edges.append({
                "source": source,
                "target": target,
                "type": rel_type
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def generate_spurious_motivation(
        self,
        source: str,
        target: str,
        rel_type: str,
        original_motivation: str
    ) -> str:
        """
        Generate a spurious motivation for a relationship that has logical flaws.
        """
        try:
            # Build prompt using PromptTree
            sys_prompt, usr_prompt, _ = self.tree.build_prompt(
                id="generateSpuriousMotivation",
                variable_dict={
                    'source': source, 
                    'target': target, 
                    'relationship': rel_type,
                    'motivation': original_motivation
                }
            )
            
            # Send request to the LLM
            logger.info(f"Generating spurious motivation for {source}-[{rel_type}]->{target}")
            response = self.nucleus.send_message(
                sys_prompt=sys_prompt,
                usr_prompt=usr_prompt,
                stream=False
            )
            
            # Parse the response
            data = self.nucleus.parse_static_response(response)
            content = data['content']
            
            if not content.strip():
                logger.warning("Empty spurious motivation from LLM; using fallback text.")
                return "This relationship is supported by flawed reasoning that incorrectly assumes causality."
            
            return content.strip()
            
        except Exception as e:
            logger.info(f"Error generating spurious motivation: {e}")
            # If generation fails, add a generic spurious statement
            return "This relationship is supported by flawed reasoning that incorrectly assumes causality."

    def _get_edge_motivation(self, source: str, target: str, rel_type: str) -> str:
        """
        Get the original motivation for an edge from the database using a session-based query.
        Returns an empty string if not found.
        """
        try:
            with self.graph_db._get_session() as session:
                query = """
                MATCH (n:variable {name: $source})-[r]->(m:variable {name: $target})
                WHERE type(r) = $rel_type 
                  AND n.session_id = $session_id
                  AND m.session_id = $session_id
                RETURN properties(r) as props
                """
                params = {
                    "source": source,
                    "target": target,
                    "rel_type": rel_type,
                    "session_id": self.session_id
                }
                
                result = session.run(query, params).single()
                
                if not result:
                    logger.warning(f"No relationship found for {source}-[{rel_type}]->{target}")
                    return ""
                
                rel_props = result["props"] or {}
                
                # Look for 'motivation' or synonyms
                for key in ["motivation", "motivations"]:
                    if key in rel_props and rel_props[key]:
                        return str(rel_props[key])
                
                logger.warning(f"No motivation property found in edge props: {rel_props}")
                return ""
                
        except Exception as e:
            logger.error(f"Error getting edge motivation: {e}")
            return ""

    def _update_edge_with_spurious_motivation(
        self, source: str, target: str, rel_type: str, spurious_motivation: str
    ) -> bool:
        """Update an edge with spurious motivation in the Neo4j database."""
        try:
            with self.graph_db._get_session() as session:
                query = """
                MATCH (s:variable {name: $source_name})-[r]->(t:variable {name: $target_name})
                WHERE type(r) = $rel_type AND s.session_id = $session_id AND t.session_id = $session_id
                SET r.spurious_motivation = $spurious_motivation, r.is_corrupted = true
                RETURN count(r) as updated_count
                """
                params = {
                    "source_name": source,
                    "target_name": target,
                    "rel_type": rel_type,
                    "session_id": self.session_id,
                    "spurious_motivation": spurious_motivation
                }
                
                result = session.run(query, params).single()
                updated_count = result["updated_count"] if result else 0
                
                if updated_count > 0:
                    logger.info(f"Added spurious motivation to relationship: {source}-[{rel_type}]->{target}")
                    return True
                else:
                    logger.warning(f"No relationship found to update for {source}-[{rel_type}]->{target}")
                    return False
        except Exception as e:
            logger.error(f"Error updating edge with spurious motivation: {e}")
            return False

    def _corrupt_relationship(self, source, target, rel_type, original_motivation):
        """Generate a spurious but convincing alternative motivation for a relationship."""
        try:
            sys_prompt, usr_prompt, _ = self.tree.build_prompt(
                id="corruptor",
                variable_dict={
                    "source": source,
                    "target": target,
                    "relationship_type": rel_type,
                    "original_motivation": original_motivation
                }
            )
            
            # Use the corruptor LLM instead of the generator
            response = self.corruptor_llm.send_message(sys_prompt, usr_prompt, stream=False)
            data = self.corruptor_llm.parse_static_response(response)
            
            # Extract the generated spurious motivation
            spurious_motivation = data["content"].strip()
            
            # Update the relationship in the graph with the spurious motivation
            self.update_relationship_properties(
                source=source,
                target=target,
                relationship_type=rel_type,
                properties={
                    "is_corrupted": True,
                    "original_motivation": original_motivation,
                    "spurious_motivation": spurious_motivation,
                    "motivation": spurious_motivation  # Replace the visible motivation
                }
            )
            
            return True, spurious_motivation
        except Exception as e:
            logger.error(f"Error corrupting relationship {source}->{target}: {e}")
            return False, None
        
    def _corrupt_relationships(self, corruption_rate: float) -> int:
        """
        Corrupt a fraction of the relationships with spurious motivations.
        
        Returns:
            The number of relationships actually corrupted.
        """
        try:
            logger.info(f"Starting to corrupt relationships with rate {corruption_rate}")
            
            # Get all relationships for this session
            with self.graph_db._get_session() as session:
                query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $session_id AND t.session_id = $session_id
                RETURN s.name as source, t.name as target, type(r) as rel_type
                """
                
                results = list(session.run(query, {"session_id": self.session_id}))
                logger.info(f"Found {len(results)} relationships to potentially corrupt")
                
                # Determine how many to corrupt
                num_to_corrupt = int(len(results) * corruption_rate)
                logger.info(f"Planning to corrupt {num_to_corrupt} relationships")
                
                corrupted_count = 0
                corrupted_data = {"count": 0, "corrupted_edges": []}
                
                # Shuffle to pick random relationships
                random.shuffle(results)
                
                for record in results:
                    if corrupted_count >= num_to_corrupt:
                        break
                    
                    source = record["source"]
                    target = record["target"]
                    rel_type = record["rel_type"]
                    
                    original_motivation = self._get_edge_motivation(source, target, rel_type)
                    if not original_motivation:
                        logger.warning(f"No motivation found for {source}-[{rel_type}]->{target}")
                        continue
                    
                    # Generate a spurious motivation
                    spurious_motivation = self.generate_spurious_motivation(
                        source, target, rel_type, original_motivation
                    )
                    
                    if not spurious_motivation:
                        logger.warning(f"Failed to generate spurious motivation for {source}-[{rel_type}]->{target}")
                        continue
                    
                    # Update in the database
                    success = self._update_edge_with_spurious_motivation(
                        source, target, rel_type, spurious_motivation
                    )
                    if not success:
                        continue
                    
                    # Add to corrupted data
                    corrupted_data["corrupted_edges"].append({
                        "source": source,
                        "target": target,
                        "type": rel_type,
                        "original_motivation": original_motivation,
                        "spurious_motivation": spurious_motivation
                    })
                    corrupted_count += 1
                    corrupted_data["count"] = corrupted_count
                
                # Save the corrupted relationships metadata
                if corrupted_count > 0:
                    metadata_file = f"corrupted_relationships_{self.session_id}.json"
                    logger.info(f"Saving corrupted metadata to {metadata_file}")
                    with open(metadata_file, 'w') as f:
                        json.dump(corrupted_data, f, indent=2)
                    
                    # Optionally store in-memory
                    self.corrupted_relationships_data = corrupted_data
            
            logger.info(f"Successfully corrupted {corrupted_count} relationships")
            return corrupted_count
        
        except Exception as e:
            logger.error(f"Error corrupting relationships: {e}")
            return 0
        

    def run(
    self,
    num_variables: int = 5,
    parallel: bool = False,
    max_workers: int = 3,
    corruption_rate: float = 0.0,
    run_judgment: bool = False,
    judge_models: Optional[Union[List[str], str]] = None,
    num_judges: Optional[int] = None, 
    starting_variables: Optional[List[str]] = None
) -> Dict:
        """
        Run the causal discovery algorithm end-to-end with optional serial judging.
        
        Args:
            num_variables: Number of variables to generate
            parallel: Whether to process relationships in parallel
            max_workers: Maximum number of worker threads for parallel processing
            corruption_rate: Rate of corrupting relationships with spurious motivations
            run_judgment: Whether to run judgment on all edges after discovery
            judge_models: List of model names to use for judging, or a single model name
            num_judges: Number of judge calls to make
        
        Returns:
            A dictionary containing session info, variables, relationships, the final graph, stats,
            and judgment results if run_judgment is True
        """


        start_time = time.time()
        logger.info(f"Starting causal discovery algorithm with session ID: {self.session_id}")
        
        try:
            # Generate variables
            if not starting_variables:
                variables = self.generate_variables(num_variables)
            else:
                variables = starting_variables
            logger.info(f"Generated {len(variables)} variables: {variables}")
            
            # Discover relationships (including inline spurious motivations if corruption_rate>0)
            relationships = self.discover_relationships(
                parallel=parallel,
                max_workers=max_workers,
                corruption_rate=corruption_rate
            )
            logger.info(f"Discovered {len(relationships)} relationships")
            
            # If corruption_rate > 0, we can do a separate pass to randomly corrupt more edges
            # (optional depending on your use case)
            corrupted_count = 0
            if corruption_rate > 0:
                # This will randomly corrupt additional edges by overwriting them
                corrupted_count = self._corrupt_relationships(corruption_rate)
                logger.info(f"Corrupted {corrupted_count} relationships in a separate pass")
            
            # Build graph representation
            graph = {
                "nodes": [
                    {
                        "id": var,
                        "label": var,
                        "target": (var == self.target_variable)
                    } 
                    for var in variables
                ],
                "edges": [
                    {
                        "source": s,
                        "target": t,
                        "type": r
                    }
                    for (s, t, r) in relationships
                ]
            }
            
            result = {
                "session_id": self.session_id,
                "target_variable": self.target_variable,
                "temporal_scale": self.temporal_scale,
                "spatial_scale": self.spatial_scale,
                "variables": variables,
                "relationships": relationships,
                "graph": graph,
                "stats": {
                    "variable_count": len(variables),
                    "relationship_count": len(relationships),
                    "corrupted_count": corrupted_count,
                    "elapsed_time": 0  # Will be updated below
                }
            }
            
            # Run judgment if requested
            if run_judgment:
                logger.info(f"Running judgment on all edges with {'serial' if num_judges or judge_models else 'single'} judge")
                judgment_start_time = time.time()
                
                judgment_results = self.judge_all_edges_serial(
                    judge_models=judge_models,
                    num_judges=num_judges
                )
                
                judgment_time = time.time() - judgment_start_time
                logger.info(f"Judgment completed in {judgment_time:.2f} seconds")
                
                # Add judgment results to the output
                result["judgment"] = {
                    "results": judgment_results,
                    "stats": {
                        "total_edges": len(judgment_results),
                        "ok_edges": sum(1 for edge in judgment_results if edge["judge_verdict"] == "OK"),
                        "inconsistent_edges": sum(1 for edge in judgment_results if edge["judge_verdict"] == "INCONSISTENT"),
                        "no_motivation_edges": sum(1 for edge in judgment_results if edge["judge_verdict"] == "NO_MOTIVATION"),
                        "error_edges": sum(1 for edge in judgment_results if edge["judge_verdict"] not in ["OK", "INCONSISTENT", "NO_MOTIVATION"]),
                        "elapsed_time": judgment_time
                    }
                }
                
                # Update graph with judgment results
                for edge in result["graph"]["edges"]:
                    matching_judgments = [j for j in judgment_results 
                                        if j["source"] == edge["source"] and 
                                            j["target"] == edge["target"] and 
                                            j["type"] == edge["type"]]
                    if matching_judgments:
                        edge["judge_verdict"] = matching_judgments[0]["judge_verdict"]
            
            # Calculate and record total elapsed time
            elapsed_time = time.time() - start_time
            result["stats"]["elapsed_time"] = elapsed_time
            logger.info(f"Causal discovery completed in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in causal discovery: {e}")
            return {
                "error": str(e),
                "session_id": self.session_id
            }

    def fill_in_missing_citations(
    self,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> None:
        """
        For every edge missing citations, ask the LLM for evidence and parse references 
        via the "citations" array from parse_static_response(). If none found, treat as NO_EVIDENCE.
        """
        import os
        import logging

        logger = logging.getLogger(__name__)
        if logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)

        #
        # 1) Decide which model to use. (Fallback to judge_llm if none given.)
        #
        judge_llm = getattr(self, "judge_llm", None)
        chosen_provider = llm_provider or getattr(judge_llm, "provider", "anthropic")  
        chosen_model    = llm_model    or getattr(judge_llm, "model",    "claude-3-7-sonnet-20250219")

        # Example environment usage
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
        OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "your-claude-api-key")
        CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
        PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "your-perplexity-api-key")
        PERPLEXITY_API_URL = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/v1/chat/completions")

        logger.info("[CITATION-FILL] Using provider=%s model=%s", chosen_provider, chosen_model)

        #
        # 2) Build the LLM client
        #
        try:
            llm = self._build_llm_client(
                provider=chosen_provider,
                model=chosen_model,
                dev_mode=getattr(self, "dev_mode", False),
                temperature=0.7,
                openai_key=OPENAI_API_KEY,
                openai_url=OPENAI_API_URL,
                claude_key=CLAUDE_API_KEY,
                claude_url=CLAUDE_API_URL,
                perplexity_key=PERPLEXITY_API_KEY,
                perplexity_url=PERPLEXITY_API_URL,
                top_p=1.0,
                role_name="citation_fill"
            )
        except Exception as e:
            logger.exception("[CITATION-FILL] Failed to initialize LLM client; aborting.")
            return

        #
        # 3) Find edges lacking citations
        #
        query = """
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id = $sid
        AND t.session_id = $sid
        AND (r.citations IS NULL OR size(r.citations) = 0)
        RETURN s.name AS src,
            t.name AS tgt,
            type(r) AS rel_type,
            r       AS edge_props
        """
        with self.graph_db._get_session() as sess:
            records = list(sess.run(query, {"sid": self.session_id}))

        if not records:
            logger.info("[CITATION-FILL] No edges found that lack citations.")
            return
        logger.info("[CITATION-FILL] Found %d edges missing citations.", len(records))

        #
        # 4) Build the system prompt
        #
        sys_prompt = (
            "You are an expert literature-search assistant. Provide 1–5 reputable scientific URLs "
            "that support the following claim, plus a brief explanation of relevance (1–2 sentences). "
            "If you find no evidence, reply exactly NO_EVIDENCE.\n"
            "Output must be in one of two forms:\n\n"
            "NO_EVIDENCE\n\n"
            "-- or --\n\n"
            "CITATION: <URL>\n"
            "RELEVANCE: <short statement>\n"
        )

        #
        # 5) For each edge, ask the LLM & parse the structured citations
        #
        for record in records:
            src   = record["src"]
            tgt   = record["tgt"]
            rel   = record["rel_type"]
            props = record.get("edge_props") or {}

            # Use spurious_motivation if available, else fallback
            claim_txt = props.get("spurious_motivation") or props.get("motivation", "")
            claim_txt = claim_txt.strip()
            if not claim_txt:
                logger.debug("[CITATION-FILL] Edge %s-%s->%s has no motivation text; skipping.", src, rel, tgt)
                continue

            # Build user prompt
            usr_prompt = f"CLAIM:\n{claim_txt}\n\nAnswer now."

            try:
                # 5a) Call the LLM
                response = llm.send_message(
                    sys_prompt=sys_prompt,
                    usr_prompt=usr_prompt,
                    stream=False
                )

                # 5b) Parse result
                data = llm.parse_static_response(response)
                content   = (data.get("content") or "").strip()
                citations = data.get("citations", [])  # a list of URLs or empty

                logger.debug("[CITATION-FILL] LLM raw content: %s", content)
                logger.debug("[CITATION-FILL] LLM extracted citations: %r", citations)

                # If no citations, treat it as NO_EVIDENCE
                if not citations:
                    if content.upper().startswith("NO_EVIDENCE"):
                        logger.info("[CITATION-FILL] Edge %s-%s->%s => NO_EVIDENCE", src, rel, tgt)
                    else:
                        logger.info("[CITATION-FILL] Edge %s-%s->%s => No citations extracted; skipping.", src, rel, tgt)
                    continue

                # citations is already a Python list; we store it directly
                self._update_edge_citation_info(src, tgt, rel, citations, "Auto-generated relevance here")
                logger.info("[CITATION-FILL] Patched %s-%s->%s => CITATION=%r", src, rel, tgt, citations)

            except Exception as e:
                logger.exception("[CITATION-FILL] Edge %s-%s->%s => LLM call or parse failed: %s",
                                src, rel, tgt, e)

        logger.info("[CITATION-FILL] Completed fill_in_missing_citations().")


    def _update_edge_citation_info(self, src, tgt, rel_type, citations, relevance):
        """
        Helper method to set r.citations as a proper Neo4j list.
        'citations' must be a Python list of URLs.
        """
        with self.graph_db._get_session() as sess:
            sess.run(
                """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $sid
                AND t.session_id = $sid
                AND s.name = $src
                AND t.name = $tgt
                AND type(r) = $rel_type
                SET r.citations = $citations,
                    r.relevance = $relevance
                """,
                {
                    "sid": self.session_id,
                    "src": src,
                    "tgt": tgt,
                    "rel_type": rel_type,
                    "citations": citations,  # must be a Python list
                    "relevance": relevance,
                }
            )

        
    def get_edge_properties(self, print_all=False) -> List:
        """
        Retrieve all edge properties in the current session for debugging.

        Args:
            print_all: If True, prints the results to stdout.

        Returns:
            A list of records with 'source', 'target', 'rel_type', and 'edge_props'.
        """
        try:
            with self.graph_db._get_session() as session:
                query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE r.session_id = $session_id
                RETURN s.name as source, t.name as target, type(r) as rel_type, 
                       properties(r) as edge_props
                """
                params = {"session_id": self.session_id}
                
                results = list(session.run(query, params))
                
                if print_all:
                    print(f"\nFound {len(results)} relationships in session {self.session_id}:")
                    for i, record in enumerate(results, 1):
                        print(f"\n{i}. {record['source']}-[{record['rel_type']}]->{record['target']}")
                        print(f"   Properties: {record['edge_props']}")
                
                return results
        except Exception as e:
            logger.error(f"Error retrieving edge properties: {e}")
            return []

    def inspect_relationships(self) -> List:
        """
        Display all relationships with their properties for the current session.
        
        Returns:
            A list of records with 'source', 'target', 'relationship_type', and 'properties'.
        """
        try:
            with self.graph_db._get_session() as session:
                query = """
                MATCH (n:variable)-[r]->(m:variable)
                WHERE n.session_id = $session_id AND m.session_id = $session_id
                RETURN n.name as source, m.name as target, type(r) as relationship_type, 
                       properties(r) as properties
                """
                
                results = list(session.run(query, {"session_id": self.session_id}))
                
                print(f"\nFound {len(results)} relationships in session {self.session_id}:")
                for i, record in enumerate(results, 1):
                    print(f"\n{i}. {record['source']}-[{record['relationship_type']}]->{record['target']}")
                    print(f"   Properties: {record['properties']}")
                
                return results
        except Exception as e:
            print(f"Error inspecting relationships: {e}")
            return []
        
    def get_corrupted_relationships(self) -> Dict:
        """
        Retrieve information about corrupted relationships for the current session.
        
        Returns:
            Dict containing:
            - "count": Number of corrupted relationships
            - "corrupted_edges": A list of dictionaries with details:
                    {
                        "source": str,
                        "target": str,
                        "type": str,
                        "original_motivation": str,
                        "spurious_motivation": str
                    }
        """
        try:
            # If we have in-memory data (from the second pass), return that directly
            if hasattr(self, 'corrupted_relationships_data'):
                return self.corrupted_relationships_data
            
            # Otherwise, check if the corrupted relationships file exists
            metadata_file = f"corrupted_relationships_{self.session_id}.json"
            if not os.path.exists(metadata_file):
                logger.warning(f"No corrupted relationships file found: {metadata_file}")
                return {"count": 0, "corrupted_edges": []}
            
            # Read the corrupted relationships from the file
            with open(metadata_file, 'r') as f:
                corrupted_data = json.load(f)
            
            # If the data is in the shape {"count": X, "corrupted_edges": [...]}:
            if "corrupted_edges" in corrupted_data:
                return corrupted_data
            else:
                # Otherwise, it's likely in the shape {edgeId: {...}, edgeId2: {...}}
                edges = []
                for _, edge_info in corrupted_data.items():
                    edges.append({
                        "source": edge_info["source"],
                        "target": edge_info["target"],
                        "type": edge_info["type"],
                        "original_motivation": edge_info["original_motivation"],
                        "spurious_motivation": edge_info["spurious_motivation"]
                    })
                return {
                    "count": len(edges),
                    "corrupted_edges": edges
                }
        except Exception as e:
            logger.error(f"Error retrieving corrupted relationships: {e}")
            return {"count": 0, "corrupted_edges": []}
        
    

    def judge_all_edges(self) -> List[Dict]:
        """
        Judge every edge in the current session. 
        If 'spurious_motivation' is present, judge that text;
        otherwise, judge the 'motivation' property.

        The judge_verdict and judge_message are stored exactly like spurious_motivation,
        i.e., directly on the edge: r.judge_verdict = <...>, r.judge_message = <...>.
        
        Returns a list of dictionaries describing each judged edge.
        """
        logger = logging.getLogger(__name__)
        judged_edges = []

        query = """
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id = $session_id
        AND t.session_id = $session_id
        RETURN s.name AS source,
            t.name AS target,
            type(r) AS rel_type,
            properties(r) AS props
        """
        params = {"session_id": self.session_id}

        try:
            with self.graph_db._get_session() as session:
                records = list(session.run(query, params))

            if not records:
                logger.info("No edges found in this session to judge.")
                return []

            for record in records:
                src = record["source"]
                tgt = record["target"]
                rel_type = record["rel_type"]
                props = record.get("props") or {}

                # 1) Decide which text to judge
                text_to_judge = props.get("spurious_motivation", "").strip()
                if not text_to_judge:
                    text_to_judge = props.get("motivation", "").strip()

                # 2) If there's nothing, store NO_MOTIVATION
                if not text_to_judge:
                    judge_verdict = "NO_MOTIVATION"
                    judge_message = "Edge has no motivation text."
                    self._update_edge_with_judge_verdict(src, tgt, rel_type, judge_verdict, judge_message)
                    judged_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "judged_text": "",
                        "judge_verdict": judge_verdict,
                        "judge_message": judge_message
                    })
                    continue

                # 3) Call your Claude-based LLM (instead of Perplexity) to get a verdict
                try:
                    sys_prompt, usr_prompt, _ = self.tree.build_prompt(
                        id="edgeVerifier",
                        variable_dict={"edge_description": text_to_judge}
                    )
                    response = self._time_llm_call(
                        llm_client=current_judge,
                        sys_prompt=sys_prompt,
                        usr_prompt=usr_prompt,
                        llm_type="judge",
                        stream=False
                    )
                    data = self.judge_llm.parse_static_response(response)
                    llm_content = data["content"].strip()
                except Exception as e:
                    logger.error(f"Error calling LLM for edge {src}->{tgt}: {e}")
                    llm_content = f"ERROR: {e}"

                # 4) Parse the LLM's result
                if llm_content.startswith("OK"):
                    judge_verdict = "OK"
                    judge_message = ""
                elif llm_content.startswith("INCONSISTENT"):
                    parts = llm_content.split(":", 1)
                    judge_verdict = "INCONSISTENT"
                    judge_message = parts[1].strip() if len(parts) > 1 else "No reason"
                else:
                    judge_verdict = "INCONSISTENT"
                    judge_message = f"Unexpected response: {llm_content[:100]}"

                # 5) Update the relationship with our verdict
                self._update_edge_with_judge_verdict(src, tgt, rel_type, judge_verdict, judge_message)

                # 6) Collect a summary for return
                judged_edges.append({
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "judged_text": text_to_judge[:100],
                    "judge_verdict": judge_verdict,
                    "judge_message": judge_message
                })

            return judged_edges

        except Exception as e:
            logger.error(f"Error in judge_all_edges: {e}")
            return []

    def _update_edge_with_judge_verdict(self, src, tgt, rel_type, verdict, message, aggregate_score=None) -> bool:
        """
        Update an edge with judge_verdict and judge_message in the same 
        style as spurious_motivation was set.
        """
        try:
            with self.graph_db._get_session() as session:
                query = """
                MATCH (s:variable {name: $source})-[r]->(t:variable {name: $target})
                WHERE type(r) = $rel_type
                AND s.session_id = $session_id
                AND t.session_id = $session_id
                SET r.judge_verdict = $verdict,
                    r.judge_message = $message,
                    r.aggregate_score = $score
                RETURN count(r) AS updated_count
                """
                params = {
                    "source": src,  # Changed from source to src
                    "target": tgt,  # Changed from target to tgt
                    "rel_type": rel_type,
                    "session_id": self.session_id,
                    "verdict": verdict,
                    "message": message,
                    "score": aggregate_score
                }
                result = session.run(query, params).single()
                updated_count = result["updated_count"] if result else 0

            if updated_count > 0:
                logger.info(f"Judge verdict updated on {src}-[{rel_type}]->{tgt}")  # Changed from source/target to src/tgt
                return True
            else:
                logger.warning(f"No matching edge found for {src}-[{rel_type}]->{tgt}")  # Changed from source/target to src/tgt
                return False

        except Exception as e:
            logger.error(f"Failed to update judge verdict for {src}->{tgt}: {e}")  # Changed from source/target to src/tgt
            return False

    
    def update_relationship_properties(
        self, source: str, target: str, relationship_type: str, properties: Dict
    ) -> bool:
        """
        Update (SET) arbitrary properties on an existing relationship.
        
        Returns True if at least one relationship was updated, False otherwise.
        """
        try:
            # Build a dynamic SET clause from the properties dictionary
            set_clause = ", ".join([f"r.{key} = ${key}" for key in properties.keys()])
            
            query = f"""
            MATCH (s:variable {{name: $source}})-[r]->(t:variable {{name: $target}})
            WHERE type(r) = $relationship_type 
              AND r.session_id = $session_id
            SET {set_clause}
            RETURN count(r) as updated_count
            """
            
            # Prepare parameters, including session and basic match constraints
            params = {
                "source": source,
                "target": target,
                "relationship_type": relationship_type,
                "session_id": self.session_id
            }
            # Add the properties themselves
            for key, value in properties.items():
                params[key] = value

            with self.graph_db._get_session() as session:
                result = session.run(query, params).single()
                updated_count = result["updated_count"] if result else 0
            
            if updated_count > 0:
                logger.info(f"Updated properties for relationship: {source}-[{relationship_type}]->{target}")
                return True
            else:
                logger.warning(f"No matching relationship found for {source}-[{relationship_type}]->{target} to update.")
                return False
        
        except Exception as e:
            logger.error(f"Error updating relationship properties for {source}-[{relationship_type}]->{target}: {e}")
            return False
        
    def judge_all_edges_serial(self, judge_models=None, num_judges=None) -> List[Dict]:
        """
        Judge every edge in the current session using multiple judges in series.
        
        Args:
            judge_models: List of model names to use for judging, or a single model name.
                        If None, uses the original judge_llm for all calls.
            num_judges: Number of judge calls to make. If None, uses the length of judge_models.
                    If judge_models is None or a string, defaults to 2.
        
        Returns a list of dictionaries describing each judged edge.
        """
        logger = logging.getLogger(__name__)
        judged_edges = []

        # Set up the judge models
        if judge_models is None:
            # Default to using the original judge_llm
            models = [getattr(self.judge_llm, 'model', 'claude-3-7-sonnet-20250219')] * (num_judges or 3)
        elif isinstance(judge_models, str):
            # Use the same model for all judges
            models = [judge_models] * (num_judges or 3)
        else:
            # Use the provided list of models - the list length determines the number of judges
            models = judge_models
            if num_judges is not None and num_judges != len(models):
                logger.warning(f"num_judges parameter ({num_judges}) is ignored when judge_models is a list. Using {len(models)} judges based on list length.")
        
        print(f"\n{'='*50}")
        print(f"SERIAL JUDGING: Starting with {len(models)} judges")
        print(f"SERIAL JUDGING: Using models: {models}")
        print(f"{'='*50}\n")
                
        # Create the judges
        judges = []
        for i, model in enumerate(models):
            if "claude" in model.lower():
                api_key = os.environ.get("CLAUDE_API_KEY", 'None')
                api_url = os.environ.get("CLAUDE_API_URL", 'None')
                judges.append(ClaudeClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.claude.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: Claude with model {model}")
            elif "gpt" in model.lower() or "openai" in model.lower():
                api_key = os.environ.get("OPENAI_API_KEY", 'None')
                api_url = os.environ.get("OPENAI_API_URL", 'None')
                judges.append(OpenAIClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.openai.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: OpenAI with model {model}")
            else:
                # Default to Perplexity for any other model name
                api_key = os.environ.get("PERPLEXITY_API_KEY", 'None')
                api_url = os.environ.get("PERPLEXITY_API_URL", 'None')
                judges.append(PerplexityClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.perplexity.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: Perplexity with model {model}")

        # Get all edges to judge
        query = """
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id = $session_id
        AND t.session_id = $session_id
        RETURN s.name AS source,
            t.name AS target,
            type(r) AS rel_type,
            properties(r) AS props
        """
        params = {"session_id": self.session_id}

        try:
            with self.graph_db._get_session() as session:
                records = list(session.run(query, params))

            if not records:
                print("SERIAL JUDGING: No edges found in this session to judge.")
                return []
            
            print(f"SERIAL JUDGING: Found {len(records)} edges to judge")
            print(f"{'='*50}")

            for record_idx, record in enumerate(records):
                src = record["source"]
                tgt = record["target"]
                rel_type = record["rel_type"]
                props = record.get("props") or {}
                
                print(f"\nSERIAL JUDGING: Edge {record_idx+1}/{len(records)}: {src}-[{rel_type}]->{tgt}")

                # 1) Decide which text to judge
                text_to_judge = props.get("spurious_motivation", "").strip()
                if not text_to_judge:
                    text_to_judge = props.get("motivation", "").strip()

                # 2) If there's nothing to judge, store NO_MOTIVATION
                if not text_to_judge:
                    judge_verdict = "NO_MOTIVATION"
                    judge_message = "Edge has no motivation text."
                    print(f"SERIAL JUDGING: No motivation text to judge")
                    self._update_edge_with_judge_verdict(src, tgt, rel_type, judge_verdict, judge_message)
                    judged_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "judged_text": "",
                        "judge_verdict": judge_verdict,
                        "judge_message": judge_message
                    })
                    continue

                # 3) Make a prompt for judging
                print(f"SERIAL JUDGING: Building prompt to judge text of length {len(text_to_judge)}")
                sys_prompt, usr_prompt, _ = self.tree.build_prompt(
                    id="edgeVerifier",
                    variable_dict={"edge_description": text_to_judge}
                )
                
                # 4) Call each judge in series
                judge_results = []
                for i, judge in enumerate(judges):
                    print(f"SERIAL JUDGING: Calling judge {i+1} ({models[i]}) for edge {src}->{tgt}")
                    try:
                        response = judge.send_message(sys_prompt, usr_prompt, stream=False)
                        data = judge.parse_static_response(response)
                        content = data["content"].strip()
                        
                        if content.startswith("OK"):
                            verdict = "OK"
                            message = ""
                        elif content.startswith("INCONSISTENT"):
                            parts = content.split(":", 1)
                            verdict = "INCONSISTENT"
                            message = parts[1].strip() if len(parts) > 1 else "No reason"
                        else:
                            verdict = "INCONSISTENT"
                            message = f"Unexpected response: {content[:100]}"
                        
                        judge_results.append({"verdict": verdict, "message": message})
                        print(f"SERIAL JUDGING: Judge {i+1} ({models[i]}) verdict: {verdict}")
                        if message:
                            print(f"SERIAL JUDGING: Judge {i+1} reason: {message[:100]}...")
                        
                    except Exception as e:
                        print(f"SERIAL JUDGING: Error with judge {i+1}: {e}")
                        judge_results.append({"verdict": "ERROR", "message": f"{e}"})
                
                # 5) Combine the judge results
                ok_count = sum(1 for r in judge_results if r["verdict"] == "OK")
                inconsistent_count = sum(1 for r in judge_results if r["verdict"] == "INCONSISTENT")
                error_count = sum(1 for r in judge_results if r["verdict"] == "ERROR")
                
                print(f"SERIAL JUDGING: Vote tally: {ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR")
                
                # Simple majority rule with conservative tie-breaking
                if ok_count > inconsistent_count + error_count:
                    judge_verdict = "OK"
                    judge_message = f"Majority of judges ({ok_count}/{len(judge_results)}) found this relationship valid."
                elif inconsistent_count > ok_count + error_count:
                    judge_verdict = "INCONSISTENT"
                    judge_message = f"Majority of judges ({inconsistent_count}/{len(judge_results)}) found issues with this relationship."
                elif ok_count == inconsistent_count:
                    # Tie between OK and INCONSISTENT - be conservative
                    judge_verdict = "INCONSISTENT"
                    judge_message = f"Split decision between judges ({ok_count} OK, {inconsistent_count} INCONSISTENT) - marking as inconsistent to be conservative."
                else:
                    # Handle cases with errors
                    if ok_count >= inconsistent_count:
                        judge_verdict = "OK"
                        judge_message = f"Relationship considered valid ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
                    else:
                        judge_verdict = "INCONSISTENT"
                        judge_message = f"Relationship considered inconsistent ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
                
                print(f"SERIAL JUDGING: Final verdict for {src}->{tgt}: {judge_verdict}")
                
                # Add detailed messages from each judge
                judge_details = []
                for i, result in enumerate(judge_results):
                    model_name = models[i].split('-')[-1] if '-' in models[i] else models[i]
                    if result["message"]:
                        judge_details.append(f"Judge {i+1} ({model_name}): {result['verdict']} - {result['message']}")
                    else:
                        judge_details.append(f"Judge {i+1} ({model_name}): {result['verdict']}")
                
                full_message = f"{judge_message} Details: {'; '.join(judge_details)}"
                
                # 6) Update the relationship with our verdict
                self._update_edge_with_judge_verdict(src, tgt, rel_type, judge_verdict, full_message)

                # 7) Collect a summary for return
                judged_edges.append({
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "judged_text": text_to_judge[:100] + "..." if len(text_to_judge) > 100 else text_to_judge,
                    "judge_verdict": judge_verdict,
                    "judge_message": full_message
                })

            print(f"\n{'='*50}")
            print(f"SERIAL JUDGING: Complete! Judged {len(judged_edges)} edges with {len(models)} judges")
            print(f"{'='*50}\n")
            return judged_edges

        except Exception as e:
            print(f"SERIAL JUDGING ERROR: {e}")
            return []
            logger.info(f"Generated {len(variables)} variables: {variables}")
            
            # Discover relationships (including inline spurious motivations if corruption_rate>0)
            relationships = self.discover_relationships(
                parallel=parallel,
                max_workers=max_workers,
                corruption_rate=corruption_rate
            )
            logger.info(f"Discovered {len(relationships)} relationships")
            
            # If corruption_rate > 0, we can do a separate pass to randomly corrupt more edges
            # (optional depending on your use case)
            corrupted_count = 0
            if corruption_rate > 0:
                # This will randomly corrupt additional edges by overwriting them
                corrupted_count = self._corrupt_relationships(corruption_rate)
                logger.info(f"Corrupted {corrupted_count} relationships in a separate pass")
            
            # Build graph representation
            graph = {
                "nodes": [
                    {
                        "id": var,
                        "label": var,
                        "target": (var == self.target_variable)
                    } 
                    for var in variables
                ],
                "edges": [
                    {
                        "source": s,
                        "target": t,
                        "type": r
                    }
                    for (s, t, r) in relationships
                ]
            }
            
            result = {
                "session_id": self.session_id,
                "target_variable": self.target_variable,
                "temporal_scale": self.temporal_scale,
                "spatial_scale": self.spatial_scale,
                "variables": variables,
                "relationships": relationships,
                "graph": graph,
                "stats": {
                    "variable_count": len(variables),
                    "relationship_count": len(relationships),
                    "corrupted_count": corrupted_count,
                    "elapsed_time": 0  # Will be updated below
                }
            }
            
            # Run judgment if requested
            if run_judgment:
                logger.info(f"Running judgment on all edges with {'serial' if num_judges or judge_models else 'single'} judge")
                judgment_start_time = time.time()
                
                judgment_results = self.judge_all_edges_serial(
                    judge_models=judge_models,
                    num_judges=num_judges
                )
                
                judgment_time = time.time() - judgment_start_time
                logger.info(f"Judgment completed in {judgment_time:.2f} seconds")
                
                # Add judgment results to the output
                result["judgment"] = {
                    "results": judgment_results,
                    "stats": {
                        "total_edges": len(judgment_results),
                        "ok_edges": sum(1 for edge in judgment_results if edge["judge_verdict"] == "OK"),
                        "inconsistent_edges": sum(1 for edge in judgment_results if edge["judge_verdict"] == "INCONSISTENT"),
                        "no_motivation_edges": sum(1 for edge in judgment_results if edge["judge_verdict"] == "NO_MOTIVATION"),
                        "error_edges": sum(1 for edge in judgment_results if edge["judge_verdict"] not in ["OK", "INCONSISTENT", "NO_MOTIVATION"]),
                        "elapsed_time": judgment_time
                    }
                }
                
                # Update graph with judgment results
                for edge in result["graph"]["edges"]:
                    matching_judgments = [j for j in judgment_results 
                                        if j["source"] == edge["source"] and 
                                            j["target"] == edge["target"] and 
                                            j["type"] == edge["type"]]
                    if matching_judgments:
                        edge["judge_verdict"] = matching_judgments[0]["judge_verdict"]
            
            # Calculate and record total elapsed time
            elapsed_time = time.time() - start_time
            result["stats"]["elapsed_time"] = elapsed_time
            logger.info(f"Causal discovery completed in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in causal discovery: {e}")
            return {
                "error": str(e),
                "session_id": self.session_id
            }
        
    def get_edge_properties(self, print_all=False) -> List:
        """
        Retrieve all edge properties in the current session for debugging.

        Args:
            print_all: If True, prints the results to stdout.

        Returns:
            A list of records with 'source', 'target', 'rel_type', and 'edge_props'.
        """
        try:
            with self.graph_db._get_session() as session:
                query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE r.session_id = $session_id
                RETURN s.name as source, t.name as target, type(r) as rel_type, 
                       properties(r) as edge_props
                """
                params = {"session_id": self.session_id}
                
                results = list(session.run(query, params))
                
                if print_all:
                    print(f"\nFound {len(results)} relationships in session {self.session_id}:")
                    for i, record in enumerate(results, 1):
                        print(f"\n{i}. {record['source']}-[{record['rel_type']}]->{record['target']}")
                        print(f"   Properties: {record['edge_props']}")
                
                return results
        except Exception as e:
            logger.error(f"Error retrieving edge properties: {e}")
            return []

    def inspect_relationships(self) -> List:
        """
        Display all relationships with their properties for the current session.
        
        Returns:
            A list of records with 'source', 'target', 'relationship_type', and 'properties'.
        """
        try:
            with self.graph_db._get_session() as session:
                query = """
                MATCH (n:variable)-[r]->(m:variable)
                WHERE n.session_id = $session_id AND m.session_id = $session_id
                RETURN n.name as source, m.name as target, type(r) as relationship_type, 
                       properties(r) as properties
                """
                
                results = list(session.run(query, {"session_id": self.session_id}))
                
                print(f"\nFound {len(results)} relationships in session {self.session_id}:")
                for i, record in enumerate(results, 1):
                    print(f"\n{i}. {record['source']}-[{record['relationship_type']}]->{record['target']}")
                    print(f"   Properties: {record['properties']}")
                
                return results
        except Exception as e:
            print(f"Error inspecting relationships: {e}")
            return []
        
    def get_corrupted_relationships(self) -> Dict:
        """
        Retrieve information about corrupted relationships for the current session.
        
        Returns:
            Dict containing:
            - "count": Number of corrupted relationships
            - "corrupted_edges": A list of dictionaries with details:
                    {
                        "source": str,
                        "target": str,
                        "type": str,
                        "original_motivation": str,
                        "spurious_motivation": str
                    }
        """
        try:
            # If we have in-memory data (from the second pass), return that directly
            if hasattr(self, 'corrupted_relationships_data'):
                return self.corrupted_relationships_data
            
            # Otherwise, check if the corrupted relationships file exists
            metadata_file = f"corrupted_relationships_{self.session_id}.json"
            if not os.path.exists(metadata_file):
                logger.warning(f"No corrupted relationships file found: {metadata_file}")
                return {"count": 0, "corrupted_edges": []}
            
            # Read the corrupted relationships from the file
            with open(metadata_file, 'r') as f:
                corrupted_data = json.load(f)
            
            # If the data is in the shape {"count": X, "corrupted_edges": [...]}:
            if "corrupted_edges" in corrupted_data:
                return corrupted_data
            else:
                # Otherwise, it's likely in the shape {edgeId: {...}, edgeId2: {...}}
                edges = []
                for _, edge_info in corrupted_data.items():
                    edges.append({
                        "source": edge_info["source"],
                        "target": edge_info["target"],
                        "type": edge_info["type"],
                        "original_motivation": edge_info["original_motivation"],
                        "spurious_motivation": edge_info["spurious_motivation"]
                    })
                return {
                    "count": len(edges),
                    "corrupted_edges": edges
                }
        except Exception as e:
            logger.error(f"Error retrieving corrupted relationships: {e}")
            return {"count": 0, "corrupted_edges": []}
        
    

    def judge_all_edges(self) -> List[Dict]:
        """
        Judge every edge in the current session. 
        If 'spurious_motivation' is present, judge that text;
        otherwise, judge the 'motivation' property.

        The judge_verdict and judge_message are stored exactly like spurious_motivation,
        i.e., directly on the edge: r.judge_verdict = <...>, r.judge_message = <...>.
        
        Returns a list of dictionaries describing each judged edge.
        """
        logger = logging.getLogger(__name__)
        judged_edges = []

        query = """
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id = $session_id
        AND t.session_id = $session_id
        RETURN s.name AS source,
            t.name AS target,
            type(r) AS rel_type,
            properties(r) AS props
        """
        params = {"session_id": self.session_id}

        try:
            with self.graph_db._get_session() as session:
                records = list(session.run(query, params))

            if not records:
                logger.info("No edges found in this session to judge.")
                return []

            for record in records:
                src = record["source"]
                tgt = record["target"]
                rel_type = record["rel_type"]
                props = record.get("props") or {}

                # 1) Decide which text to judge
                text_to_judge = props.get("spurious_motivation", "").strip()
                if not text_to_judge:
                    text_to_judge = props.get("motivation", "").strip()

                # 2) If there's nothing, store NO_MOTIVATION
                if not text_to_judge:
                    judge_verdict = "NO_MOTIVATION"
                    judge_message = "Edge has no motivation text."
                    self._update_edge_with_judge_verdict(src, tgt, rel_type, judge_verdict, judge_message)
                    judged_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "judged_text": "",
                        "judge_verdict": judge_verdict,
                        "judge_message": judge_message
                    })
                    continue

                # 3) Call your Claude-based LLM (instead of Perplexity) to get a verdict
                try:
                    sys_prompt, usr_prompt, _ = self.tree.build_prompt(
                        id="edgeVerifier",
                        variable_dict={"edge_description": text_to_judge}
                    )
                    response = self.judge_llm.send_message(sys_prompt, usr_prompt, stream=False)
                    data = self.judge_llm.parse_static_response(response)
                    llm_content = data["content"].strip()
                except Exception as e:
                    logger.error(f"Error calling LLM for edge {src}->{tgt}: {e}")
                    llm_content = f"ERROR: {e}"

                # 4) Parse the LLM's result
                if llm_content.startswith("OK"):
                    judge_verdict = "OK"
                    judge_message = ""
                elif llm_content.startswith("INCONSISTENT"):
                    parts = llm_content.split(":", 1)
                    judge_verdict = "INCONSISTENT"
                    judge_message = parts[1].strip() if len(parts) > 1 else "No reason"
                else:
                    judge_verdict = "INCONSISTENT"
                    judge_message = f"Unexpected response: {llm_content[:100]}"

                # 5) Update the relationship with our verdict
                self._update_edge_with_judge_verdict(src, tgt, rel_type, judge_verdict, judge_message)

                # 6) Collect a summary for return
                judged_edges.append({
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "judged_text": text_to_judge[:100],
                    "judge_verdict": judge_verdict,
                    "judge_message": judge_message
                })

            return judged_edges

        except Exception as e:
            logger.error(f"Error in judge_all_edges: {e}")
            return []

    def _update_edge_with_judge_verdict(
    self,
    src: str,
    tgt: str,
    rel_type: str,
    verdict: str,
    message: str,
    aggregate_score: float = None,
    aggregator_message: str = None
):
        """
        Update the edge in Neo4j with:
        - r.judge_verdict
        - r.judge_message
        - r.aggregate_score  (if provided, else skip)
        - r.aggregator_message (optional, if you want it)
        
        We keep the first five parameters the same as before, and add new
        optional parameters at the end (aggregate_score, aggregator_message).

        If aggregate_score or aggregator_message is None, we skip setting them.
        """
        # 1) Build a dynamic SET clause
        set_clauses = [
            "r.judge_verdict = $verdict",
            "r.judge_message = $message"
        ]
        if aggregate_score is not None:
            set_clauses.append("r.aggregate_score = $aggregate_score")
        if aggregator_message is not None:
            set_clauses.append("r.aggregator_message = $aggregator_message")

        set_clause_str = ", ".join(set_clauses)

        # 2) Construct the Cypher query
        query = f"""
        MATCH (s:variable {{name:$src, session_id:$session_id}})-[r:{rel_type}]->(t:variable {{name:$tgt, session_id:$session_id}})
        SET {set_clause_str}
        """

        # 3) Build params
        params = {
            "src": src,
            "tgt": tgt,
            "session_id": self.session_id,
            "verdict": verdict,
            "message": message,
            "aggregate_score": aggregate_score,
            "aggregator_message": aggregator_message
        }

        # 4) Execute
        with self.graph_db._get_session() as session:
            session.run(query, params)

        print(
            f"UPDATED EDGE: ({src})-[:{rel_type}]->({tgt}) | "
            f"verdict='{verdict}' | "
            f"aggregate_score={aggregate_score} | aggregator_message={aggregator_message}"
        )


    
    def update_relationship_properties(
        self, source: str, target: str, relationship_type: str, properties: Dict
    ) -> bool:
        """
        Update (SET) arbitrary properties on an existing relationship.
        
        Returns True if at least one relationship was updated, False otherwise.
        """
        try:
            # Build a dynamic SET clause from the properties dictionary
            set_clause = ", ".join([f"r.{key} = ${key}" for key in properties.keys()])
            
            query = f"""
            MATCH (s:variable {{name: $source}})-[r]->(t:variable {{name: $target}})
            WHERE type(r) = $relationship_type 
              AND r.session_id = $session_id
            SET {set_clause}
            RETURN count(r) as updated_count
            """
            
            # Prepare parameters, including session and basic match constraints
            params = {
                "source": source,
                "target": target,
                "relationship_type": relationship_type,
                "session_id": self.session_id
            }
            # Add the properties themselves
            for key, value in properties.items():
                params[key] = value

            with self.graph_db._get_session() as session:
                result = session.run(query, params).single()
                updated_count = result["updated_count"] if result else 0
            
            if updated_count > 0:
                logger.info(f"Updated properties for relationship: {source}-[{relationship_type}]->{target}")
                return True
            else:
                logger.warning(f"No matching relationship found for {source}-[{relationship_type}]->{target} to update.")
                return False
        
        except Exception as e:
            logger.error(f"Error updating relationship properties for {source}-[{relationship_type}]->{target}: {e}")
            return False
        
    def judge_all_edges_serial(self, judge_models=None, num_judges=None) -> List[Dict]:
        """
        Judge every edge in the current session using multiple judges in series.
        
        Args:
            judge_models: List of model names to use for judging, or a single model name.
                        If None, uses the original judge_llm for all calls.
            num_judges: Number of judge calls to make. If None, uses the length of judge_models.
                    If judge_models is None or a string, defaults to 2.
        
        Returns a list of dictionaries describing each judged edge.
        """
        logger = logging.getLogger(__name__)
        judged_edges = []

        # Set up the judge models
        if judge_models is None:
            # Default to using the original judge_llm
            models = [getattr(self.judge_llm, 'model', 'claude-3-7-sonnet-20250219')] * (num_judges or 3)
        elif isinstance(judge_models, str):
            # Use the same model for all judges
            models = [judge_models] * (num_judges or 3)
        else:
            # Use the provided list of models - the list length determines the number of judges
            models = judge_models
            if num_judges is not None and num_judges != len(models):
                logger.warning(f"num_judges parameter ({num_judges}) is ignored when judge_models is a list. Using {len(models)} judges based on list length.")
        
        print(f"\n{'='*50}")
        print(f"SERIAL JUDGING: Starting with {len(models)} judges")
        print(f"SERIAL JUDGING: Using models: {models}")
        print(f"{'='*50}\n")
                
        # Create the judges
        judges = []
        for i, model in enumerate(models):
            if "claude" in model.lower():
                api_key = os.environ.get("CLAUDE_API_KEY", 'None')
                api_url = os.environ.get("CLAUDE_API_URL", 'None')
                judges.append(ClaudeClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.claude.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: Claude with model {model}")
            elif "gpt" in model.lower() or "openai" in model.lower():
                api_key = os.environ.get("OPENAI_API_KEY", 'None')
                api_url = os.environ.get("OPENAI_API_URL", 'None')
                judges.append(OpenAIClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.openai.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: OpenAI with model {model}")
            else:
                # Default to Perplexity for any other model name
                api_key = os.environ.get("PERPLEXITY_API_KEY", 'None')
                api_url = os.environ.get("PERPLEXITY_API_URL", 'None')
                judges.append(PerplexityClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.perplexity.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: Perplexity with model {model}")

        # Get all edges to judge
        query = """
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id = $session_id
        AND t.session_id = $session_id
        RETURN s.name AS source,
            t.name AS target,
            type(r) AS rel_type,
            properties(r) AS props
        """
        params = {"session_id": self.session_id}

        try:
            with self.graph_db._get_session() as session:
                records = list(session.run(query, params))

            if not records:
                print("SERIAL JUDGING: No edges found in this session to judge.")
                return []
            
            print(f"SERIAL JUDGING: Found {len(records)} edges to judge")
            print(f"{'='*50}")

            for record_idx, record in enumerate(records):
                src = record["source"]
                tgt = record["target"]
                rel_type = record["rel_type"]
                props = record.get("props") or {}
                
                print(f"\nSERIAL JUDGING: Edge {record_idx+1}/{len(records)}: {src}-[{rel_type}]->{tgt}")

                # 1) Decide which text to judge
                text_to_judge = props.get("spurious_motivation", "").strip()
                if not text_to_judge:
                    text_to_judge = props.get("motivation", "").strip()

                # 2) If there's nothing to judge, store NO_MOTIVATION
                if not text_to_judge:
                    judge_verdict = "NO_MOTIVATION"
                    judge_message = "Edge has no motivation text."
                    print(f"SERIAL JUDGING: No motivation text to judge")
                    self._update_edge_with_judge_verdict(src, tgt, rel_type, judge_verdict, judge_message)
                    judged_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "judged_text": "",
                        "judge_verdict": judge_verdict,
                        "judge_message": judge_message
                    })
                    continue

                # 3) Make a prompt for judging
                print(f"SERIAL JUDGING: Building prompt to judge text of length {len(text_to_judge)}")
                sys_prompt, usr_prompt, _ = self.tree.build_prompt(
                    id="edgeVerifier",
                    variable_dict={"edge_description": text_to_judge}
                )
                
                # 4) Call each judge in series
                judge_results = []
                for i, judge in enumerate(judges):
                    print(f"SERIAL JUDGING: Calling judge {i+1} ({models[i]}) for edge {src}->{tgt}")
                    try:
                        response = judge.send_message(sys_prompt, usr_prompt, stream=False)
                        data = judge.parse_static_response(response)
                        content = data["content"].strip()
                        
                        if content.startswith("OK"):
                            verdict = "OK"
                            message = ""
                        elif content.startswith("INCONSISTENT"):
                            parts = content.split(":", 1)
                            verdict = "INCONSISTENT"
                            message = parts[1].strip() if len(parts) > 1 else "No reason"
                        else:
                            verdict = "INCONSISTENT"
                            message = f"Unexpected response: {content[:100]}"
                        
                        judge_results.append({"verdict": verdict, "message": message})
                        print(f"SERIAL JUDGING: Judge {i+1} ({models[i]}) verdict: {verdict}")
                        if message:
                            print(f"SERIAL JUDGING: Judge {i+1} reason: {message[:100]}...")
                        
                    except Exception as e:
                        print(f"SERIAL JUDGING: Error with judge {i+1}: {e}")
                        judge_results.append({"verdict": "ERROR", "message": f"{e}"})
                
                # 5) Combine the judge results
                ok_count = sum(1 for r in judge_results if r["verdict"] == "OK")
                inconsistent_count = sum(1 for r in judge_results if r["verdict"] == "INCONSISTENT")
                error_count = sum(1 for r in judge_results if r["verdict"] == "ERROR")
                
                print(f"SERIAL JUDGING: Vote tally: {ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR")
                
                # Simple majority rule with conservative tie-breaking
                if ok_count > inconsistent_count + error_count:
                    judge_verdict = "OK"
                    judge_message = f"Majority of judges ({ok_count}/{len(judge_results)}) found this relationship valid."
                elif inconsistent_count > ok_count + error_count:
                    judge_verdict = "INCONSISTENT"
                    judge_message = f"Majority of judges ({inconsistent_count}/{len(judge_results)}) found issues with this relationship."
                elif ok_count == inconsistent_count:
                    # Tie between OK and INCONSISTENT - be conservative
                    judge_verdict = "INCONSISTENT"
                    judge_message = f"Split decision between judges ({ok_count} OK, {inconsistent_count} INCONSISTENT) - marking as inconsistent to be conservative."
                else:
                    # Handle cases with errors
                    if ok_count >= inconsistent_count:
                        judge_verdict = "OK"
                        judge_message = f"Relationship considered valid ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
                    else:
                        judge_verdict = "INCONSISTENT"
                        judge_message = f"Relationship considered inconsistent ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
                
                print(f"SERIAL JUDGING: Final verdict for {src}->{tgt}: {judge_verdict}")
                
                # Add detailed messages from each judge
                judge_details = []
                for i, result in enumerate(judge_results):
                    model_name = models[i].split('-')[-1] if '-' in models[i] else models[i]
                    if result["message"]:
                        judge_details.append(f"Judge {i+1} ({model_name}): {result['verdict']} - {result['message']}")
                    else:
                        judge_details.append(f"Judge {i+1} ({model_name}): {result['verdict']}")
                
                full_message = f"{judge_message} Details: {'; '.join(judge_details)}"
                
                # 6) Update the relationship with our verdict
                self._update_edge_with_judge_verdict(src, tgt, rel_type, judge_verdict, full_message)

                # 7) Collect a summary for return
                judged_edges.append({
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "judged_text": text_to_judge[:100] + "..." if len(text_to_judge) > 100 else text_to_judge,
                    "judge_verdict": judge_verdict,
                    "judge_message": full_message
                })

            print(f"\n{'='*50}")
            print(f"SERIAL JUDGING: Complete! Judged {len(judged_edges)} edges with {len(models)} judges")
            print(f"{'='*50}\n")
            return judged_edges

        except Exception as e:
            print(f"SERIAL JUDGING ERROR: {e}")
            return []
    
    # def judge_all_edges_with_citations_serial(self, judge_models=None, num_judges=None) -> List[Dict]:
    #     """
    #     Judge every edge in the current session using multiple judges in series, 
    #     but also fetch citations for each edge and include them in the LLM prompt.

    #     The LLM(s) should respond in the format:
    #     - "OK"
    #     - or "INCONSISTENT: SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|UNADDRESSED <brief reason>"

    #     We gather responses from all judges and apply a majority vote:
    #     - If most say "OK", final verdict = "OK"
    #     - Otherwise final verdict = "INCONSISTENT"

    #     We store the final 'judge_verdict' and 'judge_message' on each relationship in Neo4j.
        
    #     Args:
    #         judge_models: List of model names to use for judging, or a single model name.
    #                     If None, uses the original judge_llm for all calls.
    #         num_judges: Number of judge calls to make. If None, uses the length of judge_models.
    #                     If judge_models is None or a string, defaults to 2.

    #     Returns:
    #         A list of dictionaries describing each judged edge.
    #     """
    #     logger = logging.getLogger(__name__)
    #     judged_edges = []

    #     # 1) Determine the judge models
    #     if judge_models is None:
    #         # Default to using the original judge_llm's model (arbitrary fallback name if not set)
    #         fallback_model = getattr(self.judge_llm, "model", "claude-3-7-sonnet-20250219")
    #         models = [fallback_model] * (num_judges or 2)
    #     elif isinstance(judge_models, str):
    #         # Use the same model for all judges
    #         models = [judge_models] * (num_judges or 2)
    #     else:
    #         # Use the provided list of models
    #         models = judge_models
    #         if num_judges is not None and num_judges != len(models):
    #             logger.warning(
    #                 f"num_judges parameter ({num_judges}) is ignored when judge_models is a list. "
    #                 f"Using {len(models)} judges based on list length."
    #             )

    #     print(f"\n{'='*50}")
    #     print(f"CITATION JUDGING: Starting with {len(models)} judges")
    #     print(f"CITATION JUDGING: Using models: {models}")
    #     print(f"{'='*50}\n")

    #     # 2) Create the LLM clients for each judge
    #     judges = []
    #     for i, model in enumerate(models):
    #         model_lower = model.lower()
    #         if "claude" in model_lower:
    #             api_key = os.environ.get("CLAUDE_API_KEY", "None")
    #             api_url = os.environ.get("CLAUDE_API_URL", "None")
    #             judges.append(
    #                 ClaudeClient(
    #                     api_key=api_key,
    #                     api_url=api_url,
    #                     model=model,
    #                     dev_mode=getattr(self, "dev_mode", False),
    #                     logger=logging.getLogger(f"app.llm.claude.judge{i}")
    #                 )
    #             )
    #             print(f"CITATION JUDGING: Created Judge {i+1}: Claude with model {model}")
    #         elif "gpt" in model_lower or "openai" in model_lower:
    #             api_key = os.environ.get("OPENAI_API_KEY", "None")
    #             api_url = os.environ.get("OPENAI_API_URL", "None")
    #             judges.append(
    #                 OpenAIClient(
    #                     api_key=api_key,
    #                     api_url=api_url,
    #                     model=model,
    #                     dev_mode=getattr(self, "dev_mode", False),
    #                     logger=logging.getLogger(f"app.llm.openai.judge{i}")
    #                 )
    #             )
    #             print(f"CITATION JUDGING: Created Judge {i+1}: OpenAI with model {model}")
    #         else:
    #             # Default to Perplexity for any other model name
    #             api_key = os.environ.get("PERPLEXITY_API_KEY", "None")
    #             api_url = os.environ.get("PERPLEXITY_API_URL", "None")
    #             judges.append(
    #                 PerplexityClient(
    #                     api_key=api_key,
    #                     api_url=api_url,
    #                     model=model,
    #                     dev_mode=getattr(self, "dev_mode", False),
    #                     logger=logging.getLogger(f"app.llm.perplexity.judge{i}")
    #                 )
    #             )
    #             print(f"CITATION JUDGING: Created Judge {i+1}: Perplexity with model {model}")

    #     # 3) Retrieve all edges in the session
    #     query = """
    #     MATCH (s:variable)-[r]->(t:variable)
    #     WHERE s.session_id = $session_id AND t.session_id = $session_id
    #     RETURN s.name AS source,
    #         t.name AS target,
    #         type(r) AS rel_type,
    #         properties(r) AS props
    #     """
    #     params = {"session_id": self.session_id}

    #     try:
    #         with self.graph_db._get_session() as session:
    #             records = list(session.run(query, params))

    #         if not records:
    #             print("CITATION JUDGING: No edges found in this session to judge.")
    #             return []

    #         print(f"CITATION JUDGING: Found {len(records)} edges to judge")
    #         print(f"{'='*50}")

    #         # 4) Process each edge
    #         for idx, record in enumerate(records, start=1):
    #             src = record["source"]
    #             tgt = record["target"]
    #             rel_type = record["rel_type"]
    #             props = record.get("props") or {}

    #             print(f"\nCITATION JUDGING: Edge {idx}/{len(records)}: {src}-[{rel_type}]->{tgt}")

    #             # (a) Extract text & citations
    #             text_to_judge = props.get("spurious_motivation", "") or props.get("motivation", "")
    #             citations = props.get("citations", []) or props.get("citation", [])
    #             text_to_judge = text_to_judge.strip()

    #             # (b) If no motivation or no citations => skip or mark as NO_MOTIVATION
    #             if not text_to_judge or not citations:
    #                 verdict = "NO_MOTIVATION"
    #                 reason = "Edge has no motivation or citations to judge."
    #                 print(f"CITATION JUDGING: {verdict} - {reason}")
    #                 self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
    #                 judged_edges.append(
    #                     {
    #                         "source": src,
    #                         "target": tgt,
    #                         "type": rel_type,
    #                         "judged_text": "",
    #                         "judge_verdict": verdict,
    #                         "judge_message": reason,
    #                     }
    #                 )
    #                 continue

    #             print(f"CITATION JUDGING: Found {len(citations)} citations and motivation length {len(text_to_judge)}")

    #             # (c) Fetch citation texts
    #             try:
    #                 processor = CitationProcessor(timeout=15, max_workers=3)
    #                 citation_results = processor.process_citations(citations, fetch_content=True)
    #                 citation_content = processor.prepare_for_llm(citation_results)

    #                 # Format the citations for the prompt
    #                 # Format citations for the prompt
    #                 formatted_citations = ""
    #                 if isinstance(citation_content, dict):
    #                     for url, content in citation_content.items():
    #                         formatted_citations += f"\n\n--- CITATION: {url} ---\n{content}\n---"
    #                 else:
    #                     # Handle case where it's a list
    #                     for citation in citation_content:
    #                         formatted_citations += f"\n\n--- CITATION: {citation.url} ---\n{citation.content or ''}\n---"
    #             except Exception as e:
    #                 verdict = "INCONSISTENT"
    #                 reason = f"Error fetching citations: {e}"
    #                 print(f"CITATION JUDGING ERROR: {reason}")
    #                 self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
    #                 judged_edges.append(
    #                     {
    #                         "source": src,
    #                         "target": tgt,
    #                         "type": rel_type,
    #                         "judged_text": text_to_judge[:100],
    #                         "judge_verdict": verdict,
    #                         "judge_message": reason,
    #                     }
    #                 )
    #                 continue

    #             # (d) Build the judgeCitation prompt
    #             variables = {
    #                 "source": src,
    #                 "target": tgt,
    #                 "relationship": rel_type,
    #                 "motivation": text_to_judge,
    #                 "citations": formatted_citations,
    #             }
    #             sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCitation", variables)

    #             # (e) Call each judge in series
    #             judge_results = []
    #             for j_idx, judge_client in enumerate(judges):
    #                 current_model_name = models[j_idx]
    #                 print(f"CITATION JUDGING: Calling judge {j_idx+1} ({current_model_name}) for {src}->{tgt}")
    #                 try:
    #                     response = judge_client.send_message(sys_prompt, usr_prompt, stream=False)
    #                     data = judge_client.parse_static_response(response)
    #                     llm_content = data.get("content", "").strip()

    #                     # Parse the response: expect "OK" or "INCONSISTENT:..."
    #                     if llm_content.startswith("OK"):
    #                         v = "OK"
    #                         m = ""
    #                     elif llm_content.startswith("INCONSISTENT:"):
    #                         v = "INCONSISTENT"
    #                         m = llm_content[len("INCONSISTENT:") :].strip()
    #                     else:
    #                         v = "INCONSISTENT"
    #                         m = f"Unexpected response: {llm_content[:100]}"

    #                     judge_results.append({"verdict": v, "message": m})
    #                     print(f"CITATION JUDGING: Judge {j_idx+1} verdict => {v}, reason => {m[:100]}...")

    #                 except Exception as e:
    #                     print(f"CITATION JUDGING: Error with judge {j_idx+1} ({current_model_name}): {e}")
    #                     judge_results.append({"verdict": "ERROR", "message": str(e)})

    #             # (f) Combine the results => majority vote
    #             ok_count = sum(1 for r in judge_results if r["verdict"] == "OK")
    #             inconsistent_count = sum(1 for r in judge_results if r["verdict"] == "INCONSISTENT")
    #             error_count = sum(1 for r in judge_results if r["verdict"] == "ERROR")

    #             print(f"CITATION JUDGING: Vote tally => {ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR")

    #             if ok_count > inconsistent_count + error_count:
    #                 final_verdict = "OK"
    #                 final_reason = f"Majority of judges ({ok_count}/{len(judge_results)}) found this relationship consistent."
    #             elif inconsistent_count > ok_count + error_count:
    #                 final_verdict = "INCONSISTENT"
    #                 final_reason = f"Majority of judges ({inconsistent_count}/{len(judge_results)}) found this relationship inconsistent."
    #             elif ok_count == inconsistent_count:
    #                 # Tie between OK and INCONSISTENT => be conservative => INCONSISTENT
    #                 final_verdict = "INCONSISTENT"
    #                 final_reason = (
    #                     f"Tie between OK and INCONSISTENT => marking as inconsistent to be conservative."
    #                 )
    #             else:
    #                 # Cases involving errors
    #                 # If OK >= INCONSISTENT, we consider it OK, else inconsistent
    #                 if ok_count >= inconsistent_count:
    #                     final_verdict = "OK"
    #                     final_reason = (
    #                         f"Determined OK ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
    #                     )
    #                 else:
    #                     final_verdict = "INCONSISTENT"
    #                     final_reason = (
    #                         f"Determined inconsistent ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
    #                     )

    #             # Collect details from each judge for full_message
    #             judge_details = []
    #             for j_idx, r in enumerate(judge_results):
    #                 model_name = models[j_idx]
    #                 v = r["verdict"]
    #                 msg = r["message"]
    #                 detail = f"Judge {j_idx+1} ({model_name}): {v}"
    #                 if msg:
    #                     detail += f" - {msg}"
    #                 judge_details.append(detail)

    #             full_message = f"{final_reason} Details: {'; '.join(judge_details)}"
    #             print(f"CITATION JUDGING: Final verdict => {final_verdict}")

    #             # (g) Update the relationship in DB
    #             self._update_edge_with_judge_verdict(src, tgt, rel_type, final_verdict, full_message)

    #             # (h) Collect summary
    #             judged_edges.append(
    #                 {
    #                     "source": src,
    #                     "target": tgt,
    #                     "type": rel_type,
    #                     "judged_text": text_to_judge[:100] + "..." if len(text_to_judge) > 100 else text_to_judge,
    #                     "judge_verdict": final_verdict,
    #                     "judge_message": full_message,
    #                 }
    #             )

    #         print(f"\n{'='*50}")
    #         print(f"CITATION JUDGING: Complete! Judged {len(judged_edges)} edges with {len(models)} judges")
    #         print(f"{'='*50}\n")
    #         return judged_edges

    #     except Exception as e:
    #         print(f"CITATION JUDGING ERROR (outer loop): {e}")
            # return []

    def judge_all_edges_with_citations_serial(
    self,
    judge_models=None,
    num_judges=None,
    approach: str = "per_citation_aggregate"
) -> List[Dict]:
        """
        A single function that can do EITHER mode, controlled by 'approach':

        1) "all_citations_at_once" (original logic):
        - For each edge, build one big prompt with all citations.
        - Each LLM returns either "OK" or "INCONSISTENT:..."
        - We do majority voting across the LLMs to decide final "OK" or "INCONSISTENT".

        2) "per_citation_aggregate" (new logic):
        - For each citation, call each LLM individually, parse a partial verdict:
            [SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|UNADDRESSED], then apply a
            majority vote among LLMs to pick the single-citation verdict + numeric score (1.0, 0.5, 0.0).
        - Then we do a second aggregator step, where each LLM sees the partial results
            and returns "Fully supported"/"Partially supported"/"Not supported". Again,
            we do a majority vote among LLMs for the final aggregator verdict.
        - We store a JSON with per-citation details + aggregate_score + aggregator_verdict.

        Args:
            judge_models: List of model names to use for judging, or a single model name.
                        If None, uses the original judge_llm for all calls.
            num_judges: Number of judge calls to make. If None, uses len(judge_models).
                        If judge_models is None or a string, defaults to 2.
            approach: "all_citations_at_once" or "per_citation_aggregate".

        Returns:
            A list of dicts describing the final judgment of each edge.
        """
        import os
        import re
        import json
        import logging
        from collections import Counter

        logger = logging.getLogger(__name__)
        judged_edges: List[Dict] = []

        # ---------------------------------------------------------------
        # 1) Determine which judge models to use (multi-model approach)
        # ---------------------------------------------------------------
        if judge_models is None:
            # fallback from your existing judge_llm
            fallback_model = getattr(self.judge_llm, "model", "gpt-4")
            models = [fallback_model] * (num_judges or 2)
        elif isinstance(judge_models, str):
            models = [judge_models] * (num_judges or 2)
        else:
            models = judge_models
            if num_judges is not None and num_judges != len(models):
                logger.warning(
                    f"num_judges parameter ({num_judges}) is ignored when judge_models is a list. "
                    f"Using {len(models)} judges based on list length."
                )

        print(f"\n{'='*50}")
        print(f"CITATION JUDGING (approach='{approach}'): Starting with {len(models)} judge model(s)")
        print(f"Models: {models}")
        print(f"{'='*50}\n")

        # ---------------------------------------------------------------
        # 2) Create ephemeral judge clients for each model in judge_models
        # ---------------------------------------------------------------
        ephemeral_judges = []
        for i, model_name in enumerate(models):
            model_lower = model_name.lower()
            if "claude" in model_lower:
                api_key = os.environ.get("CLAUDE_API_KEY", "None")
                api_url = os.environ.get("CLAUDE_API_URL", "None")
                ephemeral_judges.append(
                    ClaudeClient(
                        api_key=api_key,
                        api_url=api_url,
                        model=model_name,
                        dev_mode=getattr(self, "dev_mode", False),
                        logger=logging.getLogger(f"app.llm.claude.judge{i}")
                    )
                )
                print(f"CITATION JUDGING: Created ephemeral Judge {i+1}: Claude with model {model_name}")
            elif "gpt" in model_lower or "openai" in model_lower:
                api_key = os.environ.get("OPENAI_API_KEY", "None")
                api_url = os.environ.get("OPENAI_API_URL", "None")
                ephemeral_judges.append(
                    OpenAIClient(
                        api_key=api_key,
                        api_url=api_url,
                        model=model_name,
                        dev_mode=getattr(self, "dev_mode", False),
                        logger=logging.getLogger(f"app.llm.openai.judge{i}")
                    )
                )
                print(f"CITATION JUDGING: Created ephemeral Judge {i+1}: OpenAI with model {model_name}")
            else:
                # fallback => Perplexity
                api_key = os.environ.get("PERPLEXITY_API_KEY", "None")
                api_url = os.environ.get("PERPLEXITY_API_URL", "None")
                ephemeral_judges.append(
                    PerplexityClient(
                        api_key=api_key,
                        api_url=api_url,
                        model=model_name,
                        dev_mode=getattr(self, "dev_mode", False),
                        logger=logging.getLogger(f"app.llm.perplexity.judge{i}")
                    )
                )
                print(f"CITATION JUDGING: Created ephemeral Judge {i+1}: Perplexity with model {model_name}")

        # ---------------------------------------------------------------
        # 3) Query edges from the database for this session
        # ---------------------------------------------------------------
        query = """
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id = $session_id AND t.session_id = $session_id
        RETURN s.name AS source,
            t.name AS target,
            type(r) AS rel_type,
            properties(r) AS props
        """
        params = {"session_id": self.session_id}

        try:
            with self.graph_db._get_session() as session:
                records = list(session.run(query, params))

            if not records:
                print("CITATION JUDGING: No edges found in this session to judge.")
                return []

            print(f"CITATION JUDGING: Found {len(records)} edges to judge")
            print(f"{'='*50}")

            # Helper for partial verdict => numeric score
            def verdict_to_score(v_str: str) -> float:
                """
                Convert verdict strings (SUPPORTED, PARTIALLY_SUPPORTED, etc.)
                into a numeric float for averaging.
                """
                v_str = v_str.upper()
                if v_str == "SUPPORTED":
                    return 1.0
                elif v_str == "PARTIALLY_SUPPORTED":
                    return 0.5
                else:
                    # CONTRADICTED or UNADDRESSED => 0.0
                    return 0.0

            # ------------------------------------------------------------
            # 4) Process each edge with ephemeral_judges
            # ------------------------------------------------------------
            for idx, record in enumerate(records, start=1):
                src = record["source"]
                tgt = record["target"]
                rel_type = record["rel_type"]
                props = record.get("props") or {}

                print(f"\nCITATION JUDGING: Edge {idx}/{len(records)} => {src}-[{rel_type}]->{tgt}")

                motivation_text = props.get("spurious_motivation", "") or props.get("motivation", "")
                motivation_text = motivation_text.strip()
                citations = props.get("citations", []) or props.get("citation", [])

                # (a) If no motivation or no citations => skip
                if not motivation_text or not citations:
                    verdict = "NO_MOTIVATION"
                    reason = "Edge has no motivation or citations to judge."
                    print(f"CITATION JUDGING: {verdict} => {reason}")
                    self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
                    judged_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "judge_verdict": verdict,
                        "judge_message": reason
                    })
                    continue

                print(f"CITATION JUDGING: Found {len(citations)} citations; motivation length={len(motivation_text)}")

                # (b) Fetch + process citations (both approaches need the text)
                try:
                    processor = CitationProcessor(timeout=15, max_workers=3)
                    citation_meta = processor.process_citations(citations, fetch_content=True)
                    citation_text_map = processor.prepare_for_llm(citation_meta, include_metadata=True)
                except Exception as e:
                    verdict = "ERROR"
                    reason = f"Error fetching citations: {e}"
                    print(f"CITATION JUDGING ERROR: {reason}")
                    self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
                    judged_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "judge_verdict": verdict,
                        "judge_message": reason,
                    })
                    continue

                # ---------------------------------------------------------------
                # Now handle the different "approach" modes
                # ---------------------------------------------------------------
                if approach == "all_citations_at_once":
                    # 1) Format citations for single prompt
                    formatted_citations = ""
                    for url, content in citation_text_map.items():
                        formatted_citations += f"\n\n--- CITATION: {url} ---\n{content}\n---"

                    variables = {
                        "source": src,
                        "target": tgt,
                        "relationship": rel_type,
                        "motivation": motivation_text,
                        "citations": formatted_citations,
                    }
                    sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCitation", variables)

                    # 2) Each ephemeral judge => "OK" or "INCONSISTENT:..."
                    judge_results = []
                    for j_idx, judge_client in enumerate(ephemeral_judges):
                        model_name = models[j_idx]
                        print(f"[all_citations_at_once] => Judge {j_idx+1} ({model_name}) for {src}->{tgt}")
                        try:
                            response = judge_client.send_message(sys_prompt, usr_prompt, stream=False)
                            data = judge_client.parse_static_response(response)
                            llm_content = data.get("content", "").strip()

                            if llm_content.startswith("OK"):
                                v = "OK"
                                m = ""
                            elif llm_content.startswith("INCONSISTENT:"):
                                v = "INCONSISTENT"
                                m = llm_content[len("INCONSISTENT:"):].strip()
                            else:
                                v = "INCONSISTENT"
                                m = f"Unexpected response: {llm_content[:100]}"

                            judge_results.append({"verdict": v, "message": m})
                        except Exception as e:
                            err_msg = f"ERROR calling judge {j_idx+1} ({model_name}): {e}"
                            print(err_msg)
                            judge_results.append({"verdict": "ERROR", "message": err_msg})

                    # 3) Majority vote => final
                    ok_count = sum(1 for r in judge_results if r["verdict"] == "OK")
                    inc_count = sum(1 for r in judge_results if r["verdict"] == "INCONSISTENT")
                    err_count = sum(1 for r in judge_results if r["verdict"] == "ERROR")

                    print(f"[all_citations_at_once] => Votes => {ok_count} OK, {inc_count} INCONSISTENT, {err_count} ERROR")

                    if ok_count > inc_count + err_count:
                        final_verdict = "OK"
                        final_reason = "Majority of judges => OK"
                    elif inc_count > ok_count + err_count:
                        final_verdict = "INCONSISTENT"
                        final_reason = "Majority of judges => INCONSISTENT"
                    elif ok_count == inc_count:
                        # tie => conservative => INCONSISTENT
                        final_verdict = "INCONSISTENT"
                        final_reason = "Tie => mark as inconsistent."
                    else:
                        # Some tie with errors
                        if ok_count >= inc_count:
                            final_verdict = "OK"
                            final_reason = "OK tie with errors"
                        else:
                            final_verdict = "INCONSISTENT"
                            final_reason = "Inconsistent tie with errors"

                    detail_list = []
                    for j_idx, r in enumerate(judge_results):
                        detail_list.append(f"Judge {j_idx+1} => {r['verdict']} - {r['message']}")
                    full_message = f"{final_reason}. Details: {'; '.join(detail_list)}"

                    # 4) Store in DB
                    self._update_edge_with_judge_verdict(src, tgt, rel_type, final_verdict, full_message)

                    # 5) Summarize
                    judged_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "judged_text": motivation_text[:100] + "..." if len(motivation_text) > 100 else motivation_text,
                        "judge_verdict": final_verdict,
                        "judge_message": full_message
                    })

                elif approach == "per_citation_aggregate":
                    # (1) Single-citation partial judgments => majority among ephemeral_judges
                    partial_results = []
                    c_idx = 0

                    for url, content in citation_text_map.items():
                        c_idx += 1
                        single_judge_outcomes = []
                        for j_idx, judge_client in enumerate(ephemeral_judges):
                            model_name = models[j_idx]
                            sys_prompt_single = (
                                "You are a precise, fair judge evaluating if a claim (the 'motivation') "
                                "is supported by ONE citation. Base your judgment ONLY on that citation."
                            )
                            usr_prompt_single = (
                                f"MOTIVATION: {motivation_text}\n\n"
                                f"CITATION CONTENT:\n{content}\n\n"
                                "Respond EXACTLY in this format:\n"
                                "VERDICT: [SUPPORTED | PARTIALLY_SUPPORTED | CONTRADICTED | UNADDRESSED]\n"
                                "REASON: [brief explanation]"
                            )
                            try:
                                resp = judge_client.send_message(sys_prompt_single, usr_prompt_single, stream=False)
                                data = judge_client.parse_static_response(resp)
                                llm_content = data.get("content", "")
                            except Exception as e:
                                print(f"Error single-citation => Judge {j_idx+1} ({model_name}): {e}")
                                llm_content = "VERDICT: UNADDRESSED\nREASON: LLM error"

                            # parse verdict
                            v_match = re.search(
                                r"VERDICT:\s*(SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|UNADDRESSED)",
                                llm_content,
                                re.IGNORECASE
                            )
                            r_match = re.search(r"REASON:\s*(.*)", llm_content, re.IGNORECASE)
                            if v_match:
                                v_str = v_match.group(1).upper()
                            else:
                                v_str = "UNADDRESSED"
                            reason_str = r_match.group(1).strip() if r_match else ""
                            single_judge_outcomes.append({"verdict": v_str, "reason": reason_str})

                        # majority vote for single-citation
                        tallies = Counter(d["verdict"] for d in single_judge_outcomes)
                        max_count = max(tallies.values()) if tallies else 0
                        winners = [cat for cat, cnt in tallies.items() if cnt == max_count]
                        priority_order = ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNADDRESSED"]
                        final_partial_verdict = None
                        for cat in priority_order:
                            if cat in winners:
                                final_partial_verdict = cat
                                break
                        if not final_partial_verdict:
                            final_partial_verdict = "UNADDRESSED"

                        # detail reasons
                        detail_strs = []
                        for j_idx, outcome in enumerate(single_judge_outcomes):
                            detail_strs.append(f"Judge {j_idx+1}: {outcome['verdict']} - {outcome['reason']}")
                        partial_reason_concat = " | ".join(detail_strs)

                        partial_results.append({
                            "citation_idx": c_idx,
                            "url": url,
                            "verdict": final_partial_verdict,
                            "score": verdict_to_score(final_partial_verdict),
                            "reason": partial_reason_concat
                        })

                    # (2) Compute average score
                    if len(partial_results) == 0:
                        aggregate_score = 0.0
                    else:
                        sum_scores = sum(r["score"] for r in partial_results)
                        aggregate_score = sum_scores / len(partial_results)

                    # (3) aggregator step => ephemeral_judges see partial results again
                    partial_summary_text = "\n".join(
                        f"- Citation #{pr['citation_idx']} => {pr['verdict']} (score={pr['score']}), reason(s): {pr['reason']}"
                        for pr in partial_results
                    )

                    aggregator_judge_outcomes = []
                    for j_idx, judge_client in enumerate(ephemeral_judges):
                        model_name = models[j_idx]
                        sys_prompt_agg = (
                            "You are an aggregator. You see partial verdicts for multiple citations. "
                            "Each citation is judged fully, partially, or not supported. "
                            "Produce EXACTLY one final label:\n"
                            "- Fully supported\n"
                            "- Partially supported\n"
                            "- Not supported\n"
                        )
                        usr_prompt_agg = (
                            f"PARTIAL RESULTS:\n{partial_summary_text}\n\n"
                            "Please respond with EXACTLY one of these lines:\n"
                            "Fully supported\n"
                            "Partially supported\n"
                            "Not supported\n"
                        )
                        try:
                            agg_resp = judge_client.send_message(sys_prompt_agg, usr_prompt_agg, stream=False)
                            agg_data = judge_client.parse_static_response(agg_resp)
                            final_text = agg_data.get("content", "").strip().lower()
                            if "fully supported" in final_text:
                                aggregator_judge_outcomes.append("Fully supported")
                            elif "partially supported" in final_text:
                                aggregator_judge_outcomes.append("Partially supported")
                            else:
                                aggregator_judge_outcomes.append("Not supported")
                        except Exception as e:
                            print(f"Error aggregator => Judge {j_idx+1} ({model_name}): {e}")
                            aggregator_judge_outcomes.append("Not supported")

                    # majority aggregator verdict
                    aggregator_counts = Counter(aggregator_judge_outcomes)
                    max_count = max(aggregator_counts.values()) if aggregator_counts else 0
                    aggregator_winners = [k for k, c in aggregator_counts.items() if c == max_count]
                    final_agg_verdict = "Not supported"
                    for cat in ["Fully supported", "Partially supported", "Not supported"]:
                        if cat in aggregator_winners:
                            final_agg_verdict = cat
                            break

                    aggregator_details = []
                    for j_idx, decision in enumerate(aggregator_judge_outcomes):
                        aggregator_details.append(f"Judge {j_idx+1}: {decision}")
                    aggregator_reason_str = f"Aggregator majority => {final_agg_verdict} | " + " | ".join(aggregator_details)

                    final_judge_data = {
                        "citations": partial_results,
                        "aggregate_score": aggregate_score,
                        "aggregate_verdict": final_agg_verdict
                    }
                    final_json = json.dumps(final_judge_data, ensure_ascii=False)

                    # (4) Store in DB
                    self._update_edge_with_judge_verdict(src, tgt, rel_type, final_agg_verdict, final_json, aggregate_score)

                    # (5) Summarize
                    judged_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "motivation": motivation_text[:100] + "..." if len(motivation_text) > 100 else motivation_text,
                        "aggregate_score": aggregate_score,
                        "aggregate_verdict": final_agg_verdict,
                        "aggregator_message": aggregator_reason_str,
                        "details_json": final_judge_data
                    })

                elif approach == "one_causal_source_found":
                    # Single big prompt: ephemeral judges => "CAUSAL" or "NOT_CAUSAL"
                    # majority => final => store => aggregator_score = 1.0 or 0.0
                    formatted_citations = ""
                    for url, content in citation_text_map.items():
                        formatted_citations += f"\n\n--- CITATION: {url} ---\n{content}\n---"

                    sys_prompt_causal = (
                        "You are a precise judge checking if these citations provide direct causal evidence "
                        "for the motivation below. Only rely on the text below."
                    )
                    usr_prompt_causal = (
                        f"MOTIVATION: {motivation_text}\n\n"
                        f"ALL CITATIONS:\n{formatted_citations}\n\n"
                        "Respond EXACTLY with one of:\n"
                        "CAUSAL\n"
                        "NOT_CAUSAL\n"
                    )

                    judge_results = []
                    for j_idx, judge_client in enumerate(ephemeral_judges):
                        model_name = models[j_idx]
                        print(f"[one_causal_source_found] => Judge {j_idx+1} ({model_name}) for {src}->{tgt}")
                        try:
                            resp = judge_client.send_message(sys_prompt_causal, usr_prompt_causal, stream=False)
                            data = judge_client.parse_static_response(resp)
                            llm_content = data.get("content", "").strip().upper()
                            if llm_content.startswith("CAUSAL"):
                                v = "CAUSAL"
                            else:
                                v = "NOT_CAUSAL"
                            judge_results.append({"verdict": v})
                        except Exception as ex:
                            print(f"Error calling judge {j_idx+1} => {ex}")
                            judge_results.append({"verdict": "NOT_CAUSAL"})

                    tallies = Counter(r["verdict"] for r in judge_results)
                    if not tallies:
                        final_verdict = "NOT_CAUSAL"
                        aggregate_score = 0.0
                    else:
                        c_count = tallies["CAUSAL"]
                        n_count = tallies["NOT_CAUSAL"]
                        if c_count > n_count:
                            final_verdict = "CAUSAL"
                            aggregate_score = 1.0
                        else:
                            final_verdict = "NOT_CAUSAL"
                            aggregate_score = 0.0

                    detail_msg = "; ".join(f"Judge {i+1} => {r['verdict']}" for i, r in enumerate(judge_results))
                    self._update_edge_with_judge_verdict(src, tgt, rel_type, final_verdict, detail_msg, aggregate_score)
                    judged_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "motivation": motivation_text[:100] + "..." if len(motivation_text) > 100 else motivation_text,
                        "judge_verdict": final_verdict,
                        "judge_message": detail_msg,
                        "aggregate_score": aggregate_score
                    })

                else:
                    raise ValueError(f"Unknown approach mode: {approach}")

            print(f"\n{'='*50}")
            print(f"CITATION JUDGING (approach='{approach}'): Complete! Judged {len(judged_edges)} edges.")
            print(f"{'='*50}\n")

            # --------------------------------------------------------------------
            # 5) Merge ephemeral usage stats into self.judge_llm for final reporting
            # --------------------------------------------------------------------
            for ephemeral_llm in ephemeral_judges:
                ephemeral_stats = ephemeral_llm.get_stats()
                self._merge_judge_usage(ephemeral_stats)

            return judged_edges

        except Exception as e:
            print(f"CITATION JUDGING ERROR (outer loop): {e}")
            return []

    

    def _merge_judge_usage(self, ephemeral_stats: dict):
        """
        Merge ephemeral usage (token counts, status codes, time) into self.judge_llm.
        ephemeral_stats is the dict returned by ephemeral_llm.get_stats().
        """
        # 1) ephemeral status codes
        ephemeral_status = ephemeral_stats.get("status_counts", {})
        for code, count in ephemeral_status.items():
            self.judge_llm._status_counts[code] += count

        # 2) ephemeral token totals
        ephemeral_tokens = ephemeral_stats.get("token_totals", {})
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            self.judge_llm._token_totals[key] += ephemeral_tokens.get(key, 0)

        # 3) ephemeral inference time + calls
        self.judge_llm._inference_time_total += ephemeral_stats.get("inference_time_total", 0.0)
        self.judge_llm._inference_call_count += ephemeral_stats.get("inference_call_count", 0)




    def _update_edge_with_judge_verdict(
        self,
        source_name: str,
        target_name: str,
        rel_type: str,
        verdict: str,
        message: str,
        aggregate_score: float = None
    ):
        """
        Update the edge in Neo4j with the final verdict, message, and optional aggregate_score.
        If aggregate_score is None, we can set it to null in the DB. Otherwise we store it as a float.
        """
        query = f"""
        MATCH (s:variable {{name: $src, session_id: $session_id}})-[r:{rel_type}]->(t:variable {{name: $tgt, session_id: $session_id}})
        SET r.judge_verdict = $verdict,
            r.judge_message = $message,
            r.aggregate_score = $score
        """
        # If you'd rather store null vs. a default, you can do COALESCE or an IF check,
        # but this direct approach sets r.aggregate_score to whatever is in $score, or null if None.

        params = {
            "src": source_name,
            "tgt": target_name,
            "session_id": self.session_id,
            "verdict": verdict,
            "message": message,
            "score": aggregate_score
        }

        with self.graph_db._get_session() as session:
            session.run(query, params)

        print(f"UPDATED: Edge ({source_name})-[{rel_type}]->({target_name}) => "
            f"verdict='{verdict}', aggregate_score={aggregate_score}")
            
    def get_variables(self, session_id=None):
        """
        Retrieves all nodes whose `session_id` matches the provided value
        or, if none is provided, uses the object's current session ID.
        
        Args:
            session_id (str, optional): The session ID to match against.
                                        If None, attempts to use self.graph_db._session_id.
        
        Returns:
            List of matching nodes.
        """
        # Determine which session ID to use
        target_session = session_id if session_id else self.graph_db._session_id
        
        if not target_session:
            print("ERROR: No session ID available - cannot retrieve variables")
            return []
        
        query = """
        MATCH (n)
        WHERE n.session_id = $session_id
        RETURN n
        """
        
        with self.graph_db._get_session() as session:
            result = session.run(query, {"session_id": target_session})
            return [record["n"] for record in result]

    def set_session_id(self, new_session_id: str):
            """
            Override the session ID for this CausalDiscovery instance and its Neo4j client.
            This allows you to reuse a session that already exists in the database,
            or run the judging in isolation on an existing session's data.
            """
            logger = logging.getLogger(__name__)
            old_id = self.session_id
            self.session_id = new_session_id
            self.graph_db.set_session_id(new_session_id)
            logger.info(f"Session ID updated from {old_id} to {new_session_id}")

    # def judge_all_edges_with_citations_serial(self, judge_models=None, num_judges=None, indiv_cit=False, threshold=0.75, judge_temperature=0.7) -> List[Dict]:
    #     """
    #     Judge every edge in the current session, optionally per-citation, using multiple judges in series.

    #     If indiv_cit=False (default):
    #         - We judge the entire motivation & all citations at once (original behavior).
    #         - We then do a multi-judge majority vote.

    #     If indiv_cit=True:
    #         - For each edge, we fetch the motivation and the citations.
    #         - For each citation individually, we call the LLM(s) to decide "OK" or "INCONSISTENT".
    #         - We do a mini-majority vote for each individual citation. 
    #         If majority is "OK", that citation is considered consistent; otherwise inconsistent.
    #         - We compute judged_cit_consistency_score = (# citations consistent) / (total citations).
    #         - If that score >= threshold, final verdict is "OK"; else "INCONSISTENT".

    #     :param judge_models: List of model names or single model name. If None, fallback to self.judge_llm's model.
    #     :param num_judges:   Number of times to call judge. If None, uses len(judge_models) or defaults to 2.
    #     :param indiv_cit:    If True, do per-citation judging. Otherwise, do the existing “bulk” approach.
    #     :param threshold:    What fraction of citations must be “OK” for final verdict to be “OK” (if indiv_cit=True).
    #     :return:             A list of dicts with final judge info for each edge.
    #     """
    #     logger = logging.getLogger(__name__)
    #     judged_edges = []

    #     # 1) Determine the judge models
    #     if judge_models is None:
    #         # Default to using the original judge_llm's model (arbitrary fallback name if not set)
    #         fallback_model = getattr(self.judge_llm, "model", "claude-3-7-sonnet-20250219")
    #         models = [fallback_model] * (num_judges or 2)
    #     elif isinstance(judge_models, str):
    #         # Use the same model for all judges
    #         models = [judge_models] * (num_judges or 2)
    #     else:
    #         # Use the provided list of models
    #         models = judge_models
    #         if num_judges is not None and num_judges != len(models):
    #             logger.warning(
    #                 f"num_judges parameter ({num_judges}) is ignored when judge_models is a list. "
    #                 f"Using {len(models)} judges based on list length."
    #             )

    #     print(f"\n{'='*50}")
    #     print(f"CITATION JUDGING: Starting with {len(models)} judges")
    #     print(f"CITATION JUDGING: Using models: {models}")
    #     print(f"CITATION JUDGING: Per-citation mode? {indiv_cit}")
    #     print(f"CITATION JUDGING: Threshold = {threshold}")
    #     print(f"{'='*50}\n")

    #     # 2) Create the LLM clients for each judge
    #     judges = []
    #     for i, model in enumerate(models):
    #         model_lower = model.lower()
    #         if "claude" in model_lower:
    #             api_key = os.environ.get("CLAUDE_API_KEY", "None")
    #             api_url = os.environ.get("CLAUDE_API_URL", "None")
    #             judges.append(
    #                 ClaudeClient(
    #                     api_key=api_key,
    #                     api_url=api_url,
    #                     model=model,
    #                     temperature=judge_temperature,
    #                     dev_mode=getattr(self, "dev_mode", False),
    #                     logger=logging.getLogger(f"app.llm.claude.judge{i}")
    #                 )
    #             )
    #             print(f"CITATION JUDGING: Created Judge {i+1}: Claude with model {model}")
    #         elif "gpt" in model_lower or "openai" in model_lower:
    #             api_key = os.environ.get("OPENAI_API_KEY", "None")
    #             api_url = os.environ.get("OPENAI_API_URL", "None")
    #             judges.append(
    #                 OpenAIClient(
    #                     api_key=api_key,
    #                     api_url=api_url,
    #                     model=model,
    #                     dev_mode=getattr(self, "dev_mode", False),
    #                     logger=logging.getLogger(f"app.llm.openai.judge{i}")
    #                 )
    #             )
    #             print(f"CITATION JUDGING: Created Judge {i+1}: OpenAI with model {model}")
    #         else:
    #             # Default to Perplexity for any other model name
    #             api_key = os.environ.get("PERPLEXITY_API_KEY", "None")
    #             api_url = os.environ.get("PERPLEXITY_API_URL", "None")
    #             judges.append(
    #                 PerplexityClient(
    #                     api_key=api_key,
    #                     api_url=api_url,
    #                     model=model,
    #                     dev_mode=getattr(self, "dev_mode", False),
    #                     logger=logging.getLogger(f"app.llm.perplexity.judge{i}")
    #                 )
    #             )
    #             print(f"CITATION JUDGING: Created Judge {i+1}: Perplexity with model {model}")

    #     # 3) Retrieve all edges in the session
    #     query = """
    #     MATCH (s:variable)-[r]->(t:variable)
    #     WHERE s.session_id = $session_id AND t.session_id = $session_id
    #     RETURN s.name AS source,
    #         t.name AS target,
    #         type(r) AS rel_type,
    #         properties(r) AS props
    #     """
    #     params = {"session_id": self.session_id}

    #     try:
    #         with self.graph_db._get_session() as session:
    #             records = list(session.run(query, params))

    #         if not records:
    #             print("CITATION JUDGING: No edges found in this session to judge.")
    #             return []

    #         print(f"CITATION JUDGING: Found {len(records)} edges to judge")
    #         print(f"{'='*50}")

    #         # 4) Process each edge
    #         for idx, record in enumerate(records, start=1):
    #             src = record["source"]
    #             tgt = record["target"]
    #             rel_type = record["rel_type"]
    #             props = record.get("props") or {}

    #             print(f"\nCITATION JUDGING: Edge {idx}/{len(records)}: {src}-[{rel_type}]->{tgt}")

    #             # (a) Extract text & citations
    #             text_to_judge = props.get("spurious_motivation", "") or props.get("motivation", "")
    #             citations = props.get("citations", []) or props.get("citation", [])
    #             text_to_judge = text_to_judge.strip()

    #             # (b) If no motivation or no citations => skip or mark as NO_MOTIVATION
    #             if not text_to_judge or not citations:
    #                 verdict = "NO_MOTIVATION"
    #                 reason = "Edge has no motivation or citations to judge."
    #                 print(f"CITATION JUDGING: {verdict} - {reason}")

    #                 # Update DB
    #                 self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)

    #                 judged_edges.append(
    #                     {
    #                         "source": src,
    #                         "target": tgt,
    #                         "type": rel_type,
    #                         "judged_text": "",
    #                         "judge_verdict": verdict,
    #                         "judge_message": reason,
    #                     }
    #                 )
    #                 continue

    #             print(f"CITATION JUDGING: Found {len(citations)} citations and motivation length {len(text_to_judge)}")

    #             # (c) If indiv_cit=False, we do the original approach (bulk judging for all citations at once)
    #             if not indiv_cit:
    #                 try:
    #                     # Fetch *all* citation texts for a single prompt
    #                     processor = CitationProcessor(timeout=15, max_workers=3)
    #                     citation_results = processor.process_citations(citations, fetch_content=True)
    #                     citation_content = processor.prepare_for_llm(citation_results)

    #                     formatted_citations = ""
    #                     for url, content in citation_content.items():
    #                         formatted_citations += f"\n\n--- CITATION: {url} ---\n{content}\n---"

    #                 except Exception as e:
    #                     verdict = "INCONSISTENT"
    #                     reason = f"Error fetching citations: {e}"
    #                     print(f"CITATION JUDGING ERROR: {reason}")
    #                     self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
    #                     judged_edges.append(
    #                         {
    #                             "source": src,
    #                             "target": tgt,
    #                             "type": rel_type,
    #                             "judged_text": text_to_judge[:100],
    #                             "judge_verdict": verdict,
    #                             "judge_message": reason,
    #                         }
    #                     )
    #                     continue

    #                 # Build the judgeCitation prompt (bulk)
    #                 variables = {
    #                     "source": src,
    #                     "target": tgt,
    #                     "relationship": rel_type,
    #                     "motivation": text_to_judge,
    #                     "citations": formatted_citations,
    #                 }
    #                 sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCitation", variables)

    #                 # Call each judge in series (bulk)
    #                 judge_results = []
    #                 for j_idx, judge_client in enumerate(judges):
    #                     current_model_name = models[j_idx]
    #                     print(f"CITATION JUDGING: Calling judge {j_idx+1} ({current_model_name}) for {src}->{tgt}")
    #                     try:
    #                         response = judge_client.send_message(sys_prompt, usr_prompt, stream=False)
    #                         data = judge_client.parse_static_response(response)
    #                         llm_content = data.get("content", "").strip()

    #                         # Parse the response: expect "OK" or "INCONSISTENT:..."
    #                         if llm_content.startswith("OK"):
    #                             v = "OK"
    #                             m = ""
    #                         elif llm_content.startswith("INCONSISTENT:"):
    #                             v = "INCONSISTENT"
    #                             m = llm_content[len("INCONSISTENT:") :].strip()
    #                         else:
    #                             v = "INCONSISTENT"
    #                             m = f"Unexpected response: {llm_content[:100]}"

    #                         judge_results.append({"verdict": v, "message": m})
    #                         print(f"CITATION JUDGING: Judge {j_idx+1} => {v}, reason => {m[:100]}...")
    #                     except Exception as e:
    #                         print(f"CITATION JUDGING: Error with judge {j_idx+1} ({current_model_name}): {e}")
    #                         judge_results.append({"verdict": "ERROR", "message": str(e)})

    #                 # Combine the results => majority
    #                 ok_count = sum(1 for r in judge_results if r["verdict"] == "OK")
    #                 inconsistent_count = sum(1 for r in judge_results if r["verdict"] == "INCONSISTENT")
    #                 error_count = sum(1 for r in judge_results if r["verdict"] == "ERROR")

    #                 if ok_count > inconsistent_count + error_count:
    #                     final_verdict = "OK"
    #                     final_reason = f"Majority of judges ({ok_count}/{len(judge_results)}) found this relationship consistent."
    #                 elif inconsistent_count > ok_count + error_count:
    #                     final_verdict = "INCONSISTENT"
    #                     final_reason = f"Majority of judges ({inconsistent_count}/{len(judge_results)}) found this relationship inconsistent."
    #                 elif ok_count == inconsistent_count:
    #                     final_verdict = "INCONSISTENT"
    #                     final_reason = "Tie between OK and INCONSISTENT => marking as inconsistent (conservative)."
    #                 else:
    #                     # Cases with errors
    #                     if ok_count >= inconsistent_count:
    #                         final_verdict = "OK"
    #                         final_reason = (
    #                             f"Determined OK ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
    #                         )
    #                     else:
    #                         final_verdict = "INCONSISTENT"
    #                         final_reason = (
    #                             f"Determined inconsistent ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
    #                         )

    #                 # Collect details from each judge
    #                 judge_details = []
    #                 for j_idx, r in enumerate(judge_results):
    #                     model_name = models[j_idx]
    #                     v = r["verdict"]
    #                     msg = r["message"]
    #                     detail = f"Judge {j_idx+1} ({model_name}): {v}"
    #                     if msg:
    #                         detail += f" - {msg}"
    #                     judge_details.append(detail)

    #                 full_message = f"{final_reason} Details: {'; '.join(judge_details)}"
    #                 print(f"CITATION JUDGING: Final verdict => {final_verdict}")

    #                 # Update DB (no separate judged_cit_consistency_score in bulk mode)
    #                 self._update_edge_with_judge_verdict(src, tgt, rel_type, final_verdict, full_message)

    #                 judged_edges.append(
    #                     {
    #                         "source": src,
    #                         "target": tgt,
    #                         "type": rel_type,
    #                         "judged_text": text_to_judge[:100] + "..." if len(text_to_judge) > 100 else text_to_judge,
    #                         "judge_verdict": final_verdict,
    #                         "judge_message": full_message,
    #                     }
    #                 )

    #             else:
    #                 # ===============================================
    #                 # =========== Per-Citation Judging =============
    #                 # ===============================================
    #                 consistent_count = 0
    #                 total_citations = len(citations)

    #                 # We still want to fetch the full text for each citation individually
    #                 # in case we want to feed it into the LLM one by one.
    #                 processor = CitationProcessor(timeout=15, max_workers=3)
    #                 try:
    #                     citation_results = processor.process_citations(citations, fetch_content=True)
    #                     # citation_results is typically a list or dict of { "url": <content> }, etc.
    #                 except Exception as e:
    #                     verdict = "INCONSISTENT"
    #                     reason = f"Error fetching citations: {e}"
    #                     print(f"CITATION JUDGING ERROR: {reason}")
    #                     # Store results
    #                     self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
    #                     judged_edges.append(
    #                         {
    #                             "source": src,
    #                             "target": tgt,
    #                             "type": rel_type,
    #                             "judged_text": text_to_judge[:100],
    #                             "judge_verdict": verdict,
    #                             "judge_message": reason,
    #                         }
    #                     )
    #                     continue

    #                 # For each citation, we do a mini "bulk" judge call
    #                 for citation_url, citation_content in citation_results.items():
    #                     # Build a prompt with just the single citation
    #                     formatted_citations = f"\n\n--- CITATION: {citation_url} ---\n{citation_content}\n---"

    #                     variables = {
    #                         "source": src,
    #                         "target": tgt,
    #                         "relationship": rel_type,
    #                         "motivation": text_to_judge,
    #                         "citations": formatted_citations,  # single citation
    #                     }
    #                     sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCitation", variables)

    #                     # Now call each judge for THIS citation
    #                     per_citation_judge_verdicts = []
    #                     for j_idx, judge_client in enumerate(judges):
    #                         current_model_name = models[j_idx]
    #                         try:
    #                             response = judge_client.send_message(sys_prompt, usr_prompt, stream=False)
    #                             data = judge_client.parse_static_response(response)
    #                             llm_content = data.get("content", "").strip()
    #                             if llm_content.startswith("OK"):
    #                                 v = "OK"
    #                             elif llm_content.startswith("INCONSISTENT:"):
    #                                 v = "INCONSISTENT"
    #                             else:
    #                                 # If it's weird or can't parse, default to inconsistent
    #                                 v = "INCONSISTENT"
    #                             per_citation_judge_verdicts.append(v)
    #                         except Exception as e:
    #                             print(f"ERROR judging citation {citation_url} with judge {j_idx+1}: {e}")
    #                             per_citation_judge_verdicts.append("INCONSISTENT")

    #                     # majority for the single citation
    #                     ok_count_cit = per_citation_judge_verdicts.count("OK")
    #                     inconsistent_count_cit = per_citation_judge_verdicts.count("INCONSISTENT")
    #                     if ok_count_cit > inconsistent_count_cit:
    #                         # citation is considered consistent
    #                         consistent_count += 1

    #                 # done with all citations => compute ratio
    #                 if total_citations > 0:
    #                     consistency_ratio = consistent_count / total_citations
    #                 else:
    #                     # theoretically shouldn't happen due to earlier check, but just in case
    #                     consistency_ratio = 0.0

    #                 print(
    #                     f"CITATION JUDGING (Per-Citation): Edge {src}-{rel_type}->{tgt} => "
    #                     f"{consistent_count} / {total_citations} citations consistent = {consistency_ratio:.2f}"
    #                 )

    #                 # store that ratio as an edge property
    #                 # final verdict depends on threshold
    #                 if consistency_ratio >= threshold:
    #                     final_verdict = "OK"
    #                     reason = (
    #                         f"{consistent_count}/{total_citations} citations consistent"
    #                         f" ({consistency_ratio:.2f} >= threshold {threshold})"
    #                     )
    #                 else:
    #                     final_verdict = "INCONSISTENT"
    #                     reason = (
    #                         f"{consistent_count}/{total_citations} citations consistent"
    #                         f" ({consistency_ratio:.2f} < threshold {threshold})"
    #                     )

    #                 # Update DB, including the new judged_cit_consistency_score
    #                 self._update_edge_with_judge_verdict_and_score(
    #                     src, tgt, rel_type, final_verdict, reason, consistency_ratio
    #                 )

    #                 judged_edges.append(
    #                     {
    #                         "source": src,
    #                         "target": tgt,
    #                         "type": rel_type,
    #                         "judged_text": text_to_judge[:100] + "..." if len(text_to_judge) > 100 else text_to_judge,
    #                         "judge_verdict": final_verdict,
    #                         "judge_message": reason,
    #                         "judged_cit_consistency_score": consistency_ratio,
    #                     }
    #                 )

    #         print(f"\n{'='*50}")
    #         print(f"CITATION JUDGING: Complete! Judged {len(judged_edges)} edges with {len(models)} judges")
    #         print(f"{'='*50}\n")
    #         return judged_edges

    #     except Exception as e:
    #         print(f"CITATION JUDGING ERROR (outer loop): {e}")
    #         return []


# def _update_edge_with_judge_verdict_and_score(
#     self,
#     src_name: str,
#     tgt_name: str,
#     rel_type: str,
#     verdict: str,
#     message: str,
#     score: float
# ):
#     """
#     Updates the edge with both the final judge verdict/message and the
#     judged_cit_consistency_score property (float between 0 and 1).
#     """
#     query = """
#     MATCH (s:variable)-[r]->(t:variable)
#     WHERE s.session_id = $session_id
#       AND t.session_id = $session_id
#       AND s.name = $src_name
#       AND t.name = $tgt_name
#       AND type(r) = $rel_type
#     SET r.judge_verdict = $verdict,
#         r.judge_message = $message,
#         r.judged_cit_consistency_score = $score
#     RETURN r
#     """
#     params = {
#         "session_id": self.session_id,
#         "src_name": src_name,
#         "tgt_name": tgt_name,
#         "rel_type": rel_type,
#         "verdict": verdict,
#         "message": message,
#         "score": score,
#     }
#     with self.graph_db._get_session() as session:
#         session.run(query, params)

# def judge_all_edges_with_citations_serial(
#     self,
#     judge_models=None,
#     num_judges=None,
#     indiv_cit=False,
#     threshold=0.75
# ) -> List[Dict]:
#     """
#     Judge every edge in the current session, optionally per-citation, using multiple judges in series.

#     If indiv_cit=False (default):
#         - We judge the entire motivation & all citations at once (original behavior).
#         - We then do a multi-judge majority vote.

#     If indiv_cit=True:
#         - For each edge, we fetch the motivation and the citations.
#         - For each citation individually, we call the LLM(s) to decide "OK" or "INCONSISTENT".
#         - We do a mini-majority vote for each individual citation. 
#           If majority is "OK", that citation is considered consistent; otherwise inconsistent.
#         - We compute judged_cit_consistency_score = (# citations consistent) / (total citations).
#         - If that score >= threshold, final verdict is "OK"; else "INCONSISTENT".

#     :param judge_models: List of model names or single model name. If None, fallback to self.judge_llm's model.
#     :param num_judges:   Number of times to call judge. If None, uses len(judge_models) or defaults to 2.
#     :param indiv_cit:    If True, do per-citation judging. Otherwise, do the existing “bulk” approach.
#     :param threshold:    What fraction of citations must be “OK” for final verdict to be “OK” (if indiv_cit=True).
#     :return:             A list of dicts with final judge info for each edge.
#     """
#     logger = logging.getLogger(__name__)
#     judged_edges = []

#     # 1) Determine the judge models
#     if judge_models is None:
#         # Default to using the original judge_llm's model (arbitrary fallback name if not set)
#         fallback_model = getattr(self.judge_llm, "model", "claude-3-7-sonnet-20250219")
#         models = [fallback_model] * (num_judges or 2)
#     elif isinstance(judge_models, str):
#         # Use the same model for all judges
#         models = [judge_models] * (num_judges or 2)
#     else:
#         # Use the provided list of models
#         models = judge_models
#         if num_judges is not None and num_judges != len(models):
#             logger.warning(
#                 f"num_judges parameter ({num_judges}) is ignored when judge_models is a list. "
#                 f"Using {len(models)} judges based on list length."
#             )

#     print(f"\n{'='*50}")
#     print(f"CITATION JUDGING: Starting with {len(models)} judges")
#     print(f"CITATION JUDGING: Using models: {models}")
#     print(f"CITATION JUDGING: Per-citation mode? {indiv_cit}")
#     print(f"CITATION JUDGING: Threshold = {threshold}")
#     print(f"{'='*50}\n")

#     # 2) Create the LLM clients for each judge
#     judges = []
#     for i, model in enumerate(models):
#         model_lower = model.lower()
#         if "claude" in model_lower:
#             api_key = os.environ.get("CLAUDE_API_KEY", "None")
#             api_url = os.environ.get("CLAUDE_API_URL", "None")
#             judges.append(
#                 ClaudeClient(
#                     api_key=api_key,
#                     api_url=api_url,
#                     model=model,
#                     dev_mode=getattr(self, "dev_mode", False),
#                     logger=logging.getLogger(f"app.llm.claude.judge{i}")
#                 )
#             )
#             print(f"CITATION JUDGING: Created Judge {i+1}: Claude with model {model}")
#         elif "gpt" in model_lower or "openai" in model_lower:
#             api_key = os.environ.get("OPENAI_API_KEY", "None")
#             api_url = os.environ.get("OPENAI_API_URL", "None")
#             judges.append(
#                 OpenAIClient(
#                     api_key=api_key,
#                     api_url=api_url,
#                     model=model,
#                     dev_mode=getattr(self, "dev_mode", False),
#                     logger=logging.getLogger(f"app.llm.openai.judge{i}")
#                 )
#             )
#             print(f"CITATION JUDGING: Created Judge {i+1}: OpenAI with model {model}")
#         else:
#             # Default to Perplexity for any other model name
#             api_key = os.environ.get("PERPLEXITY_API_KEY", "None")
#             api_url = os.environ.get("PERPLEXITY_API_URL", "None")
#             judges.append(
#                 PerplexityClient(
#                     api_key=api_key,
#                     api_url=api_url,
#                     model=model,
#                     dev_mode=getattr(self, "dev_mode", False),
#                     logger=logging.getLogger(f"app.llm.perplexity.judge{i}")
#                 )
#             )
#             print(f"CITATION JUDGING: Created Judge {i+1}: Perplexity with model {model}")

#     # 3) Retrieve all edges in the session
#     query = """
#     MATCH (s:variable)-[r]->(t:variable)
#     WHERE s.session_id = $session_id AND t.session_id = $session_id
#     RETURN s.name AS source,
#            t.name AS target,
#            type(r) AS rel_type,
#            properties(r) AS props
#     """
#     params = {"session_id": self.session_id}

#     try:
#         with self.graph_db._get_session() as session:
#             records = list(session.run(query, params))

#         if not records:
#             print("CITATION JUDGING: No edges found in this session to judge.")
#             return []

#         print(f"CITATION JUDGING: Found {len(records)} edges to judge")
#         print(f"{'='*50}")

#         # 4) Process each edge
#         for idx, record in enumerate(records, start=1):
#             src = record["source"]
#             tgt = record["target"]
#             rel_type = record["rel_type"]
#             props = record.get("props") or {}

#             print(f"\nCITATION JUDGING: Edge {idx}/{len(records)}: {src}-[{rel_type}]->{tgt}")

#             # (a) Extract text & citations
#             text_to_judge = props.get("spurious_motivation", "") or props.get("motivation", "")
#             citations = props.get("citations", []) or props.get("citation", [])
#             text_to_judge = text_to_judge.strip()

#             # (b) If no motivation or no citations => skip or mark as NO_MOTIVATION
#             if not text_to_judge or not citations:
#                 verdict = "NO_MOTIVATION"
#                 reason = "Edge has no motivation or citations to judge."
#                 print(f"CITATION JUDGING: {verdict} - {reason}")

#                 # Update DB
#                 self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)

#                 judged_edges.append(
#                     {
#                         "source": src,
#                         "target": tgt,
#                         "type": rel_type,
#                         "judged_text": "",
#                         "judge_verdict": verdict,
#                         "judge_message": reason,
#                     }
#                 )
#                 continue

#             print(f"CITATION JUDGING: Found {len(citations)} citations and motivation length {len(text_to_judge)}")

#             # (c) If indiv_cit=False, we do the original approach (bulk judging for all citations at once)
#             if not indiv_cit:
#                 try:
#                     # Fetch *all* citation texts for a single prompt
#                     processor = CitationProcessor(timeout=15, max_workers=3)
#                     citation_results = processor.process_citations(citations, fetch_content=True)
#                     citation_content = processor.prepare_for_llm(citation_results)

#                     formatted_citations = ""
#                     for url, content in citation_content.items():
#                         formatted_citations += f"\n\n--- CITATION: {url} ---\n{content}\n---"

#                 except Exception as e:
#                     verdict = "INCONSISTENT"
#                     reason = f"Error fetching citations: {e}"
#                     print(f"CITATION JUDGING ERROR: {reason}")
#                     self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
#                     judged_edges.append(
#                         {
#                             "source": src,
#                             "target": tgt,
#                             "type": rel_type,
#                             "judged_text": text_to_judge[:100],
#                             "judge_verdict": verdict,
#                             "judge_message": reason,
#                         }
#                     )
#                     continue

#                 # Build the judgeCitation prompt (bulk)
#                 variables = {
#                     "source": src,
#                     "target": tgt,
#                     "relationship": rel_type,
#                     "motivation": text_to_judge,
#                     "citations": formatted_citations,
#                 }
#                 sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCitation", variables)

#                 # Call each judge in series (bulk)
#                 judge_results = []
#                 for j_idx, judge_client in enumerate(judges):
#                     current_model_name = models[j_idx]
#                     print(f"CITATION JUDGING: Calling judge {j_idx+1} ({current_model_name}) for {src}->{tgt}")
#                     try:
#                         response = judge_client.send_message(sys_prompt, usr_prompt, stream=False)
#                         data = judge_client.parse_static_response(response)
#                         llm_content = data.get("content", "").strip()

#                         # Parse the response: expect "OK" or "INCONSISTENT:..."
#                         if llm_content.startswith("OK"):
#                             v = "OK"
#                             m = ""
#                         elif llm_content.startswith("INCONSISTENT:"):
#                             v = "INCONSISTENT"
#                             m = llm_content[len("INCONSISTENT:") :].strip()
#                         else:
#                             v = "INCONSISTENT"
#                             m = f"Unexpected response: {llm_content[:100]}"

#                         judge_results.append({"verdict": v, "message": m})
#                         print(f"CITATION JUDGING: Judge {j_idx+1} => {v}, reason => {m[:100]}...")
#                     except Exception as e:
#                         print(f"CITATION JUDGING: Error with judge {j_idx+1} ({current_model_name}): {e}")
#                         judge_results.append({"verdict": "ERROR", "message": str(e)})

#                 # Combine the results => majority
#                 ok_count = sum(1 for r in judge_results if r["verdict"] == "OK")
#                 inconsistent_count = sum(1 for r in judge_results if r["verdict"] == "INCONSISTENT")
#                 error_count = sum(1 for r in judge_results if r["verdict"] == "ERROR")

#                 if ok_count > inconsistent_count + error_count:
#                     final_verdict = "OK"
#                     final_reason = f"Majority of judges ({ok_count}/{len(judge_results)}) found this relationship consistent."
#                 elif inconsistent_count > ok_count + error_count:
#                     final_verdict = "INCONSISTENT"
#                     final_reason = f"Majority of judges ({inconsistent_count}/{len(judge_results)}) found this relationship inconsistent."
#                 elif ok_count == inconsistent_count:
#                     final_verdict = "INCONSISTENT"
#                     final_reason = "Tie between OK and INCONSISTENT => marking as inconsistent (conservative)."
#                 else:
#                     # Cases with errors
#                     if ok_count >= inconsistent_count:
#                         final_verdict = "OK"
#                         final_reason = (
#                             f"Determined OK ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
#                         )
#                     else:
#                         final_verdict = "INCONSISTENT"
#                         final_reason = (
#                             f"Determined inconsistent ({ok_count} OK, {inconsistent_count} INCONSISTENT, {error_count} ERROR)."
#                         )

#                 # Collect details from each judge
#                 judge_details = []
#                 for j_idx, r in enumerate(judge_results):
#                     model_name = models[j_idx]
#                     v = r["verdict"]
#                     msg = r["message"]
#                     detail = f"Judge {j_idx+1} ({model_name}): {v}"
#                     if msg:
#                         detail += f" - {msg}"
#                     judge_details.append(detail)

#                 full_message = f"{final_reason} Details: {'; '.join(judge_details)}"
#                 print(f"CITATION JUDGING: Final verdict => {final_verdict}")

#                 # Update DB (no separate judged_cit_consistency_score in bulk mode)
#                 self._update_edge_with_judge_verdict(src, tgt, rel_type, final_verdict, full_message)

#                 judged_edges.append(
#                     {
#                         "source": src,
#                         "target": tgt,
#                         "type": rel_type,
#                         "judged_text": text_to_judge[:100] + "..." if len(text_to_judge) > 100 else text_to_judge,
#                         "judge_verdict": final_verdict,
#                         "judge_message": full_message,
#                     }
#                 )

#             else:
#                 # ===============================================
#                 # =========== Per-Citation Judging =============
#                 # ===============================================
#                 consistent_count = 0
#                 total_citations = len(citations)

#                 # We still want to fetch the full text for each citation individually
#                 # in case we want to feed it into the LLM one by one.
#                 processor = CitationProcessor(timeout=15, max_workers=3)
#                 try:
#                     citation_results = processor.process_citations(citations, fetch_content=True)
#                     # citation_results is typically a list or dict of { "url": <content> }, etc.
#                 except Exception as e:
#                     verdict = "INCONSISTENT"
#                     reason = f"Error fetching citations: {e}"
#                     print(f"CITATION JUDGING ERROR: {reason}")
#                     # Store results
#                     self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
#                     judged_edges.append(
#                         {
#                             "source": src,
#                             "target": tgt,
#                             "type": rel_type,
#                             "judged_text": text_to_judge[:100],
#                             "judge_verdict": verdict,
#                             "judge_message": reason,
#                         }
#                     )
#                     continue

#                 # For each citation, we do a mini "bulk" judge call
#                 for citation_url, citation_content in citation_results.items():
#                     # Build a prompt with just the single citation
#                     formatted_citations = f"\n\n--- CITATION: {citation_url} ---\n{citation_content}\n---"

#                     variables = {
#                         "source": src,
#                         "target": tgt,
#                         "relationship": rel_type,
#                         "motivation": text_to_judge,
#                         "citations": formatted_citations,  # single citation
#                     }
#                     sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCitation", variables)

#                     # Now call each judge for THIS citation
#                     per_citation_judge_verdicts = []
#                     for j_idx, judge_client in enumerate(judges):
#                         current_model_name = models[j_idx]
#                         try:
#                             response = judge_client.send_message(sys_prompt, usr_prompt, stream=False)
#                             data = judge_client.parse_static_response(response)
#                             llm_content = data.get("content", "").strip()
#                             if llm_content.startswith("OK"):
#                                 v = "OK"
#                             elif llm_content.startswith("INCONSISTENT:"):
#                                 v = "INCONSISTENT"
#                             else:
#                                 # If it's weird or can't parse, default to inconsistent
#                                 v = "INCONSISTENT"
#                             per_citation_judge_verdicts.append(v)
#                         except Exception as e:
#                             print(f"ERROR judging citation {citation_url} with judge {j_idx+1}: {e}")
#                             per_citation_judge_verdicts.append("INCONSISTENT")

#                     # majority for the single citation
#                     ok_count_cit = per_citation_judge_verdicts.count("OK")
#                     inconsistent_count_cit = per_citation_judge_verdicts.count("INCONSISTENT")
#                     if ok_count_cit > inconsistent_count_cit:
#                         # citation is considered consistent
#                         consistent_count += 1

#                 # done with all citations => compute ratio
#                 if total_citations > 0:
#                     consistency_ratio = consistent_count / total_citations
#                 else:
#                     # theoretically shouldn't happen due to earlier check, but just in case
#                     consistency_ratio = 0.0

#                 print(
#                     f"CITATION JUDGING (Per-Citation): Edge {src}-{rel_type}->{tgt} => "
#                     f"{consistent_count} / {total_citations} citations consistent = {consistency_ratio:.2f}"
#                 )

#                 # store that ratio as an edge property
#                 # final verdict depends on threshold
#                 if consistency_ratio >= threshold:
#                     final_verdict = "OK"
#                     reason = (
#                         f"{consistent_count}/{total_citations} citations consistent"
#                         f" ({consistency_ratio:.2f} >= threshold {threshold})"
#                     )
#                 else:
#                     final_verdict = "INCONSISTENT"
#                     reason = (
#                         f"{consistent_count}/{total_citations} citations consistent"
#                         f" ({consistency_ratio:.2f} < threshold {threshold})"
#                     )

#                 # Update DB, including the new judged_cit_consistency_score
#                 self._update_edge_with_judge_verdict_and_score(
#                     src, tgt, rel_type, final_verdict, reason, consistency_ratio
#                 )

#                 judged_edges.append(
#                     {
#                         "source": src,
#                         "target": tgt,
#                         "type": rel_type,
#                         "judged_text": text_to_judge[:100] + "..." if len(text_to_judge) > 100 else text_to_judge,
#                         "judge_verdict": final_verdict,
#                         "judge_message": reason,
#                         "judged_cit_consistency_score": consistency_ratio,
#                     }
#                 )

#         print(f"\n{'='*50}")
#         print(f"CITATION JUDGING: Complete! Judged {len(judged_edges)} edges with {len(models)} judges")
#         print(f"{'='*50}\n")
#         return judged_edges

#     except Exception as e:
        # print(f"CITATION JUDGING ERROR (outer loop): {e}")
        # return []




    # def _update_edge_with_judge_verdict_and_score(
    #     self,
    #     src_name: str,
    #     tgt_name: str,
    #     rel_type: str,
    #     verdict: str,
    #     message: str,
    #     score: float
    # ):
    #     """
    #     Updates the edge with both the final judge verdict/message and the
    #     judged_cit_consistency_score property (float between 0 and 1).
    #     """
    #     query = """
    #     MATCH (s:variable)-[r]->(t:variable)
    #     WHERE s.session_id = $session_id
    #       AND t.session_id = $session_id
    #       AND s.name = $src_name
    #       AND t.name = $tgt_name
    #       AND type(r) = $rel_type
    #     SET r.judge_verdict = $verdict,
    #         r.judge_message = $message,
    #         r.judged_cit_consistency_score = $score
    #     RETURN r
    #     """
    #     params = {
    #         "session_id": self.session_id,
    #         "src_name": src_name,
    #         "tgt_name": tgt_name,
    #         "rel_type": rel_type,
    #         "verdict": verdict,
    #         "message": message,
    #         "score": score,
    #     }
    #     with self.graph_db._get_session() as session:
    #         session.run(query, params)




    ### Per citation judge:


        
    ###############################################################################
    # 2) Verdict Parsing + Score Mapping
    ###############################################################################

    def parse_verdict_and_reason(llm_content: str) -> (str, str):
        """
        Looks for:
        VERDICT: [SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|UNADDRESSED]
        REASON: ...
        Returns (verdict, reason).
        """
        verdict_match = re.search(
            r"VERDICT:\s*(SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|UNADDRESSED)",
            llm_content,
            re.IGNORECASE
        )
        reason_match = re.search(r"REASON:\s*(.*)", llm_content, re.IGNORECASE)

        if verdict_match:
            verdict = verdict_match.group(1).upper()  # e.g., "SUPPORTED"
        else:
            verdict = "UNADDRESSED"

        reason = reason_match.group(1).strip() if reason_match else ""
        return verdict, reason

    def verdict_to_score(verdict: str) -> float:
        """
        Map LLM verdict to a numeric score:
        - SUPPORTED => 1.0
        - PARTIALLY_SUPPORTED => 0.5
        - CONTRADICTED or UNADDRESSED => 0.0
        """
        if verdict == "SUPPORTED":
            return 1.0
        elif verdict == "PARTIALLY_SUPPORTED":
            return 0.5
        else:
            return 0.0

    ###############################################################################
    # 3) Aggregate with an Additional LLM Call
    #    We ask the LLM to interpret partial results and produce a final label
    ###############################################################################

    def get_aggregate_verdict_from_llm(
        citation_results: List[Dict[str, Any]],
        llm_client,
    ) -> str:
        """
        Perform one extra LLM call passing the partial verdicts/ scores,
        returning a single final label: "Fully supported", "Partially supported", or "Not supported".
        """
        # We'll build a short summary text of each citation's verdict & score:
        summary_lines = []
        for idx, c in enumerate(citation_results, start=1):
            summary_lines.append(
                f"- Citation #{idx} => verdict={c['verdict']}, score={c['score']}, reason='{c['reason']}'"
            )
        partial_summary = "\n".join(summary_lines)

        # System + user prompt example
        sys_prompt = (
            "You are an expert aggregator. You receive partial verdicts (with numeric scores) "
            "for multiple citations. Each citation is judged as fully, partially, or not supported. "
            "Your job is to produce a single label for the entire claim: "
            "'Fully supported', 'Partially supported', or 'Not supported'."
        )

        usr_prompt = (
            f"Here are the partial citation judgments:\n\n{partial_summary}\n\n"
            "Please respond with EXACTLY one of these options:\n\n"
            "- Fully supported\n"
            "- Partially supported\n"
            "- Not supported\n"
        )

        # Send to LLM
        resp = llm_client.send_message(sys_prompt, usr_prompt, stream=False)
        data = llm_client.parse_static_response(resp)
        text = data.get("content", "").strip()

        # We'll do a simple parse:
        # If text includes "Fully supported" => final
        # If text includes "Partially supported" => final
        # else => "Not supported"
        # Adjust to suit your usage or parse more robustly.
        text_lower = text.lower()
        if "fully supported" in text_lower:
            return "Fully supported"
        elif "partially supported" in text_lower:
            return "Partially supported"
        else:
            return "Not supported"

    ###############################################################################
    # 4) The Main Judge Function
    ###############################################################################

    def judge_motivation_with_citations(motivation: str, citations: List[str]) -> Dict[str, Any]:
        """
        1) Process the citations with CitationProcessor (fetch content).
        2) For each citation, do a single-citation LLM call => verdict + reason + score.
        3) Compute average score.
        4) Make one more LLM call with all partial results => 'aggregate_verdict'.
        5) Return final dictionary with the structure you want.

        Example return structure:

        {
        "citations": [
            {
            "citation_idx": 1,
            "url": ...,
            "verdict": "SUPPORTED",
            "reason": "...",
            "score": 1.0
            },
            {
            "citation_idx": 2,
            "url": ...,
            "verdict": "PARTIALLY_SUPPORTED",
            "reason": "...",
            "score": 0.5
            },
            ...
        ],
        "aggregate_score": 0.75,
        "aggregate_verdict": "Fully supported"
        }
        """
        # 1) CitationProcessor usage
        from your_module import CitationProcessor  # or wherever you have it
        processor = CitationProcessor()
        citation_meta = processor.process_citations(citations, fetch_content=True)
        citation_texts = processor.prepare_for_llm(citation_meta)

        # 2) For each citation: single-citation LLM call
        llm_client = MockLLMClient()  # or your real LLM
        sys_prompt_single = (
            "You are a precise, fair judge evaluating if a claim (the 'motivation') "
            "is supported by ONE citation. Base your judgment ONLY on the citation text."
        )

        citation_judgments = []
        for i, (url, content) in enumerate(citation_texts.items(), start=1):
            usr_prompt_single = (
                f"Evaluate if the MOTIVATION is supported by this single citation.\n\n"
                f"MOTIVATION: {motivation}\n\n"
                f"CITATION CONTENT:\n{content}\n\n"
                "Respond exactly in this format:\n"
                "VERDICT: [SUPPORTED | PARTIALLY_SUPPORTED | CONTRADICTED | UNADDRESSED]\n"
                "REASON: [brief explanation]"
            )

            response = llm_client.send_message(sys_prompt_single, usr_prompt_single)
            data = llm_client.parse_static_response(response)
            llm_content = data.get("content", "")
            verdict, reason = parse_verdict_and_reason(llm_content)
            score = verdict_to_score(verdict)

            citation_judgments.append({
                "citation_idx": i,
                "url": url,
                "verdict": verdict,
                "reason": reason,
                "score": score
            })

        # 3) Compute average score
        if len(citation_judgments) == 0:
            aggregate_score = 0.0
        else:
            sum_scores = sum(c["score"] for c in citation_judgments)
            aggregate_score = sum_scores / len(citation_judgments)

        # 4) Make an extra LLM call with partial results => final label
        aggregate_verdict = get_aggregate_verdict_from_llm(citation_judgments, llm_client)

        # 5) Return the final dictionary
        result = {
            "citations": citation_judgments,
            "aggregate_score": aggregate_score,
            "aggregate_verdict": aggregate_verdict
        }
        return result

    ###############################################################################
    # 5) Store in Graph
    ###############################################################################

    def update_edge_with_judge_results(graph, src, tgt, rel_type, judge_data: Dict[str, Any]):
        """
        Convert judge_data to JSON and store as node property in Neo4j or whatever DB.
        Example property: r.judge_results_json
        """
        import json
        judge_json = json.dumps(judge_data, ensure_ascii=False)

        query = f"""
        MATCH (s:variable {{name:$src}})-[r:{rel_type}]->(t:variable {{name:$tgt}})
        SET r.judge_results_json = $judge_json,
            r.aggregate_score = $aggregate_score,
            r.aggregate_verdict = $aggregate_verdict
        """
        params = {
            "src": src,
            "tgt": tgt,
            "judge_json": judge_json,
            "aggregate_score": judge_data["aggregate_score"],
            "aggregate_verdict": judge_data["aggregate_verdict"]
        }

        # Execute query on your actual Neo4j connection
        with graph.session() as session:
            session.run(query, params)

        print(f"Edge updated with:\n{judge_json}")

# Configuration parameters
# Edit these parameters as needed
num_variables = 5
output_file = 'causal_graph.json'
target_variable = 'Lower Back Pain'
temporal_scale = 'Years'
spatial_scale = 'Community Dwelling Elderly'
parallel_processing = False
max_workers = 7
yaml_path = "../backend/configs/prompts.yaml"
dev_mode = False
corruption_rate = 1

# Environment-specific configurations
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "your-api-key")
PERPLEXITY_API_URL = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/chat/completions")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-medium-online")
# Get Claude API credentials from environment
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "your-claude-api-key")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")


# Get OpenAI API credentials from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")


print(f"Configuration set for target: {target_variable}")
print(f"Will generate {num_variables} variables with parallel processing: {parallel_processing}")


def inspect_relationships_debug(self):
    """Display all relationships with their properties for the current session with detailed debugging."""
    try:
        with self.graph_db._get_session() as session:
            query = """
            MATCH (n:variable)-[r]->(m:variable)
            WHERE n.session_id = $session_id AND m.session_id = $session_id
            RETURN n.name as source, m.name as target, type(r) as relationship_type, 
                   properties(r) as properties
            """
            
            print(f"\n[DEBUG] Running query with session_id: {self.session_id}")
            print(f"[DEBUG] Cypher query: {query}")
            
            results = list(session.run(query, {"session_id": self.session_id}))
            
            print(f"\nFound {len(results)} relationships in session {self.session_id}:")
            for i, record in enumerate(results, 1):
                print(f"\n{i}. {record['source']}-[{record['relationship_type']}]->{record['target']}")
                print(f"   Properties: {record['properties']}")
                
                # IMPORTANT DEBUGGING: Try to retrieve this exact relationship using _get_edge_motivation
                print(f"   [DEBUG] Now trying to retrieve the same relationship with _get_edge_motivation:")
                motivation = self._get_edge_motivation_debug(
                    record['source'], 
                    record['target'], 
                    record['relationship_type']
                )
                print(f"   [DEBUG] _get_edge_motivation returned: '{motivation}'")
            
            return results
    except Exception as e:
        print(f"Error inspecting relationships: {e}")
        return []
    
def mean_and_95ci_bounds(values):
    """Return mean and the actual lower/upper bounds of the 95% CI"""
    if not values:
        return 0.0, 0.0, 0.0
    values_array = np.array(values)
    mu = np.mean(values_array)
    # Calculate actual 2.5th and 97.5th percentiles
    lower = np.percentile(values_array, 2.5)
    upper = np.percentile(values_array, 97.5)
    return mu, lower, upper

def safe_bool(x):
    return True if x is True else False

def run_discovery_with_predefined_variables(
    variables,
    target_variable,
    temporal_scale="2-5 years",
    spatial_scale="Community Dwelling Elderly",
    yaml_path="../backend/configs/prompts.yaml",
    corruption_rate=0.5,
    run_judgment=True,
    judge_models=["claude-3-7-sonnet-20250219"],
    num_judges=1,
    parallel_processing=False,
    max_workers=3,
    dev_mode=False
):
    """
    Run causal discovery with a predefined list of variables.
    
    Args:
        variables (list): List of variable names to use
        target_variable (str): The target variable name (must be in the variables list)
        temporal_scale (str): Temporal scale for the causal relationships
        spatial_scale (str): Spatial scale for the causal relationships
        yaml_path (str): Path to the prompts YAML file
        corruption_rate (float): Rate of corruption for the edges (0-1)
        run_judgment (bool): Whether to run judgment on the edges
        judge_models (list or str): Models to use for judgment
        num_judges (int): Number of judges to use
        parallel_processing (bool): Whether to use parallel processing
        max_workers (int): Maximum number of workers for parallel processing
        dev_mode (bool): Whether to run in development mode
        
    Returns:
        dict: Result dictionary with session_id and stats
    """
    # Load environment variables
    load_dotenv("../.env.dev", override=True)
    
    # Check that target_variable is in the variables list
    if target_variable not in variables:
        raise ValueError(f"Target variable '{target_variable}' must be in the variables list")
    
    # Initialize CausalDiscovery instance
    discovery = CausalDiscovery(
        target_variable=target_variable,
        temporal_scale=temporal_scale,
        spatial_scale=spatial_scale,
        yaml_path=yaml_path,
        dev_mode=dev_mode,
        generator_model=os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219"),
        corruptor_model=os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219"),
        judge_model=os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
    )
    
    # Create nodes for predefined variables
    session_id = str(uuid.uuid4())
    discovery.session_id = session_id
    
    # Add variables to the graph database
    with discovery.graph_db._get_session() as session:
        # First add the target variable
        query = """
        CREATE (n:variable {name: $name, description: $description, session_id: $session_id, target: $is_target})
        """
        session.run(query, {
            "name": target_variable,
            "description": f"Target variable: {target_variable}",
            "session_id": session_id,
            "is_target": True
        })
        
        # Then add the other variables
        for var in variables:
            if var != target_variable:
                session.run(query, {
                    "name": var,
                    "description": f"Variable: {var}",
                    "session_id": session_id,
                    "is_target": False
                })
    
    print(f"Created {len(variables)} variables with session ID: {session_id}")
    
    # Generate relationships between variables
    relationship_count = 0
    corrupted_count = 0
    
    # Create all possible pairs of variables
    from itertools import permutations
    variable_pairs = []
    for source, target in permutations(variables, 2):
        variable_pairs.append((source, target))
    
    # Generate relationships
    print("Generating relationships between variables...")
    for source, target in variable_pairs:
        # Skip if relationship already exists
        existing = discovery._relationship_exists(source, target)
        if existing:
            continue
            
        # Generate relationship
        relationship = discovery._generate_relationship(source, target)
        if relationship and relationship != "NONE":
            relationship_count += 1
            
            # Corrupt relationship if needed
            if corruption_rate > 0 and discovery._should_corrupt(corruption_rate):
                discovery._corrupt_relationship(source, target, relationship)
                corrupted_count += 1
    
    # Run judgment if requested
    judged_edges = []
    if run_judgment:
        print("Running judgment on relationships...")
        if parallel_processing:
            judged_edges = discovery.judge_all_edges(judge_models=judge_models, num_judges=num_judges)
        else:
            judged_edges = discovery.judge_all_edges_serial(judge_models=judge_models)
    
    # Create result dictionary
    result = {
        "session_id": session_id,
        "stats": {
            "variable_count": len(variables),
            "relationship_count": relationship_count,
            "corrupted_count": corrupted_count,
            "judged_count": len(judged_edges)
        }
    }
    
    # Debug: Inspect relationships
    discovery.inspect_relationships()
    
    return result


def extract_networkx_from_session(
    discovery,
    session_ids=None,
    plot_graph=False,
    only_cited_edges=False,
    only_consistent_edges=False
):
    """
    Extract a NetworkX graph from the Neo4j database based on session ID(s).
    
    Args:
        discovery: CausalDiscovery object with graph_db for Neo4j queries
        session_ids: Session ID or list of session IDs, None to use discovery.session_id
        plot_graph: Whether to visualize the graph after building
        only_cited_edges: If True, only include edges with citations
        only_consistent_edges: If True, only include edges with "consistent" judge verdict
        
    Returns:
        NetworkX DiGraph containing the nodes and edges
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Handle session_ids: None, single ID, or list of IDs
    if session_ids is None:
        session_ids = discovery.session_id
    if isinstance(session_ids, str):
        session_ids = [session_ids]
    
    # Build Neo4j query
    query = """
    MATCH (s:variable)-[r]->(t:variable)
    WHERE s.session_id = $session_id AND t.session_id = $session_id
    """
    
    # Add filters if needed
    if only_cited_edges:
        query += " AND r.citation IS NOT NULL AND r.citation <> ''"
    if only_consistent_edges:
        query += " AND r.judge_verdict = 'OK'"
    
    query += """
    RETURN s.name AS source, t.name AS target
    """
    
    # For each session ID, fetch nodes and edges
    for session_id in session_ids:
        params = {"session_id": session_id}
        
        # Execute query with current session ID
        with discovery.graph_db._get_session() as session:
            results = list(session.run(query, params))
        
        # Add nodes and edges to the graph
        for record in results:
            src = record["source"]
            tgt = record["target"]
            
            G.add_node(src)
            G.add_node(tgt)
            G.add_edge(src, tgt)
    
    # Plot the graph if requested
    if plot_graph:
        pos = nx.spring_layout(G, seed=42, k=2)
        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='lightblue',
            edge_color='gray',
            arrows=True
        )
        
        # Create descriptive title based on filters
        title = "Graph from Neo4j Session"
        if only_cited_edges:
            title += " (Cited Edges Only)"
        if only_consistent_edges:
            title += " (Consistent Edges Only)"
            
        plt.title(title)
        plt.axis("off")
        plt.show()
    
    return G

import json
import networkx as nx
import matplotlib.pyplot as plt


def build_networkx_graph(
    vars_json_path: str = None,
    edges_json_path: str = None,
    node_list: list = None,
    edge_list: list = None,
    plot_graph: bool = True
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from either:
      1) JSON files containing unique_variables and edges, OR
      2) Python lists of nodes & edges, OR
      3) A mix of the two.

    The resulting graph only includes edges whose (source, target) 
    both exist in the final node set.

    Args:
        vars_json_path: Path to a JSON file with variables list. Optional.
        edges_json_path: Path to a JSON file with edges list. Optional.
        node_list: List of node names or items to use (strings). Optional.
        edge_list: List of edges (tuples or dicts). 
                   - If tuple, must be (source, target).
                   - If dict, must have {"source": X, "target": Y}.
        plot_graph: If True, plot the resulting DiGraph using Matplotlib.

    Returns:
        A networkx.DiGraph containing the specified nodes & edges.
        (Directed—if you want undirected, replace nx.DiGraph with nx.Graph.)
    """
    # 1) Gather nodes
    if node_list is not None:
        # If a node_list is provided, use it
        print("[DEBUG] node_list provided -> skipping vars_json_path.")
        final_nodes = set(node_list)
    else:
        final_nodes = set()
        if vars_json_path:
            print(f"[DEBUG] Loading variables from JSON: {vars_json_path}")
            with open(vars_json_path, "r") as fvars:
                vars_data = json.load(fvars)
            
            # Check the format of vars_data
            if isinstance(vars_data, dict) and "unique_variables" in vars_data:
                # Extract "unique_variables" as a list (original format)
                file_vars = vars_data.get("unique_variables", [])
                final_nodes = set(file_vars)
            elif isinstance(vars_data, list):
                # Direct list of variables (new format)
                if all(isinstance(item, str) for item in vars_data):
                    # Simple list of strings
                    final_nodes = set(vars_data)
                elif all(isinstance(item, dict) and "name" in item for item in vars_data):
                    # List of variable objects with "name" field
                    final_nodes = set(item["name"] for item in vars_data)
            else:
                print("[DEBUG] Unable to determine variables format in JSON")
        else:
            print("[DEBUG] No node_list or vars_json_path provided; no nodes to add initially.")

    # 2) Gather edges
    if edge_list is not None:
        # If an edge_list is provided, use it
        print("[DEBUG] edge_list provided -> skipping edges_json_path.")
        raw_edges = edge_list
    else:
        # Otherwise read from edges JSON if given
        raw_edges = []
        if edges_json_path:
            print(f"[DEBUG] Loading edges from JSON: {edges_json_path}")
            with open(edges_json_path, "r") as fedges:
                edges_data = json.load(fedges)
            
            # Check the format of edges_data
            if isinstance(edges_data, dict) and "edges" in edges_data:
                # Extract "edges" as a list (original format)
                file_edges = edges_data.get("edges", [])
                raw_edges = file_edges
            elif isinstance(edges_data, list):
                # Direct list of edges (new format)
                raw_edges = edges_data
            else:
                print("[DEBUG] Unable to determine edges format in JSON")
        else:
            print("[DEBUG] No edge_list or edges_json_path provided; no edges to add.")

    print(f"[DEBUG] final_nodes count: {len(final_nodes)}")
    print(f"[DEBUG] raw_edges count: {len(raw_edges)}")

    # 3) Create a directed graph
    G = nx.DiGraph()

    # 4) Add nodes to the graph
    for node in final_nodes:
        # If it's a string, treat as node name
        if isinstance(node, str):
            G.add_node(node)
        else:
            # Possibly handle custom node dict if needed
            # For now, we assume they're strings
            print(f"[DEBUG] Skipping invalid node (not a string): {node}")

    # 5) Process edges, filtering out those not in final_nodes
    valid_edge_count = 0
    for edge in raw_edges:
        if isinstance(edge, tuple) and len(edge) == 2:
            src, tgt = edge
        elif isinstance(edge, dict) and "source" in edge and "target" in edge:
            src = edge["source"]
            tgt = edge["target"]
        else:
            print(f"[DEBUG] Skipping invalid edge: {edge}")
            continue

        # Filter: only add if src,tgt are in final_nodes
        if src in final_nodes and tgt in final_nodes:
            # For dict edges, store extra attributes
            if isinstance(edge, dict):
                edge_attrs = dict(edge)
                edge_attrs.pop("source")
                edge_attrs.pop("target")
                G.add_edge(src, tgt, **edge_attrs)
            else:
                # tuple edge => no extra attrs
                G.add_edge(src, tgt)

            valid_edge_count += 1
        else:
            print(f"[DEBUG] Skipping edge {src}->{tgt} because src/tgt not in final_nodes")

    print(f"[DEBUG] Graph now has {G.number_of_nodes()} nodes, {valid_edge_count} edges added (valid).")

    # 6) If requested, plot the graph
    if plot_graph:
        print("[DEBUG] plot_graph=True, plotting the DiGraph now...")
        pos = nx.spring_layout(G, seed=42,k=2)
        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='lightblue',
            edge_color='gray',
            arrows=True
        )
        plt.title("Graph from JSON and/or Python Lists")
        plt.axis("off")
        plt.show()

    return G




import networkx as nx
import matplotlib.pyplot as plt

def filter_validation_graph_to_session_nodes(
    G_validation: nx.DiGraph,
    session_nodes: set,
    plot_graph: bool = False
) -> nx.DiGraph:
    """
    Return a *copy* of the G_validation graph that contains only the nodes
    which appear in session_nodes (and edges among those nodes).

    If plot_graph is True, we also display a basic spring-layout plot
    of the resulting filtered graph.
    """
    # Make a copy to avoid mutating the original
    G_filtered = nx.DiGraph()

    # Only add nodes that are in the session graph
    for node in G_validation.nodes():
        if node in session_nodes:
            # Copy over the node data as well
            G_filtered.add_node(node, **G_validation.nodes[node])

    # Only add edges if both src and tgt are in session_nodes
    for (src, tgt) in G_validation.edges():
        if src in session_nodes and tgt in session_nodes:
            # Copy over edge data too
            G_filtered.add_edge(src, tgt, **G_validation[src][tgt])

    print(f"[DEBUG] Filtered validation graph: {G_filtered.number_of_nodes()} nodes, "
          f"{G_filtered.number_of_edges()} edges.")

    # Plot the filtered graph if requested
    if plot_graph:
        print("[DEBUG] plot_graph=True, plotting the filtered validation graph.")
        pos = nx.spring_layout(G_filtered, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw(
            G_filtered,
            pos,
            with_labels=True,
            node_color='lightblue',
            edge_color='gray',
            arrows=True
        )
        plt.title("Filtered Validation Graph (only session nodes)")
        plt.axis("off")
        plt.show()

    return G_filtered


def compare_session_graph_to_validation(
    discovery,
    session_ids=None,
    validation_vars_json_path=None,
    validation_edges_json_path=None,
    plot_session_graph=False,
    plot_validation_graph=False,
    only_cited_edges=False,
    only_consistent_edges=False, 
    filter_validation_nodes_to_session_nodes=True,
    compare_variable_overlap = None,
):
    """
    Compare a session graph (from Neo4j) and a validation graph (from JSON) without
    removing or filtering out any disconnected nodes. We simply count edges that appear
    in both graphs, or only in one, to produce confusion-matrix metrics.

    Args:
        discovery: Your CausalDiscovery object with .graph_db for Neo4j queries.
        session_ids: A single session ID string or list of session IDs, or None 
                     to default to discovery.session_id.
        validation_vars_json_path: Path to validation JSON for nodes.
        validation_edges_json_path: Path to validation JSON for edges.
        plot_session_graph: Whether to visualize the session graph when built.
        plot_validation_graph: Whether to visualize the validation graph when built.
        only_cited_edges: If True, only edges with citation data are extracted from Neo4j.
        only_consistent_edges: If True, only edges with "consistent" judge verdict are extracted.

    Returns:
        A dictionary with:
            - session_graph_edges (set)
            - validation_graph_edges (set)
            - session_nodes (set)
            - validation_nodes (set)
            - tp, fp, fn, tn
            - precision, recall, f1, accuracy
    """


    # 1)
    G_session = extract_networkx_from_session(
        discovery=discovery,
        session_ids=session_ids,
        plot_graph=plot_session_graph,
        only_cited_edges=only_cited_edges,
        only_consistent_edges=only_consistent_edges
    )
    session_edges = set(G_session.edges())
    session_nodes = set(G_session.nodes())

    # 2) Build the validation graph from JSON
    G_validation = build_networkx_graph(
        vars_json_path=validation_vars_json_path,
        edges_json_path=validation_edges_json_path,
        plot_graph=plot_validation_graph
    )


    validation_edges = set(G_validation.edges())
    validation_nodes = set(G_validation.nodes())

    # Compare sameness of variables:
    if compare_variable_overlap:
        PROMPT = """
        You are an expert in data science and variable semantics analysis.

        Your task is to determine if two variables are semantically equivalent, meaning they represent the same underlying concept, even if named differently.

        For example:
        - 'age' and 'patient_age' are semantically equivalent
        - 'gender' and 'sex' are semantically equivalent
        - 'income' and 'salary' are semantically equivalent
        - 'temperature_c' and 'temperature_f' are NOT semantically equivalent (different units)
        - 'first_name' and 'full_name' are NOT semantically equivalent (different scope)

        When making your determination:
        1. Focus on the underlying concept, not just the variable name
        2. Consider common synonyms and domain-specific terminology
        3. Be consistent in your judgments
        4. Provide a clear True/False answer

        Respond with a boolean true/false **and** a short explanation of your reasoning.
        """

        class InputTwoEquivalent(BaseModel):
            target: str = Field(..., description="First variable name")
            query: str = Field(..., description="Second variable name to compare")

        class OutputEquivalent(BaseModel):
            MeansEquivalent: bool = Field(..., description="True if the two variables are semantically equivalent")
            Reasoning: str    = Field(..., description="Short explanation of the decision")

        Equivalency_Checker_agent = Agent(
            name="EquivalencyChecker",
            instructions=PROMPT,
            model="gpt-4o",
            output_type=OutputEquivalent,   # The agent will return this schema
        )
    # 3) Setup agent for semantic comparison
        correct = 0
        incorrect = 0
        total_comparisons = 0


        for session_var in session_nodes:
            for validation_var in validation_nodes:
                try:
                    print(f"Comparing: {session_var} vs {validation_var}")
                    
                    # Call the agent with proper configuration and input format
                    result = Runner.run_sync(
                        Equivalency_Checker_agent,
                        [
                            {
                                "role": "user", 
                                "content": f"Are these two variables semantically equivalent?\n\ntarget: {session_var}\nquery: {validation_var}"
                            }
                        ],
                    )
                    
                    # Debug information
                    print(f"Result type: {type(result.final_output)}")
                    
                    # Access the result properly
                    if hasattr(result.final_output, 'MeansEquivalent'):
                        agent_says_equivalent = result.final_output.MeansEquivalent
                        print(f"Equivalent: {agent_says_equivalent}, Reasoning: {result.final_output.Reasoning}")
                    else:
                        # Fallback parsing if needed
                        print(f"WARNING: Unexpected response format: {result.final_output}")
                        response_text = str(result.final_output).upper()
                        agent_says_equivalent = any(word in response_text for word in ["TRUE", "YES", "EQUIVALENT"])
                        
                    # Update counts
                    if agent_says_equivalent:
                        correct += 1
                    else:
                        incorrect += 1
                        
                    total_comparisons += 1
                    
                except Exception as e:
                    print(f"Error comparing '{session_var}' and '{validation_var}': {e}")
                    traceback.print_exc()  # Print full stack trace for debugging

        # Calculate ratio
        ratio_vars_correct = correct / total_comparisons if total_comparisons > 0 else 0.0
        print(f"Ratio of correct variables: {ratio_vars_correct:.2f} ({correct}/{total_comparisons})")
    else:
        ratio_vars_correct = 'n/a'
        total_comparisons = 'n/a'
        correct = 'n/a'
        incorrect = 'n/a'
        

    
    if filter_validation_nodes_to_session_nodes:
        G_filtered_validation = filter_validation_graph_to_session_nodes(G_validation, session_nodes, plot_graph=True)
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
    #    (since we're not filtering/discarding anything).
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
        "ratio_vars_correct": ratio_vars_correct,
        "correct vars": correct,
        "incorrect vars": incorrect,
        "total_comparisons": total_comparisons,
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


def plot_validation_graph_for_variables(
    variable_list,
    validation_vars_json_path,
    validation_edges_json_path,
    figsize=(12, 8),
    seed=42
):
    """
    Build and plot a validation graph that only includes the specified variables.
    
    Args:
        variable_list: List of variable names to include in the graph
        validation_vars_json_path: Path to JSON file with full list of variables
        validation_edges_json_path: Path to JSON file with edges
        figsize: Size of the figure as a tuple (width, height)
        seed: Random seed for graph layout reproducibility
        
    Returns:
        The filtered NetworkX DiGraph object
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # 1) Build the full validation graph from JSON
    print(f"[DEBUG] Building validation graph from JSON files")
    G_validation_full = build_networkx_graph(
        vars_json_path=validation_vars_json_path,
        edges_json_path=validation_edges_json_path,
        plot_graph=False  # Don't plot the full graph
    )
    
    # 2) Create a set of variables to keep
    variables_to_keep = set(variable_list)
    print(f"[DEBUG] Filtering validation graph to {len(variables_to_keep)} variables")
    
    # 3) Create a new filtered graph
    G_filtered = nx.DiGraph()
    
    # 4) Add nodes from the variable list that exist in the full graph
    for node in variables_to_keep:
        if node in G_validation_full.nodes():
            # Copy node attributes if they exist
            attrs = G_validation_full.nodes.get(node, {})
            G_filtered.add_node(node, **attrs)
    
    # 5) Add edges where both source and target are in our variable list
    for src, tgt in G_validation_full.edges():
        if src in variables_to_keep and tgt in variables_to_keep:
            # Copy edge attributes
            edge_attrs = G_validation_full[src][tgt]
            G_filtered.add_edge(src, tgt, **edge_attrs)
    
    # 6) Plot the filtered graph
    print(f"[DEBUG] Plotting filtered graph with {G_filtered.number_of_nodes()} nodes and {G_filtered.number_of_edges()} edges")
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G_filtered, seed=seed)
    nx.draw(
        G_filtered,
        pos,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        arrows=True
    )
    plt.title(f"Validation Graph (filtered to {len(variables_to_keep)} variables)")
    plt.axis("off")
    plt.show()
    
    return G_filtered

def find_connected_variable_subsets(
    validation_vars_json_path,
    validation_edges_json_path,
    subset_size=5,
    min_connectivity_pct=40,
    max_attempts=10000,
    return_count=5
):
    """Find random subsets of variables with sufficient connectivity in validation graph."""
    import random
    import json
    import networkx as nx
    
    # Load variables
    with open(validation_vars_json_path, "r") as f:
        all_variables = json.load(f).get("unique_variables", [])
        
    # Load edges
    with open(validation_edges_json_path, "r") as f:
        edges_data = json.load(f).get("edges", [])
    
    # Build validation graph
    G_validation = nx.DiGraph()
    G_validation.add_nodes_from(all_variables)
    for edge in edges_data:
        if isinstance(edge, dict) and "source" in edge and "target" in edge:
            G_validation.add_edge(edge["source"], edge["target"])
    
    connected_subsets = []
    attempts = 0
    
    while attempts < max_attempts and len(connected_subsets) < return_count:
        attempts += 1
        
        # Generate random subset
        variable_subset = random.sample(all_variables, subset_size)
        
        # Build subgraph
        sub_nodes = set(variable_subset)
        sub_edges = [(s, t) for s, t in G_validation.edges() 
                     if s in sub_nodes and t in sub_nodes]
        
        # Calculate connectivity percentage
        max_possible_edges = len(sub_nodes) * (len(sub_nodes) - 1)
        connectivity_pct = (len(sub_edges) / max_possible_edges * 100) if max_possible_edges > 0 else 0
        
        if connectivity_pct >= min_connectivity_pct:
            connected_subsets.append((variable_subset, connectivity_pct))
            print(f"Found connected subset: {connectivity_pct:.1f}% connected")
    
    # Sort by connectivity (highest first)
    return sorted(connected_subsets, key=lambda x: x[1], reverse=True)


def run_temperature_experiments(
    gen_temps=[0.2, 0.4, 0.8],
    corr_temps=None,  
    judge_temp=0.4,  
    connected_subsets=None,
    validation_vars_json_path="Cillian_CLD_unique_vars_data.json",
    validation_edges_json_path="Cillian_CLD_edges_data.json",
    llm_params=None,
    citations_only=False,
    judge_models=None,
    num_judges=3,
    visualize_comparison=True,  # Control visualizations
    runs_per_subgraph=1,        # NEW: Number of runs per subgraph
    num_subgraphs=None          # NEW: Number of subgraphs to use (default: len(gen_temps))
):
    """Run causal discovery experiments with different temperatures on connected subgraphs."""
    import os
    import json
    import datetime
    from dotenv import load_dotenv
    
    # Setup
    load_dotenv("../.env.dev", override=True)
    results = {}
    
    # If corruptor temps not specified, use same as generator temps
    if corr_temps is None:
        corr_temps = gen_temps
    
    # Default to one subgraph per temperature combination if not specified
    if num_subgraphs is None:
        num_subgraphs = len(gen_temps)
    
    # Find connected subsets if not provided
    if connected_subsets is None:
        connected_subsets = find_connected_variable_subsets(
            validation_vars_json_path, validation_edges_json_path,
            subset_size=5, min_connectivity_pct=15, return_count=num_subgraphs
        )
    
    # Default parameters
    if llm_params is None:
        llm_params = {
            "generator_model": os.environ.get("PERPLEXITY_MODEL", "sonar-pro"),
        }
    if judge_models is None:
        judge_models = [os.environ.get("PERPLEXITY_MODEL", "sonar-pro")] * num_judges
    
    # For each subgraph
    for subgraph_idx in range(min(num_subgraphs, len(connected_subsets))):
        variable_subset, connectivity = connected_subsets[subgraph_idx]
        print(f"\n{'#'*80}")
        print(f"SUBGRAPH #{subgraph_idx+1}: {variable_subset} with {connectivity:.1f}% connectivity")
        print(f"{'#'*80}")
        
        # For each temperature combination
        for temp_idx, (gen_temp, corr_temp) in enumerate(zip(gen_temps, corr_temps)):
            print(f"\n{'='*30} TEMPERATURE SET #{temp_idx+1} {'='*30}")
            print(f"Generator Temp: {gen_temp}, Corruptor Temp: {corr_temp}, Judge Temp: {judge_temp}")
            
            # Run multiple times per subgraph-temperature combination
            for run_idx in range(runs_per_subgraph):
                print(f"\n{'-'*30} RUN #{run_idx+1}/{runs_per_subgraph} {'-'*30}")
                
                # Initialize model with proper temperatures
                discovery = CausalDiscovery(
                    target_variable="",  # Use first variable as target
                    temporal_scale="Years",
                    spatial_scale="Community Dwelling Elderly",
                    yaml_path="../backend/configs/prompts.yaml", 
                    dev_mode=False,
                    generator_temperature=gen_temp,
                    corruptor_temperature=corr_temp,
                    judge_temperature=judge_temp,
                    **llm_params
                )
                
                # Add variables to graph
                session_id = discovery.session_id
                discovery.variables = variable_subset.copy()
                
                # Plot validation graph first if visualization is enabled
                # Create nodes in Neo4j
                with discovery.graph_db._get_session() as session:
                    for var in variable_subset:
                        session.run("""
                        CREATE (n:variable {name: $name, description: $desc, session_id: $sid, target: $is_target})
                        """, {
                            "name": var, 
                            "desc": f"Variable: {var}", 
                            "sid": session_id,
                            "is_target": var == variable_subset[0]
                        })
                
                # Generate relationships
                print(f"Discovering relationships...")
                rels = discovery.discover_relationships(parallel=False)
                
                # Judge the edges if requested
                if num_judges > 0:
                    print(f"Judging edges with {num_judges} judges")
                    judged = discovery.judge_all_edges_serial(judge_models=judge_models)
                
                # # Visualize the discovered graph
                # if show_visualization:
                #     print("\nDiscovered Graph:")
                #     G_session = extract_networkx_from_session(
                #         discovery=discovery,
                #         session_ids=session_id,
                #         plot_graph=True
                #     )
                    
                    
                # Get metrics compared to validation graph
                if visualize_comparison:
                            plot_session_graph = True
                            plot_validation_graph = True
                            
                        # Get metrics compared to validation graph
                        # Get metrics compared to validation graph
                metrics = compare_session_graph_to_validation(discovery= discovery,session_ids=session_id,validation_vars_json_path="Cillian_CLD_unique_vars_data.json",validation_edges_json_path="Cillian_CLD_edges_data.json", plot_session_graph=True, plot_validation_graph=True, only_cited_edges=False, only_consistent_edges=False, filter_validation_nodes_to_session_nodes=False)
                
                # Filter by citations if requested
                if citations_only:
                    # ... (same citation handling code as before)
                    with discovery.graph_db._get_session() as session:
                        query = """
                        MATCH (s)-[r]->(t) 
                        WHERE s.session_id = $sid AND t.session_id = $sid 
                        AND (exists(r.motivation) OR exists(r.citations))
                        RETURN s.name AS src, t.name AS tgt, 
                               CASE WHEN exists(r.citations) THEN r.citations ELSE [] END AS citations,
                               CASE WHEN exists(r.motivation) THEN r.motivation ELSE '' END AS motivation
                        """
                        edge_data = list(session.run(query, {"sid": session_id}))
                        cited_edges = []
                        for record in edge_data:
                            has_citation = len(record["citations"]) > 0 if record["citations"] else False
                            has_reference = "reference" in record["motivation"].lower() if record["motivation"] else False
                            if has_citation or has_reference:
                                cited_edges.append((record["src"], record["tgt"]))
                        
                        metrics["cited_edges"] = cited_edges
                        metrics["cited_edge_count"] = len(cited_edges)
                
                # Store results with detailed key to distinguish runs
                exp_key = f"subgraph{subgraph_idx+1}_gen{gen_temp}_corr{corr_temp}_run{run_idx+1}"
                results[exp_key] = {
                    "subgraph_idx": subgraph_idx + 1,
                    "temperature_idx": temp_idx + 1,
                    "run_idx": run_idx + 1,
                    "variables": variable_subset,
                    "connectivity": connectivity,
                    "session_id": session_id,
                    "metrics": metrics,
                    "generator_temperature": gen_temp,
                    "corruptor_temperature": corr_temp,
                    "judge_temperature": judge_temp,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                print(f"Completed run #{run_idx+1} for subgraph #{subgraph_idx+1}, temperature set #{temp_idx+1}")
    
    # Compute aggregated metrics if multiple runs were performed
    if runs_per_subgraph > 1:
        aggregated_results = {}
        
        # Group by subgraph and temperature combination
        for subgraph_idx in range(min(num_subgraphs, len(connected_subsets))):
            for temp_idx, (gen_temp, corr_temp) in enumerate(zip(gen_temps, corr_temps)):
                # Collect all runs for this combination
                runs_metrics = []
                
                for run_idx in range(runs_per_subgraph):
                    key = f"subgraph{subgraph_idx+1}_gen{gen_temp}_corr{corr_temp}_run{run_idx+1}"
                    if key in results:
                        runs_metrics.append(results[key]["metrics"])
                
                # Calculate averages
                if runs_metrics:
                    avg_metrics = {
                        "precision": sum(m["precision"] for m in runs_metrics) / len(runs_metrics),
                        "recall": sum(m["recall"] for m in runs_metrics) / len(runs_metrics),
                        "f1": sum(m["f1"] for m in runs_metrics) / len(runs_metrics),
                        "tp_avg": sum(m["tp"] for m in runs_metrics) / len(runs_metrics),
                        "fp_avg": sum(m["fp"] for m in runs_metrics) / len(runs_metrics),
                        "fn_avg": sum(m["fn"] for m in runs_metrics) / len(runs_metrics),
                        "tn_avg": sum(m["tn"] for m in runs_metrics) / len(runs_metrics),
                        "run_count": len(runs_metrics)
                    }
                    
                    agg_key = f"subgraph{subgraph_idx+1}_gen{gen_temp}_corr{corr_temp}_AVERAGE"
                    aggregated_results[agg_key] = avg_metrics
        
        # Add aggregated results to the full results
        results["aggregated"] = aggregated_results
    
    return results

def visualize_temperature_experiment_results(results, show_plots=True, save_plots=False, output_dir="experiment_results"):
    """
    Visualize and summarize the results from temperature experiments.
    
    Args:
        results: Dictionary output from run_temperature_experiments
        show_plots: Whether to display plots interactively
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots (if save_plots is True)
    """
    # Create output directory if saving plots
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Organize results into pandas DataFrames
    full_data = []
    
    # Process regular results (exclude 'aggregated')
    for key, data in results.items():
        if key == "aggregated":
            continue
            
        # Extract key components
        if 'subgraph_idx' in data:
            # New format with explicit indexes
            subgraph_idx = data['subgraph_idx']
            temp_idx = data['temperature_idx'] 
            run_idx = data['run_idx']
        else:
            # Parse from the key (older format)
            key_parts = key.split('_')
            subgraph_idx = int(key_parts[0].replace('subgraph', '')) if 'subgraph' in key_parts[0] else 1
            run_idx = int(key_parts[-1].replace('run', '')) if 'run' in key_parts[-1] else 1
            temp_idx = 1  # Default if not specified
            
        metrics = data['metrics']
        
        # Create a row with all relevant data
        row = {
            'Subgraph': subgraph_idx,
            'Run': run_idx,
            'Generator Temp': data.get('generator_temperature', 0.0),
            'Corruptor Temp': data.get('corruptor_temperature', 0.0),
            'Judge Temp': data.get('judge_temperature', 0.0),
            'Precision': metrics.get('precision', 0.0),
            'Recall': metrics.get('recall', 0.0),
            'F1': metrics.get('f1', 0.0),
            'Accuracy': metrics.get('accuracy', 0.0),  # Add accuracy
            'TP': metrics.get('tp', 0),
            'FP': metrics.get('fp', 0),
            'FN': metrics.get('fn', 0),
            'TN': metrics.get('tn', 0),
            'Session ID': data.get('session_id', '')
        }
        full_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(full_data)
    
    # If no data, return empty DataFrame
    if len(df) == 0:
        print("No results to visualize.")
        return df, None
    
    # Process aggregated results if available
    agg_df = None
    if "aggregated" in results and results["aggregated"]:
        agg_data = []
        
        for key, metrics in results["aggregated"].items():
            # Parse key components
            key_parts = key.split('_')
            subgraph_idx = int(key_parts[0].replace('subgraph', ''))
            gen_temp = float(key_parts[1].replace('gen', ''))
            corr_temp = float(key_parts[2].replace('corr', ''))
            
            # Create a row with aggregated data
            row = {
                'Subgraph': subgraph_idx,
                'Generator Temp': gen_temp,
                'Corruptor Temp': corr_temp,
                'Precision': metrics.get('precision', 0.0),
                'Recall': metrics.get('recall', 0.0),
                'F1': metrics.get('f1', 0.0),
                'Accuracy': metrics.get('accuracy', 0.0),  # Add accuracy
                'TP Avg': metrics.get('tp_avg', 0),
                'FP Avg': metrics.get('fp_avg', 0),
                'FN Avg': metrics.get('fn_avg', 0),
                'TN Avg': metrics.get('tn_avg', 0),
                'Run Count': metrics.get('run_count', 0)
            }
            agg_data.append(row)
        
        agg_df = pd.DataFrame(agg_data)
    
    # =====================================================
    # Visualizations
    # =====================================================
    
    # Plot 1: Metrics by temperature for each subgraph
    if show_plots or save_plots:
        metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
        colors = ['blue', 'green', 'red', 'purple']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get data for plotting
            x = stats_by_temp['Generator Temp']
            y = stats_by_temp[f'{metric}_mean']
            y_min = stats_by_temp[f'{metric}_min']
            y_max = stats_by_temp[f'{metric}_max']
            
            # Calculate error bar sizes
            yerr_lower = y - y_min
            yerr_upper = y_max - y
            yerr = np.array([yerr_lower, yerr_upper])
            
            # Plot with error bars
            ax.errorbar(
                x, y, 
                yerr=yerr,
                fmt='o-',
                capsize=6, 
                color=colors[i],
                markersize=8,
                linewidth=2,
                elinewidth=1.5
            )
            
            ax.set_title(f'{metric}', fontsize=12)
            ax.set_xlabel('Generator Temperature')
            ax.set_ylabel('Score')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Set y-axis limits with padding
            ax.set_ylim(max(0, min(y_min) - 0.05), min(1.05, max(y_max) + 0.05))
            
            # Annotate with values
            for j, (x_val, y_val) in enumerate(zip(x, y)):
                ax.annotate(f'{y_val:.2f}', 
                           (x_val, y_val), 
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center')
        
        plt.suptitle('Metrics vs Generator Temperature with Min/Max Error Bars', fontsize=14)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/metrics_faceted_{timestamp}.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Option 2: Staggered Points (horizontal jitter)
    if show_plots or save_plots:
        plt.figure(figsize=(12, 8))
        
        metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
        colors = ['blue', 'green', 'red', 'purple']
        markers = ['o', 's', '^', 'D']
        
        # Define offsets to avoid overlap
        offsets = [-0.015, -0.005, 0.005, 0.015]
        
        for i, metric in enumerate(metrics):
            # Get data for plotting
            x = stats_by_temp['Generator Temp']
            y = stats_by_temp[f'{metric}_mean']
            y_min = stats_by_temp[f'{metric}_min']
            y_max = stats_by_temp[f'{metric}_max']
            
            # Apply offset to x values
            x_offset = [x_val + offsets[i] for x_val in x]
            
            # Calculate error bar sizes
            yerr_lower = y - y_min
            yerr_upper = y_max - y
            yerr = np.array([yerr_lower, yerr_upper])
            
            plt.errorbar(
                x_offset, y, 
                yerr=yerr,
                fmt=markers[i],
                capsize=5, 
                color=colors[i],
                label=metric,
                markersize=8,
                linewidth=1.5,
                elinewidth=1
            )
            
            # Connect points with lines using original x (not offset)
            plt.plot(x, y, color=colors[i], alpha=0.5, linewidth=1.2)
        
        plt.xlabel('Generator Temperature', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Metrics vs Generator Temperature with Min/Max Error Bars', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{output_dir}/metrics_staggered_{timestamp}.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    # Option 3: Grouped Bar Chart
    if show_plots or save_plots:
        plt.figure(figsize=(14, 8))
        
        metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
        colors = ['blue', 'green', 'red', 'purple']
        
        x = np.arange(len(stats_by_temp['Generator Temp']))
        width = 0.18
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            y = stats_by_temp[f'{metric}_mean']
            yerr_lower = y - stats_by_temp[f'{metric}_min']
            yerr_upper = stats_by_temp[f'{metric}_max'] - y
            yerr = np.array([yerr_lower, yerr_upper])
            
            offset = width * (i - 1.5)
            plt.bar(x + offset, y, width, label=metric, color=colors[i], alpha=0.7)
            
            # Add error bars
            plt.errorbar(x + offset, y, yerr=yerr, fmt='none', ecolor='black', capsize=5)
            
            # Add value labels
            for j, y_val in enumerate(y):
                plt.text(x[j] + offset, y_val + 0.02, f'{y_val:.2f}', 
                         ha='center', va='bottom', rotation=0, fontsize=9)
        
        # Set x-axis labels
        plt.xticks(x, [f"{temp:.1f}" for temp in stats_by_temp['Generator Temp']])
        
        plt.xlabel('Generator Temperature', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Metrics Comparison Across Generator Temperatures', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{output_dir}/metrics_bar_chart_{timestamp}.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Return the processed DataFrames for further analysis
    return df, agg_df



# def export_edges_comparison_to_excel(
#     discovery,
#     session_ids=None,
#     validation_vars_json_path=None,
#     validation_edges_json_path=None,
#     output_filename=None
# ):
#     """
#     Export edge evaluations for two distinct scenarios in a well-organized Excel file:
#     1. All Edges + Summary Stats 
#     2. Cited Edges Only + Summary Stats
#     3. Consistent Edges Only + Summary Stats
    
#     Args:
#         discovery: CausalDiscovery object with graph_db for Neo4j queries
#         session_ids: Session ID or list of session IDs, None to use discovery.session_id
#         validation_vars_json_path: Path to validation JSON for nodes
#         validation_edges_json_path: Path to validation JSON for edges
#         output_filename: Name of the Excel file to create
        
#     Returns:
#         Path to the created Excel file
#     """

#     # Add extension if not present
#     if not output_filename.endswith('.xlsx'):
#         output_filename = f"{output_filename}.xlsx"
    
#     # Default filename if none provided
#     if output_filename is None:
#         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         output_filename = f"edges_evaluation_{timestamp}.xlsx"
    
#     # Define the three scenarios
#     scenarios = [
#         {"name": "All Edges", "only_cited_edges": False, "only_consistent_edges": False},
#         {"name": "Cited Edges Only", "only_cited_edges": True, "only_consistent_edges": False},
#         {"name": "Consistent Edges Only", "only_cited_edges": False, "only_consistent_edges": True}
#     ]
    
#     # Create the Excel writer
#     with pd.ExcelWriter(output_filename) as writer:
#         # Process each scenario separately
#         for i, scenario in enumerate(scenarios):
#             try:
#                 # Run the analysis for this scenario
#                 metrics = compare_session_graph_to_validation(
#                     discovery=discovery,
#                     session_ids=session_ids,
#                     validation_vars_json_path=validation_vars_json_path,
#                     validation_edges_json_path=validation_edges_json_path,
#                     only_cited_edges=scenario["only_cited_edges"],
#                     only_consistent_edges=scenario["only_consistent_edges"]  # Consistently named!
#                 )
                
#                 # Extract edge sets
#                 session_edges = metrics["session_graph_edges"]
#                 validation_edges = metrics["validation_graph_edges"]
                
#                 # Get classification sets
#                 tp_edges = session_edges.intersection(validation_edges)
#                 fp_edges = session_edges - validation_edges
#                 fn_edges = validation_edges - session_edges
                
#                 # Get all possible node pairs for TN calculation
#                 union_nodes = metrics["session_nodes"].union(metrics["validation_nodes"])
#                 all_possible_pairs = {
#                     (src, tgt)
#                     for src in union_nodes
#                     for tgt in union_nodes
#                     if src != tgt
#                 }
                
#                 # True negatives are pairs that appear in neither graph
#                 combined_edges = session_edges.union(validation_edges)
#                 tn_edges = all_possible_pairs - combined_edges
                
#                 # Get edge data from Neo4j
#                 edge_data = {}
#                 if session_edges:
#                     # Query Neo4j for relationships and their properties
#                     # Now including judge_verdict and judge_message
#                     query = """
#                     MATCH (s:variable)-[r]->(t:variable)
#                     WHERE s.session_id = $session_id AND t.session_id = $session_id
#                     """
                    
#                     # Apply filters based on scenario
#                     if scenario["only_cited_edges"]:
#                         query += " AND r.citation IS NOT NULL AND r.citation <> ''"
#                     if scenario["only_consistent_edges"]:
#                         query += " AND r.judge_verdict = 'OK'"
                        
#                     query += """
#                     RETURN s.name AS source, t.name AS target, 
#                            r.motivation AS motivation,
#                            r.citation AS citation,
#                            r.citation_type AS citation_type,
#                            r.judge_verdict AS judge_verdict,
#                            r.judge_message AS judge_message
#                     """
#                     params = {"session_id": session_ids or discovery.session_id}
                    
#                     with discovery.graph_db._get_session() as session:
#                         results = list(session.run(query, params))
                    
#                     for record in results:
#                         src = record["source"]
#                         tgt = record["target"]
#                         motivation = record["motivation"] or "No motivation provided"
#                         citation = record["citation"] or ""
#                         citation_type = record["citation_type"] or ""
#                         judge_verdict = record["judge_verdict"] or "Not judged"
#                         judge_message = record["judge_message"] or ""
                        
#                         edge_data[(src, tgt)] = {
#                             "motivation": motivation,
#                             "citation": citation,
#                             "citation_type": citation_type,
#                             "has_citation": bool(citation),
#                             "judge_verdict": judge_verdict,
#                             "judge_message": judge_message
#                         }
                
#                 # Create data for each edge category
#                 edges_data = []
                
#                 # True Positives (TP)
#                 for edge in tp_edges:
#                     if edge in edge_data:
#                         edges_data.append({
#                             "Source": edge[0],
#                             "Target": edge[1],
#                             "Motivation": edge_data[edge]["motivation"],
#                             "Citation": edge_data[edge]["citation"],
#                             "Judge Verdict": edge_data[edge]["judge_verdict"],
#                             "Judge Message": edge_data[edge]["judge_message"],
#                             "In Session Graph": "Yes",
#                             "In Validation Graph": "Yes",
#                             "Classification": "TP"
#                         })
                
#                 # False Positives (FP)
#                 for edge in fp_edges:
#                     if edge in edge_data:
#                         edges_data.append({
#                             "Source": edge[0],
#                             "Target": edge[1],
#                             "Motivation": edge_data[edge]["motivation"],
#                             "Citation": edge_data[edge]["citation"],
#                             "Judge Verdict": edge_data[edge]["judge_verdict"],
#                             "Judge Message": edge_data[edge]["judge_message"],
#                             "In Session Graph": "Yes",
#                             "In Validation Graph": "No",
#                             "Classification": "FP"
#                         })
                
#                 # False Negatives (FN)
#                 for edge in fn_edges:
#                     edges_data.append({
#                         "Source": edge[0],
#                         "Target": edge[1],
#                         "Motivation": "Not discovered",
#                         "Citation": "",
#                         "Judge Verdict": "N/A",
#                         "Judge Message": "",
#                         "In Session Graph": "No",
#                         "In Validation Graph": "Yes",
#                         "Classification": "FN"
#                     })
                
#                 # Create a DataFrame for all edges in this scenario
#                 all_edges_df = pd.DataFrame(edges_data)
                
#                 # Create a summary DataFrame
#                 summary_data = {
#                     "Metric": [
#                         "True Positives (TP)", 
#                         "False Positives (FP)", 
#                         "False Negatives (FN)", 
#                         "True Negatives (TN)",
#                         "Total Discovered Edges",
#                         "Total Validation Edges",
#                         "Variables Correct Score", 
#                         "correct variables",
#                         "incorrect variables",
#                         "Total Comparisons",
#                         "Precision",
#                         "Recall",
#                         "F1 Score",
#                         "Accuracy"
#                     ],
#                     "Value": [
#                         metrics["tp"],
#                         metrics["fp"],
#                         metrics["fn"],
#                         metrics["tn"],
#                         len(session_edges),
#                         len(validation_edges),
#                         metrics.get("ratio_vars_correct", "N/A"),
#                         metrics.get("correct vars", "N/A"),
#                         metrics.get("incorrect vars", "N/A"),
#                         metrics.get("total_comparisons", "N/A"),
#                         metrics["precision"],
#                         metrics["recall"],
#                         metrics["f1"],
#                         metrics["accuracy"]
#                     ]
#                 }
#                 summary_df = pd.DataFrame(summary_data)
                
#                 # Write sheets for this scenario with proper names
#                 detail_sheet_name = scenario["name"]
#                 summary_sheet_name = f"{scenario['name']} Summary"
                
#                 all_edges_df.to_excel(writer, sheet_name=detail_sheet_name, index=False)
#                 summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)
#             except Exception as e:
#                 print(f"Error processing scenario '{scenario['name']}': {str(e)}")
    
#     print(f"Excel file created: {output_filename}")
#     print("The file contains 6 sheets:")
#     print("  All Edges - All edges and their details (includes judge verdicts)")
#     print("  All Edges Summary - Summary statistics for all edges")
#     print("  Cited Edges Only - Edges with citations only")
#     print("  Cited Edges Only Summary - Summary statistics for cited edges only")
#     print("  Consistent Edges Only - Edges with consistent judge verdict only") 
#     print("  Consistent Edges Only Summary - Summary statistics for consistent edges only")
    
#     return output_filename


# def export_edges_comparison_to_excel(
#     discovery,
#     session_ids=None,
#     validation_vars_json_path=None,
#     validation_edges_json_path=None,
#     output_filename=None
# ):
#     """
#     Export edge evaluations to Excel, generating multiple scenario-based sheets:
#       1) All Edges + Summary
#       2) Cited Edges Only + Summary
#       3) Consistent Edges Only (judge_verdict='OK') + Summary
#       4) Fully Supported Only (judge_verdict in ['OK','Fully supported']) + Summary
#       5) Partially Supported (judge_verdict in ['OK','Fully supported','Partially supported']) + Summary

#     Returns:
#         Path to the created Excel file
#     """
#     import datetime
#     import pandas as pd
#     # ^^ adjust import as needed

#     # 1) Prepare the output filename
#     if not output_filename:
#         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         output_filename = f"edges_evaluation_{timestamp}.xlsx"
#     elif not output_filename.endswith('.xlsx'):
#         output_filename += ".xlsx"

#     # 2) Define the scenarios
#     #    Each scenario dictates how we filter the edges out of Neo4j
#     #    We’ll carry an optional "verdict_list" for additional filtering
#     scenarios = [
#         {
#             "name": "All Edges",
#             "only_cited_edges": False,
#             "only_consistent_edges": False,
#             "verdict_list": None  # means no filter on verdict
#         },
#         {
#             "name": "Cited Edges Only",
#             "only_cited_edges": True,
#             "only_consistent_edges": False,
#             "verdict_list": None
#         },
#         {
#             "name": "Consistent Edges Only",  # the original
#             "only_cited_edges": False,
#             "only_consistent_edges": True,    # means 'OK'
#             "verdict_list": None
#         },
#         {
#             "name": "Fully Supported Only",
#             "only_cited_edges": False,
#             "only_consistent_edges": False,  
#             "verdict_list": ["OK", "Fully supported"]
#         },
#         {
#             "name": "Partially Supported",
#             "only_cited_edges": False,
#             "only_consistent_edges": False,
#             "verdict_list": ["OK", "Fully supported", "Partially supported"]
#         }
#     ]

#     # 3) Create the Excel writer
#     with pd.ExcelWriter(output_filename) as writer:

#         # 4) Process each scenario separately
#         for scenario in scenarios:
#             scenario_name = scenario["name"]
#             try:
#                 # === (A) Run the analysis for this scenario
#                 metrics = compare_session_graph_to_validation(
#                     discovery=discovery,
#                     session_ids=session_ids,
#                     validation_vars_json_path=validation_vars_json_path,
#                     validation_edges_json_path=validation_edges_json_path,
#                     only_cited_edges=scenario["only_cited_edges"],
#                     only_consistent_edges=scenario["only_consistent_edges"],
#                     verdict_list=scenario["verdict_list"],
#                 )

#                 # === (B) Extract the relevant sets
#                 session_edges = metrics["session_graph_edges"]
#                 validation_edges = metrics["validation_graph_edges"]
#                 session_nodes = metrics["session_nodes"]
#                 validation_nodes = metrics["validation_nodes"]

#                 tp_edges = session_edges.intersection(validation_edges)
#                 fp_edges = session_edges - validation_edges
#                 fn_edges = validation_edges - session_edges

#                 # For TN calculation: union of all nodes
#                 union_nodes = session_nodes.union(validation_nodes)
#                 all_pairs = {
#                     (src, tgt)
#                     for src in union_nodes
#                     for tgt in union_nodes
#                     if src != tgt
#                 }
#                 combined_edges = session_edges.union(validation_edges)
#                 tn_edges = all_pairs - combined_edges

#                 # === (C) Fetch extra property data for edges from DB
#                 #    This step is optional if compare_session_graph_to_validation 
#                 #    already returned them. Otherwise:
#                 edge_data = {}
#                 if session_edges:
#                     with discovery.graph_db._get_session() as session:
#                         # Build a base query
#                         query = """
#                         MATCH (s:variable)-[r]->(t:variable)
#                         WHERE s.session_id = $session_id
#                           AND t.session_id = $session_id
#                         """

#                         # Filter out edges if scenario says "only_cited_edges"
#                         if scenario["only_cited_edges"]:
#                             query += " AND r.citation IS NOT NULL AND r.citation <> ''"

#                         # Filter if scenario says "only_consistent_edges"
#                         # (the original approach for judge_verdict='OK')
#                         if scenario["only_consistent_edges"]:
#                             query += " AND r.judge_verdict = 'OK'"

#                         # Additional 'verdict_list' approach
#                         if scenario["verdict_list"]:
#                             # e.g. r.judge_verdict in ['OK','Fully supported']
#                             query += " AND r.judge_verdict IN $verdict_list"

#                         query += """
#                         RETURN s.name AS source, t.name AS target,
#                                r.motivation AS motivation,
#                                r.citation AS citation,
#                                r.citation_type AS citation_type,
#                                r.judge_verdict AS judge_verdict,
#                                r.judge_message AS judge_message
#                         """

#                         params = {
#                             "session_id": session_ids or discovery.session_id,
#                             "verdict_list": scenario["verdict_list"],
#                         }

#                         results = list(session.run(query, params))

#                     for record in results:
#                         src = record["source"]
#                         tgt = record["target"]
#                         edge_data[(src, tgt)] = {
#                             "motivation": record["motivation"] or "No motivation provided",
#                             "citation": record["citation"] or "",
#                             "citation_type": record["citation_type"] or "",
#                             "judge_verdict": record["judge_verdict"] or "Not judged",
#                             "judge_message": record["judge_message"] or ""
#                         }

#                 # === (D) Build row data for all edges
#                 edges_data = []

#                 # True Positives
#                 for e in tp_edges:
#                     row = {
#                         "Source": e[0],
#                         "Target": e[1],
#                         "In Session Graph": "Yes",
#                         "In Validation Graph": "Yes",
#                         "Classification": "TP",
#                     }
#                     # Add any property data if we have it
#                     if e in edge_data:
#                         row["Motivation"] = edge_data[e]["motivation"]
#                         row["Citation"]   = edge_data[e]["citation"]
#                         row["Judge Verdict"]  = edge_data[e]["judge_verdict"]
#                         row["Judge Message"]  = edge_data[e]["judge_message"]
#                     else:
#                         row["Motivation"] = "N/A"
#                         row["Citation"] = ""
#                         row["Judge Verdict"] = "N/A"
#                         row["Judge Message"] = ""
#                     edges_data.append(row)

#                 # False Positives
#                 for e in fp_edges:
#                     row = {
#                         "Source": e[0],
#                         "Target": e[1],
#                         "In Session Graph": "Yes",
#                         "In Validation Graph": "No",
#                         "Classification": "FP",
#                     }
#                     if e in edge_data:
#                         row["Motivation"] = edge_data[e]["motivation"]
#                         row["Citation"]   = edge_data[e]["citation"]
#                         row["Judge Verdict"]  = edge_data[e]["judge_verdict"]
#                         row["Judge Message"]  = edge_data[e]["judge_message"]
#                     else:
#                         row["Motivation"] = "N/A"
#                         row["Citation"] = ""
#                         row["Judge Verdict"] = "N/A"
#                         row["Judge Message"] = ""
#                     edges_data.append(row)

#                 # False Negatives
#                 for e in fn_edges:
#                     row = {
#                         "Source": e[0],
#                         "Target": e[1],
#                         "In Session Graph": "No",
#                         "In Validation Graph": "Yes",
#                         "Classification": "FN",
#                         "Motivation": "Not discovered",
#                         "Citation": "",
#                         "Judge Verdict": "N/A",
#                         "Judge Message": "",
#                     }
#                     edges_data.append(row)

#                 # === (E) Build a DataFrame for all edges
#                 all_edges_df = pd.DataFrame(edges_data)

#                 # === (F) Build the summary table
#                 summary_data = {
#                     "Metric": [
#                         "True Positives (TP)",
#                         "False Positives (FP)",
#                         "False Negatives (FN)",
#                         "True Negatives (TN)",
#                         "Total Discovered Edges",
#                         "Total Validation Edges",
#                         "Variables Correct Score",
#                         "correct variables",
#                         "incorrect variables",
#                         "Total Comparisons",
#                         "Precision",
#                         "Recall",
#                         "F1 Score",
#                         "Accuracy",
#                     ],
#                     "Value": [
#                         metrics["tp"],
#                         metrics["fp"],
#                         metrics["fn"],
#                         metrics["tn"],
#                         len(session_edges),
#                         len(validation_edges),
#                         metrics.get("ratio_vars_correct", "N/A"),
#                         metrics.get("correct vars", "N/A"),
#                         metrics.get("incorrect vars", "N/A"),
#                         metrics.get("total_comparisons", "N/A"),
#                         metrics["precision"],
#                         metrics["recall"],
#                         metrics["f1"],
#                         metrics["accuracy"],
#                     ],
#                 }
#                 summary_df = pd.DataFrame(summary_data)

#                 # === (G) Write these as two sheets
#                 detail_sheet_name = scenario_name
#                 summary_sheet_name = f"{scenario_name} Summary"

#                 all_edges_df.to_excel(writer, sheet_name=detail_sheet_name, index=False)
#                 summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)

#             except Exception as e:
#                 print(f"Error processing scenario '{scenario_name}': {e}")

#     # 5) Print a final message
#     print(f"Excel file created: {output_filename}")
#     print("The file contains the following sheets:")
#     print("  - All Edges + All Edges Summary")
#     print("  - Cited Edges Only + Cited Edges Only Summary")
#     print("  - Consistent Edges Only + Consistent Edges Only Summary")
#     print("  - Fully Supported Only + Fully Supported Only Summary")
#     print("  - Partially Supported + Partially Supported Summary\n")
#     return output_filename

# def export_edges_comparison_to_excel(
#     discovery,
#     session_ids=None,
#     validation_vars_json_path=None,
#     validation_edges_json_path=None,
#     output_filename=None
# ):
#     """
#     Export edge evaluations to Excel, generating multiple scenario-based sheets:

#       1) All Edges + Summary
#       2) Cited Edges Only + Summary
#       3) Consistent Edges Only (i.e. judge_verdict='OK') + Summary
#       4) Fully Supported Only (judge_verdict in ['OK','Fully supported']) + Summary
#       5) Partially Supported (judge_verdict in ['OK','Fully supported','Partially supported']) + Summary

#     Returns:
#         Path to the created Excel file
#     """
#     import datetime
#     import pandas as pd
#     import traceback

#     # If you have the function in the same file, no need for import;
#     # otherwise import your compare_session_graph_to_validation function
#     # from .whatever_module import compare_session_graph_to_validation

#     # 1) Prepare the output filename
#     if not output_filename:
#         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         output_filename = f"edges_evaluation_{timestamp}.xlsx"
#     elif not output_filename.endswith('.xlsx'):
#         output_filename += ".xlsx"

#     # Define your 5 scenarios
#     scenarios = [
#         {
#             "name": "All Edges",
#             "only_cited_edges": False,
#             "only_consistent_edges": False,
#             "verdict_filter": None  # means no local filter
#         },
#         {
#             "name": "Cited Edges Only",
#             "only_cited_edges": True,
#             "only_consistent_edges": False,
#             "verdict_filter": None
#         },
#         {
#             "name": "Consistent Edges Only",  # the original "OK" scenario
#             "only_cited_edges": False,
#             "only_consistent_edges": True,  # means r.judge_verdict='OK' inside the DB query
#             "verdict_filter": None
#         },
#         {
#             "name": "Fully Supported Only",
#             "only_cited_edges": False,
#             # We'll retrieve all edges from DB, ignoring "only_consistent_edges"
#             "only_consistent_edges": False,
#             # Then we do a local filter: in ["OK","Fully supported"]
#             "verdict_filter": ["OK", "Fully supported"]
#         },
#         {
#             "name": "Partially Supported",
#             "only_cited_edges": False,
#             "only_consistent_edges": False,
#             # local filter in ["OK","Fully supported","Partially supported"]
#             "verdict_filter": ["OK", "Fully supported", "Partially supported"]
#         }
#     ]

#     with pd.ExcelWriter(output_filename) as writer:
#         for scenario in scenarios:
#             scenario_name = scenario["name"]
#             try:
#                 # === (A) Call compare_session_graph_to_validation using the scenario's
#                 #     only_cited_edges / only_consistent_edges
#                 metrics = compare_session_graph_to_validation(
#                     discovery=discovery,
#                     session_ids=session_ids,
#                     validation_vars_json_path=validation_vars_json_path,
#                     validation_edges_json_path=validation_edges_json_path,
#                     only_cited_edges=scenario["only_cited_edges"],
#                     only_consistent_edges=scenario["only_consistent_edges"],
#                     plot_session_graph=False,  # or True if you want
#                     plot_validation_graph=False
#                 )
                
#                 # We'll get:
#                 # metrics["session_graph_edges"]   # a set of (src,tgt) from the DB
#                 # metrics["validation_graph_edges"]# a set from the validation JSON
#                 # plus metrics["session_nodes"], etc.

#                 session_edges = metrics["session_graph_edges"]
#                 validation_edges = metrics["validation_graph_edges"]
#                 session_nodes = metrics["session_nodes"]
#                 validation_nodes = metrics["validation_nodes"]

#                 # === (B) If scenario has a verdict_filter, do local subsetting
#                 # We also need "edge_data" from Neo4j to check judge_verdict in Python.

#                 # 1) fetch all edges & their properties from DB
#                 edge_data = {}
#                 # This query *does not* force r.judge_verdict='OK', etc. Because we want everything
#                 # (except for scenario #2 which is "cited edges only," so we do that).
#                 query = """
#                 MATCH (s:variable)-[r]->(t:variable)
#                 WHERE s.session_id = $session_id AND t.session_id = $session_id
#                 """
#                 # If scenario says only_cited_edges, we add that condition
#                 if scenario["only_cited_edges"]:
#                     query += " AND r.citation IS NOT NULL AND r.citation <> ''"

#                 # If scenario originally had "only_consistent_edges" we do that
#                 # but for e.g. "Fully Supported" scenario, we set only_consistent_edges=False,
#                 # so no condition here
#                 if scenario["only_consistent_edges"]:
#                     query += " AND r.judge_verdict = 'OK'"

#                 query += """
#                 RETURN s.name AS source,
#                        t.name AS target,
#                        r.judge_verdict AS judge_verdict,
#                        r.motivation AS motivation,
#                        r.citation AS citation,
#                        r.judge_message AS judge_message
#                 """
#                 params = {"session_id": session_ids or discovery.session_id}
                
#                 with discovery.graph_db._get_session() as s:
#                     results = s.run(query, params)
#                     for rec in results:
#                         src = rec["source"]
#                         tgt = rec["target"]
#                         edge_data[(src,tgt)] = {
#                             "judge_verdict": rec["judge_verdict"] or "N/A",
#                             "motivation": rec["motivation"] or "",
#                             "citation": rec["citation"] or "",
#                             "judge_message": rec["judge_message"] or ""
#                         }

#                 # 2) If scenario has a verdict_filter, refine session_edges
#                 if scenario["verdict_filter"] is not None:
#                     new_session_edges = set()
#                     for (src, tgt) in session_edges:
#                         if (src, tgt) in edge_data:
#                             v = edge_data[(src, tgt)]["judge_verdict"]
#                             if v in scenario["verdict_filter"]:
#                                 new_session_edges.add((src,tgt))
#                     # Replace session_edges with the filtered version
#                     session_edges = new_session_edges

#                 # === (C) Recompute confusion matrix with this final `session_edges`
#                 tp_edges = session_edges.intersection(validation_edges)
#                 fp_edges = session_edges - validation_edges
#                 fn_edges = validation_edges - session_edges

#                 # TN => among union of nodes
#                 union_nodes = session_nodes.union(validation_nodes)
#                 all_pairs = {
#                     (a,b) for a in union_nodes for b in union_nodes if a != b
#                 }
#                 combined_edges = session_edges.union(validation_edges)
#                 tn_edges = all_pairs - combined_edges

#                 # Build local metrics (tp,fp,fn,tn) => precision, recall, f1, accuracy
#                 tp = len(tp_edges)
#                 fp = len(fp_edges)
#                 fn = len(fn_edges)
#                 tn = len(tn_edges)
#                 # The original code from compare_session_graph_to_validation might have
#                 # these, but we want them re-computed after local filter
#                 precision = tp / (tp + fp) if (tp+fp)>0 else 0.0
#                 recall = tp / (tp + fn) if (tp+fn)>0 else 0.0
#                 f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
#                 accuracy = (tp + tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0.0

#                 # We keep "Variables Correct Score" etc. from the existing metrics if you want,
#                 # or we do fresh. Below we just reuse from the original but you can do similarly:
#                 variable_correct_score = metrics.get("ratio_vars_correct","N/A")
#                 correct_vars = metrics.get("correct vars","N/A")
#                 incorrect_vars = metrics.get("incorrect vars","N/A")
#                 total_comparisons = metrics.get("total_comparisons","N/A")

#                 # === (D) Build up the edges_data rows for Excel
#                 # We'll classify each edge as TP / FP / FN
#                 all_rows = []
#                 # 1) True Positives
#                 for e in tp_edges:
#                     row = _build_edge_row(e, "TP", session_edges, validation_edges, edge_data)
#                     all_rows.append(row)
#                 # 2) False Positives
#                 for e in fp_edges:
#                     row = _build_edge_row(e, "FP", session_edges, validation_edges, edge_data)
#                     all_rows.append(row)
#                 # 3) False Negatives
#                 for e in fn_edges:
#                     # If it's not in local edge_data, there's no DB entry
#                     row = {
#                         "Source": e[0],
#                         "Target": e[1],
#                         "Classification": "FN",
#                         "In Session Graph": "No",
#                         "In Validation Graph": "Yes",
#                         "Judge Verdict": "N/A",
#                         "Motivation": "Not discovered",
#                         "Citation": "",
#                         "Judge Message": "",
#                     }
#                     all_rows.append(row)
#                 # (We usually don't list the TN edges in Excel, to avoid huge bloat.)

#                 # Convert to DataFrame
#                 all_edges_df = pd.DataFrame(all_rows)

#                 # === (E) Build summary
#                 summary_data = {
#                     "Metric": [
#                         "True Positives (TP)",
#                         "False Positives (FP)",
#                         "False Negatives (FN)",
#                         "False Negatives (TN)",
#                         "Total Discovered Edges",
#                         "Total Validation Edges",
#                         "Variables Correct Score",
#                         "correct variables",
#                         "incorrect variables",
#                         "Total Comparisons",
#                         "Precision",
#                         "Recall",
#                         "F1 Score",
#                         "Accuracy"
#                     ],
#                     "Value": [
#                         tp, fp, fn, tn,
#                         len(session_edges),
#                         len(validation_edges),
#                         variable_correct_score,
#                         correct_vars,
#                         incorrect_vars,
#                         total_comparisons,
#                         precision,
#                         recall,
#                         f1,
#                         accuracy
#                     ]
#                 }
#                 summary_df = pd.DataFrame(summary_data)

#                 # === (F) Write to Excel
#                 detail_sheet = scenario_name
#                 summary_sheet = f"{scenario_name} Summary"

#                 all_edges_df.to_excel(writer, sheet_name=detail_sheet, index=False)
#                 summary_df.to_excel(writer, sheet_name=summary_sheet, index=False)

#             except Exception as exc:
#                 print(f"Error processing scenario '{scenario_name}': {exc}")
#                 traceback.print_exc()

#     print(f"\nExcel file created: {output_filename}")
#     print("Created scenario sheets:\n"
#           "  1) All Edges / All Edges Summary\n"
#           "  2) Cited Edges Only / Cited Edges Only Summary\n"
#           "  3) Consistent Edges Only / Consistent Edges Only Summary\n"
#           "  4) Fully Supported Only / Fully Supported Only Summary\n"
#           "  5) Partially Supported / Partially Supported Summary\n")
#     return output_filename


# def export_edges_comparison_to_excel(
#     discovery,
#     session_ids=None,
#     validation_vars_json_path=None,
#     validation_edges_json_path=None,
#     output_filename=None
# ):
#     """
#     Export edge evaluations to Excel, generating multiple scenario-based sheets:

#       1) All Edges + Summary
#       2) Cited Edges Only + Summary
#       3) Consistent Edges Only (i.e. judge_verdict='OK') + Summary
#       4) Fully Supported Only (judge_verdict in ['OK','Fully supported','CAUSAL'])
#       5) Partially Supported (judge_verdict in ['OK','Fully supported','Partially supported','CAUSAL'])

#     Returns:
#         Path to the created Excel file
#     """
#     import datetime
#     import pandas as pd
#     import traceback

#     # If you have the function in the same file, no need for import;
#     # otherwise import your compare_session_graph_to_validation function
#     # from .whatever_module import compare_session_graph_to_validation

#     # 1) Prepare the output filename
#     if not output_filename:
#         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         output_filename = f"edges_evaluation_{timestamp}.xlsx"
#     elif not output_filename.endswith('.xlsx'):
#         output_filename += ".xlsx"

#     # Define your 5 scenarios
#     scenarios = [
#         {
#             "name": "All Edges",
#             "only_cited_edges": False,
#             "only_consistent_edges": False,
#             "verdict_filter": None  # means no local filter
#         },
#         {
#             "name": "Cited Edges Only",
#             "only_cited_edges": True,
#             "only_consistent_edges": False,
#             "verdict_filter": None
#         },
#         {
#             "name": "Consistent Edges Only",  # the original "OK" scenario
#             "only_cited_edges": False,
#             "only_consistent_edges": True,  # means r.judge_verdict='OK' inside the DB query
#             "verdict_filter": None
#         },
#         {
#             "name": "Fully Supported Only",
#             "only_cited_edges": False,
#             # We'll retrieve all edges from DB, ignoring "only_consistent_edges"
#             "only_consistent_edges": False,
#             # Then we do a local filter: in ["OK","Fully supported","CAUSAL"]
#             "verdict_filter": ["OK", "Fully supported", "CAUSAL"]
#         },
#         {
#             "name": "Partially Supported",
#             "only_cited_edges": False,
#             "only_consistent_edges": False,
#             # local filter in ["OK","Fully supported","Partially supported","CAUSAL"]
#             "verdict_filter": ["OK", "Fully supported", "Partially supported", "CAUSAL"]
#         }
#     ]

#     with pd.ExcelWriter(output_filename) as writer:
#         for scenario in scenarios:
#             scenario_name = scenario["name"]
#             try:
#                 # === (A) Call compare_session_graph_to_validation using the scenario's
#                 #     only_cited_edges / only_consistent_edges
#                 metrics = compare_session_graph_to_validation(
#                     discovery=discovery,
#                     session_ids=session_ids,
#                     validation_vars_json_path=validation_vars_json_path,
#                     validation_edges_json_path=validation_edges_json_path,
#                     only_cited_edges=scenario["only_cited_edges"],
#                     only_consistent_edges=scenario["only_consistent_edges"],
#                     plot_session_graph=False,  # or True if you want
#                     plot_validation_graph=False
#                 )
                
#                 # We'll get:
#                 # metrics["session_graph_edges"]   # a set of (src,tgt) from the DB
#                 # metrics["validation_graph_edges"]# a set from the validation JSON
#                 # plus metrics["session_nodes"], etc.

#                 session_edges = metrics["session_graph_edges"]
#                 validation_edges = metrics["validation_graph_edges"]
#                 session_nodes = metrics["session_nodes"]
#                 validation_nodes = metrics["validation_nodes"]

#                 # === (B) If scenario has a verdict_filter, do local subsetting
#                 # We also need "edge_data" from Neo4j to check judge_verdict in Python.

#                 # 1) fetch all edges & their properties from DB
#                 edge_data = {}
#                 # This query *does not* force r.judge_verdict='OK', etc. Because we want everything
#                 # (except for scenario #2 which is "cited edges only," so we do that).
#                 query = """
#                 MATCH (s:variable)-[r]->(t:variable)
#                 WHERE s.session_id = $session_id AND t.session_id = $session_id
#                 """
#                 # If scenario says only_cited_edges, we add that condition
#                 if scenario["only_cited_edges"]:
#                     query += " AND r.citation IS NOT NULL AND r.citation <> ''"

#                 # If scenario originally had "only_consistent_edges" we do that
#                 if scenario["only_consistent_edges"]:
#                     query += " AND r.judge_verdict = 'OK'"

#                 query += """
#                 RETURN s.name AS source,
#                        t.name AS target,
#                        r.judge_verdict AS judge_verdict,
#                        r.motivation AS motivation,
#                        r.citation AS citation,
#                        r.judge_message AS judge_message
#                 """
#                 params = {"session_id": session_ids or discovery.session_id}
                
#                 with discovery.graph_db._get_session() as s:
#                     results = s.run(query, params)
#                     for rec in results:
#                         src = rec["source"]
#                         tgt = rec["target"]
#                         edge_data[(src,tgt)] = {
#                             "judge_verdict": rec["judge_verdict"] or "N/A",
#                             "motivation": rec["motivation"] or "",
#                             "citation": rec["citation"] or "",
#                             "judge_message": rec["judge_message"] or ""
#                         }

#                 # 2) If scenario has a verdict_filter, refine session_edges
#                 if scenario["verdict_filter"] is not None:
#                     new_session_edges = set()
#                     for (src, tgt) in session_edges:
#                         if (src, tgt) in edge_data:
#                             v = edge_data[(src, tgt)]["judge_verdict"]
#                             if v in scenario["verdict_filter"]:
#                                 new_session_edges.add((src,tgt))
#                     # Replace session_edges with the filtered version
#                     session_edges = new_session_edges

#                 # === (C) Recompute confusion matrix with this final `session_edges`
#                 tp_edges = session_edges.intersection(validation_edges)
#                 fp_edges = session_edges - validation_edges
#                 fn_edges = validation_edges - session_edges

#                 # TN => among union of nodes
#                 union_nodes = session_nodes.union(validation_nodes)
#                 all_pairs = {
#                     (a,b) for a in union_nodes for b in union_nodes if a != b
#                 }
#                 combined_edges = session_edges.union(validation_edges)
#                 tn_edges = all_pairs - combined_edges

#                 # Build local metrics (tp,fp,fn,tn) => precision, recall, f1, accuracy
#                 tp = len(tp_edges)
#                 fp = len(fp_edges)
#                 fn = len(fn_edges)
#                 tn = len(tn_edges)
#                 # The original code from compare_session_graph_to_validation might have
#                 # these, but we want them re-computed after local filter
#                 precision = tp / (tp + fp) if (tp+fp)>0 else 0.0
#                 recall = tp / (tp + fn) if (tp+fn)>0 else 0.0
#                 f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
#                 accuracy = (tp + tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0.0

#                 # We keep "Variables Correct Score" etc. from the existing metrics if you want
#                 variable_correct_score = metrics.get("ratio_vars_correct","N/A")
#                 correct_vars = metrics.get("correct vars","N/A")
#                 incorrect_vars = metrics.get("incorrect vars","N/A")
#                 total_comparisons = metrics.get("total_comparisons","N/A")

#                 # === (D) Build up the edges_data rows for Excel
#                 all_rows = []
#                 # 1) True Positives
#                 for e in tp_edges:
#                     row = _build_edge_row(e, "TP", session_edges, validation_edges, edge_data)
#                     all_rows.append(row)
#                 # 2) False Positives
#                 for e in fp_edges:
#                     row = _build_edge_row(e, "FP", session_edges, validation_edges, edge_data)
#                     all_rows.append(row)
#                 # 3) False Negatives
#                 for e in fn_edges:
#                     # If it's not in local edge_data, there's no DB entry
#                     row = {
#                         "Source": e[0],
#                         "Target": e[1],
#                         "Classification": "FN",
#                         "In Session Graph": "No",
#                         "In Validation Graph": "Yes",
#                         "Judge Verdict": "N/A",
#                         "Motivation": "Not discovered",
#                         "Citation": "",
#                         "Judge Message": "",
#                     }
#                     all_rows.append(row)

#                 # Convert to DataFrame
#                 all_edges_df = pd.DataFrame(all_rows)

#                 # === (E) Build summary
#                 summary_data = {
#                     "Metric": [
#                         "True Positives (TP)",
#                         "False Positives (FP)",
#                         "False Negatives (FN)",
#                         "False Negatives (TN)",
#                         "Total Discovered Edges",
#                         "Total Validation Edges",
#                         "Variables Correct Score",
#                         "correct variables",
#                         "incorrect variables",
#                         "Total Comparisons",
#                         "Precision",
#                         "Recall",
#                         "F1 Score",
#                         "Accuracy"
#                     ],
#                     "Value": [
#                         tp, fp, fn, tn,
#                         len(session_edges),
#                         len(validation_edges),
#                         variable_correct_score,
#                         correct_vars,
#                         incorrect_vars,
#                         total_comparisons,
#                         precision,
#                         recall,
#                         f1,
#                         accuracy
#                     ]
#                 }
#                 summary_df = pd.DataFrame(summary_data)

#                 # === (F) Write to Excel
#                 detail_sheet = scenario_name
#                 summary_sheet = f"{scenario_name} Summary"

#                 all_edges_df.to_excel(writer, sheet_name=detail_sheet, index=False)
#                 summary_df.to_excel(writer, sheet_name=summary_sheet, index=False)

#             except Exception as exc:
#                 print(f"Error processing scenario '{scenario_name}': {exc}")
#                 traceback.print_exc()

#     print(f"\nExcel file created: {output_filename}")
#     print("Created scenario sheets:\n"
#           "  1) All Edges / All Edges Summary\n"
#           "  2) Cited Edges Only / Cited Edges Only Summary\n"
#           "  3) Consistent Edges Only / Consistent Edges Only Summary\n"
#           "  4) Fully Supported Only / Fully Supported Only Summary\n"
#           "  5) Partially Supported / Partially Supported Summary\n")
#     return output_filename


def _build_edge_row(e, classification, session_edges, validation_edges, edge_data):
    """
    Helper to build a dictionary row for a single edge in the final Excel.
    classification is one of "TP","FP","FN".
    """
    (src, tgt) = e
    row = {
        "Source": src,
        "Target": tgt,
        "Classification": classification,
        "In Session Graph": "Yes" if e in session_edges else "No",
        "In Validation Graph": "Yes" if e in validation_edges else "No",
        "Judge Verdict": "N/A",
        "Motivation": "",
        "Citation": "",
        "Judge Message": ""
    }
    if e in edge_data:
        row["Judge Verdict"] = edge_data[e]["judge_verdict"]
        row["Motivation"]    = edge_data[e]["motivation"]
        row["Citation"]      = edge_data[e]["citation"]
        row["Judge Message"] = edge_data[e]["judge_message"]
    return row


def export_edges_comparison_to_excel(
    discovery,
    session_ids=None,
    validation_vars_json_path=None,
    validation_edges_json_path=None,
    output_filename=None,
    lm_stats: dict = None
):
    """
    Export edge evaluations to Excel, generating multiple scenario-based sheets:

      1) All Edges + Summary
      2) Cited Edges Only + Summary
      3) Consistent Edges Only (i.e. judge_verdict='OK') + Summary
      4) Fully Supported Only (judge_verdict in ['OK','Fully supported','CAUSAL'])
      5) Partially Supported (judge_verdict in ['OK','Fully supported','Partially supported','CAUSAL'])

    If lm_stats is provided, creates an extra "LLM Usage Stats" sheet 
    that shows generator/corruptor/judge usage data (status codes & token totals).

    Returns:
        Path to the created Excel file
    """
    import datetime
    import pandas as pd
    import traceback
    import openpyxl
    from openpyxl import Workbook

    # ------------------------------------------------------------------
    # Helper function to build a row (TP, FP edges) that exist in the DB
    # ------------------------------------------------------------------
    def _build_edge_row(edge, classification, session_edges, validation_edges, edge_data):
        """Build a dictionary representing a single edge's row in the output."""
        src, tgt = edge
        data = edge_data.get((src, tgt), {})
        return {
            "Source": src,
            "Target": tgt,
            "Classification": classification,
            "In Session Graph": "Yes" if classification in ("TP","FP") else "No",
            "In Validation Graph": "Yes" if edge in validation_edges else "No",
            "Judge Verdict": data.get("judge_verdict", "N/A"),
            "Motivation": data.get("motivation", ""),
            "Citation": data.get("citation", ""),
            "Judge Message": data.get("judge_message", ""),
            "Aggregate Score": data.get("aggregate_score", "N/A"),
        }

    # 1) Prepare the output filename
    if not output_filename:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"edges_evaluation_{timestamp}.xlsx"
    elif not output_filename.endswith('.xlsx'):
        output_filename += ".xlsx"

    # Define your 5 scenarios
    scenarios = [
        {
            "name": "All Edges",
            "only_cited_edges": False,
            "only_consistent_edges": False,
            "verdict_filter": None  # means no local filter
        },
        {
            "name": "Cited Edges Only",
            "only_cited_edges": True,
            "only_consistent_edges": False,
            "verdict_filter": None
        },
        {
            "name": "Consistent Edges Only",  # the original "OK" scenario
            "only_cited_edges": False,
            "only_consistent_edges": True,  # means r.judge_verdict='OK' inside the DB query
            "verdict_filter": None
        },
        {
            "name": "Fully Supported Only",
            "only_cited_edges": False,
            "only_consistent_edges": False,
            "verdict_filter": ["OK", "Fully supported", "CAUSAL"]
        },
        {
            "name": "Partially Supported",
            "only_cited_edges": False,
            "only_consistent_edges": False,
            "verdict_filter": ["OK", "Fully supported", "Partially supported", "CAUSAL"]
        }
    ]

    # Create and use a context manager for the writer
    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:

        # 2) For each scenario, build "detail" & "summary" data frames
        for scenario in scenarios:
            scenario_name = scenario["name"]
            try:
                # === (A) Call compare_session_graph_to_validation 
                metrics = compare_session_graph_to_validation(
                    discovery=discovery,
                    session_ids=session_ids,
                    validation_vars_json_path=validation_vars_json_path,
                    validation_edges_json_path=validation_edges_json_path,
                    only_cited_edges=scenario["only_cited_edges"],
                    only_consistent_edges=scenario["only_consistent_edges"],
                    plot_session_graph=False,  # or True if you want
                    plot_validation_graph=False
                )
                
                # We'll get:
                # metrics["session_graph_edges"]   # a set of (src,tgt) from the DB
                # metrics["validation_graph_edges"]# a set from the validation JSON
                # plus metrics["session_nodes"], etc.
                session_edges = metrics["session_graph_edges"]
                validation_edges = metrics["validation_graph_edges"]
                session_nodes = metrics["session_nodes"]
                validation_nodes = metrics["validation_nodes"]

                # === (B) If scenario has a verdict_filter, do local subsetting
                # We also need "edge_data" from Neo4j to check judge_verdict in Python.
                edge_data = {}
                query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $session_id AND t.session_id = $session_id
                """
                # If scenario says only_cited_edges, add that condition
                if scenario["only_cited_edges"]:
                    query += " AND r.citation IS NOT NULL AND r.citation <> ''"

                # If scenario originally had "only_consistent_edges"
                if scenario["only_consistent_edges"]:
                    query += " AND r.judge_verdict = 'OK'"

                # IMPORTANT: now also fetch r.aggregate_score
                query += """
                RETURN s.name AS source,
                       t.name AS target,
                       r.judge_verdict AS judge_verdict,
                       r.motivation AS motivation,
                       r.citation AS citation,
                       r.judge_message AS judge_message,
                       r.aggregate_score AS aggregate_score
                """

                params = {"session_id": session_ids or discovery.session_id}
                
                with discovery.graph_db._get_session() as s:
                    results = s.run(query, params)
                    for rec in results:
                        src = rec["source"]
                        tgt = rec["target"]
                        edge_data[(src,tgt)] = {
                            "judge_verdict": rec["judge_verdict"] or "N/A",
                            "motivation": rec["motivation"] or "",
                            "citation": rec["citation"] or "",
                            "judge_message": rec["judge_message"] or "",
                            "aggregate_score": rec["aggregate_score"] if rec["aggregate_score"] is not None else "N/A"
                        }

                # If scenario has a verdict_filter, refine session_edges
                if scenario["verdict_filter"] is not None:
                    new_session_edges = set()
                    for (src, tgt) in session_edges:
                        if (src, tgt) in edge_data:
                            v = edge_data[(src, tgt)]["judge_verdict"]
                            if v in scenario["verdict_filter"]:
                                new_session_edges.add((src,tgt))
                    # Replace session_edges with the filtered version
                    session_edges = new_session_edges

                # === (C) Recompute confusion matrix with this final session_edges
                tp_edges = session_edges.intersection(validation_edges)
                fp_edges = session_edges - validation_edges
                fn_edges = validation_edges - session_edges

                # TN => among union of nodes
                union_nodes = session_nodes.union(validation_nodes)
                all_pairs = {
                    (a,b) for a in union_nodes for b in union_nodes if a != b
                }
                combined_edges = session_edges.union(validation_edges)
                tn_edges = all_pairs - combined_edges

                # Build local metrics (tp,fp,fn,tn) => precision, recall, f1, accuracy
                tp = len(tp_edges)
                fp = len(fp_edges)
                fn = len(fn_edges)
                tn = len(tn_edges)

                precision = tp / (tp + fp) if (tp+fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp+fn) > 0 else 0.0
                f1 = (2 * precision * recall)/(precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn)/(tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

                variable_correct_score = metrics.get("ratio_vars_correct","N/A")
                correct_vars = metrics.get("correct vars","N/A")
                incorrect_vars = metrics.get("incorrect vars","N/A")
                total_comparisons = metrics.get("total_comparisons","N/A")

                # === (D) Build up the edges_data rows for Excel
                all_rows = []
                # 1) True Positives
                for e in tp_edges:
                    row = _build_edge_row(e, "TP", session_edges, validation_edges, edge_data)
                    all_rows.append(row)
                # 2) False Positives
                for e in fp_edges:
                    row = _build_edge_row(e, "FP", session_edges, validation_edges, edge_data)
                    all_rows.append(row)
                # 3) False Negatives
                for e in fn_edges:
                    row = {
                        "Source": e[0],
                        "Target": e[1],
                        "Classification": "FN",
                        "In Session Graph": "No",
                        "In Validation Graph": "Yes",
                        "Judge Verdict": "N/A",
                        "Motivation": "Not discovered",
                        "Citation": "",
                        "Judge Message": "",
                        "Aggregate Score": "N/A",
                    }
                    all_rows.append(row)

                # Convert to DataFrame
                all_edges_df = pd.DataFrame(all_rows)

                # === (E) Build summary
                summary_data = {
                    "Metric": [
                        "True Positives (TP)",
                        "False Positives (FP)",
                        "False Negatives (FN)",
                        "False Negatives (TN)",
                        "Total Discovered Edges",
                        "Total Validation Edges",
                        "Variables Correct Score",
                        "correct variables",
                        "incorrect variables",
                        "Total Comparisons",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "Accuracy"
                    ],
                    "Value": [
                        tp, fp, fn, tn,
                        len(session_edges),
                        len(validation_edges),
                        variable_correct_score,
                        correct_vars,
                        incorrect_vars,
                        total_comparisons,
                        precision,
                        recall,
                        f1,
                        accuracy
                    ]
                }
                summary_df = pd.DataFrame(summary_data)

                # === (F) Write these two DataFrames (detail & summary) to new sheets
                detail_sheet = scenario_name
                summary_sheet = f"{scenario_name} Summary"

                all_edges_df.to_excel(writer, sheet_name=detail_sheet, index=False)
                summary_df.to_excel(writer, sheet_name=summary_sheet, index=False)

            except Exception as exc:
                print(f"Error processing scenario '{scenario_name}': {exc}")
                traceback.print_exc()

        # 3) If we have usage stats, create an extra "LLM Usage Stats" sheet
        if lm_stats:
            usage_rows = []
            # We expect keys like "generator", "corruptor", "judge"
            for role_name in ["generator", "corruptor", "judge"]:
                role_data = lm_stats.get(role_name, {})
                row = {
                    "Role": role_name.capitalize(),
                    "Status Counts": str(role_data.get("status_counts", {})),
                    "Token Totals": str(role_data.get("token_totals", {})),
                    "Inference Time (s)": role_data.get("inference_time_total", 0.0),
                    "Inference Calls": role_data.get("inference_call_count", 0),
                }
                usage_rows.append(row)

            usage_df = pd.DataFrame(usage_rows)
            usage_df.to_excel(writer, sheet_name="LLM Usage Stats", index=False)

    # When we exit the 'with' block, the file is fully written and closed
    print(f"\nExcel file created: {output_filename}")
    print("Created scenario sheets:\n"
          "  1) All Edges / All Edges Summary\n"
          "  2) Cited Edges Only / Cited Edges Only Summary\n"
          "  3) Consistent Edges Only / Consistent Edges Only Summary\n"
          "  4) Fully Supported Only / Fully Supported Only Summary\n"
          "  5) Partially Supported / Partially Supported Summary\n"
          "  6) (Optional) LLM Usage Stats\n")

    return output_filename






def check_judge_verdicts(discovery, session_id):
    query = """
    MATCH (s:variable)-[r]->(t:variable)
    WHERE s.session_id = $session_id AND t.session_id = $session_id
    RETURN DISTINCT r.judge_verdict as verdict, count(*) as count
    """
    params = {"session_id": session_id}
    
    with discovery.graph_db._get_session() as session:
        results = list(session.run(query, params))
    
    print("Judge verdict values in database:")
    for record in results:
        print(f"  {record['verdict']} - {record['count']} edges")


# Main eval pipeline

# def run_discovery_experiment(
#     retrieved_session_id: str = 'False',
#     excel_path: str = "ground_truth_CLDs/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
#     export_json: bool = True,
#     plot_graph: bool = True,
#     # variables_json_path: str = "ground_truth_CLDs/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_vars_data.json",
#     # edges_json_path: str = "ground_truth_CLDs/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_edges_data.json",
#     target_variable_key: str = "Target",
#     temporal_scale_key: str = "Temporal Scale",
#     spatial_scale_key: str = "Spatial Scale",
#     context: str = "",
#     yaml_path: str = "../backend/configs/prompts.yaml",
#     dev_mode: bool = False,
#     generator_model_env: str = "PERPLEXITY_MODEL",
#     corruptor_model_env: str = "PERPLEXITY_MODEL",
#     judge_model_env: str = "CLAUDE_MODEL",
#     corruption_rate: float = 0.0,
#     judge_edges: bool = False,
#     judge_models: list = ['claude-3-7-sonnet-20250219'],
#     num_judges: int = 3,
#     generator_temperature=0.7,
#     generator_top_p=1.0,
#     corruptor_temperature=0.7,
#     corruptor_top_p=1.0,
#     judge_temperature=0.7,
#     judge_top_p=1.0,
#     parallel: bool = True,
#     max_workers: int = 3,
#     load_env: bool = True,
#     plot_session_graph: bool = True,
#     plot_validation_graph: bool = True,
#     only_cited_edges: bool = False,
#     only_consistent_edges: bool = False,
#     filter_validation_nodes_to_session_nodes: bool = True,
#     experiment_description: str = "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
#     output_json_prefix: str = "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
#     compare_variable_overlap: bool = False,
#     result_excel_path: str = "result_excel_path",
# ):
#     """
#     Runs a discovery experiment for causal relationships among variables.

#     Args:
#         excel_path (str): Path to the Excel file for CLD data.
#         export_json (bool): Whether to export CLD data to JSON.
#         plot_graph (bool): Whether to plot the CLD graph upon loading.
#         variables_json_path (str): Path to the JSON file of unique variables.
#         edges_json_path (str): Path to the JSON file of known/validation edges.
#         target_variable_key (str): The key in CLD context dict that identifies the target variable.
#         temporal_scale_key (str): The key in CLD context dict that identifies the temporal scale.
#         spatial_scale_key (str): The key in CLD context dict that identifies the spatial scale.
#         context (str): Additional context passed to CausalDiscovery.
#         yaml_path (str): Path to the prompts/configuration YAML for CausalDiscovery.
#         dev_mode (bool): Whether to enable dev_mode in CausalDiscovery.
#         generator_model_env (str): Environment variable name for the generator model.
#         corruptor_model_env (str): Environment variable name for the corruptor model.
#         judge_model_env (str): Environment variable name for the judge model.
#         corruption_rate (float): Rate of spurious relationship corruption (0.0 to 1.0).
#         judge_edges (bool): Whether to judge edges.
#         judge_models (list): List of model names for multi-judge logic.
#         num_judges (int): Number of judges to use in multi-judge logic.
#         parallel (bool): Whether to discover relationships in parallel.
#         max_workers (int): Number of workers for parallel relationship discovery.
#         load_env (bool): Whether to load environment variables via dotenv.
#         plot_session_graph (bool): Whether to plot the generated session graph in validation step.
#         plot_validation_graph (bool): Whether to plot the validation/ground-truth graph.
#         only_cited_edges (bool): If True, only include edges that have citations in the final comparison.
#         only_consistent_edges (bool): If True, only consider consistent edges in the final comparison.
#         filter_validation_nodes_to_session_nodes (bool): Only compare validation nodes that appear in session.
#         experiment_description (str): Description to include in the final experiment info.
#         output_json_prefix (str): Prefix for the final output JSON filename.

#     Returns:
#         dict: A dictionary with experiment metadata, including variables, session ID, stats, etc.
#     """
#     # 1) Optionally load environment variables
#     if load_env:
#         load_dotenv(".env.dev", override=True)

#     # Initialize these so we can refer to them later without errors
#     relationship_count = 0
#     corrupted_count = 0
    
#     # 2) Convert Excel -> CLD -> JSON in a temporary directory
#     with tempfile.TemporaryDirectory() as tmpdir:
#         # We'll force export_json=True here so we get the JSON for edges/variables
#         cld_data = load_cld_from_excel(
#             excel_path,
#             plot_graph=plot_graph,
#             export_json=True,
#             output_dir=tmpdir
#         )

#         # Extract the ephemeral JSON files we need
#         variables_json_path = cld_data["json_files"].get("variables", None)
#         edges_json_path = cld_data["json_files"].get("edges", None)

#         # 3) Load variables from the ephemeral JSON
#         with open(variables_json_path, "r") as f:
#             data = json.load(f)
#         variables = data
#         if not variables:
#             raise ValueError(
#                 f"No data found in {variables_json_path}. "
#                 "Please ensure your Excel file has the correct structure."
#             )

#         # 4) Extract relevant context from the CLD data
#         target_variable = cld_data['context'].get(target_variable_key, None)
#         if isinstance(target_variable, dict) and "name" in target_variable:
#             target_variable = target_variable["name"]
#         if not target_variable:
#             raise ValueError(
#                 f"Could not find target variable under key '{target_variable_key}' in CLD data."
#             )

#         temporal_scale = cld_data['context'].get(temporal_scale_key, None)
#         spatial_scale = cld_data['context'].get(spatial_scale_key, None)

#         # 5) Setup judge_models if not provided
#         if judge_models is None:
#             judge_models = []

#         print(f"generator_model_env run_discovery_experiment: {generator_model_env}")
#         # 6) Initialize CausalDiscovery
#         discovery = CausalDiscovery(
#             target_variable=target_variable,
#             temporal_scale=temporal_scale,
#             spatial_scale=spatial_scale,
#             context=context,
#             yaml_path=yaml_path,
#             dev_mode=dev_mode,
#             generator_model=generator_model_env,
#             corruptor_model=corruptor_model_env,
#             judge_model=judge_model_env,
#             generator_temperature=generator_temperature,
#             generator_top_p=generator_top_p,
#             corruptor_temperature=corruptor_temperature,
#             corruptor_top_p=corruptor_top_p,
#             judge_temperature=judge_temperature,
#             judge_top_p=judge_top_p

#         )

#         # 7) Create or retrieve session ID
#         if retrieved_session_id == 'False':
#             session_id = discovery.session_id
#             print(f"Using session ID: {session_id}")


#             # 8) Create variables (nodes) in Neo4j under this session
#             discovery.variables = []  # Reset any existing variables list

#             # Single transaction for all variable creation
#             with discovery.graph_db._get_session() as session:
#                 create_query = """
#                 CREATE (n:variable {
#                     name: $name,
#                     description: $description,
#                     session_id: $session_id,
#                     target: $is_target,
#                     deleted: false
#                 })
#                 """
#                 # Add the target variable node
#                 session.run(create_query, {
#                     "name": target_variable,
#                     "description": f"Target variable: {target_variable}",
#                     "session_id": session_id,
#                     "is_target": True
#                 })
#                 discovery.variables.append(target_variable)

#                 # Add all other variables
#                 for var in variables:
#                     var_name = var["name"]
#                     var_definition = var["definition"]
#                     if var_name == target_variable:
#                         # Skip if it is the same as target
#                         continue

#                     session.run(create_query, {
#                         "name": var_name,
#                         "description": f"definition: {var_definition}",
#                         "session_id": session_id,
#                         "is_target": False
#                     })
#                     discovery.variables.append(var_name)

#                 print(f"Created {len(discovery.variables)} variables with session ID: {session_id}")

#                 # 9) Discover relationships
#                 relationships = discovery.discover_relationships(
#                     parallel=parallel,
#                     max_workers=max_workers,
#                     corruption_rate=corruption_rate
#                 )

#                 relationship_count = len(relationships)
#                 corrupted_count = discovery.corrupted_count
#                 print(f"Discovered {relationship_count} relationships; {corrupted_count} were corrupted/spurious.")
#         else:
#             session_id = retrieved_session_id
#             print(f"Using Retrieved session ID: {session_id}")

#             discovery.set_session_id(session_id)

#         if judge_edges:

#         # 10) Multi-judge logic with citations if possible
#             try:
#                 print("Running judgment with citations on relationships...")
#                 judged_edges = discovery.judge_all_edges_with_citations_serial(
#                     judge_models=judge_models,
#                     num_judges=num_judges
#                 )
#             except:
#                 print("Citation-based judgment failed; running basic judgment on relationships...")
#                 judged_edges = discovery.judge_all_edges_serial(
#                     judge_models=judge_models,
#                     num_judges=num_judges,
#                     judge_temperature=judge_temperature
#                 )

#             # Summarize partial results
#             result = {
#                 "session_id": session_id,
#                 "stats": {
#                     "variable_count": len(discovery.variables),
#                     "relationship_count": relationship_count,
#                     "corrupted_count": corrupted_count,
#                     "judged_count": len(judged_edges)
#                 }
#             }
#         else:
#             result = {
#                 "session_id": session_id,
#                 "stats": {
#                     "variable_count": len(discovery.variables),
#                     "relationship_count": relationship_count,
#                     "corrupted_count": corrupted_count,
#                     "judged_count": 0
#                 }
#             }


#             # 11) Inspect relationships
#             discovery.inspect_relationships()

#             print(
#                 f"\nGenerated {result['stats']['variable_count']} variables "
#                 f"and {result['stats']['relationship_count']} relationships"
#             )
#             print(f"Created {result['stats']['corrupted_count']} spurious relationships")
#             print(f"Judged {result['stats']['judged_count']} edges")

            
#             # Add this section to define the result variable when using a retrieved session
#             # Fetch stats for the existing session
#             with discovery.graph_db._get_session() as session:
#                 # Count variables
#                 var_query = """
#                 MATCH (n:variable)
#                 WHERE n.session_id = $session_id AND n.deleted = false
#                 RETURN count(n) as variable_count
#                 """
#                 var_count = session.run(var_query, {"session_id": session_id}).single()["variable_count"]
                
#                 # Count relationships
#                 rel_query = """
#                 MATCH (s:variable)-[r]->(t:variable)
#                 WHERE s.session_id = $session_id AND t.session_id = $session_id
#                 AND s.deleted = false AND t.deleted = false
#                 RETURN count(r) as relationship_count
#                 """
#                 rel_count = session.run(rel_query, {"session_id": session_id}).single()["relationship_count"]
                
#                 # Get judged count if available
#                 judged_query = """
#                 MATCH (s:variable)-[r]->(t:variable)
#                 WHERE s.session_id = $session_id AND t.session_id = $session_id
#                 AND s.deleted = false AND t.deleted = false
#                 AND r.judge_verdict IS NOT NULL
#                 RETURN count(r) as judged_count
#                 """
#                 judged_count = session.run(judged_query, {"session_id": session_id}).single()["judged_count"]
            
#             # Set discovery.variables to allow other functions to work
#             discovery.variables = []
#             node_query = """
#             MATCH (n:variable)
#             WHERE n.session_id = $session_id AND n.deleted = false
#             RETURN n.name as name
#             """
#             with discovery.graph_db._get_session() as session:
#                 for record in session.run(node_query, {"session_id": session_id}):
#                     discovery.variables.append(record["name"])
            
#             # Create result dictionary
#             result = {
#                 "session_id": session_id,
#                 "stats": {
#                     "variable_count": var_count,
#                     "relationship_count": rel_count,
#                     "corrupted_count": 0,  # Can't determine for retrieved session
#                     "judged_count": judged_count
#                 }
#             }

#         # 12) Provide helpful Cypher query
#         cypher_query = f"""
#         MATCH (n)-[r]->(m)
#         WHERE n.session_id = '{session_id}' AND m.session_id = '{session_id}'
#         RETURN n, r, m
#         """
#         print("\nRun this Cypher query in Neo4j Browser:")
#         print(cypher_query)

#         # 13) Save session info to JSON
#         timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         output_filename = f"{output_json_prefix}_{timestamp_str}.json"

#         experiment_info = {
#             "timestamp": timestamp_str,
#             "description": experiment_description,
#             "variables": discovery.variables,
#             "target_variable": target_variable,
#             "session_id": session_id,
#             "stats": result["stats"],
#             "temporal_scale": temporal_scale,
#             "spatial_scale": spatial_scale
#         }

#         with open(output_filename, "w") as f:
#             json.dump(experiment_info, f, indent=2)

#         print(f"\nSaved session information to: {output_filename}")


#         print(f'variables_json_path: {variables_json_path}')
#         print(f'edges_json_path: {edges_json_path}')

#         # 14) Compare session graph to validation data
#         compare_session_graph_to_validation(
#             discovery,
#             session_ids=session_id,
#             validation_vars_json_path=variables_json_path,
#             validation_edges_json_path=edges_json_path,
#             plot_session_graph=plot_session_graph,
#             plot_validation_graph=plot_validation_graph,
#             only_cited_edges=only_cited_edges,
#             only_consistent_edges=only_consistent_edges,
#             filter_validation_nodes_to_session_nodes=filter_validation_nodes_to_session_nodes,
#             compare_variable_overlap=compare_variable_overlap
#         )

#         # Export edge data to excel
#         export_edges_comparison_to_excel(
#             discovery=discovery,
#             session_ids=session_id,  #
#             validation_vars_json_path=variables_json_path,
#             validation_edges_json_path=edges_json_path,
#             output_filename=result_excel_path
#         )

#             # Return the experiment info for further use
#     return experiment_info






def run_discovery_experiment(
    retrieved_session_id: str = 'False',
    excel_path: str = "ground_truth_CLDs/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
    export_json: bool = True,
    plot_graph: bool = True,
    # variables_json_path: str = "ground_truth_CLDs/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_vars_data.json",
    # edges_json_path: str = "ground_truth_CLDs/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_edges_data.json",
    target_variable_key: str = "Target",
    temporal_scale_key: str = "Temporal Scale",
    spatial_scale_key: str = "Spatial Scale",
    context: str = "",
    yaml_path: str = "../backend/configs/prompts.yaml",
    dev_mode: bool = False,
    generator_model_env: str = "PERPLEXITY_MODEL",
    corruptor_model_env: str = "PERPLEXITY_MODEL",
    judge_model_env: str = "CLAUDE_MODEL",
    corruption_rate: float = 0.0,
    judge_edges: bool = False,
    judge_models: list = ['claude-3-7-sonnet-20250219'],
    num_judges: int = 3,
    generator_temperature=0.7,
    generator_top_p=1.0,
    corruptor_temperature=0.7,
    corruptor_top_p=1.0,
    judge_temperature=0.7,
    judge_top_p=1.0,
    parallel: bool = True,
    max_workers: int = 3,
    load_env: bool = True,
    plot_session_graph: bool = True,
    plot_validation_graph: bool = True,
    only_cited_edges: bool = False,
    only_consistent_edges: bool = False,
    filter_validation_nodes_to_session_nodes: bool = True,
    experiment_description: str = "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
    output_json_prefix: str = "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
    compare_variable_overlap: bool = False,
    result_excel_path: str = "result_excel_path",
):
    """
    Runs a discovery experiment for causal relationships among variables.

    Args:
        excel_path (str): Path to the Excel file for CLD data.
        export_json (bool): Whether to export CLD data to JSON.
        plot_graph (bool): Whether to plot the CLD graph upon loading.
        variables_json_path (str): Path to the JSON file of unique variables.
        edges_json_path (str): Path to the JSON file of known/validation edges.
        target_variable_key (str): The key in CLD context dict that identifies the target variable.
        temporal_scale_key (str): The key in CLD context dict that identifies the temporal scale.
        spatial_scale_key (str): The key in CLD context dict that identifies the spatial scale.
        context (str): Additional context passed to CausalDiscovery.
        yaml_path (str): Path to the prompts/configuration YAML for CausalDiscovery.
        dev_mode (bool): Whether to enable dev_mode in CausalDiscovery.
        generator_model_env (str): Environment variable name for the generator model.
        corruptor_model_env (str): Environment variable name for the corruptor model.
        judge_model_env (str): Environment variable name for the judge model.
        corruption_rate (float): Rate of spurious relationship corruption (0.0 to 1.0).
        judge_edges (bool): Whether to judge edges.
        judge_models (list): List of model names for multi-judge logic.
        num_judges (int): Number of judges to use in multi-judge logic.
        parallel (bool): Whether to discover relationships in parallel.
        max_workers (int): Number of workers for parallel relationship discovery.
        load_env (bool): Whether to load environment variables via dotenv.
        plot_session_graph (bool): Whether to plot the generated session graph in validation step.
        plot_validation_graph (bool): Whether to plot the validation/ground-truth graph.
        only_cited_edges (bool): If True, only include edges that have citations in the final comparison.
        only_consistent_edges (bool): If True, only consider consistent edges in the final comparison.
        filter_validation_nodes_to_session_nodes (bool): Only compare validation nodes that appear in session.
        experiment_description (str): Description to include in the final experiment info.
        output_json_prefix (str): Prefix for the final output JSON filename.

    Returns:
        dict: A dictionary with experiment metadata, including variables, session ID, stats, etc.
    """
    if load_env:
        load_dotenv(".env.dev", override=True)

    relationship_count = 0
    corrupted_count = 0

    # 1) Convert Excel -> JSON
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cld_data = load_cld_from_excel(
            excel_path,
            plot_graph=plot_graph,
            export_json=True,
            output_dir=tmpdir
        )
        variables_json_path = cld_data["json_files"].get("variables", None)
        edges_json_path = cld_data["json_files"].get("edges", None)

        # 2) Load variables from ephemeral JSON
        with open(variables_json_path, "r") as f:
            data = json.load(f)
        variables = data  # list of { "name":..., "definition":... }

        # 3) Extract context (target, temporal, spatial)
        target_variable = cld_data['context'].get(target_variable_key, None)
        if isinstance(target_variable, dict) and "name" in target_variable:
            target_variable = target_variable["name"]
        temporal_scale = cld_data['context'].get(temporal_scale_key, None)
        spatial_scale = cld_data['context'].get(spatial_scale_key, None)

        # 4) Build the discovery
        generator_model = generator_model_env
        corruptor_model = corruptor_model_env
        judge_model = judge_model_env

        discovery = CausalDiscovery(
            target_variable=target_variable,
            temporal_scale=temporal_scale,
            spatial_scale=spatial_scale,
            yaml_path=yaml_path,
            dev_mode=dev_mode,
            generator_model=generator_model,
            corruptor_model=corruptor_model,
            judge_model=judge_model,
            context=context,
            generator_temperature=generator_temperature,
            generator_top_p=generator_top_p,
            corruptor_temperature=corruptor_temperature,
            corruptor_top_p=corruptor_top_p,
            judge_temperature=judge_temperature,
            judge_top_p=judge_top_p
        )

        if retrieved_session_id == 'False':
            session_id = discovery.session_id
            # Create nodes for each variable
            with discovery.graph_db._get_session() as session:
                create_query = """
                CREATE (n:variable {
                    name: $name,
                    description: $description,
                    session_id: $session_id,
                    target: $is_target,
                    deleted: false
                })
                """
                # target node
                session.run(create_query, {
                    "name": target_variable,
                    "description": f"Target variable: {target_variable}",
                    "session_id": session_id,
                    "is_target": True
                })
                discovery.variables.append(target_variable)

                # non-target
                for var in variables:
                    var_name = var["name"]
                    var_def = var["definition"]
                    if var_name == target_variable:
                        continue
                    session.run(create_query, {
                        "name": var_name,
                        "description": f"definition: {var_def}",
                        "session_id": session_id,
                        "is_target": False
                    })
                    discovery.variables.append(var_name)

            # 5) Discover
            relationships = discovery.discover_relationships(
                parallel=parallel,
                max_workers=max_workers,
                corruption_rate=corruption_rate
            )
            relationship_count = len(relationships)
            corrupted_count = discovery.corrupted_count
        else:
            # We skip the creation, re‑use session
            session_id = retrieved_session_id
            discovery.set_session_id(session_id)
            # we might want to fetch counts
            with discovery.graph_db._get_session() as session:
                vq = """
                MATCH (n:variable)
                WHERE n.session_id = $sid AND n.deleted=false
                RETURN count(n) as c
                """
                var_count = session.run(vq, {"sid": session_id}).single()["c"]
                rq = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $sid AND t.session_id=$sid
                AND s.deleted=false AND t.deleted=false
                RETURN count(r) as c
                """
                rel_count = session.run(rq, {"sid": session_id}).single()["c"]
            relationship_count = rel_count
            # Rebuild discovery.variables
            discovery.variables = []
            with discovery.graph_db._get_session() as session:
                nodes = session.run(
                    """
                    MATCH (n:variable)
                    WHERE n.session_id=$sid AND n.deleted=false
                    RETURN n.name as nm
                    """,
                    {"sid": session_id}
                )
                for rec in nodes:
                    discovery.variables.append(rec["nm"])

        # 6) Possibly judge
        judged_count = 0
        if judge_edges:
            try:
                # Attempt citation-based multi-judge
                judged_edges = discovery.judge_all_edges_with_citations_serial(
                    judge_models=judge_models,
                    num_judges=num_judges
                )
            except:
                # fallback
                judged_edges = discovery.judge_all_edges_serial(
                    judge_models=judge_models,
                    num_judges=num_judges
                )
            judged_count = len(judged_edges)

        # -------------- GET LLM STATS for each role --------------
        def safe_get_stats(llm_client):
                    if hasattr(llm_client, 'get_stats'):
                        return llm_client.get_stats()
                    # Return empty structure
                    return {
                        "status_counts": {},
                        "token_totals": {},
                        "inference_time_total": 0.0,
                        "inference_call_count": 0
                    }

        gen_stats = safe_get_stats(discovery.nucleus)          # <-- ADDED LLM STATS HERE
        corr_stats = safe_get_stats(discovery.corruptor_llm)   # <-- ADDED LLM STATS HERE
        judge_stats = safe_get_stats(discovery.judge_llm)      # <-- ADDED LLM STATS HERE

        print("\nLLM Usage Stats [Generator]:", gen_stats)      # <-- ADDED LLM STATS HERE
        print("LLM Usage Stats [Corruptor]:", corr_stats)
        print("LLM Usage Stats [Judge]:", judge_stats)
        # ---------------------------------------------------------

        # 7) Summarize
        result = {
            "session_id": session_id,
            "stats": {
                "variable_count": len(discovery.variables),
                "relationship_count": relationship_count,
                "corrupted_count": corrupted_count,
                "judged_count": judged_count
            },
            # Include the LLM usage stats in the final result:
            "lm_stats": {                                   # <-- ADDED LLM STATS HERE
                "generator": gen_stats,
                "corruptor": corr_stats,
                "judge": judge_stats
            }
        }

        # Inspect
        discovery.inspect_relationships()
        print(f"\nGenerated {result['stats']['variable_count']} variables; {result['stats']['relationship_count']} relationships.")
        print(f"Corrupted: {result['stats']['corrupted_count']}; Judged: {result['stats']['judged_count']}")

        # Provide a helpful Cypher
        print(f"\nSample Cypher:\nMATCH (n)-[r]->(m) WHERE n.session_id='{session_id}' RETURN n,r,m")

        # 8) Save session info
        ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_fname = f"{output_json_prefix}_{ts_str}.json"
        experiment_info = {
            "timestamp": ts_str,
            "description": experiment_description,
            "variables": discovery.variables,
            "target_variable": target_variable,
            "session_id": session_id,
            "stats": result["stats"],
            "temporal_scale": temporal_scale,
            "spatial_scale": spatial_scale,
            "lm_stats": result["lm_stats"]  # keep the LLM stats
        }
        with open(out_fname, "w") as f:
            json.dump(experiment_info, f, indent=2)
        print(f"Saved session info to {out_fname}")

        # 9) Compare with validation
        print(f'variables_json_path: {variables_json_path}')
        print(f'edges_json_path: {edges_json_path}')
        compare_session_graph_to_validation(
            discovery,
            session_ids=session_id,
            validation_vars_json_path=variables_json_path,
            validation_edges_json_path=edges_json_path,
            plot_session_graph=plot_session_graph,
            plot_validation_graph=plot_validation_graph,
            only_cited_edges=only_cited_edges,
            only_consistent_edges=only_consistent_edges,
            filter_validation_nodes_to_session_nodes=filter_validation_nodes_to_session_nodes,
            compare_variable_overlap=compare_variable_overlap
        )

        # 10) Export edges to Excel with scenario sheets
        export_edges_comparison_to_excel(
            discovery=discovery,
            session_ids=session_id,
            validation_vars_json_path=variables_json_path,
            validation_edges_json_path=edges_json_path,
            output_filename=result_excel_path,
            lm_stats=experiment_info["lm_stats"])

        return experiment_info


# ============= The multi-run version: run_temperature_experiments =============
def run_temperature_experiments(
    gen_temps=[0.2, 0.4, 0.8],
    corr_temps=None,
    judge_temp=0.4,
    connected_subsets=None,
    validation_vars_json_path="Cillian_CLD_unique_vars_data.json",
    validation_edges_json_path="Cillian_CLD_edges_data.json",
    llm_params=None,
    citations_only=False,
    judge_models=None,
    num_judges=3,
    visualize_comparison=True,
    runs_per_subgraph=1,
    num_subgraphs=None
):
    """
    Multi-run version: for each subgraph & temperature combo, we do discovery + optional judging,
    then measure vs. validation. Now also capturing & storing LLM usage stats for each run.
    """
    from . import compare_session_graph_to_validation
    load_dotenv("../.env.dev", override=True)
    results = {}

    if corr_temps is None:
        corr_temps = gen_temps
    if num_subgraphs is None:
        num_subgraphs = len(gen_temps)

    # If user didn’t provide connected_subsets, we find them:
    if connected_subsets is None:
        from . import find_connected_variable_subsets
        connected_subsets = find_connected_variable_subsets(
            validation_vars_json_path, validation_edges_json_path,
            subset_size=5, min_connectivity_pct=15, return_count=num_subgraphs
        )
    if llm_params is None:
        llm_params = {
            "generator_model": os.environ.get("PERPLEXITY_MODEL", "sonar-pro"),
        }
    if judge_models is None:
        judge_models = [os.environ.get("PERPLEXITY_MODEL", "sonar-pro")] * num_judges

    # For each subgraph
    for subgraph_idx in range(min(num_subgraphs, len(connected_subsets))):
        variable_subset, connectivity = connected_subsets[subgraph_idx]
        print(f"\n{'#'*80}")
        print(f"SUBGRAPH #{subgraph_idx+1}: {variable_subset} with {connectivity:.1f}% connectivity")
        print(f"{'#'*80}")

        # For each (gen_temp, corr_temp)
        for temp_idx, (gen_temp, c_temp) in enumerate(zip(gen_temps, corr_temps)):
            print(f"\n{'='*30} TEMPERATURE SET #{temp_idx+1} {'='*30}")
            print(f"Generator Temp: {gen_temp}, Corruptor Temp: {c_temp}, Judge Temp: {judge_temp}")

            for run_idx in range(runs_per_subgraph):
                print(f"\n{'-'*30} RUN #{run_idx+1}/{runs_per_subgraph} {'-'*30}")

                # 1) Build a new CausalDiscovery for this run
                discovery = CausalDiscovery(
                    target_variable=variable_subset[0],
                    temporal_scale="Years",
                    spatial_scale="Community Dwelling Elderly",
                    yaml_path="../backend/configs/prompts.yaml",
                    dev_mode=False,
                    generator_model=llm_params.get("generator_model","sonar-pro"),
                    corruptor_model=llm_params.get("corruptor_model", "sonar-pro"),
                    judge_model=llm_params.get("judge_model","claude-3"),
                    generator_temperature=gen_temp,
                    generator_top_p=1.0,
                    corruptor_temperature=c_temp,
                    corruptor_top_p=1.0,
                    judge_temperature=judge_temp,
                    judge_top_p=1.0
                )

                # Insert the subgraph’s variables into the DB
                session_id = discovery.session_id
                with discovery.graph_db._get_session() as session:
                    create_query = """
                    CREATE (n:variable {
                        name: $name,
                        description: $desc,
                        session_id: $sid,
                        target: $is_target,
                        deleted: false
                    })
                    """
                    # The first is the target
                    session.run(create_query, {
                        "name": variable_subset[0],
                        "desc": f"Variable: {variable_subset[0]}",
                        "sid": session_id,
                        "is_target": True
                    })
                    discovery.variables.append(variable_subset[0])

                    # The rest
                    for var in variable_subset[1:]:
                        session.run(create_query, {
                            "name": var,
                            "desc": f"Variable: {var}",
                            "sid": session_id,
                            "is_target": False
                        })
                        discovery.variables.append(var)

                # 2) Discover relationships
                rels = discovery.discover_relationships(parallel=False)
                # 3) Optional judging
                if num_judges > 0:
                    judged = discovery.judge_all_edges_serial(judge_models=judge_models, num_judges=num_judges)
                else:
                    judged = []

                # 4) Compare to validation
                metrics = compare_session_graph_to_validation(
                    discovery=discovery,
                    session_ids=session_id,
                    validation_vars_json_path=validation_vars_json_path,
                    validation_edges_json_path=validation_edges_json_path,
                    plot_session_graph=visualize_comparison,
                    plot_validation_graph=visualize_comparison,
                    only_cited_edges=False,
                    only_consistent_edges=False,
                    filter_validation_nodes_to_session_nodes=False
                )

                # 5) CAPTURE LLM STATS
                def safe_get_stats(llm_client):
                    if hasattr(llm_client, 'get_stats'):
                        return llm_client.get_stats()
                    return {"status_counts": {}, "token_totals": {}}
                gen_stats = safe_get_stats(discovery.nucleus)         # <-- ADDED LLM STATS HERE
                corr_stats = safe_get_stats(discovery.corruptor_llm)
                judge_stats = safe_get_stats(discovery.judge_llm)

                print("\n[Multi-Run] LLM Stats [Generator]:", gen_stats)    # <-- ADDED LLM STATS HERE
                print("[Multi-Run] LLM Stats [Corruptor]:", corr_stats)
                print("[Multi-Run] LLM Stats [Judge]:", judge_stats)

                # 6) Store results
                exp_key = f"subgraph{subgraph_idx+1}_gen{gen_temp}_corr{c_temp}_run{run_idx+1}"
                results[exp_key] = {
                    "subgraph_idx": subgraph_idx + 1,
                    "temperature_idx": temp_idx + 1,
                    "run_idx": run_idx + 1,
                    "variables": variable_subset,
                    "connectivity": connectivity,
                    "session_id": session_id,
                    "metrics": metrics,
                    "generator_temperature": gen_temp,
                    "corruptor_temperature": c_temp,
                    "judge_temperature": judge_temp,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    # Store the LLM stats in the result
                    "lm_stats": {
                        "generator": gen_stats,
                        "corruptor": corr_stats,
                        "judge": judge_stats
                    }
                }

    return results


import os
import re
import glob
import statistics
import openpyxl

def summarize_by_param_combo_and_prompt(
    input_folder: str,
    experiment_id: str,
    output_excel: str = "summarized_CLD_metrics.xlsx"
):
    """
    Summarize the 'All Edges Summary' sheets for all .xlsx files in `input_folder`
    that start with `experiment_id`, grouping side-by-side by param_combo_nr
    AND adding a one-column gap between param combos for clarity.

    Layout:
      Param_combo_nr_1               Param_combo_nr_2           ...
      Prompt: current deployed       Prompt: current deployed   ...
      CLD 1    CLD 2  CLD 3  [gap]   CLD 1  CLD 2  CLD 3  [gap]  ...
      filename1 ... filename3        filename4 ... filename6
      Metric Value Metric Value ...
      ...
                        Average accuracy ...
    """

    # Mapping prompt tokens -> display labels
    PROMPT_MAP = {
        "prompts_current": "Currrent deployed",
        "prompts_Nitai_A": "Nitai A",
        "prompts_Nitai_B": "Nitai B",
    }

    # In a fixed order
    METRICS = [
        "True Positives (TP)",
        "False Positives (FP)",
        "False Negatives (FN)",
        "True Negatives (TN)",
        "Total Discovered Edges",
        "Total Validation Edges",
        "Variables Correct Score",
        "correct variables",
        "incorrect variables",
        "Total Comparisons",
        "Precision",
        "Recall",
        "F1 Score",
        "Accuracy",
    ]

    # 1) Collect relevant .xlsx files
    pattern = os.path.join(input_folder, f"{experiment_id}*.xlsx")
    all_xlsx = glob.glob(pattern)
    if not all_xlsx:
        print(f"No XLSX files found in '{input_folder}' matching '{experiment_id}*.xlsx'.")
        return

    # 2) Build a nested data structure: data[param_combo][prompt][scenario]
    data = {}

    for fpath in all_xlsx:
        fname = os.path.basename(fpath)

        # Parse param_combo
        combo_match = re.search(r"_param_combo_nr_(\d+)_", fname)
        if not combo_match:
            # If no match, skip
            continue
        param_combo = combo_match.group(1)

        # Parse prompt
        prompt_found = None
        for token, label in PROMPT_MAP.items():
            if token in fname:
                prompt_found = label
                break
        if not prompt_found:
            # Optionally skip if we can’t map it
            continue

        # Parse scenario from tail
        scenario_match = re.search(r"_run\d+_(.+)\.xlsx$", fname)
        if scenario_match:
            scenario_name = scenario_match.group(1)
        else:
            scenario_name = fname  # fallback

        # Read the "All Edges Summary" sheet
        try:
            wb = openpyxl.load_workbook(fpath, data_only=True)
            if "All Edges Summary" not in wb.sheetnames:
                wb.close()
                continue

            ws = wb["All Edges Summary"]
            sheet_metrics = {}
            for row in ws.iter_rows(min_row=2, values_only=True):
                metric_name, metric_val = row[0], row[1]
                if metric_name:
                    sheet_metrics[metric_name] = metric_val

            wb.close()

            # Store
            if param_combo not in data:
                data[param_combo] = {}
            if prompt_found not in data[param_combo]:
                data[param_combo][prompt_found] = {}
            data[param_combo][prompt_found][scenario_name] = {
                "filename": fname,
                "metrics": sheet_metrics
            }

        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # 3) Create output workbook
    wb_out = openpyxl.Workbook()
    ws_out = wb_out.active
    ws_out.title = "Summarized"

    # Sort param_combo numerically
    all_param_combos = sorted(data.keys(), key=lambda x: int(x))

    # Gather union of all prompts across combos
    prompt_set = set()
    for combo in all_param_combos:
        prompt_set.update(data[combo].keys())
    sorted_prompts = sorted(list(prompt_set))

    current_row = 1

    # 4) For each prompt, we’ll write blocks side-by-side (one block per param_combo)
    for prompt in sorted_prompts:

        # Union of all scenario names for this prompt across all combos
        all_scenarios_for_prompt = set()
        for combo in all_param_combos:
            if prompt in data[combo]:
                all_scenarios_for_prompt.update(data[combo][prompt].keys())
        scenario_list = sorted(list(all_scenarios_for_prompt))

        # Each param combo block is (2 * len(scenario_list)) columns for the data
        # plus 1 blank column to separate from the next block.
        col_block_width = 2 * len(scenario_list) + 1

        # --- Row 1: Param_combo_nr_x labels ---
        for i, combo in enumerate(all_param_combos):
            col_start = 1 + i * col_block_width
            ws_out.cell(row=current_row, column=col_start, value=f"Param_combo_nr_{combo}")
        current_row += 1

        # --- Row 2: "Prompt: <prompt>" ---
        for i, combo in enumerate(all_param_combos):
            col_start = 1 + i * col_block_width
            ws_out.cell(row=current_row, column=col_start, value=f"Prompt: {prompt}")
        current_row += 1

        # --- Row 3: "CLD 1, CLD 2, ..." for each scenario block
        for i, combo in enumerate(all_param_combos):
            col_start = 1 + i * col_block_width
            if prompt not in data[combo]:
                continue
            for j, scenario in enumerate(scenario_list):
                cld_col = col_start + 2 * j
                ws_out.cell(row=current_row, column=cld_col, value=f"CLD  {j+1}")
        current_row += 1

        # --- Row 4: Filenames
        for i, combo in enumerate(all_param_combos):
            col_start = 1 + i * col_block_width
            if prompt not in data[combo]:
                continue
            for j, scenario in enumerate(scenario_list):
                cld_col = col_start + 2 * j
                info = data[combo][prompt].get(scenario)
                if info is not None:
                    ws_out.cell(row=current_row, column=cld_col, value=info["filename"])
        current_row += 1

        # --- Row 5: "Metric, Value"
        for i, combo in enumerate(all_param_combos):
            col_start = 1 + i * col_block_width
            if prompt not in data[combo]:
                continue
            for j, scenario in enumerate(scenario_list):
                cld_col = col_start + 2 * j
                ws_out.cell(row=current_row, column=cld_col,   value="Metric")
                ws_out.cell(row=current_row, column=cld_col+1, value="Value")
        current_row += 1

        # --- Next: Each metric repeated across each param combo’s scenario block
        for metric in METRICS:
            row_start_for_this_metric = current_row
            for i, combo in enumerate(all_param_combos):
                col_start = 1 + i * col_block_width
                # If this prompt not in data[combo], skip
                if prompt not in data[combo]:
                    continue
                for j, scenario in enumerate(scenario_list):
                    cld_col = col_start + 2 * j
                    ws_out.cell(row=current_row, column=cld_col, value=metric)
                    info = data[combo][prompt].get(scenario)
                    if info is not None:
                        val = info["metrics"].get(metric, None)
                        ws_out.cell(row=current_row, column=cld_col + 1, value=val)
            current_row += 1

        # After metrics, add an "Average accuracy" row for each param combo block
        for i, combo in enumerate(all_param_combos):
            if prompt not in data[combo]:
                continue
            # Gather accuracy across all scenarios
            accuracies = []
            for scenario, info in data[combo][prompt].items():
                acc = info["metrics"].get("Accuracy", None)
                if acc is not None:
                    try:
                        accuracies.append(float(acc))
                    except:
                        pass
            if accuracies:
                avg_acc = statistics.mean(accuracies)
                # Place label + value in the last scenario's columns
                col_start = 1 + i * col_block_width
                last_scenario_idx = len(scenario_list) - 1
                label_col = col_start + 2 * last_scenario_idx
                ws_out.cell(row=current_row, column=label_col, value="Average accuracy")
                ws_out.cell(row=current_row, column=label_col + 1, value=avg_acc)
        current_row += 2  # blank rows before next prompt

    wb_out.save(output_excel)
    print(f"Summary table written to: {output_excel}")

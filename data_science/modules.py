# imports
import os
import sys
import gc

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# imports continued
import re
import glob
import statistics
import openpyxl
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
import seaborn as sns
from datetime import datetime
import logging
import sys
import traceback
import requests
from bs4 import BeautifulSoup
import asyncio
from pydantic import BaseModel, Field
import datetime
import uuid
from memory_profiler import profile
from dotenv import load_dotenv
from itertools import permutations
import traceback
import logging
from backend.db_clients.neo4j_client import Neo4jClient
from data_loading_scripts.excel_cld_loader_v3 import load_cld_from_excel
import tempfile
import re
import glob
import statistics
import openpyxl
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
import glob
import csv
import yaml
import itertools
import datetime
import uuid
import yaml


# Import from existing codebase
from backend.db_clients.neo4j_client import Neo4jClient
from backend.lm_clients.perplexity_client import PerplexityClient
from backend.prompt_tree import PromptTree
from backend.lm_clients.claude_client import ClaudeClient
from backend.lm_clients.openai_client_working import OpenAIClient
from dotenv import load_dotenv

# Import ContentScraper from platform for full PDF support
try:
    import sys
    import os
    # Platform is at /home/nitai/code/platform (same parent as causalix.ai)
    platform_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "platform", "backend"))
    if platform_backend_path not in sys.path:
        sys.path.insert(0, platform_backend_path)
    from websearch_clients.content_scraper import ContentScraper
except (ImportError, ModuleNotFoundError) as e:
    print(f"⚠️ Warning: Could not import ContentScraper from platform: {e}")
    ContentScraper = None

# Context-insensitive metrics utilities
try:
    from logit_metrics import compute_logit_metrics, compute_alignment
except Exception:
    try:
        from data_science.logit_metrics import compute_logit_metrics, compute_alignment
    except Exception:
        compute_logit_metrics = None
        compute_alignment = None

# Logprob metrics adapter (OpenAI-like → compute_logit_metrics schema)
def _compute_and_trace_metrics_from_logprobs(log_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    # All CI metrics with None defaults (includes metrics from arXiv:2405.19648v1 and CHAIR)
    empty_metrics = {
        "avg_nll": None, "perplexity": None, "min_prob": None, "max_window_entropy": None,
        "mean_token_prob": None, "prob_variance": None, "prob_std": None, "max_prob_diff": None,
        "token_prob_slope": None
    }
    if not log_data or not compute_logit_metrics:
        return empty_metrics
    # If already standard
    if isinstance(log_data, dict) and "token_logprobs" in log_data:
        return compute_logit_metrics(log_data)
    token_logprobs: List[Optional[float]] = []
    top_logprobs: List[Dict[str, float]] = []
    try:
        content = log_data.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                token_logprobs.append(item.get("logprob"))
                topk = {}
                tlp = item.get("top_logprobs")
                if isinstance(tlp, list):
                    for entry in tlp[:5]:
                        if isinstance(entry, dict):
                            tok = entry.get("token")
                            lp = entry.get("logprob")
                            if isinstance(tok, str) and isinstance(lp, (float, int)):
                                topk[tok] = lp
                top_logprobs.append(topk)
        std = {"tokens": [], "token_logprobs": token_logprobs, "top_logprobs": top_logprobs}
        return compute_logit_metrics(std)
    except Exception:
        return empty_metrics



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("causal_discovery.log")
    ]
)

# Suppress verbose HTTP/API logging from external libraries
# This prevents full article content from being printed in terminal
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)  
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("anthropic._base_client").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("causal_discovery")


# Fix for notebook environment - use current working directory
# notebook_path = os.getcwd()
# parent_path = os.path.dirname(notebook_path)
# sys.path.append(parent_path)

load_dotenv("../.env.dev", override=True)

# heck if the API key is set
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


def compute_comprehensive_stats(values: list, name: str = "") -> dict:
    """
    Compute comprehensive statistics for repeated measurements.
    
    For n < 30, reports mean [min, max] as primary (assumption-free).
    Also includes SD and 95% CI (t-distribution) for reference.
    
    Args:
        values: List of observed values (e.g., F1 scores across runs/folds)
        name: Optional name for the metric
    
    Returns:
        Dictionary with all statistics and LaTeX formatting
    """
    from scipy import stats as scipy_stats
    
    values = list(values)  # Ensure list
    n = len(values)
    
    if n == 0:
        return {
            'name': name, 'n': 0, 'values': [],
            'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'se': np.nan, 't_crit': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
            'latex_primary': 'N/A', 'latex_full': 'N/A'
        }
    
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    
    # 95% CI using t-distribution (for reference - not recommended for n < 30)
    se = std_val / np.sqrt(n) if n > 1 else 0.0
    t_crit = float(scipy_stats.t.ppf(0.975, n - 1)) if n > 1 else np.nan
    ci_lower = mean_val - t_crit * se if n > 1 else mean_val
    ci_upper = mean_val + t_crit * se if n > 1 else mean_val
    
    # LaTeX formatting
    latex_primary = f"{mean_val:.3f} [{min_val:.3f}, {max_val:.3f}]"
    latex_full = (
        f"{mean_val:.3f} [{min_val:.3f}, {max_val:.3f}] "
        f"(SD={std_val:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}])"
    )
    
    return {
        'name': name,
        'n': n,
        'values': [float(v) for v in values],
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'se': float(se),
        't_crit': float(t_crit) if not np.isnan(t_crit) else None,
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'latex_primary': latex_primary,
        'latex_full': latex_full
    }


def safe_bool(x):
    return True if x is True else False


def fetch_citations_with_jina(
    citations: List[str],
    timeout: int = 60,
    max_retries: int = 2,
    fallback_to_scraper: bool = True,
    logger = None,
    api_key: str = None
) -> Dict[str, str]:
    """
    Fetch citation content using Jina AI Reader with fallback to ContentScraper.
    
    Args:
        citations: List of URLs to fetch
        timeout: Timeout for Jina AI requests (default: 60s)
        max_retries: Number of retries for failed requests (default: 2)
        fallback_to_scraper: Whether to fallback to ContentScraper on failure
        logger: Logger instance
        api_key: Jina AI API key (optional, improves rate limits)
    
    Returns:
        Dict mapping URLs to their text content
    """
    import requests
    import time
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY", "")
    
    citation_text_map = {}
    
    def fetch_single_citation(url: str) -> tuple:
        """Fetch a single citation with Jina AI."""
        for attempt in range(max_retries + 1):
            try:
                jina_url = f"https://r.jina.ai/{url}"
                logger.debug(f"Fetching via Jina AI: {url} (attempt {attempt + 1}/{max_retries + 1})")
                
                # Add delay to respect rate limits (with API key: ~200 req/min, without: ~10 req/min)
                if api_key:
                    time.sleep(0.3)  # 0.3 second delay with API key
                else:
                    time.sleep(1)  # 1 second delay without API key
                
                # Add API key to headers if provided
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                response = requests.get(jina_url, headers=headers, timeout=timeout)
                response.raise_for_status()
                
                content = response.text
                
                # Check if content is meaningful (not just an error page)
                if content and len(content) > 200 and "[ERROR]" not in content[:500]:
                    logger.info(f"✅ Jina AI success: {url} ({len(content):,} chars)")
                    return url, content, "jina"
                else:
                    logger.warning(f"⚠️ Jina AI returned short/error content for: {url} ({len(content)} chars)")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return url, None, "jina_failed"
                        
            except requests.exceptions.Timeout:
                logger.warning(f"⏱️ Jina AI timeout for: {url} (attempt {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return url, None, "timeout"
                    
            except Exception as e:
                logger.warning(f"❌ Jina AI error for {url}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return url, None, "error"
        
        return url, None, "failed"
    
    # Fetch all citations in parallel with Jina AI
    if api_key:
        logger.info(f"Fetching {len(citations)} citations with Jina AI using API key (timeout={timeout}s, retries={max_retries})")
        max_workers = 5  # Higher limit with API key
    else:
        logger.info(f"Fetching {len(citations)} citations with Jina AI (free tier, timeout={timeout}s, retries={max_retries})")
        max_workers = 2  # Lower limit without API key
    
    jina_results = {}
    failed_urls = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_single_citation, url): url for url in citations}
        
        for future in as_completed(future_to_url):
            url, content, status = future.result()
            if content:
                jina_results[url] = content
            else:
                failed_urls.append(url)
    
    logger.info(f"Jina AI results: {len(jina_results)}/{len(citations)} successful ({len(jina_results)/len(citations)*100:.1f}%)")
    
    # Fallback to ContentScraper for failed URLs
    if fallback_to_scraper and failed_urls:
        logger.info(f"Falling back to ContentScraper for {len(failed_urls)} failed URLs...")
        try:
            # Try to import ContentScraper dynamically
            ContentScraper_local = None
            try:
                import sys
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                platform_path = os.path.join(project_root, "platform")
                if platform_path not in sys.path:
                    sys.path.insert(0, platform_path)
                from backend.websearch_clients.content_scraper import ContentScraper as ContentScraper_local
            except (ImportError, ModuleNotFoundError):
                logger.warning("ContentScraper not available, skipping fallback")
                return jina_results
            
            processor = ContentScraper_local(
                timeout=15,
                max_workers=3,
                max_content_length=10_000_000,
                pdf_extraction_mode="local",
                disable_agentic_fallback=True,
                access_all_domains=False  # Enable whitelist: only trusted domains
            )
            citation_meta = processor.process_citations(failed_urls, fetch_content=True)
            scraper_results = processor.prepare_for_llm(citation_meta, include_metadata=True)
            
            # Filter out error messages from ContentScraper
            for url, content in scraper_results.items():
                if content and len(content) > 200 and "[ERROR]" not in content[:100]:
                    jina_results[url] = content
                    logger.info(f"✅ ContentScraper success: {url} ({len(content):,} chars)")
                else:
                    logger.warning(f"⚠️ ContentScraper also failed for: {url}")
            
            logger.info(f"ContentScraper recovered {len(scraper_results)}/{len(failed_urls)} URLs")
            
        except Exception as e:
            logger.error(f"ContentScraper fallback failed: {e}")
    
    logger.info(f"Total successful: {len(jina_results)}/{len(citations)} ({len(jina_results)/len(citations)*100:.1f}%)")
    
    return jina_results


# Structured output support for LLM judges
try:
    from judge_models import CorrectnessJudgment, CitationJudgment
    from judge_response_adapter import JudgeResponseAdapter
    STRUCTURED_OUTPUTS_AVAILABLE = True
except ImportError:
    STRUCTURED_OUTPUTS_AVAILABLE = False
    logger.warning("Structured output support not available (judge_models or judge_response_adapter not found)")

# Structured output support for LLM corrector
try:
    from corrector_response_models import CorrectorResponse, CorrectorAction, EdgeType
    CORRECTOR_STRUCTURED_OUTPUTS_AVAILABLE = True
except ImportError:
    CORRECTOR_STRUCTURED_OUTPUTS_AVAILABLE = False
    logger.warning("Corrector structured output support not available (corrector_response_models not found)")


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
                generator_config: Optional[Dict[str, str]] = None,
                corruptor_config: Optional[Dict[str, str]] = None,
                judge_config: Optional[Dict[str, str]] = None,
                context=None,

                # Sampling parameters for each role
                generator_temperature=0.7,
                generator_top_p=1.0,
                corruptor_temperature=0.7,
                corruptor_top_p=1.0,
                judge_temperature=0.7,
                judge_top_p=1.0,
                judge_seed=None,
                corruptor_seed: Optional[int] = None,
                corrector_seed: Optional[int] = None,

                # Web search parameters for each role (parity with legacy)
                generator_enable_web_search: bool = False,
                corruptor_enable_web_search: bool = False,
                judge_enable_web_search: bool = True,
                citation_fill_enable_web_search: bool = True
            ):
                """
                Initialize the causal discovery algorithm.

                Args:
                    target_variable: The target variable for causal discovery
                    temporal_scale: Temporal scale for analysis
                    spatial_scale: Spatial scale for analysis
                    yaml_path: Path to the prompt template YAML file
                    dev_mode: Whether to run in development mode

                    generator_config (dict): e.g. {"provider": "openai", "model": "gpt-4"}
                    corruptor_config (dict): e.g. {"provider": "openai", "model": "gpt-4"}
                    judge_config (dict): e.g. {"provider": "claude", "model": "claude-3-7-sonnet"}

                    generator_temperature: Temperature for the generator LLM
                    generator_top_p: Top-p for the generator LLM
                    corruptor_temperature: Temperature for the corruptor LLM
                    corruptor_top_p: Top-p for the corruptor LLM
                    judge_temperature: Temperature for the judge LLM
                    judge_top_p: Top-p for the judge LLM
                """
                # 1) Connect to Neo4j
                self.graph_db = Neo4jClient(
                    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    username=os.getenv("NEO4J_USERNAME", "neo4j"),
                    password=os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz"),
                    logger=logging.getLogger('app.db.neo4j')
                )

                # Store the target; skip if it's None or empty
                if not target_variable or target_variable.strip() == "":
                    self.target_variable = None
                    logger.warning("No valid target variable specified (None or empty) => skipping target node creation.")
                else:
                    self.target_variable = target_variable.strip()

                # 2) Load environment variables for each provider
                perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY", None)
                perplexity_api_url = os.environ.get("PERPLEXITY_API_URL", None)
                perplexity_default_model = os.environ.get("PERPLEXITY_MODEL", None)

                claude_api_key = os.environ.get("CLAUDE_API_KEY", None)
                claude_api_url = os.environ.get("CLAUDE_API_URL", 'https://api.anthropic.com/v1/messages')
                claude_default_model = os.environ.get("CLAUDE_MODEL", None)

                openai_api_key = os.environ.get("OPENAI_API_KEY", None)
                openai_api_url = os.environ.get("OPENAI_API_URL", None)
                openai_default_model = os.environ.get("OPENAI_MODEL", None)

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

                
                print(f"generator_model CausalDiscovery: {generator_config}")

    # Make sure we have some default config if user didn't provide it
                if generator_config is None:
                    generator_config = {"provider": "openai", "model": "gpt-4"}
                if corruptor_config is None:
                    corruptor_config = {"provider": "openai", "model": "gpt-4"}
                if judge_config is None:
                    judge_config = {"provider": "anthropic", "model": "claude-3-7-sonnet"}

                # Persist web-search flags
                self.generator_enable_web_search = generator_enable_web_search
                self.corruptor_enable_web_search = corruptor_enable_web_search
                self.judge_enable_web_search = judge_enable_web_search
                self.citation_fill_enable_web_search = citation_fill_enable_web_search
                
                # Citation filling parallelization settings (default: parallel with 5 workers)
                self.citation_fill_parallel = True
                self.citation_fill_max_workers = 5

                # 3) Instantiate the generator LLM client
                g_provider = generator_config["provider"].lower().strip()
                g_model = generator_config["model"]
                self.generator_llm = self._build_llm_client(
                    provider=g_provider,
                    model=g_model,
                    dev_mode=dev_mode,
                    temperature=generator_temperature,
                    top_p=generator_top_p,
                    openai_key=openai_api_key,
                    openai_url=openai_api_url,
                    claude_key=claude_api_key,
                    claude_url=claude_api_url,
                    perplexity_key=perplexity_api_key,
                    perplexity_url=perplexity_api_url,
                    role_name="generator",
                    enable_web_search=self.generator_enable_web_search,
                )
                logger.info(f"Generator LLM => provider={g_provider}, model={g_model}")

                # 4) Instantiate the corruptor LLM
                c_provider = corruptor_config["provider"].lower().strip()
                c_model = corruptor_config["model"]
                # if corruptor is the same as generator, you can reuse or not, up to you:
                if (c_provider == g_provider) and (c_model == g_model):
                    self.corruptor_llm = self.generator_llm
                    logger.info(f"Corruptor LLM => reusing generator LLM: {c_provider}/{c_model}")
                else:
                    self.corruptor_llm = self._build_llm_client(
                        provider=c_provider,
                        model=c_model,
                        dev_mode=dev_mode,
                        temperature=corruptor_temperature,
                        top_p=corruptor_top_p,
                        openai_key=openai_api_key,
                        openai_url=openai_api_url,
                        claude_key=claude_api_key,
                        claude_url=claude_api_url,
                        perplexity_key=perplexity_api_key,
                        perplexity_url=perplexity_api_url,
                        role_name="corruptor",
                        enable_web_search=self.corruptor_enable_web_search,
                    )
                    logger.info(f"Corruptor LLM => provider={c_provider}, model={c_model}")

                # 5) Instantiate the judge LLM(s)
                # Support both single judge (dict) and ensemble (list of dicts)
                if isinstance(judge_config, list):
                    # Heterogeneous ensemble: create multiple judge LLM clients
                    self.judge_llms = []
                    for idx, jcfg in enumerate(judge_config):
                        j_provider = jcfg["provider"].lower().strip()
                        j_model = jcfg["model"]
                        j_temp = jcfg.get("temperature", judge_temperature)
                        judge_client = self._build_llm_client(
                            provider=j_provider,
                            model=j_model,
                            dev_mode=dev_mode,
                            temperature=j_temp,
                            top_p=judge_top_p,
                            openai_key=openai_api_key,
                            openai_url=openai_api_url,
                            claude_key=claude_api_key,
                            claude_url=claude_api_url,
                            perplexity_key=perplexity_api_key,
                            perplexity_url=perplexity_api_url,
                            role_name=f"judge_{idx+1}",
                            enable_web_search=self.judge_enable_web_search,
                        )
                        self.judge_llms.append(judge_client)
                        logger.info(f"Judge {idx+1} LLM => provider={j_provider}, model={j_model}")
                    # For compatibility, set judge_llm to the first judge
                    self.judge_llm = self.judge_llms[0]
                    # Store judge config for corrector to use
                    self.judge_provider = judge_config[0]["provider"].lower().strip()
                    self.judge_model = judge_config[0]["model"]
                else:
                    # Single judge
                    j_provider = judge_config["provider"].lower().strip()
                    j_model = judge_config["model"]
                    self.judge_llm = self._build_llm_client(
                        provider=j_provider,
                        model=j_model,
                        dev_mode=dev_mode,
                        temperature=judge_temperature,
                        top_p=judge_top_p,
                        openai_key=openai_api_key,
                        openai_url=openai_api_url,
                        claude_key=claude_api_key,
                        claude_url=claude_api_url,
                        perplexity_key=perplexity_api_key,
                        perplexity_url=perplexity_api_url,
                        role_name="judge",
                        enable_web_search=self.judge_enable_web_search,
                    )
                    logger.info(f"Judge LLM => provider={j_provider}, model={j_model}")
                    # Store judge config for corrector to use
                    self.judge_provider = j_provider
                    self.judge_model = j_model


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
                self.judge_seed = judge_seed
                self.corruptor_seed = corruptor_seed
                self.corrector_seed = corrector_seed

                # Session & variables
                self.session_id = str(uuid.uuid4())
                self.variables = []
                self.processed_pairs = set()

                # Set the session ID in Neo4j
                self.graph_db.set_session_id(self.session_id)

                logger.info(f"Initialized CausalDiscovery with session_id={self.session_id}")
                logger.info(f"Target: {self.target_variable}, Temporal: {self.temporal_scale}, Spatial: {self.spatial_scale}")
                logger.info(f"Generator temperature={generator_temperature}, top_p={generator_top_p}")
                logger.info(f"Corruptor temperature={corruptor_temperature}, top_p={corruptor_top_p}, seed={corruptor_seed}")
                logger.info(f"Judge temperature={judge_temperature}, top_p={judge_top_p}, seed={judge_seed}")
                logger.info(f"Corrector seed={corrector_seed}")

    def generate_variables(
        self, 
        num_variables: int = 5,
        target_units: str = None,
        target_definition: str = None,
        context_list: list = None
    ) -> List[str]:
        """
        Generate variables related to the (optional) target variable.
        If target_variable is None or empty, skip creating a target node entirely.
        
        Args:
            num_variables: Number of variables to generate
            target_units: Units of measurement for the target variable
            target_definition: Definition of the target variable
            context_list: List of contextual factors
        """
        logger.info(f"Generating up to {num_variables} variables...")
        self.variables = []

        ## NEW CODE: Skip the target creation if self.target_variable is None
        if self.target_variable:
            # Create the target node in the graph
            try:
                self.graph_db.create_node(
                    node_label="variable",
                    node_properties={
                        "name": self.target_variable,
                        "definition": f"Definition for {self.target_variable}",
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
        else:
            logger.warning("Skipping target node creation (target_variable is None or empty).")

        # The rest of the code for generating additional variables
        variable_count = 1 if self.target_variable else 0

        while variable_count < num_variables:
            try:
                # Use generateNodes prompt (Andrew's naming convention)
                # Ensure all template variables are not None to avoid .title() errors
                target_for_prompt = self.target_variable
                if not target_for_prompt:
                    target_for_prompt = "outcome variable"
                    logger.warning("⚠️  Using default target='outcome variable' (no target specified)")
                    
                units_for_prompt = target_units if target_units else "standard units"
                if not target_units:
                    logger.warning("⚠️  Using default target_units='standard units'")
                    
                definition_for_prompt = target_definition if target_definition else "The primary outcome variable"
                if not target_definition:
                    logger.warning("⚠️  Using default target_definition")
                    
                temporal_for_prompt = self.temporal_scale if self.temporal_scale else "medium term"
                if not self.temporal_scale:
                    logger.warning("⚠️  Using default temporal='medium term'")
                    
                spatial_for_prompt = self.spatial_scale if self.spatial_scale else "community level"
                if not self.spatial_scale:
                    logger.warning("⚠️  Using default spatial='community level'")
                
                sys_prompt, usr_prompt, usr_prompt_extension = self.tree.build_prompt(
                    id="generateNodes",
                    variable_dict={
                        "target": target_for_prompt,
                        "target_units": units_for_prompt,
                        "target_definition": definition_for_prompt,
                        "number_of_variables": num_variables,
                        "temporal": temporal_for_prompt,
                        "spatial": spatial_for_prompt,
                        "context": context_list if context_list is not None else []
                    }
                )

                excluded_str = ", ".join(self.variables)
                full_usr_prompt = f"{usr_prompt}\n{usr_prompt_extension} {excluded_str}"
                # Send request. Skip logprobs for models that forbid it (e.g., GPT-5)
                # Claude and Perplexity also don't support extra_body parameter.
                model_name = getattr(self.generator_llm, "model", "").lower()
                provider = getattr(self.generator_llm, "__class__", object).__name__.lower()
                is_openai = "openai" in provider
                supports_logprobs = is_openai and not model_name.startswith("gpt-5")
                if supports_logprobs:
                    response = self.generator_llm.send_message(
                        sys_prompt=sys_prompt,
                        usr_prompt=full_usr_prompt,
                        stream=False,
                        extra_body={"logprobs": True, "top_logprobs": 5},
                    )
                else:
                    response = self.generator_llm.send_message(
                        sys_prompt=sys_prompt,
                        usr_prompt=full_usr_prompt,
                        stream=False,
                        max_tokens=1000,  # Enough for JSON response
                    )

                data = self.generator_llm.parse_static_response(response)
                # Extract and compute logprob metrics if available
                try:
                    log_data = data.get("logprobs") or {}
                    log_metrics = _compute_and_trace_metrics_from_logprobs(log_data)
                except Exception:
                    log_metrics = {
                        "avg_nll": None, "perplexity": None, "min_prob": None, "max_window_entropy": None,
                        "mean_token_prob": None, "prob_variance": None, "prob_std": None, "max_prob_diff": None,
                        "token_prob_slope": None
                    }
                content = data['content']
                citations = data.get('citations', [])

                # Parse variable response (returns 4 values: name, definition, units, citations)
                name, definition, units, citations = self.tree.read_variable_response(content, citations)

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

                time.sleep(1)

            except Exception as e:
                logger.error(f"Error generating variable {variable_count+1}: {e}")
                # Retry or skip
                continue

        logger.info(f"Finished generating variables => total: {len(self.variables)}")
        logger.info(f"Variables: {self.variables}")
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

            # Initialize logprob captures (avoid lints)
            log_data: Dict[str, Any] = {}
            log_metrics: Dict[str, Optional[float]] = {
                "avg_nll": None, "perplexity": None, "min_prob": None, "max_window_entropy": None
            }

            # Send request to the LLM. Some models (e.g., GPT-5) disallow logprobs.
            # Claude and Perplexity also don't support extra_body parameter.
            logger.info(f"Analyzing relationship between {source} and {target}")
            model_name = getattr(self.generator_llm, "model", "").lower()
            provider = getattr(self.generator_llm, "__class__", object).__name__.lower()
            # Only OpenAI models (except GPT-5) support extra_body with logprobs
            is_openai = "openai" in provider
            supports_logprobs = is_openai and not model_name.startswith("gpt-5")
            if supports_logprobs:
                response = self.generator_llm.send_message(
                    sys_prompt=sys_prompt,
                    usr_prompt=usr_prompt,
                    stream=False,
                    extra_body={"logprobs": True, "top_logprobs": 5},
                )
            else:
                response = self.generator_llm.send_message(
                    sys_prompt=sys_prompt,
                    usr_prompt=usr_prompt,
                    stream=False,
                    max_tokens=1000,  # Enough for JSON response
                )



            # Parse the response
            data = self.generator_llm.parse_static_response(response)
            # Truncate logprobs for logging to avoid massive output
            data_excerpt = {
                'content': data.get('content', '')[:200] + '...' if len(data.get('content', '')) > 200 else data.get('content', ''),
                'metadata': data.get('metadata', {}),
                'citations': data.get('citations', [])[:3],  # Show max 3 citations
                'logprobs': '... (truncated)' if data.get('logprobs') else None
            }
            logger.info(f"Returned data: {data_excerpt}")
            # Extract logprobs & compute metrics (if available)
            try:
                log_data = data.get("logprobs") or {}
                log_metrics = _compute_and_trace_metrics_from_logprobs(log_data)
            except Exception:
                pass
            content = data['content']
            citations = data.get('citations', [])
            logger.info(f"Citations: {citations}")
            print('citations', citations)

            # Process the relationship
            relationship, motivation, citations = self.tree.read_relationships(content, citations)

            # Include NONE relationships - they should be stored and judged too!
            # The LLM might be wrong about classifying something as NONE
            if relationship is not False and (not isinstance(relationship, str) or relationship.lower() != "false"):
                # Handle NONE relationships specially but still store them
                if relationship == "NONE":
                    logger.info(f"Storing NONE relationship for judging: {source}-[NONE]->{target}")
                    logger.info(f"NONE explanation: {motivation}")
                
    # create relationship code
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
                # Store motivation, citations, and (if available) logprob-derived metrics
                try:
                    self.graph_db.create_relationships(
                        node1_name=source,
                        node2_name=target,
                        node_label="variable",
                        relationship_type=relationship,
                        motivation=edge_motivation,
                        citations=citations,
                        avg_nll=(log_metrics.get("avg_nll") if 'log_metrics' in locals() and isinstance(log_metrics, dict) else None),
                        perplexity=(log_metrics.get("perplexity") if 'log_metrics' in locals() and isinstance(log_metrics, dict) else None),
                        min_prob=(log_metrics.get("min_prob") if 'log_metrics' in locals() and isinstance(log_metrics, dict) else None),
                        max_window_entropy=(log_metrics.get("max_window_entropy") if 'log_metrics' in locals() and isinstance(log_metrics, dict) else None),
                        # New metrics from arXiv:2405.19648v1
                        mean_token_prob=(log_metrics.get("mean_token_prob") if 'log_metrics' in locals() and isinstance(log_metrics, dict) else None),
                        prob_variance=(log_metrics.get("prob_variance") if 'log_metrics' in locals() and isinstance(log_metrics, dict) else None),
                        prob_std=(log_metrics.get("prob_std") if 'log_metrics' in locals() and isinstance(log_metrics, dict) else None),
                        max_prob_diff=(log_metrics.get("max_prob_diff") if 'log_metrics' in locals() and isinstance(log_metrics, dict) else None),
                        # Temporal slope metric
                        token_prob_slope=(log_metrics.get("token_prob_slope") if 'log_metrics' in locals() and isinstance(log_metrics, dict) else None)
                    )
                    
                    # If we used spurious motivation, mark the edge as corrupted
                    if spurious_motivation:
                        print(f"🔴 CORRUPTION: Setting is_corrupted=True for {source}-[{relationship}]->{target}")
                        print(f"   Original motivation: {motivation[:50]}...")
                        print(f"   Spurious motivation: {spurious_motivation[:50]}...")
                        success = self.update_relationship_properties(
                            source=source,
                            target=target,
                            relationship_type=relationship,
                            properties={
                                "is_corrupted": True,
                                "original_motivation": motivation,
                                "spurious_motivation": spurious_motivation
                            }
                        )
                        print(f"   Corruption flag update result: {success}")
                except TypeError:
                    # Fallback in case older signature is present
                    self.graph_db.create_relationships(
                        node1_name=source,
                        node2_name=target,
                        node_label="variable",
                        relationship_type=relationship,
                        motivation=edge_motivation,
                        citations=citations,
                    )

                # Persist only aggregate logprob metrics as separate properties
                try:
                    self.update_relationship_properties(
                        source=source,
                        target=target,
                        relationship_type=relationship,
                        properties={
                            "perplexity": log_metrics.get("perplexity"),
                            "min_prob": log_metrics.get("min_prob"),
                            "max_window_entropy": log_metrics.get("max_window_entropy"),
                        }
                    )
                except Exception:
                    pass

                logger.info(f"Created relationship: {source}-[{relationship}]->{target}")

                # sleep to avoid rate limits
                time.sleep(1)

                # Return the entire set
                return (source, target, relationship, motivation, spurious_motivation)

            else:
                # Store NONE relationships for judging - LLM might be wrong!
                if relationship == "NONE":
                    logger.info(f"Storing NONE relationship for judging: {source}-[NONE]->{target}")
                    logger.info(f"NONE explanation: {motivation}")
                    
                    # Create NONE relationship in the graph for judging
                    try:
                        self.graph_db.create_relationships(
                            node1_name=source,
                            node2_name=target,
                            node_label="variable",
                            relationship_type="NONE",
                            motivation=motivation,
                            citations=citations,
                        )
                        logger.info(f"Created NONE relationship: {source}-[NONE]->{target}")
                        return (source, target, "NONE", motivation, None)
                    except Exception as e:
                        logger.error(f"Error creating NONE relationship: {e}")
                        
                logger.debug(f"No relationship found between {source} and {target}")
                return None

        except Exception as e:
            logger.error(f"Error processing pair {source}->{target}: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            return None

    
    def discover_relationships(
        self,
        generation_parallel: bool = False,
        generation_max_workers: int = 3,
        corruption_rate: float = 0
    ) -> List[Tuple[str, str, str]]:
        """
        Discover causal relationships between variables with optional spurious motivations.
        
        Args:
            generation_parallel: Whether to discover relationships in parallel.
            generation_max_workers: Number of workers for parallel relationship discovery.
            corruption_rate: Rate of corruption (deprecated, kept for compatibility).
        
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
        if generation_parallel and len(pairs) > 1:
            with ThreadPoolExecutor(max_workers=generation_max_workers) as executor:
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
                    
                    # Log NONE relationships for tracking
                    if rel_type == "NONE":
                        logger.info(f"NONE relationship discovered: {src} → {tgt} (explanation: {orig_motivation[:50]}...)")
                    
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
        
    def _build_ephemeral_llm(
    self,
    provider: str,
    model: str,
    temperature: float = 0.7,
    top_p: float = 1.0,
    dev_mode: bool = False
    ):
        """
        Build and return a short-lived (ephemeral) LLM client based on the given provider and model.

        The caller can specify a temperature, top_p, and dev_mode to override defaults.
        This method does not store the resulting client as a persistent attribute of 'self'—
        it merely returns an LLM instance for one-off usage.

        Example usage:
            ephemeral_llm = self._build_ephemeral_llm(provider="openai", model="gpt-4")
            response = ephemeral_llm.send_message(sys_prompt, usr_prompt)
        """
        import os
        import logging

        # Gather environment credentials for each possible provider
        openai_key = os.environ.get("OPENAI_API_KEY", "None")
        openai_url = os.environ.get("OPENAI_API_URL", "None")
        claude_key = os.environ.get("CLAUDE_API_KEY", "None")
        claude_url = os.environ.get("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
        perplexity_key = os.environ.get("PERPLEXITY_API_KEY", "None")
        perplexity_url = os.environ.get("PERPLEXITY_API_URL", "None")

        logger = logging.getLogger(f"ephemeral_llm.{provider}")

        # Pick the correct LLM subclass based on provider
        if provider.lower() == "openai":
            from backend.lm_clients.openai_client_working import OpenAIClient
            return OpenAIClient(
                api_key=openai_key,
                api_url=openai_url,
                model=model,
                dev_mode=dev_mode,
                logger=logger,
                temperature=temperature,
                top_p=top_p
            )
        elif provider.lower() == "anthropic":
            from backend.lm_clients.claude_client import ClaudeClient
            return ClaudeClient(
                api_key=claude_key,
                api_url=claude_url,
                model=model,
                dev_mode=dev_mode,
                logger=logger,
                temperature=temperature,
                top_p=top_p
            )
        elif provider.lower() == "perplexity":
            from backend.lm_clients.perplexity_client import PerplexityClient
            return PerplexityClient(
                api_key=perplexity_key,
                api_url=perplexity_url,
                model=model,
                dev_mode=dev_mode,
                logger=logger,
                temperature=temperature,
                top_p=top_p
            )
        else:
            raise ValueError(f"Unsupported ephemeral LLM provider: '{provider}'.")
    
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
            
            # Send request to the corruptor LLM
            logger.info(f"Generating spurious motivation for {source}-[{rel_type}]->{target}")
            response = self.corruptor_llm.send_message(
                sys_prompt=sys_prompt,
                usr_prompt=usr_prompt,
                stream=False,
                seed=self.corruptor_seed
            )
            
            # Parse the response
            data = self.corruptor_llm.parse_static_response(response)
            content = data['content']
            
            if not content.strip():
                logger.warning("Empty spurious motivation from LLM; using fallback text.")
                return "This relationship is supported by flawed reasoning that incorrectly assumes causality."
            
            return content.strip()
            
        except Exception as e:
            logger.info(f"Error generating spurious motivation: {e}")
            # If generation fails, add a generic spurious statement
            return "This relationship is supported by flawed reasoning that incorrectly assumes causality."
    
    def _build_llm_client(
        self,
        provider: str,
        model: str,
        dev_mode: bool,
        temperature: float,
        top_p: float,
        openai_key: str,
        openai_url: str,
        claude_key: str,
        claude_url: str,
        perplexity_key: str,
        perplexity_url: str,
        role_name: str = "",
        enable_web_search: bool = False
    ):
        """
        Build and return the appropriate LLM client instance based on provider + model.
        """
        # Example provider checks
        if provider == "openai":
            return OpenAIClient(
                api_key=openai_key,
                api_url=openai_url or "https://api.openai.com/v1/chat/completions",  # Default URL if None
                model=model,
                dev_mode=dev_mode,
                logger=logging.getLogger(f'app.llm.openai.{role_name}'),
                temperature=temperature,
                top_p=top_p,
                enable_websearch=enable_web_search
            )
        elif provider == "anthropic":
            return ClaudeClient(
                api_key=claude_key,
                api_url=claude_url,
                model=model,
                dev_mode=dev_mode,
                logger=logging.getLogger(f'app.llm.anthropic.{role_name}'),
                temperature=temperature,
                top_p=top_p
            )
        elif provider == "perplexity":
            return PerplexityClient(
                api_key=perplexity_key,
                api_url=perplexity_url,
                model=model,
                dev_mode=dev_mode,
                logger=logging.getLogger(f'app.llm.perplexity.{role_name}'),
                temperature=temperature,
                top_p=top_p
            )
        else:
            raise ValueError(f"Unsupported provider '{provider}' for LLM role: {role_name}.")

    def _is_openai_compatible_model(self, model: str) -> bool:
        """
        Check if a model should use OpenAI-compatible client (e.g., local Ollama models).
        
        Returns True if:
        - OPENAI_API_URL is set to a localhost URL (local LLM mode)
        - Model name contains known local LLM identifiers (qwen, llama, mistral, etc.)
        """
        openai_api_url = os.environ.get("OPENAI_API_URL", "")
        model_lower = model.lower()
        
        # Check for local LLM mode via environment
        is_local_url = openai_api_url.startswith("http://localhost") or openai_api_url.startswith("http://127.0.0.1")
        
        # Check for known local LLM model names
        local_model_patterns = ["qwen", "llama", "mistral", "phi", "gemma", "deepseek", "yi-", "vicuna"]
        is_local_model = any(pattern in model_lower for pattern in local_model_patterns)
        
        return is_local_url or is_local_model

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
        self, source: str, target: str, rel_type: str, spurious_motivation: str, original_motivation: str = ""
    ) -> bool:
        """Update an edge with spurious motivation in the Neo4j database."""
        try:
            with self.graph_db._get_session() as session:
                query = """
                MATCH (s:variable {name: $source_name})-[r]->(t:variable {name: $target_name})
                WHERE type(r) = $rel_type AND s.session_id = $session_id AND t.session_id = $session_id
                SET r.motivation = $spurious_motivation,
                    r.spurious_motivation = $spurious_motivation, 
                    r.original_motivation = $original_motivation,
                    r.is_corrupted = true,
                    r.corruption_type = 'motivation',
                    r.corruption_subtype = 'spurious_motivation'
                RETURN count(r) as updated_count
                """
                params = {
                    "source_name": source,
                    "target_name": target,
                    "rel_type": rel_type,
                    "session_id": self.session_id,
                    "spurious_motivation": spurious_motivation,
                    "original_motivation": original_motivation
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
            response = self.corruptor_llm.send_message(sys_prompt, usr_prompt, stream=False, seed=self.corruptor_seed)
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
        
    def _convert_none_to_spurious(self, source: str, target: str) -> bool:
        """Convert a NONE edge into a spurious causal edge (false positive)."""
        try:
            # Generate spurious causal relationship using LLM with YAML prompt
            sys_prompt, usr_prompt, _ = self.tree.build_prompt(
                id="generateSpuriousEdge",
                variable_dict={
                    'source': source,
                    'target': target
                }
            )
            
            response = self.corruptor_llm.send_message(sys_prompt, usr_prompt, stream=False, seed=self.corruptor_seed)
            data = self.corruptor_llm.parse_static_response(response)
            content = data["content"].strip()
            
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            edge_data = json.loads(content)
            new_rel_type = edge_data["NewRelationship"]
            spurious_motivation = edge_data["Explanation"]
            
            # Delete NONE edge and create new causal edge
            with self.graph_db._get_session() as session:
                # Get original motivation from NONE edge before deleting
                original_mot_result = session.run("""
                    MATCH (s:variable {name: $source, session_id: $session_id})
                          -[r:NONE]->
                          (t:variable {name: $target, session_id: $session_id})
                    RETURN r.motivation as original_motivation
                    """, {"source": source, "target": target, "session_id": self.session_id}).single()
                original_motivation = original_mot_result["original_motivation"] if original_mot_result else ""
                
                session.run("""
                    MATCH (s:variable {name: $source, session_id: $session_id})
                          -[r:NONE]->
                          (t:variable {name: $target, session_id: $session_id})
                    DELETE r
                    """, {"source": source, "target": target, "session_id": self.session_id})
                
                session.run(f"""
                    MATCH (s:variable {{name: $source, session_id: $session_id}})
                    MATCH (t:variable {{name: $target, session_id: $session_id}})
                    CREATE (s)-[r:{new_rel_type}]->(t)
                    SET r.motivation = $motivation,
                        r.spurious_motivation = $motivation,
                        r.original_motivation = $original_motivation,
                        r.corruption_type = 'structural',
                        r.corruption_subtype = 'spurious_edge',
                        r.is_corrupted = TRUE,
                        r.session_id = $session_id
                    """, {"source": source, "target": target, "session_id": self.session_id, 
                          "motivation": spurious_motivation, "original_motivation": original_motivation})
            
            logger.info(f"Converted NONE to spurious: {source}-[{new_rel_type}]->{target}")
            return True
        except Exception as e:
            logger.error(f"Error converting NONE edge {source}->{target}: {e}")
            return False
    
    def _flip_edge_polarity(self, source: str, target: str, old_type: str) -> bool:
        """Flip POSITIVE ↔ NEGATIVE polarity of an edge."""
        try:
            new_type = "NEGATIVE" if old_type == "POSITIVE" else "POSITIVE"
            
            # Generate spurious motivation for flipped polarity using YAML prompt
            sys_prompt, usr_prompt, _ = self.tree.build_prompt(
                id="generateFlippedPolarity",
                variable_dict={
                    'source': source,
                    'target': target,
                    'old_type': old_type,
                    'new_type': new_type
                }
            )
            
            response = self.corruptor_llm.send_message(sys_prompt, usr_prompt, stream=False, seed=self.corruptor_seed)
            data = self.corruptor_llm.parse_static_response(response)
            content = data["content"].strip()
            
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            edge_data = json.loads(content)
            spurious_motivation = edge_data["Explanation"]
            
            # Delete old edge and create new one with flipped polarity
            with self.graph_db._get_session() as session:
                # Get original motivation before deleting old edge
                original_mot_result = session.run(f"""
                    MATCH (s:variable {{name: $source, session_id: $session_id}})
                          -[r:{old_type}]->
                          (t:variable {{name: $target, session_id: $session_id}})
                    RETURN r.motivation as original_motivation
                    """, {"source": source, "target": target, "session_id": self.session_id}).single()
                original_motivation = original_mot_result["original_motivation"] if original_mot_result else ""
                
                session.run(f"""
                    MATCH (s:variable {{name: $source, session_id: $session_id}})
                          -[r:{old_type}]->
                          (t:variable {{name: $target, session_id: $session_id}})
                    DELETE r
                    """, {"source": source, "target": target, "session_id": self.session_id})
                
                session.run(f"""
                    MATCH (s:variable {{name: $source, session_id: $session_id}})
                    MATCH (t:variable {{name: $target, session_id: $session_id}})
                    CREATE (s)-[r:{new_type}]->(t)
                    SET r.motivation = $motivation,
                        r.spurious_motivation = $motivation,
                        r.original_motivation = $original_motivation,
                        r.corruption_type = 'structural',
                        r.corruption_subtype = 'polarity_flip',
                        r.original_polarity = $old_type,
                        r.is_corrupted = TRUE,
                        r.session_id = $session_id
                    """, {"source": source, "target": target, "old_type": old_type, 
                         "session_id": self.session_id, "motivation": spurious_motivation,
                         "original_motivation": original_motivation})
            
            logger.info(f"Flipped polarity: {source}-[{old_type}→{new_type}]->{target}")
            return True
        except Exception as e:
            logger.error(f"Error flipping edge {source}-[{old_type}]->{target}: {e}")
            return False
    
    def _corrupt_single_edge(self, source: str, target: str, rel_type: str) -> dict:
        """Helper function to corrupt a single edge (for parallel execution)."""
        try:
            original_motivation = self._get_edge_motivation(source, target, rel_type)
            if not original_motivation:
                return {"success": False, "reason": "No motivation found"}
            
            # Generate a spurious motivation
            spurious_motivation = self.generate_spurious_motivation(
                source, target, rel_type, original_motivation
            )
            
            if not spurious_motivation:
                return {"success": False, "reason": "Failed to generate spurious motivation"}
            
            # Update in the database
            success = self._update_edge_with_spurious_motivation(
                source, target, rel_type, spurious_motivation, original_motivation
            )
            
            if not success:
                return {"success": False, "reason": "Failed to update database"}
            
            return {
                "success": True,
                "source": source,
                "target": target,
                "type": rel_type,
                "original_motivation": original_motivation,
                "spurious_motivation": spurious_motivation
            }
        except Exception as e:
            logger.error(f"Error corrupting {source}-[{rel_type}]->{target}: {e}")
            return {"success": False, "reason": str(e)}

    def _corrupt_relationships(self, corruption_rate: float, parallel: bool = True, max_workers: int = 10, seed: int = None) -> int:
        """
        Corrupt a fraction of the relationships with THREE types of corruption:
        1. Motivation corruption: Replace explanation with spurious text
        2. Spurious new edges: Convert NONE → POSITIVE/NEGATIVE (false positive)
        3. Polarity flip: Flip POSITIVE ↔ NEGATIVE
        
        For each edge to corrupt, randomly select ONE corruption type.
        
        Args:
            corruption_rate: Fraction of edges to corrupt (0-1), e.g. 0.3 = 30% total
            parallel: Whether to corrupt in parallel
            max_workers: Number of parallel workers
            seed: Random seed for reproducible corruption (optional)
        
        Returns:
            The number of relationships actually corrupted.
        """
        try:
            # Set random seed for reproducible corruption
            if seed is not None:
                random.seed(seed)
                logger.info(f"Starting mixed corruption with rate {corruption_rate} (parallel={parallel}, seed={seed})")
            else:
                logger.info(f"Starting mixed corruption with rate {corruption_rate} (parallel={parallel})")
            
            # Get all relationships for this session
            with self.graph_db._get_session() as session:
                query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $session_id AND t.session_id = $session_id
                RETURN s.name as source, t.name as target, type(r) as rel_type
                """
                
                results = list(session.run(query, {"session_id": self.session_id}))
                logger.info(f"Found {len(results)} total relationships")
                
                # Group edges by type for STRATIFIED sampling (balanced classes)
                none_edges = [e for e in results if e["rel_type"] == "NONE"]
                causal_edges = [e for e in results if e["rel_type"] in ["POSITIVE", "NEGATIVE"]]
                
                logger.info(f"Edge distribution: {len(none_edges)} NONE, {len(causal_edges)} causal")
                
                # Calculate how many edges to corrupt with balanced classes
                # Target: equal numbers of spurious_edge, polarity_flip, and motivation
                num_to_corrupt = int(len(results) * corruption_rate)
                target_per_class = num_to_corrupt // 3
                
                # Sample edges to ensure balanced corruption types:
                # - spurious_edge: sample from NONE edges
                # - polarity_flip + motivation: sample from causal edges (split 50/50)
                
                random.shuffle(none_edges)
                random.shuffle(causal_edges)
                
                # Sample for balanced classes
                spurious_sample = none_edges[:min(target_per_class, len(none_edges))]
                causal_sample = causal_edges[:min(target_per_class * 2, len(causal_edges))]
                
                edges_to_corrupt = spurious_sample + causal_sample
                actual_total = len(edges_to_corrupt)
                
                logger.info(f"Stratified sampling: {len(spurious_sample)} NONE (→spurious), {len(causal_sample)} causal (→flip/motivation)")
                logger.info(f"Planning to corrupt {actual_total} edges (target ~{target_per_class} per class)")
                
                # Categorize edges by type for corruption assignment
                motivation_candidates = []  # Can apply motivation corruption
                spurious_candidates = []    # NONE edges (can convert to causal)
                flip_candidates = []        # POSITIVE/NEGATIVE edges (can flip polarity)
                
                for edge in edges_to_corrupt:
                    rel_type = edge["rel_type"]
                    if rel_type == "NONE":
                        spurious_candidates.append(edge)
                    elif rel_type in ["POSITIVE", "NEGATIVE"]:
                        flip_candidates.append(edge)
                        motivation_candidates.append(edge)
                    else:
                        motivation_candidates.append(edge)
                
                logger.info(f"Corruption candidates: {len(motivation_candidates)} motivation, {len(spurious_candidates)} spurious, {len(flip_candidates)} flip")
                
                # Assign corruption types with BALANCED distribution
                # Goal: Split corruption budget equally across 3 types
                corruption_assignments = []
                target_per_type = num_to_corrupt // 3
                remainder = num_to_corrupt % 3
                
                # Count how many of each type we want
                target_counts = {
                    "spurious_edge": target_per_type + (1 if remainder > 0 else 0),
                    "polarity_flip": target_per_type + (1 if remainder > 1 else 0),
                    "motivation": target_per_type
                }
                
                # Track actual assignments
                assigned_counts = {"spurious_edge": 0, "polarity_flip": 0, "motivation": 0}
                
                # Shuffle edges to randomize which ones get which corruption
                edges_shuffled = edges_to_corrupt.copy()
                random.shuffle(edges_shuffled)
                
                for edge in edges_shuffled:
                    rel_type = edge["rel_type"]
                    
                    # Determine available corruption types for this edge
                    available_types = []
                    if rel_type == "NONE":
                        available_types = ["spurious_edge"]
                    elif rel_type in ["POSITIVE", "NEGATIVE"]:
                        available_types = ["polarity_flip", "motivation"]
                    else:
                        available_types = ["motivation"]
                    
                    # Pick the corruption type that's most under-represented
                    # (among the available types for this edge)
                    eligible_types = [
                        t for t in available_types 
                        if assigned_counts[t] < target_counts[t]
                    ]
                    
                    if not eligible_types:
                        # If all targets met, fall back to any available type
                        eligible_types = available_types
                    
                    # Choose type with lowest count (to balance)
                    corruption_type = min(eligible_types, key=lambda t: assigned_counts[t])
                    assigned_counts[corruption_type] += 1
                    
                    corruption_assignments.append({
                        "source": edge["source"],
                        "target": edge["target"],
                        "rel_type": edge["rel_type"],
                        "corruption_type": corruption_type
                    })
                
                # Count by type
                motivation_count = sum(1 for a in corruption_assignments if a["corruption_type"] == "motivation")
                spurious_count = sum(1 for a in corruption_assignments if a["corruption_type"] == "spurious_edge")
                flip_count = sum(1 for a in corruption_assignments if a["corruption_type"] == "polarity_flip")
                
                logger.info(f"Corruption distribution: {motivation_count} motivation, {spurious_count} spurious, {flip_count} polarity flips")
                
                # Apply corruptions (parallel or sequential)
                corrupted_count = 0
                corrupted_data = {"count": 0, "corrupted_edges": []}
                
                def apply_single_corruption(assignment):
                    """Wrapper to apply one corruption (for parallel execution)."""
                    source = assignment["source"]
                    target = assignment["target"]
                    rel_type = assignment["rel_type"]
                    corruption_type = assignment["corruption_type"]
                    
                    try:
                        if corruption_type == "motivation":
                            result = self._corrupt_single_edge(source, target, rel_type)
                            return result if result["success"] else None
                        
                        elif corruption_type == "spurious_edge":
                            success = self._convert_none_to_spurious(source, target)
                            return {"success": success, "type": "spurious_edge"} if success else None
                        
                        elif corruption_type == "polarity_flip":
                            success = self._flip_edge_polarity(source, target, rel_type)
                            return {"success": success, "type": "polarity_flip"} if success else None
                    
                    except Exception as e:
                        logger.error(f"Error applying {corruption_type} to {source}->{target}: {e}")
                        return None
                
                # Execute in parallel or sequential
                if parallel and len(corruption_assignments) > 1:
                    logger.info(f"🚀 Parallel corruption with {max_workers} workers")
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(apply_single_corruption, assignment) 
                                 for assignment in corruption_assignments]
                        
                        for future in as_completed(futures):
                            result = future.result()
                            if result and result.get("success"):
                                if "original_motivation" in result:  # Motivation corruption
                                    corrupted_data["corrupted_edges"].append(result)
                                corrupted_count += 1
                else:
                    logger.info("Sequential corruption")
                    for assignment in corruption_assignments:
                        result = apply_single_corruption(assignment)
                        if result and result.get("success"):
                            if "original_motivation" in result:  # Motivation corruption
                                corrupted_data["corrupted_edges"].append(result)
                            corrupted_count += 1
                
                corrupted_data["count"] = corrupted_count
                
                # Save metadata
                if corrupted_count > 0:
                    metadata_file = f"corrupted_relationships_{self.session_id}.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(corrupted_data, f, indent=2)
                    self.corrupted_relationships_data = corrupted_data
            
            logger.info(f"✅ Total corruption: {corrupted_count}/{num_to_corrupt} edges ({corrupted_count/len(results)*100:.1f}% of all edges)")
            
            return corrupted_count
        
        except Exception as e:
            logger.error(f"Error corrupting relationships: {e}")
            return 0

    def _add_spurious_structural_edges(self, corruption_rate: float, parallel: bool = False, max_workers: int = 10) -> int:
        """
        Add entirely spurious edges to the graph (structural corruption).
        Two types:
        1. New edges between random variable pairs (not in ground truth)
        2. Existing ground truth edges but with wrong polarity (POSITIVE<->NEGATIVE)
        
        Args:
            corruption_rate: Fraction of edges to corrupt
            parallel: Whether to corrupt in parallel (currently sequential for LLM-guided corruption)
            max_workers: Number of parallel workers (reserved for future use)
        
        Returns:
            The number of structural corruptions added.
        """
        try:
            logger.info(f"Starting structural corruption with rate {corruption_rate}")
            
            with self.graph_db._get_session() as session:
                # Get all variables in this session
                var_query = """
                MATCH (v:variable)
                WHERE v.session_id = $session_id
                RETURN v.name as var_name
                """
                variables = [rec["var_name"] for rec in session.run(var_query, {"session_id": self.session_id})]
                logger.info(f"Found {len(variables)} variables for structural corruption")
                
                # Get all existing edges (to avoid duplicates)
                edge_query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $session_id AND t.session_id = $session_id
                RETURN s.name as source, t.name as target, type(r) as rel_type
                """
                existing_edges = set()
                existing_edge_details = []
                for rec in session.run(edge_query, {"session_id": self.session_id}):
                    existing_edges.add((rec["source"], rec["target"]))
                    existing_edge_details.append(rec)
                
                logger.info(f"Found {len(existing_edges)} existing edges")
                
                # Calculate corruption budget based on CAUSAL edges only
                # (for polarity flips) and NONE edges (for false positives)
                none_edges_initial = [e for e in existing_edge_details if e["rel_type"] == "NONE"]
                polar_edges_initial = [e for e in existing_edge_details if e["rel_type"] in ["POSITIVE", "NEGATIVE"]]
                
                # New edges: corrupt a percentage of NONE edges (convert to false causal)
                new_edge_budget = int(len(none_edges_initial) * corruption_rate)
                # Polarity flips: corrupt a percentage of existing causal edges
                flip_polarity_budget = int(len(polar_edges_initial) * corruption_rate)
                total_budget = new_edge_budget + flip_polarity_budget
                
                logger.info(f"Structural corruption budget: {total_budget} total ({new_edge_budget} new edges, {flip_polarity_budget} polarity flips)")
                
                corrupted_count = 0
                
                # Type 1: Create spurious edges by converting NONE edges to causal edges using LLM
                # Find NONE edges that can be converted to false positive causal relationships
                none_edges = [e for e in existing_edge_details if e["rel_type"] == "NONE"]
                
                attempts = 0
                max_attempts = min(new_edge_budget * 3, len(none_edges) + new_edge_budget)
                
                while corrupted_count < new_edge_budget and attempts < max_attempts:
                    attempts += 1
                    
                    # Use LLM to convert a NONE edge into a spurious causal edge
                    try:
                        # Get causal edges for context (not NONE)
                        causal_edges_str = "\n".join([
                            f"- {e['source']} -[{e['rel_type']}]-> {e['target']}" 
                            for e in existing_edge_details if e['rel_type'] != 'NONE'
                        ])
                        
                        # Get NONE edges as candidates
                        none_edges_str = "\n".join([
                            f"{i+1}. {e['source']} -[NONE]-> {e['target']}"
                            for i, e in enumerate(none_edges)
                        ])
                        
                        sys_prompt = """You are a causal discovery system that sometimes makes false positive inferences (hallucinations).
Your task is to convert a correctly identified "NONE" relationship into a spurious but plausible causal relationship (POSITIVE or NEGATIVE).
This represents the system incorrectly inferring causality where none actually exists."""
                        
                        usr_prompt = f"""Given these variables in a CLD about '{self.target_variable}':
{', '.join(variables)}

Existing causal relationships:
{causal_edges_str}

Correctly identified NONE relationships (no causal link):
{none_edges_str}

Select ONE NONE relationship and convert it into a plausible but FALSE causal relationship (false positive).

Respond in JSON format:
{{
  "NoneEdgeNumber": number (1-{len(none_edges)}),
  "NewRelationship": "POSITIVE or NEGATIVE",
  "Explanation": "2-3 sentence plausible but scientifically incorrect motivation for why you think there IS a causal relationship"
}}"""
                        
                        response = self.corruptor_llm.send_message(sys_prompt, usr_prompt, stream=False, seed=self.corruptor_seed)
                        data = self.corruptor_llm.parse_static_response(response)
                        content = data["content"].strip()
                        
                        # Parse JSON response
                        import json
                        if content.startswith("```json"):
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif content.startswith("```"):
                            content = content.split("```")[1].split("```")[0].strip()
                        
                        edge_data = json.loads(content)
                        edge_idx = edge_data["NoneEdgeNumber"] - 1
                        rel_type = edge_data["NewRelationship"]
                        spurious_motivation = edge_data["Explanation"]
                        
                        # Validate
                        if edge_idx < 0 or edge_idx >= len(none_edges):
                            logger.warning(f"Invalid NONE edge number: {edge_idx + 1}")
                            continue
                        
                        edge = none_edges[edge_idx]
                        source = edge["source"]
                        target = edge["target"]
                        
                        if rel_type not in ["POSITIVE", "NEGATIVE"]:
                            logger.warning(f"Invalid relationship type: {rel_type}")
                            continue
                        if len(spurious_motivation) < 20:
                            logger.warning(f"Motivation too short")
                            continue
                        
                        logger.info(f"LLM converting NONE to spurious causal edge: {source}-[NONE→{rel_type}]->{target}")
                        
                    except Exception as e:
                        logger.error(f"Error generating LLM-powered spurious edge: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # Delete NONE edge and create spurious causal edge
                    convert_query = f"""
                    MATCH (s:variable {{name: $source, session_id: $session_id}})-[old_r:NONE]->(t:variable {{name: $target, session_id: $session_id}})
                    DELETE old_r
                    CREATE (s)-[r:{rel_type}]->(t)
                    SET r.motivation = $motivation,
                        r.is_corrupted = true,
                        r.corruption_type = 'structural',
                        r.corruption_subtype = 'new_edge',
                        r.original_relationship = 'NONE',
                        r.session_id = $session_id
                    RETURN r
                    """
                    result = session.run(convert_query, {
                        "source": source,
                        "target": target,
                        "session_id": self.session_id,
                        "motivation": spurious_motivation
                    })
                    
                    if result.single():
                        corrupted_count += 1
                        # Remove from none_edges list so it can't be converted again
                        none_edges.pop(edge_idx)
                        logger.info(f"Created LLM-generated spurious edge (false positive): {source}-[{rel_type}]->{target}")
                
                # Type 2: Flip polarity of existing edges using LLM
                # Find edges with polarity (POSITIVE or NEGATIVE)
                polar_edges = [e for e in existing_edge_details if e["rel_type"] in ["POSITIVE", "NEGATIVE"]]
                
                flipped = 0
                attempts = 0
                max_flip_attempts = flip_polarity_budget * 3
                
                while flipped < flip_polarity_budget and attempts < max_flip_attempts and polar_edges:
                    attempts += 1
                    
                    # Use LLM to decide which edge to flip and generate spurious motivation
                    try:
                        # Build list of polar edges for LLM
                        edges_str = "\n".join([
                            f"{i+1}. {e['source']} -[{e['rel_type']}]-> {e['target']}"
                            for i, e in enumerate(polar_edges)
                        ])
                        
                        sys_prompt = """You are a causal discovery system that sometimes makes plausible but incorrect polarity assignments (hallucinations).
Your task is to identify an edge whose polarity should be flipped, and provide a plausible but scientifically incorrect justification for the opposite polarity."""
                        
                        usr_prompt = f"""Given these existing causal relationships in a CLD about '{self.target_variable}':

{edges_str}

Select ONE edge whose polarity you would flip (POSITIVE ↔ NEGATIVE), and provide a plausible but FALSE motivation for why the flipped polarity is correct.

Respond in JSON format:
{{
  "EdgeNumber": number (1-{len(polar_edges)}),
  "NewPolarity": "POSITIVE or NEGATIVE (opposite of current)",
  "Explanation": "2-3 sentence plausible but scientifically incorrect motivation for the new polarity"
}}"""
                        
                        response = self.corruptor_llm.send_message(sys_prompt, usr_prompt, stream=False, seed=self.corruptor_seed)
                        data = self.corruptor_llm.parse_static_response(response)
                        content = data["content"].strip()
                        
                        # Parse JSON response
                        import json
                        if content.startswith("```json"):
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif content.startswith("```"):
                            content = content.split("```")[1].split("```")[0].strip()
                        
                        flip_data = json.loads(content)
                        edge_idx = flip_data["EdgeNumber"] - 1  # Convert to 0-based index
                        new_type = flip_data["NewPolarity"]
                        spurious_motivation = flip_data["Explanation"]
                        
                        # Validate
                        if edge_idx < 0 or edge_idx >= len(polar_edges):
                            logger.warning(f"Invalid edge number: {edge_idx + 1}")
                            continue
                        
                        edge = polar_edges[edge_idx]
                        source = edge["source"]
                        target = edge["target"]
                        old_type = edge["rel_type"]
                        
                        # Validate new polarity is actually flipped
                        expected_flip = "NEGATIVE" if old_type == "POSITIVE" else "POSITIVE"
                        if new_type != expected_flip:
                            logger.warning(f"LLM didn't flip polarity correctly: {old_type} -> {new_type}, expected {expected_flip}")
                            new_type = expected_flip  # Force correct flip
                        
                        if len(spurious_motivation) < 20:
                            logger.warning(f"Motivation too short")
                            continue
                        
                        logger.info(f"LLM selected edge to flip: {source}-[{old_type}→{new_type}]->{target}")
                        
                    except Exception as e:
                        logger.error(f"Error in LLM-powered polarity flip: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # Delete old edge and create new one with flipped polarity
                    flip_query = f"""
                    MATCH (s:variable {{name: $source, session_id: $session_id}})-[old_r:{old_type}]->(t:variable {{name: $target, session_id: $session_id}})
                    DELETE old_r
                    CREATE (s)-[new_r:{new_type}]->(t)
                    SET new_r.motivation = $motivation,
                        new_r.is_corrupted = true,
                        new_r.corruption_type = 'structural',
                        new_r.corruption_subtype = 'polarity_flip',
                        new_r.original_polarity = $old_type,
                        new_r.session_id = $session_id
                    RETURN new_r
                    """
                    result = session.run(flip_query, {
                        "source": source,
                        "target": target,
                        "session_id": self.session_id,
                        "motivation": spurious_motivation,
                        "old_type": old_type
                    })
                    
                    if result.single():
                        flipped += 1
                        corrupted_count += 1
                        # Remove this edge from the list so it can't be flipped again
                        polar_edges.pop(edge_idx)
                        logger.info(f"LLM-powered polarity flip: {source}-[{old_type}→{new_type}]->{target}")
                
                logger.info(f"Structural corruption complete: {corrupted_count} total ({corrupted_count - flipped} new edges, {flipped} polarity flips)")
                return corrupted_count
                
        except Exception as e:
            logger.error(f"Error in structural corruption: {e}")
            import traceback
            traceback.print_exc()
            return 0
        

    def run(
    self,
    num_variables: int = 5,
    generation_parallel: bool = False,
    generation_max_workers: int = 3,
    corruption_rate: float = 0.0,
    corruption_seed: int = None,
    run_judgment: bool = False,
    judge_models: Optional[Union[List[str], str]] = None,
    num_judges: Optional[int] = None, 
    starting_variables: Optional[List[str]] = None
) -> Dict:
        """
        Run the causal discovery algorithm end-to-end with optional serial judging.
        
        Args:
            num_variables: Number of variables to generate
            generation_parallel: Whether to discover relationships in parallel
            generation_max_workers: Maximum number of worker threads for parallel relationship discovery
            corruption_rate: Rate of corrupting relationships with spurious motivations
            corruption_seed: Random seed for reproducible corruption (optional)
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
                generation_parallel=generation_parallel,
                generation_max_workers=generation_max_workers,
                corruption_rate=corruption_rate
            )
            logger.info(f"Discovered {len(relationships)} relationships")
            
            # If corruption_rate > 0, we can do a separate pass to randomly corrupt more edges
            # (optional depending on your use case)
            corrupted_count = 0
            if corruption_rate > 0:
                # This will randomly corrupt additional edges by overwriting them
                corrupted_count = self._corrupt_relationships(corruption_rate, seed=corruption_seed)
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
                    # Send request. Skip logprobs for models that forbid it (e.g., GPT-5)
                    judge_model_name = getattr(self.judge_llm, "model", "").lower()
                    judge_supports_logprobs = not judge_model_name.startswith("gpt-5")
                    if judge_supports_logprobs:
                        response = self.judge_llm.send_message(
                            sys_prompt=sys_prompt,
                            usr_prompt=usr_prompt,
                            stream=False,
                            extra_body={"logprobs": True, "top_logprobs": 5},
                        )
                    else:
                        response = self.judge_llm.send_message(
                            sys_prompt=sys_prompt,
                            usr_prompt=usr_prompt,
                            stream=False,
                        )
                    data = self.judge_llm.parse_static_response(response)
                    # compute judge-side logprob metrics
                    judge_log_data = data.get("logprobs") or {}
                    judge_log_metrics = _compute_and_trace_metrics_from_logprobs(judge_log_data)
                    llm_content = (data.get("content") or "").strip()
                except Exception as e:
                    logger.error(f"Error calling LLM for edge {src}->{tgt}: {e}")
                    judge_log_data = {}
                    judge_log_metrics = {"avg_nll": None, "perplexity": None, "min_prob": None, "max_window_entropy": None}
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
                # Persist judge-side logprob metrics
                try:
                    self.update_relationship_properties(
                        source=src,
                        target=tgt,
                        relationship_type=rel_type,
                        properties={
                            "judge_perplexity": judge_log_metrics.get("perplexity"),
                            "judge_min_prob": judge_log_metrics.get("min_prob"),
                            "judge_max_window_entropy": judge_log_metrics.get("max_window_entropy"),
                            # New metrics from arXiv:2405.19648v1
                            "judge_mean_token_prob": judge_log_metrics.get("mean_token_prob"),
                            "judge_prob_variance": judge_log_metrics.get("prob_variance"),
                            "judge_prob_std": judge_log_metrics.get("prob_std"),
                            "judge_max_prob_diff": judge_log_metrics.get("max_prob_diff"),
                            # Temporal slope metric
                            "judge_token_prob_slope": judge_log_metrics.get("token_prob_slope"),
                        }
                    )
                except Exception:
                    pass

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

    def fill_in_missing_citations(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        search_provider: Optional[str] = None,  # NEW: Allow search provider selection
        citation_fill_parallel: bool = True,
        citation_fill_max_workers: int = 5
    ) -> None:
        """
        For every edge missing citations, ask the LLM for evidence and parse references 
        via the "citations" array from parse_static_response(). If none found, use search provider
        to find citations, or treat as NO_EVIDENCE.
        
        Args:
            llm_provider: LLM provider to use for generating responses
            llm_model: Specific model to use
            search_provider: Search provider to use ('brave', 'perplexity', or None for LLM-only)
        """
        import os
        import logging

        logger = logging.getLogger(__name__)
        if logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)

        # Environment variables
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
        OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "your-claude-api-key")
        CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
        PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "your-perplexity-api-key")
        PERPLEXITY_API_URL = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/v1/chat/completions")
        BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "your-brave-search-api-key")

        # Skip LLM if provider is None (direct search mode)
        if llm_provider is None:
            logger.info("[CITATION-FILL] Skipping LLM, using search_provider=%s directly", search_provider)
            llm = None
            chosen_provider = None
            chosen_model = None
        else:
            #
            # 1) Decide which model to use. (Fallback to judge_llm if none given.)
            #
            judge_llm = getattr(self, "judge_llm", None)
            chosen_provider = llm_provider or getattr(judge_llm, "provider", None)
            chosen_model    = llm_model    or getattr(judge_llm, "model",    None)
            
            if not chosen_provider or not chosen_model:
                raise ValueError(
                    "No judge model selected. Please provide llm_provider and llm_model, "
                    "or ensure judge_llm is configured with provider and model attributes."
                )

            logger.info("[CITATION-FILL] Using provider=%s model=%s search_provider=%s", 
                       chosen_provider, chosen_model, search_provider)

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
        # 3) Initialize search client if needed
        #
        search_client = None
        if search_provider == "brave":
            search_client = self._build_brave_search_client(BRAVE_SEARCH_API_KEY, logger)
        elif search_provider == "perplexity":
            # Use existing perplexity client for search if available
            search_client = llm if chosen_provider == "perplexity" else None

        #
        # 4) Find edges lacking citations
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
            "You are an expert literature-search assistant. Provide max 3 reutable scientific URLs "
            "that support the following claim, plus a brief explanation of relevance (1–2 sentences). "
            "If you find no evidence, reply exactly NO_EVIDENCE.\n"
            "Output must be in one of two forms:\n\n"
            "NO_EVIDENCE\n\n"
            "-- or --\n\n"
            "CITATION: <URL>\n"
            "RELEVANCE: <short statement>\n"
        )

        #
        # 6) Helper method to process a single edge citation (for parallelization)
        #
        def _process_single_citation(record, sys_prompt, llm, search_client, search_provider, logger):
            """Process citation filling for a single edge with retry logic."""
            import time
            
            src   = record["src"]
            tgt   = record["tgt"]
            rel   = record["rel_type"]
            props = record.get("edge_props") or {}

            # Use spurious_motivation if available, else fallback
            claim_txt = props.get("spurious_motivation") or props.get("motivation", "")
            claim_txt = claim_txt.strip()
            if not claim_txt:
                logger.debug("[CITATION-FILL] Edge %s-%s->%s has no motivation text; skipping.", src, rel, tgt)
                return None

            # Build user prompt
            usr_prompt = f"CLAIM:\n{claim_txt}\n\nAnswer now."

            valid_citations = []
            max_retries = 3
            base_delay = 2.0
            
            # If LLM is available, try it first with retry logic
            if llm is not None:
                for attempt in range(max_retries):
                    try:
                        # Call the LLM
                        response = llm.send_message(
                            sys_prompt=sys_prompt,
                            usr_prompt=usr_prompt,
                            stream=False
                        )

                        # Parse result
                        data = llm.parse_static_response(response)
                        content   = (data.get("content") or "").strip()
                        citations = data.get("citations", [])

                        logger.debug("[CITATION-FILL] LLM raw content: %s", content)
                        logger.debug("[CITATION-FILL] LLM extracted citations: %r", citations)

                        # Filter out empty/invalid citations
                        if citations:
                            for citation in citations:
                                if citation and isinstance(citation, str) and citation.strip():
                                    citation = citation.strip()
                                    if citation.startswith(('http://', 'https://')):
                                        valid_citations.append(citation)
                                    elif citation.startswith('www.'):
                                        valid_citations.append('https://' + citation)
                        
                        logger.debug("[CITATION-FILL] Valid citations after filtering: %r", valid_citations)
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        error_str = str(e).lower()
                        # Check if it's a rate limit error
                        if 'rate limit' in error_str or '429' in error_str or 'too many requests' in error_str:
                            if attempt < max_retries - 1:
                                # Exponential backoff: 2s, 4s, 8s
                                delay = base_delay * (2 ** attempt)
                                logger.warning("[CITATION-FILL] Rate limit hit for %s-%s->%s, retrying in %.1fs (attempt %d/%d)", 
                                             src, rel, tgt, delay, attempt + 1, max_retries)
                                time.sleep(delay)
                                continue
                            else:
                                logger.error("[CITATION-FILL] Max retries exceeded for %s-%s->%s due to rate limits", src, rel, tgt)
                        else:
                            logger.exception("[CITATION-FILL] Edge %s-%s->%s => LLM call failed: %s", src, rel, tgt, e)
                        break

            # If no valid citations from LLM (or LLM was skipped), try search provider
            if not valid_citations and search_client:
                logger.info("[CITATION-FILL] %s citations for %s-%s->%s, trying %s search", 
                           "No LLM" if llm is None else "No valid LLM", src, rel, tgt, search_provider)
                search_citations = self._search_for_citations(
                    search_client, search_provider, claim_txt, src, tgt, rel, logger
                )
                if search_citations:
                    valid_citations = search_citations

            # If still no citations, return None
            if not valid_citations:
                logger.info("[CITATION-FILL] Edge %s-%s->%s => No citations found from %s", 
                           src, rel, tgt, "search" if llm is None else "LLM or search")
                return None

            return (src, tgt, rel, valid_citations)

        #
        # 7) Process edges in parallel (or sequentially if citation_fill_parallel=False)
        #
        if citation_fill_parallel and len(records) > 1:
            logger.info("[CITATION-FILL] Processing %d edges in PARALLEL with %d workers", len(records), citation_fill_max_workers)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=citation_fill_max_workers) as executor:
                # Submit all citation filling tasks
                future_to_edge = {
                    executor.submit(_process_single_citation, record, sys_prompt, llm, search_client, search_provider, logger): record
                    for record in records
                }
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_edge):
                    try:
                        result = future.result()
                        if result:
                            src, tgt, rel, valid_citations = result
                            self._update_edge_citation_info(src, tgt, rel, valid_citations, "Auto-generated relevance here")
                            logger.info("[CITATION-FILL] Patched %s-%s->%s => CITATION=%r", src, rel, tgt, valid_citations)
                    except Exception as e:
                        logger.exception("[CITATION-FILL] Task failed with exception: %s", e)
        else:
            # Sequential processing (fallback for single edge or if parallel disabled)
            logger.info("[CITATION-FILL] Processing %d edges SEQUENTIALLY", len(records))
            for record in records:
                result = _process_single_citation(record, sys_prompt, llm, search_client, search_provider, logger)
                if result:
                    src, tgt, rel, valid_citations = result
                    self._update_edge_citation_info(src, tgt, rel, valid_citations, "Auto-generated relevance here")
                    logger.info("[CITATION-FILL] Patched %s-%s->%s => CITATION=%r", src, rel, tgt, valid_citations)

        logger.info("[CITATION-FILL] Completed fill_in_missing_citations().")

    def _build_brave_search_client(self, api_key: str, logger):
        """
        Build Brave Search client using the same pattern as the platform implementation.
        """
        try:
            # Import the SearchEngine and CitationWhitelist from root directory
            import sys
            import os
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in sys.path:
                sys.path.append(root_dir)
            from search_engine_copy import SearchEngine
            from citation_whitelist_copy import CitationWhitelist
            
            # Enable citation whitelist filtering for academic sources
            citation_whitelist = CitationWhitelist()
            trusted_domains = citation_whitelist.get_domains()
            # trusted_domains = []  # Empty list = no domain filtering (WRONG - blocks everything!)
            
            # Create SearchEngine instance
            search_client = SearchEngine(
                api_key=api_key,
                trusted_domains=trusted_domains,
                logger=logger
            )
            
            logger.info("[CITATION-FILL] Brave Search client initialized WITHOUT domain filtering (whitelist disabled)")
            return search_client
            
        except ImportError as e:
            logger.error("[CITATION-FILL] Failed to import Brave Search components: %s", e)
            logger.error("[CITATION-FILL] Make sure platform backend modules are available")
            return None
        except Exception as e:
            logger.error("[CITATION-FILL] Failed to initialize Brave Search client: %s", e)
            return None

    def _search_for_citations(self, search_client, search_provider: str, claim_txt: str, 
                             src: str, tgt: str, rel: str, logger) -> list:
        """
        Use search provider to find citations for a claim.
        
        Args:
            search_client: The search client instance
            search_provider: The search provider type ('brave' or 'perplexity')
            claim_txt: The claim text to search for
            src: Source variable name
            tgt: Target variable name  
            rel: Relationship type
            logger: Logger instance
            
        Returns:
            List of citation URLs
        """
        try:
            if search_provider == "brave":
                return self._search_with_brave(search_client, claim_txt, src, tgt, rel, logger)
            elif search_provider == "perplexity":
                return self._search_with_perplexity(search_client, claim_txt, src, tgt, rel, logger)
            else:
                logger.warning("[CITATION-FILL] Unknown search provider: %s", search_provider)
                return []
                
        except Exception as e:
            logger.error("[CITATION-FILL] Search failed for %s-%s->%s: %s", src, rel, tgt, e)
            return []

    def _search_with_brave(self, search_client, claim_txt: str, src: str, tgt: str, rel: str, logger) -> list:
        """
        Search for citations using Brave Search API.
        """
        try:
            # Build search query combining the variables and claim
            search_query = f"{src} {rel} {tgt} {claim_txt}"
            
            # Truncate to respect Brave API limits (50 words AND 400 characters)
            words = search_query.split()
            if len(words) > 50:
                search_query = ' '.join(words[:50]) + "..."
            
            # Also check character limit
            if len(search_query) > 400:
                search_query = search_query[:397] + "..."
            
            logger.debug("[CITATION-FILL] Brave search query: %s", search_query)
            
            # Perform search
            search_results = search_client.search(
                query=search_query,
                result_count=5,  # Get up to 5 results
                raw=False  # Get processed results (only trusted domains)
            )
            
            # Extract URLs from search results
            citations = []
            if search_results:
                citations = [result.get('url', '') for result in search_results if result.get('url')]
                citations = [url for url in citations if url.strip()]  # Remove empty URLs
            
            logger.info("[CITATION-FILL] Brave search found %d citations for %s-%s->%s", 
                       len(citations), src, rel, tgt)
            return citations
            
        except Exception as e:
            logger.error("[CITATION-FILL] Brave search error: %s", e)
            return []

    def _search_with_perplexity(self, search_client, claim_txt: str, src: str, tgt: str, rel: str, logger) -> list:
        """
        Search for citations using Perplexity's built-in search (if using Perplexity as LLM).
        """
        try:
            # Build a search-focused prompt
            search_prompt = (
                f"Find scientific evidence for the claim: {claim_txt}\n"
                f"Focus on the relationship between {src} and {tgt}.\n"
                f"Provide only URLs of reputable scientific sources."
            )
            
            # Use Perplexity's web search capabilities
            response = search_client.send_message(
                sys_prompt="You are a scientific literature search assistant. Provide only URLs.",
                usr_prompt=search_prompt,
                stream=False
            )
            
            # Parse response for URLs
            data = search_client.parse_static_response(response)
            content = data.get("content", "")
            citations = data.get("citations", [])
            
            # If no structured citations, try to extract URLs from content
            if not citations and content:
                import re
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                found_urls = re.findall(url_pattern, content)
                citations = found_urls[:5]  # Limit to 5 URLs
            
            logger.info("[CITATION-FILL] Perplexity search found %d citations for %s-%s->%s", 
                       len(citations), src, rel, tgt)
            return citations
            
        except Exception as e:
            logger.error("[CITATION-FILL] Perplexity search error: %s", e)
            return []


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


    def fill_in_missing_citations_with_hints(
        self,
        hint_map: Dict[Tuple[str, str], List[str]],
        search_provider: str = "semantic_scholar",
        max_urls_per_citation: int = 1,
        use_trusted_domains: bool = False
    ) -> int:
        """
        Fill citations using full bibliographic references as search hints.
        
        This method is designed for expert-validated CLDs where we have full
        bibliographic citations (author, year, title, journal, DOI) and want
        to find web URLs for them.
        
        Args:
            hint_map: Dictionary mapping (source, target) tuples to lists of full citations.
                      Each citation should be a complete bibliographic reference.
                      Example: {("Var1", "Var2"): ["Author, A. (2020). Title. Journal. DOI:10.xxx"]}
            search_provider: Search provider to use:
                - 'semantic_scholar': Semantic Scholar API (free, academic-focused, best for title search)
                - 'pubmed': PubMed E-utilities (free, medical papers, best for biomedical)
                - 'perplexity': Perplexity AI (paid, general academic search)
                - 'brave': Brave Search (needs API key, general web search)
                - None: DOI-only mode
            max_urls_per_citation: Maximum URLs to keep per citation (default 1)
            use_trusted_domains: Whether to filter search results by trusted academic domains (default False)
            
        Search strategy for each full citation:
        1. Extract DOI if present -> construct https://doi.org/... URL directly
        2. Extract paper title -> search "{title}" for highest precision  
        3. Fallback to author+year search if needed
        
        Returns:
            Number of edges updated with citations
        """
        import os
        import re
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("[CITATION-FILL-HINTS] Starting citation filling with bibliographic hints")
        logger.info(f"[CITATION-FILL-HINTS] Processing {len(hint_map)} edges with hints")
        
        # Initialize search client if needed
        search_client = None
        
        if search_provider == "brave":
            BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "your-brave-search-api-key")
            
            try:
                # Import search engine from root directory (same as _build_brave_search_client)
                import sys
                import os
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if root_dir not in sys.path:
                    sys.path.append(root_dir)
                
                from search_engine_copy import SearchEngine
                
                # Configure domain filtering based on parameter
                if use_trusted_domains:
                    from citation_whitelist_copy import CitationWhitelist
                    citation_whitelist = CitationWhitelist()
                    trusted_domains = citation_whitelist.get_domains()
                    logger.info("[CITATION-FILL-HINTS] Brave Search with trusted domain filtering enabled")
                else:
                    trusted_domains = []  # Empty list = no domain filtering
                    logger.info("[CITATION-FILL-HINTS] Brave Search with domain filtering DISABLED")
                
                # Create SearchEngine
                search_client = SearchEngine(
                    api_key=BRAVE_SEARCH_API_KEY,
                    trusted_domains=trusted_domains,
                    logger=logger
                )
                logger.info("[CITATION-FILL-HINTS] Brave Search initialized successfully")
                
            except ImportError as e:
                logger.error(f"[CITATION-FILL-HINTS] Failed to import Brave Search: {e}")
                search_client = None
            except Exception as e:
                logger.error(f"[CITATION-FILL-HINTS] Failed to initialize Brave Search: {e}")
                search_client = None
            
            if not search_client:
                logger.warning("[CITATION-FILL-HINTS] Brave Search unavailable, will use DOI-only mode")
                
        elif search_provider == "perplexity":
            # Use Perplexity for LLM-based academic search
            PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "your-perplexity-api-key")
            PERPLEXITY_API_URL = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/v1/chat/completions")
            
            try:
                search_client = self._build_llm_client(
                    provider="perplexity",
                    model="sonar",  # Perplexity's search-optimized model
                    dev_mode=getattr(self, "dev_mode", False),
                    temperature=0.1,  # Low temperature for factual search
                    openai_key="",
                    openai_url="",
                    claude_key="",
                    claude_url="",
                    perplexity_key=PERPLEXITY_API_KEY,
                    perplexity_url=PERPLEXITY_API_URL,
                    top_p=1.0,
                    role_name="citation_search"
                )
                logger.info("[CITATION-FILL-HINTS] Perplexity AI search client initialized")
            except Exception as e:
                logger.error(f"[CITATION-FILL-HINTS] Failed to initialize Perplexity: {e}")
                search_client = None
                logger.warning("[CITATION-FILL-HINTS] Perplexity unavailable, will use DOI-only mode")
                
        elif search_provider == "semantic_scholar":
            # Semantic Scholar API - no initialization needed, just use requests
            search_client = "semantic_scholar"  # Marker that we're using Semantic Scholar
            logger.info("[CITATION-FILL-HINTS] Using Semantic Scholar API (free, no auth required)")
            
        elif search_provider == "pubmed":
            # PubMed E-utilities - no initialization needed, just use requests
            search_client = "pubmed"  # Marker that we're using PubMed
            logger.info("[CITATION-FILL-HINTS] Using PubMed E-utilities API (free, no auth required)")
            
        elif search_provider == "openalex":
            # OpenAlex API - no initialization needed, just use requests
            search_client = "openalex"  # Marker that we're using OpenAlex
            logger.info("[CITATION-FILL-HINTS] Using OpenAlex API (free, 10 req/sec with polite pool)")
        
        edges_updated = 0
        total_citations_found = 0
        doi_links = 0
        search_links = 0
        
        # Process each edge in the hint map
        for edge_key, full_citations in hint_map.items():
            # Handle both tuple keys (src, tgt) and string keys "src -> tgt"
            if isinstance(edge_key, tuple):
                src, tgt = edge_key
            elif isinstance(edge_key, str) and " -> " in edge_key:
                src, tgt = edge_key.split(" -> ", 1)
            else:
                logger.warning(f"[CITATION-FILL-HINTS] Invalid edge key format: {edge_key}, skipping")
                continue
                
            if not full_citations:
                logger.debug(f"[CITATION-FILL-HINTS] No hints for {src} -> {tgt}, skipping")
                continue
                
            logger.info(f"[CITATION-FILL-HINTS] Processing edge: {src} -> {tgt} ({len(full_citations)} citations)")
            
            # Collect URLs for this edge (1 URL per citation)
            edge_urls = []
            
            for citation in full_citations:
                # Track if we found a URL for this citation
                citation_url_found = False
                # Handle both old format (string) and new format (dict with short/full)
                if isinstance(citation, dict):
                    short_cite = citation.get("short", "")
                    full_cite = citation.get("full", "")
                    confidence = citation.get("confidence", 0.0)
                    citation_text = full_cite if full_cite else short_cite
                elif isinstance(citation, str):
                    short_cite = citation
                    full_cite = citation
                    citation_text = citation
                    confidence = 1.0
                else:
                    continue
                    
                if not citation_text or not citation_text.strip():
                    continue
                
                citation_text = citation_text.strip()
                logger.debug(f"[CITATION-FILL-HINTS] Processing citation (conf={confidence:.2f}): {citation_text[:100]}...")
                
                # Strategy 1: Extract DOI and create direct URL
                # Common DOI patterns: doi:10.xxx, DOI: 10.xxx, doi.org/10.xxx
                doi_match = re.search(r'(?:doi[:\s]+|doi\.org/)(\S*10\.\S+)', citation_text, re.IGNORECASE)
                if doi_match:
                    doi = doi_match.group(1).strip()
                    # Clean up DOI (remove trailing punctuation)
                    doi = re.sub(r'[.,;]$', '', doi)
                    doi_url = f"https://doi.org/{doi}"
                    edge_urls.append(doi_url)
                    doi_links += 1
                    citation_url_found = True
                    logger.info(f"[CITATION-FILL-HINTS] Found DOI for citation: {doi_url}")
                    continue  # Move to next citation
                
                # Strategy 2: Extract title from full citation and search for it
                if search_client:
                    # Try to extract title (usually between year and journal/volume)
                    # Pattern 1: Author (Year). Title. Journal...
                    # Pattern 2: Author Last Name. Title. Journal Year;vol
                    title = None
                    author = None
                    year = None
                    
                    # First try standard pattern: Author (Year). Title. Journal
                    title_match = re.search(r'\(\d{4}[a-z]?\)\.\s*([^.]+)\.', citation_text)
                    if title_match:
                        title = title_match.group(1).strip()
                        # Extract author and year from full citation
                        author_year_match = re.search(r'^([A-Za-zÀ-ÿ\s&\'-]+?)[,\s]+.*?\((\d{4}[a-z]?)\)', citation_text)
                        if author_year_match:
                            author = author_year_match.group(1).strip().split()[0]
                            year = author_year_match.group(2).strip()
                    
                    # If no title found, try extracting from short citation
                    if not title and short_cite:
                        # Extract author and year from short citation
                        short_match = re.match(r'^([A-Za-zÀ-ÿ\s&\'-]+?)\s*\((\d{4}[a-z]?)\)', short_cite.strip())
                        if short_match:
                            author = short_match.group(1).strip()
                            year = short_match.group(2).strip()
                    
                    if title:
                        logger.debug(f"[CITATION-FILL-HINTS] Extracted title: {title[:80]}...")
                        
                        # Search for the title
                        try:
                            if search_provider == "brave":
                                # Brave Search
                                search_results = search_client.search(
                                    query=title,
                                    result_count=1,
                                    raw=False
                                )
                                
                                if search_results:
                                    url = search_results[0].get('url', '').strip()
                                    if url:
                                        edge_urls.append(url)
                                        search_links += 1
                                        citation_url_found = True
                                        logger.info(f"[CITATION-FILL-HINTS] Found via Brave title: {url}")
                                        continue  # Move to next citation
                                    
                            elif search_provider == "semantic_scholar":
                                # Semantic Scholar API search
                                found_urls = self._search_semantic_scholar(title, author, year, logger)
                                if found_urls:
                                    # Take only the first (best) URL
                                    url = found_urls[0]
                                    edge_urls.append(url)
                                    search_links += 1
                                    citation_url_found = True
                                    logger.info(f"[CITATION-FILL-HINTS] Found via Semantic Scholar title: {url}")
                                    continue  # Move to next citation
                                    
                            elif search_provider == "pubmed":
                                # PubMed API search
                                found_urls = self._search_pubmed(title, author, year, logger)
                                if found_urls:
                                    # Take only the first (best) URL
                                    url = found_urls[0]
                                    edge_urls.append(url)
                                    search_links += 1
                                    citation_url_found = True
                                    logger.info(f"[CITATION-FILL-HINTS] Found via PubMed title: {url}")
                                    continue  # Move to next citation
                                    
                            elif search_provider == "openalex":
                                # OpenAlex API search
                                found_urls = self._search_openalex(title, author, year, logger)
                                if found_urls:
                                    # Take only the first (best) URL
                                    url = found_urls[0]
                                    edge_urls.append(url)
                                    search_links += 1
                                    citation_url_found = True
                                    logger.info(f"[CITATION-FILL-HINTS] Found via OpenAlex title: {url}")
                                    continue  # Move to next citation
                                    
                            elif search_provider == "perplexity":
                                # Perplexity AI search
                                search_prompt = f"Find the URL for this academic paper: {title}\nProvide only the URL, no explanation."
                                response = search_client.send_message(
                                    sys_prompt="You are a scientific literature search assistant. Provide only paper URLs from academic sources like PubMed, DOI, arXiv, Google Scholar, etc.",
                                    usr_prompt=search_prompt,
                                    stream=False
                                )
                                
                                # Parse URLs from response
                                data = search_client.parse_static_response(response)
                                content = data.get("content", "")
                                citations_from_resp = data.get("citations", [])
                                
                                # Extract URLs from content and citations
                                import re as re_module
                                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                                found_urls = re_module.findall(url_pattern, content)
                                all_urls = citations_from_resp + found_urls
                                
                                if all_urls:
                                    # Take only the first (best) URL
                                    url = all_urls[0].strip()
                                    if url:
                                        edge_urls.append(url)
                                        search_links += 1
                                        citation_url_found = True
                                        logger.info(f"[CITATION-FILL-HINTS] Found via Perplexity title: {url}")
                                        continue  # Move to next citation
                                    
                        except Exception as e:
                            logger.warning(f"[CITATION-FILL-HINTS] Title search failed: {e}")
                    
                    # Strategy 3: Fallback to author+year search (only if no URL found yet)
                    if not citation_url_found and search_client and author and year:
                        search_query = f"{author} {year}"
                        logger.debug(f"[CITATION-FILL-HINTS] Fallback author/year search: {search_query}")
                        
                        try:
                            if search_provider == "brave":
                                # Brave Search
                                search_results = search_client.search(
                                    query=search_query,
                                    result_count=1,
                                    raw=False
                                )
                                
                                if search_results:
                                    url = search_results[0].get('url', '').strip()
                                    if url:
                                        edge_urls.append(url)
                                        search_links += 1
                                        logger.info(f"[CITATION-FILL-HINTS] Found via Brave author/year: {url}")
                                            
                            elif search_provider == "semantic_scholar":
                                # Semantic Scholar API fallback search
                                found_urls = self._search_semantic_scholar(None, author, year, logger)
                                if found_urls:
                                    edge_urls.append(found_urls[0])
                                    search_links += 1
                                    logger.info(f"[CITATION-FILL-HINTS] Found via Semantic Scholar author/year: {found_urls[0]}")
                                    
                            elif search_provider == "pubmed":
                                # PubMed API fallback search
                                found_urls = self._search_pubmed(None, author, year, logger)
                                if found_urls:
                                    edge_urls.append(found_urls[0])
                                    search_links += 1
                                    logger.info(f"[CITATION-FILL-HINTS] Found via PubMed author/year: {found_urls[0]}")
                                    
                            elif search_provider == "openalex":
                                # OpenAlex API fallback search
                                found_urls = self._search_openalex(None, author, year, logger)
                                if found_urls:
                                    edge_urls.append(found_urls[0])
                                    search_links += 1
                                    logger.info(f"[CITATION-FILL-HINTS] Found via OpenAlex author/year: {found_urls[0]}")
                                    
                            elif search_provider == "perplexity":
                                # Perplexity AI search
                                fallback_prompt = f"Find URL for academic paper by {author} published in {year}. Provide only the URL."
                                response = search_client.send_message(
                                    sys_prompt="You are a scientific literature search assistant. Provide only paper URLs from academic sources.",
                                    usr_prompt=fallback_prompt,
                                    stream=False
                                )
                                
                                # Parse URLs from response
                                data = search_client.parse_static_response(response)
                                content = data.get("content", "")
                                citations_from_resp = data.get("citations", [])
                                
                                # Extract URLs
                                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                                found_urls = re.findall(url_pattern, content)
                                all_urls = citations_from_resp + found_urls
                                
                                if all_urls:
                                    url = all_urls[0].strip()
                                    if url:
                                        edge_urls.append(url)
                                        search_links += 1
                                        logger.info(f"[CITATION-FILL-HINTS] Found via Perplexity author/year: {url}")
                                        
                        except Exception as e:
                            logger.warning(f"[CITATION-FILL-HINTS] Author/year search failed: {e}")
            
            # Update Neo4j edge with collected URLs
            if edge_urls:
                # Remove duplicates while preserving order
                edge_urls = list(dict.fromkeys(edge_urls))
                
                # Find the edge relationship type
                with self.graph_db._get_session() as sess:
                    result = sess.run(
                        """
                        MATCH (s:variable {name: $src, session_id: $sid})-[r]->(t:variable {name: $tgt, session_id: $sid})
                        RETURN type(r) as rel_type
                        LIMIT 1
                        """,
                        {"src": src, "tgt": tgt, "sid": self.session_id}
                    )
                    record = result.single()
                    if record:
                        rel_type = record["rel_type"]
                        self._update_edge_citation_info(src, tgt, rel_type, edge_urls, "Expert-validated citation")
                        edges_updated += 1
                        total_citations_found += len(edge_urls)
                        logger.info(f"[CITATION-FILL-HINTS] Updated {src} -[{rel_type}]-> {tgt} with {len(edge_urls)} citations")
                    else:
                        logger.warning(f"[CITATION-FILL-HINTS] Edge not found in Neo4j: {src} -> {tgt}")
            else:
                logger.warning(f"[CITATION-FILL-HINTS] No URLs found for {src} -> {tgt}")
        
        # Print summary
        logger.info("="*80)
        logger.info("[CITATION-FILL-HINTS] SUMMARY")
        logger.info(f"  Edges processed: {len(hint_map)}")
        logger.info(f"  Edges updated: {edges_updated}")
        logger.info(f"  Total citations found: {total_citations_found}")
        logger.info(f"  DOI links: {doi_links}")
        logger.info(f"  Search-found links: {search_links}")
        logger.info("="*80)
        
        return edges_updated

    def _search_semantic_scholar(self, title: str, author: str = None, year: str = None, logger=None) -> list:
        """
        Search for papers using Semantic Scholar API.
        API: https://api.semanticscholar.org/graph/v1/paper/search
        Docs: https://www.semanticscholar.org/product/api
        
        Rate Limit: ~100 requests per 5 minutes (VERY CONSERVATIVE: 1 request every 10 seconds)
        """
        import requests
        import time
        
        urls = []
        max_retries = 3
        retry_delay = 10  # Start with 10 seconds for rate limit hits, then exponential backoff
        
        # Rate limiting: Wait 10 seconds between calls to be VERY conservative
        # This gives us max ~30 requests per 5 minutes, well under the 100 limit
        if not hasattr(self, '_last_semantic_scholar_call'):
            self._last_semantic_scholar_call = 0
        
        time_since_last_call = time.time() - self._last_semantic_scholar_call
        if time_since_last_call < 10.0:
            sleep_time = 10.0 - time_since_last_call
            if logger:
                logger.debug(f"[SEMANTIC-SCHOLAR] Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        try:
            # Semantic Scholar search endpoint
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            
            # Build query
            query = title
            if author and year:
                query = f"{author} {year} {title}"
            
            params = {
                "query": query,
                "limit": 3,
                "fields": "title,url,externalIds,openAccessPdf"
            }
            
            for attempt in range(max_retries):
                self._last_semantic_scholar_call = time.time()
                response = requests.get(base_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    papers = data.get("data", [])
                    
                    for paper in papers:
                        # Try to get DOI first (most reliable)
                        external_ids = paper.get("externalIds", {})
                        if external_ids and external_ids.get("DOI"):
                            urls.append(f"https://doi.org/{external_ids['DOI']}")
                        elif external_ids and external_ids.get("PubMed"):
                            urls.append(f"https://pubmed.ncbi.nlm.nih.gov/{external_ids['PubMed']}/")
                        elif external_ids and external_ids.get("ArXiv"):
                            urls.append(f"https://arxiv.org/abs/{external_ids['ArXiv']}")
                        # Try open access PDF
                        elif paper.get("openAccessPdf") and paper["openAccessPdf"].get("url"):
                            urls.append(paper["openAccessPdf"]["url"])
                        # Finally, Semantic Scholar URL
                        elif paper.get("url"):
                            urls.append(paper["url"])
                    break  # Success, exit retry loop
                            
                elif response.status_code == 429:
                    if logger:
                        logger.warning(f"[SEMANTIC-SCHOLAR] Rate limit hit (attempt {attempt+1}/{max_retries}), waiting {retry_delay}s")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff: 10s → 20s → 40s
                    else:
                        if logger:
                            logger.error("[SEMANTIC-SCHOLAR] Max retries exceeded after rate limits")
                else:
                    if logger:
                        logger.warning(f"[SEMANTIC-SCHOLAR] HTTP {response.status_code}")
                    break  # Don't retry on other errors
                
        except Exception as e:
            if logger:
                logger.warning(f"[SEMANTIC-SCHOLAR] Search failed: {e}")
        
        return urls[:2]
    
    def _search_pubmed(self, title: str, author: str = None, year: str = None, logger=None) -> list:
        """
        Search for papers using PubMed E-utilities API.
        API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
        Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
        
        Rate Limit: Max 3 requests/second without API key (1 request every 0.34 seconds)
        """
        import requests
        import xml.etree.ElementTree as ET
        import time
        
        urls = []
        
        # Rate limiting: Wait 0.35 seconds between calls to respect 3 req/sec limit
        if not hasattr(self, '_last_pubmed_call'):
            self._last_pubmed_call = 0
        
        time_since_last_call = time.time() - self._last_pubmed_call
        if time_since_last_call < 0.35:
            sleep_time = 0.35 - time_since_last_call
            if logger:
                logger.debug(f"[PUBMED] Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        try:
            # PubMed ESearch to get PMIDs
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            
            # Build query
            query_parts = [title]
            if author:
                query_parts.append(f"AND {author}[Author]")
            if year:
                query_parts.append(f"AND {year}[PDAT]")
            
            query = " ".join(query_parts)
            
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": 3,
                "retmode": "json",
                "tool": "causalix_citation_refiller",
                "email": "research@causalix.ai"  # NCBI requires contact info
            }
            
            self._last_pubmed_call = time.time()
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                esearch_result = data.get("esearchresult", {})
                pmids = esearch_result.get("idlist", [])
                
                # Convert PMIDs to URLs
                for pmid in pmids[:2]:  # Top 2 results
                    urls.append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            else:
                if logger:
                    logger.warning(f"[PUBMED] HTTP {response.status_code}")
                        
        except Exception as e:
            if logger:
                logger.warning(f"[PUBMED] Search failed: {e}")
        
        return urls
    
    def _validate_url(self, url: str, logger=None, timeout: int = 5) -> bool:
        """
        Check if a URL is accessible (returns 200 or redirects to 200).
        
        Args:
            url: URL to validate
            logger: Logger instance
            timeout: Request timeout in seconds
            
        Returns:
            True if URL is accessible, False otherwise
        """
        import requests
        
        try:
            # Use HEAD request to be lightweight
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            
            # Consider 200, 301, 302, 307, 308 as valid
            if response.status_code in [200, 301, 302, 307, 308]:
                return True
            
            # Some sites block HEAD requests, try GET
            if response.status_code == 405 or response.status_code == 403:
                response = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
                if response.status_code == 200:
                    response.close()  # Close to avoid downloading full content
                    return True
            
            if logger:
                logger.debug(f"[URL-VALIDATION] {url} returned {response.status_code}")
            return False
            
        except requests.Timeout:
            if logger:
                logger.debug(f"[URL-VALIDATION] {url} timed out")
            return False
        except Exception as e:
            if logger:
                logger.debug(f"[URL-VALIDATION] {url} failed: {str(e)[:100]}")
            return False
    
    def _search_openalex(self, title: str, author: str = None, year: str = None, logger=None) -> list:
        """
        Search for papers using OpenAlex API.
        API: https://api.openalex.org/works
        Docs: https://docs.openalex.org/
        
        Rate Limit: 100,000 requests per day (polite pool - no auth needed)
        With mailto: 10 requests/second (much faster than Semantic Scholar)
        """
        import requests
        import time
        
        urls = []
        
        # Rate limiting: 10 req/sec with polite pool = 0.1s between calls
        if not hasattr(self, '_last_openalex_call'):
            self._last_openalex_call = 0
        
        time_since_last_call = time.time() - self._last_openalex_call
        if time_since_last_call < 0.11:
            sleep_time = 0.11 - time_since_last_call
            if logger:
                logger.debug(f"[OPENALEX] Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        try:
            # OpenAlex works search endpoint
            base_url = "https://api.openalex.org/works"
            
            # Build query - OpenAlex uses title.search for fuzzy matching
            query = f'title.search:"{title}"'
            if author:
                query += f' author.search:"{author}"'
            if year:
                query += f' publication_year:{year}'
            
            params = {
                "filter": query if not (author or year) else None,
                "search": title if (author or year) else None,
                "per_page": 3,
                "mailto": "research@causalix.ai"  # Polite pool - faster rate limits
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            self._last_openalex_call = time.time()
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                for work in results:
                    # Try to get DOI first (most reliable)
                    doi = work.get("doi")
                    if doi:
                        # DOI comes as full URL from OpenAlex
                        urls.append(doi)
                    # Try PMIDs
                    elif work.get("ids", {}).get("pmid"):
                        pmid = work["ids"]["pmid"].replace("https://pubmed.ncbi.nlm.nih.gov/", "")
                        urls.append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                    # Open access PDF
                    elif work.get("open_access", {}).get("oa_url"):
                        urls.append(work["open_access"]["oa_url"])
                    # OpenAlex landing page
                    elif work.get("id"):
                        urls.append(work["id"])
                        
            elif response.status_code == 429:
                if logger:
                    logger.warning("[OPENALEX] Rate limit hit (shouldn't happen with polite pool)")
            else:
                if logger:
                    logger.warning(f"[OPENALEX] HTTP {response.status_code}")
                        
        except Exception as e:
            if logger:
                logger.warning(f"[OPENALEX] Search failed: {e}")
        
        return urls[:2]


    
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

        # Check if we have pre-created judge LLMs (heterogeneous ensemble)
        if hasattr(self, 'judge_llms') and self.judge_llms:
            # Use pre-created judge LLM clients from __init__
            judges = self.judge_llms
            models = [getattr(judge, 'model', 'unknown') for judge in judges]
            print(f"\n{'='*50}")
            print(f"SERIAL JUDGING: Using {len(judges)} pre-created judge LLMs")
            print(f"SERIAL JUDGING: Using models: {models}")
            print(f"{'='*50}\n")
        else:
            # Legacy path: create judges from judge_models parameter
            # Set up the judge models
            if judge_models is None:
                raise ValueError(
                    "No judge model selected. Please provide judge_models parameter, "
                    "or ensure judge_llms is configured with a list of LLM clients."
                )
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
                api_url = os.environ.get("CLAUDE_API_URL", 'https://api.anthropic.com/v1/messages')
                judges.append(ClaudeClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.claude.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: Claude with model {model}")
            elif "gpt" in model.lower() or "openai" in model.lower() or self._is_openai_compatible_model(model):
                api_key = os.environ.get("OPENAI_API_KEY", 'None')
                api_url = os.environ.get("OPENAI_API_URL", 'None')
                judges.append(OpenAIClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.openai.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: OpenAI-compatible with model {model}")
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
    aggregator_message: str = None,
    judge_perplexity: float = None,
    judge_min_prob: float = None,
    judge_max_window_entropy: float = None,
    judge_mean_token_prob: float = None,
    judge_prob_variance: float = None,
    judge_prob_std: float = None,
    judge_max_prob_diff: float = None,
    judge_token_prob_slope: float = None
):
        """
        Update the edge in Neo4j with:
        - r.judge_verdict
        - r.judge_message
        - r.aggregate_score  (if provided, else skip)
        - r.aggregator_message (optional, if you want it)
        - r.judge_perplexity, r.judge_min_prob, r.judge_max_window_entropy (judge logit metrics)
        
        We keep the first five parameters the same as before, and add new
        optional parameters at the end.

        If any optional parameter is None, we skip setting it.
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
        if judge_perplexity is not None:
            set_clauses.append("r.judge_perplexity = $judge_perplexity")
        if judge_min_prob is not None:
            set_clauses.append("r.judge_min_prob = $judge_min_prob")
        if judge_max_window_entropy is not None:
            set_clauses.append("r.judge_max_window_entropy = $judge_max_window_entropy")
        if judge_mean_token_prob is not None:
            set_clauses.append("r.judge_mean_token_prob = $judge_mean_token_prob")
        if judge_prob_variance is not None:
            set_clauses.append("r.judge_prob_variance = $judge_prob_variance")
        if judge_prob_std is not None:
            set_clauses.append("r.judge_prob_std = $judge_prob_std")
        if judge_max_prob_diff is not None:
            set_clauses.append("r.judge_max_prob_diff = $judge_max_prob_diff")
        if judge_token_prob_slope is not None:
            set_clauses.append("r.judge_token_prob_slope = $judge_token_prob_slope")

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
            "aggregator_message": aggregator_message,
            "judge_perplexity": judge_perplexity,
            "judge_min_prob": judge_min_prob,
            "judge_max_window_entropy": judge_max_window_entropy,
            "judge_mean_token_prob": judge_mean_token_prob,
            "judge_prob_variance": judge_prob_variance,
            "judge_prob_std": judge_prob_std,
            "judge_max_prob_diff": judge_max_prob_diff,
            "judge_token_prob_slope": judge_token_prob_slope
        }

        # 4) Execute
        with self.graph_db._get_session() as session:
            session.run(query, params)

        print(
            f"UPDATED EDGE: ({src})-[:{rel_type}]->({tgt}) | "
            f"verdict='{verdict}' | "
            f"aggregate_score={aggregate_score} | "
            f"judge_perplexity={judge_perplexity} | "
            f"judge_min_prob={judge_min_prob} | "
            f"judge_max_window_entropy={judge_max_window_entropy}"
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

        # Check if we have pre-created judge LLMs (heterogeneous ensemble)
        if hasattr(self, 'judge_llms') and self.judge_llms:
            # Use pre-created judge LLM clients from __init__
            judges = self.judge_llms
            models = [getattr(judge, 'model', 'unknown') for judge in judges]
            print(f"\n{'='*50}")
            print(f"SERIAL JUDGING: Using {len(judges)} pre-created judge LLMs")
            print(f"SERIAL JUDGING: Using models: {models}")
            print(f"{'='*50}\n")
        else:
            # Legacy path: create judges from judge_models parameter
            # Set up the judge models
            if judge_models is None:
                raise ValueError(
                    "No judge model selected. Please provide judge_models parameter, "
                    "or ensure judge_llms is configured with a list of LLM clients."
                )
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
                api_url = os.environ.get("CLAUDE_API_URL", 'https://api.anthropic.com/v1/messages')
                judges.append(ClaudeClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.claude.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: Claude with model {model}")
            elif "gpt" in model.lower() or "openai" in model.lower() or self._is_openai_compatible_model(model):
                api_key = os.environ.get("OPENAI_API_KEY", 'None')
                api_url = os.environ.get("OPENAI_API_URL", 'None')
                judges.append(OpenAIClient(
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    dev_mode=getattr(self, 'dev_mode', False),
                    logger=logging.getLogger(f'app.llm.openai.judge{i}')
                ))
                print(f"SERIAL JUDGING: Created Judge {i+1}: OpenAI-compatible with model {model}")
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
    

    def _process_single_edge_judgment(
    self,
        idx: int,
        total_edges: int,
        record,
        ephemeral_judges,
        models,
        approach: str,
        citation_fetcher: str = "scraper",
        jina_timeout: int = 60,
        jina_retries: int = 2,
        jina_api_key: str = None,
        judge_config: Optional[Dict[str, Any]] = None  # ← Added for structured outputs support
    ) -> Dict:
        """
        Process a single edge judgment. This method is designed to be called
        either sequentially or in parallel from judge_all_edges_with_citations_serial.

        Args:
            idx: Index of the edge being processed (1-based)
            total_edges: Total number of edges to process
            record: Neo4j record containing edge data
            ephemeral_judges: List of LLM judge clients
            models: List of model names corresponding to ephemeral_judges
            approach: Judging approach ("all_citations_at_once", "per_citation_aggregate", "one_causal_source_found", or "correctness")

        Returns:
            Dictionary with judgment result or None if processing failed
        """
        import re
        import json
        import logging
        from collections import Counter

        logger = logging.getLogger(__name__)
        
        src = record["source"]
        tgt = record["target"]
        rel_type = record["rel_type"]
        props = record.get("props") or {}

        motivation_text = props.get("spurious_motivation", "") or props.get("motivation", "")
        motivation_text = motivation_text.strip()
        
        # Helper for partial verdict => numeric score (used by per_citation_aggregate)
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
        
        # Skip if no motivation
        if not motivation_text:
            verdict = "NO_MOTIVATION"
            reason = "Edge has no motivation to judge."
            print(f"\nJUDGING: Edge {idx}/{total_edges} => {src}-[{rel_type}]->{tgt} - {verdict}")
            self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
            # Still construct a claim based on edge structure
            if rel_type.upper() == "POSITIVE":
                no_mot_claim = f"{src} directly causes an increase in {tgt}"
            elif rel_type.upper() == "NEGATIVE":
                no_mot_claim = f"{src} directly causes a decrease in {tgt}"
            else:
                no_mot_claim = f"{src} directly causes changes in {tgt}"
            return {
                "source": src,
                "target": tgt,
                "type": rel_type,
                "judged_claim": no_mot_claim,
                "judge_verdict": verdict,
                "judge_message": reason
            }

        # ---------------------------------------------------------------
        # Handle different "approach" modes
        # ---------------------------------------------------------------
        if approach == "correctness":
            # ============================================================
            # APPROACH 3: correctness (no citations, evaluate reasoning quality)
            # ============================================================
            print(f"CORRECTNESS JUDGING: Evaluating overall correctness for {src}->{tgt}")
            
            # Build prompt using judgeCorrectness
            variables = {
                "source": src,
                "target": tgt,
                "relationship": rel_type,
                "motivation": motivation_text,
            }
            sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCorrectness", variables)
            
            # Each judge evaluates correctness
            judge_results = []
            judge_scores = []
            all_judge_logprobs = []  # Collect logprobs for CI metrics
            
            for j_idx, judge_client in enumerate(ephemeral_judges):
                model_name = models[j_idx]
                print(f"  Correctness Judge {j_idx+1} ({model_name})")
                
                try:
                    # Check if structured outputs should be used
                    use_structured = (
                        STRUCTURED_OUTPUTS_AVAILABLE and
                        judge_config and
                        judge_config.get("use_structured_outputs", False) and
                        JudgeResponseAdapter.supports_structured_outputs(model_name)
                    )
                    
                    # Check if this judge supports logprobs
                    judge_supports_logprobs = "gpt" in model_name.lower() or "o1" in model_name.lower() or "o3" in model_name.lower()
                    
                    if use_structured:
                        # Use structured outputs with Pydantic models
                        response = judge_client.send_message(
                            sys_prompt, usr_prompt, stream=False,
                            seed=self.judge_seed,
                            response_model=CorrectnessJudgment
                        )
                        
                        # Parse and convert to legacy format
                        judgment = JudgeResponseAdapter.parse_structured_response(response, CorrectnessJudgment)
                        result = JudgeResponseAdapter.to_legacy_format(judgment)
                        
                        verdict = result["verdict"]
                        score = result["score"]
                        reason = result["reason"]
                        
                        # Note: logprobs not available with structured outputs
                        judge_log_data = {}
                        
                        judge_results.append({
                            "judge_id": j_idx + 1,
                            "model": model_name,
                            "verdict": verdict,
                            "score": score,
                            "reason": reason
                        })
                        judge_scores.append(score)
                        print(f"    → {verdict} (score={score}) [structured]")
                        
                    else:
                        # Use existing regex parsing (backward compatible) with retry on parse failure
                        max_retries = 2
                        retry_count = 0
                        parsed_successfully = False
                        
                        # Extract initial max_tokens from judge_config (default: 5000)
                        initial_max_tokens = judge_config.get('max_tokens', 5000) if judge_config else 5000
                        
                        while retry_count <= max_retries and not parsed_successfully:
                            # Calculate current max_tokens (double on each retry)
                            current_max_tokens = initial_max_tokens * (2 ** retry_count)
                            if retry_count > 0:
                                print(f"    ⚠️  Retry {retry_count}/{max_retries} with max_tokens={current_max_tokens}")
                            
                            # Send request with max_tokens parameter
                            if judge_supports_logprobs:
                                response = judge_client.send_message(
                                    sys_prompt, usr_prompt, stream=False,
                                    max_tokens=current_max_tokens,
                                    seed=self.judge_seed,
                                    extra_body={"logprobs": True, "top_logprobs": 5}
                                )
                            else:
                                response = judge_client.send_message(
                                    sys_prompt, usr_prompt, stream=False,
                                    max_tokens=current_max_tokens,
                                    seed=self.judge_seed
                                )
                            
                            data = judge_client.parse_static_response(response)
                            llm_content = data.get("content", "").strip()
                            
                            # Extract logprobs for CI metrics
                            judge_log_data = data.get("logprobs") or {}
                            if judge_log_data:
                                all_judge_logprobs.append(judge_log_data)
                            
                            # Parse verdict and score (with CoT support)
                            # Strategy: Look for VERDICT/SCORE/REASON anywhere in the response
                            # This handles both:
                            # 1. Direct format: "VERDICT: ... SCORE: ... REASON: ..."
                            # 2. CoT format: "Step 1: ... Step 2: ... VERDICT: ... SCORE: ... REASON: ..."
                            verdict_match = re.search(r'VERDICT:\s*(\w+)', llm_content, re.IGNORECASE)
                            score_match = re.search(r'SCORE:\s*([\d.]+)', llm_content)
                            reason_match = re.search(r'REASON:\s*(.+?)(?:\n\n|$)', llm_content, re.DOTALL | re.IGNORECASE)
                            
                            if verdict_match and score_match:
                                # Success!
                                verdict = verdict_match.group(1).upper()
                                score = float(score_match.group(1))
                                reason = reason_match.group(1).strip() if reason_match else ""
                                
                                judge_results.append({
                                    "judge_id": j_idx + 1,
                                    "model": model_name,
                                    "verdict": verdict,
                                    "score": score,
                                    "reason": reason
                                })
                                judge_scores.append(score)
                                retry_suffix = f" (retry {retry_count})" if retry_count > 0 else ""
                                print(f"    → {verdict} (score={score}){retry_suffix}")
                                parsed_successfully = True
                            else:
                                # Parse failed
                                retry_count += 1
                                if retry_count > max_retries:
                                    # Final failure after all retries
                                    print(f"    → Error: Could not parse verdict/score after {max_retries} retries")
                                    judge_results.append({
                                        "judge_id": j_idx + 1,
                                        "model": model_name,
                                        "verdict": "ERROR",
                                        "score": 0.0,
                                        "reason": f"Parse error after {max_retries} retries: {llm_content[:100]}"
                                    })
                                    judge_scores.append(0.0)
                
                except Exception as e:
                    print(f"    → Error: {e}")
                    judge_results.append({
                        "judge_id": j_idx + 1,
                        "model": model_name,
                        "verdict": "ERROR",
                        "score": 0.0,
                        "reason": str(e)
                    })
                    judge_scores.append(0.0)
            
            # Aggregate: average score and majority verdict
            aggregate_score = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
            
            # Compute Judge CI metrics from all logprobs
            judge_perplexity = None
            judge_min_prob = None
            judge_max_window_entropy = None
            judge_mean_token_prob = None
            judge_prob_variance = None
            judge_prob_std = None
            judge_max_prob_diff = None
            judge_token_prob_slope = None
            
            if all_judge_logprobs:
                all_metrics = []
                for logprob_dict in all_judge_logprobs:
                    metrics = _compute_and_trace_metrics_from_logprobs(logprob_dict)
                    if metrics and metrics.get("perplexity") is not None:
                        all_metrics.append(metrics)

                if all_metrics:
                    # Core metrics (original 3)
                    avg_nlls = [m["avg_nll"] for m in all_metrics if m.get("avg_nll") is not None]
                    min_probs = [m["min_prob"] for m in all_metrics if m.get("min_prob") is not None]
                    max_entropies = [m["max_window_entropy"] for m in all_metrics if m.get("max_window_entropy") is not None]
                    
                    # Extended metrics (arXiv:2405.19648v1)
                    mean_token_probs = [m["mean_token_prob"] for m in all_metrics if m.get("mean_token_prob") is not None]
                    prob_variances = [m["prob_variance"] for m in all_metrics if m.get("prob_variance") is not None]
                    max_prob_diffs = [m["max_prob_diff"] for m in all_metrics if m.get("max_prob_diff") is not None]
                    token_prob_slopes = [m["token_prob_slope"] for m in all_metrics if m.get("token_prob_slope") is not None]

                    # Aggregate metrics with proper math:
                    # - Perplexity: average in log-space (avg_nll), then exp()
                    # - Min/Max metrics: take min/max, not average
                    # - Variance: average is acceptable approximation
                    # - Std: compute from averaged variance
                    if avg_nlls:
                        judge_avg_nll = sum(avg_nlls) / len(avg_nlls)
                        judge_perplexity = math.exp(judge_avg_nll)
                    if min_probs:
                        judge_min_prob = min(min_probs)  # Take minimum of all minimums
                    if max_entropies:
                        judge_max_window_entropy = max(max_entropies)  # Take maximum of all maxes
                    if mean_token_probs:
                        judge_mean_token_prob = sum(mean_token_probs) / len(mean_token_probs)
                    if prob_variances:
                        judge_prob_variance = sum(prob_variances) / len(prob_variances)
                        judge_prob_std = math.sqrt(judge_prob_variance)  # Std from averaged variance
                    if max_prob_diffs:
                        judge_max_prob_diff = max(max_prob_diffs)  # Take maximum of all maxes
                    if token_prob_slopes:
                        judge_token_prob_slope = sum(token_prob_slopes) / len(token_prob_slopes)
            
            # Majority voting on verdict
            verdict_counts = Counter([r["verdict"] for r in judge_results if r["verdict"] != "ERROR"])
            if verdict_counts:
                aggregate_verdict = verdict_counts.most_common(1)[0][0]
            else:
                aggregate_verdict = "ERROR"
            
            print(f"  AGGREGATE: verdict={aggregate_verdict}, score={aggregate_score:.3f}")
            
            # Store results
            judge_message_json = json.dumps({
                "approach": "correctness",
                "judge_results": judge_results,
                "aggregate_score": aggregate_score,
                "aggregate_verdict": aggregate_verdict
            })
            
            self._update_edge_with_judge_verdict(
                src, tgt, rel_type, aggregate_verdict, judge_message_json,
                aggregate_score=aggregate_score,
                judge_perplexity=judge_perplexity,
                judge_min_prob=judge_min_prob,
                judge_max_window_entropy=judge_max_window_entropy,
                judge_mean_token_prob=judge_mean_token_prob,
                judge_prob_variance=judge_prob_variance,
                judge_prob_std=judge_prob_std,
                judge_max_prob_diff=judge_max_prob_diff,
                judge_token_prob_slope=judge_token_prob_slope
            )
            
            return {
                "source": src,
                "target": tgt,
                "type": rel_type,
                "judge_verdict": aggregate_verdict,
                "judge_message": judge_message_json,
                "aggregate_score": aggregate_score
            }

        elif approach in ["all_citations_at_once", "per_citation_aggregate"]:
            # ============================================================
            # CITATION-BASED APPROACHES: Fetch and process citations first
            # ============================================================
            print(f"\nCITATION JUDGING: Edge {idx}/{total_edges} => {src}-[{rel_type}]->{tgt}")
            
            citations = props.get("citations", []) or props.get("citation", [])

            # Build the claim first (needed even for NO_CITATION edges for consistency)
            if motivation_text and motivation_text.lower() not in ["expert-validated", "expert validated", "n/a", "na"]:
                judged_claim_text = motivation_text
            else:
                # Construct explicit direct causal claim from edge details
                if rel_type.upper() == "POSITIVE":
                    judged_claim_text = f"{src} directly causes an increase in {tgt}"
                elif rel_type.upper() == "NEGATIVE":
                    judged_claim_text = f"{src} directly causes a decrease in {tgt}"
                else:
                    judged_claim_text = f"{src} directly causes changes in {tgt}"

            # Check if we have citations
            if not citations:
                verdict = "NO_CITATION"
                reason = "Edge has no citations to judge."
                print(f"CITATION JUDGING: {verdict} => {reason}")
                self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
                return {
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "judged_claim": judged_claim_text,
                    "judge_verdict": verdict,
                    "judge_message": reason
                }

            print(f"CITATION JUDGING: Found {len(citations)} citations; motivation length={len(motivation_text)}")

            # Fetch + process citations
            try:
                if citation_fetcher.lower() == "jina":
                    # Use Jina AI Reader ONLY (no fallback)
                    import logging
                    logger = logging.getLogger(__name__)
                    print(f"CITATION JUDGING: Using Jina AI Reader ONLY (timeout={jina_timeout}s, retries={jina_retries}, no fallback)")
                    citation_text_map = fetch_citations_with_jina(
                        citations=citations,
                        timeout=jina_timeout,
                        max_retries=jina_retries,
                        fallback_to_scraper=False,  # No fallback - test Jina AI alone
                        logger=logger,
                        api_key=jina_api_key
                    )
                elif citation_fetcher.lower() == "scraper":
                    # Use ContentScraper directly (with whitelist: only trusted domains)
                    if ContentScraper is None:
                        raise ImportError("ContentScraper is not available. Cannot proceed with citation fetching.")
                    print(f"CITATION JUDGING: Using ContentScraper with whitelist")
                    processor = ContentScraper(timeout=15, max_workers=3, max_content_length=10_000_000, pdf_extraction_mode="local", disable_agentic_fallback=True, access_all_domains=False)
                    citation_meta = processor.process_citations(citations, fetch_content=True)
                    scraper_results = processor.prepare_for_llm(citation_meta, include_metadata=True)
                    
                    # Filter out None values, None keys, and error messages from ContentScraper
                    citation_text_map = {}
                    if scraper_results:
                        for url, content in scraper_results.items():
                            # Skip None keys, None values, string "None", and invalid content
                            if url is None or url == "None" or not isinstance(url, str):
                                continue
                            if not url.startswith(('http://', 'https://')):
                                print(f"⚠️ ContentScraper returned invalid URL format: {url}")
                                continue
                            if content is None or not isinstance(content, str):
                                continue
                            if len(content) <= 200 or "[ERROR]" in content[:100]:
                                print(f"⚠️ ContentScraper returned invalid content for: {url} (content length: {len(content)})")
                                continue
                            citation_text_map[url] = content
                    
                    # Final safety check: remove any None keys that might have slipped through
                    citation_text_map = {k: v for k, v in citation_text_map.items() if k is not None and k != "None" and isinstance(k, str) and k.startswith(('http://', 'https://'))}
                    
                    print(f"CITATION JUDGING: ContentScraper returned {len(citation_text_map)}/{len(citations)} valid citations")
                    
                    # If no valid citations, return early with NO_CITATION verdict
                    if not citation_text_map:
                        verdict = "NO_CITATION"
                        reason = "All citations failed to fetch valid content from ContentScraper."
                        print(f"CITATION JUDGING: {verdict} => {reason}")
                        self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
                        # Construct judged_claim for consistency
                        if rel_type.upper() == "POSITIVE":
                            judged_claim_text = f"{src} directly causes an increase in {tgt}"
                        elif rel_type.upper() == "NEGATIVE":
                            judged_claim_text = f"{src} directly causes a decrease in {tgt}"
                        else:
                            judged_claim_text = f"{src} directly causes changes in {tgt}"
                        return {
                            "source": src,
                            "target": tgt,
                            "type": rel_type,
                            "judged_claim": judged_claim_text,
                            "judge_verdict": verdict,
                            "judge_message": reason
                        }
                else:
                    raise ValueError(f"Unknown citation_fetcher: {citation_fetcher}. Must be 'jina' or 'scraper'")
            except Exception as e:
                verdict = "ERROR"
                reason = f"Error fetching citations: {e}"
                print(f"CITATION JUDGING ERROR: {reason}")
                self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
                return {
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "judge_verdict": verdict,
                    "judge_message": reason,
                }

            # Now handle the specific citation-based approach
            if approach == "all_citations_at_once":
                # Similar logic to what was in the original function
                # (keeping the original implementation for this approach)
                    formatted_citations = ""
                    for url, content in citation_text_map.items():
                        # Double-check: skip None URLs (shouldn't happen after filtering, but be safe)
                        if url is None or url == "None" or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                            print(f"⚠️ Skipping invalid URL in citation_text_map: {url}")
                            continue
                        formatted_citations += f"\n\n--- CITATION: {url} ---\n{content}\n---"

                    # Build the actual causal claim to judge
                    # If motivation is just metadata (e.g., "Expert-validated"), construct the claim from edge details
                    if motivation_text and motivation_text.lower() not in ["expert-validated", "expert validated", "n/a", "na"]:
                        claim_text = motivation_text
                    else:
                        # Construct explicit direct causal claim from edge details
                        if rel_type.upper() == "POSITIVE":
                            claim_text = f"{src} directly causes an increase in {tgt}"
                        elif rel_type.upper() == "NEGATIVE":
                            claim_text = f"{src} directly causes a decrease in {tgt}"
                        else:
                            claim_text = f"{src} directly causes changes in {tgt}"
                    
                    variables = {
                        "source": src,
                        "target": tgt,
                        "relationship": rel_type,
                        "motivation": claim_text,
                        "citations": formatted_citations,
                    }
                    sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCitation", variables)

                    judge_results = []
                    for j_idx, judge_client in enumerate(ephemeral_judges):
                        model_name = models[j_idx]
                        print(f"[all_citations_at_once] => Judge {j_idx+1} ({model_name}) for {src}->{tgt}")
                        try:
                            response = judge_client.send_message(sys_prompt, usr_prompt, stream=False, seed=self.judge_seed)
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
                        final_verdict = "INCONSISTENT"
                        final_reason = "Tie => mark as inconsistent."
                    else:
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

                    self._update_edge_with_judge_verdict(src, tgt, rel_type, final_verdict, full_message)

                    return {
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "judged_text": motivation_text[:100] + "..." if len(motivation_text) > 100 else motivation_text,
                        "judged_claim": claim_text,
                        "judge_verdict": final_verdict,
                        "judge_message": full_message
                    }

            elif approach == "per_citation_aggregate":
                # Similar logic to what was in the original function
                # (keeping the original implementation for this approach)
                
                # Build the actual causal claim to judge BEFORE the loop so we can include it in results
                if motivation_text and motivation_text.lower() not in ["expert-validated", "expert validated", "n/a", "na"]:
                    claim_to_judge = motivation_text
                else:
                    # Construct explicit direct causal claim from edge details
                    if rel_type.upper() == "POSITIVE":
                        claim_to_judge = f"{src} directly causes an increase in {tgt}"
                    elif rel_type.upper() == "NEGATIVE":
                        claim_to_judge = f"{src} directly causes a decrease in {tgt}"
                    else:
                        claim_to_judge = f"{src} directly causes changes in {tgt}"
                
                partial_results = []
                all_judge_logprobs = []
                c_idx = 0

                for url, content in citation_text_map.items():
                        # Double-check: skip None URLs (shouldn't happen after filtering, but be safe)
                        if url is None or url == "None" or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                            print(f"⚠️ Skipping invalid URL in citation_text_map: {url}")
                            continue
                        
                        c_idx += 1
                        single_judge_outcomes = []
                        judge_logprobs_list = []
                        for j_idx, judge_client in enumerate(ephemeral_judges):
                            model_name = models[j_idx]
                            
                            # Build prompt from YAML using judgeCitation prompt_id
                            variables = {
                                "source": src,
                                "target": tgt,
                                "relationship": rel_type,
                                "motivation": claim_to_judge,
                                "citations": content,  # Single citation content
                            }
                            sys_prompt_single, usr_prompt_single, _ = self.tree.build_prompt("judgeCitation", variables)
                            
                            # Check if structured outputs should be used
                            use_structured = (
                                STRUCTURED_OUTPUTS_AVAILABLE and
                                judge_config and
                                judge_config.get("use_structured_outputs", False) and
                                JudgeResponseAdapter.supports_structured_outputs(model_name)
                            )
                            
                            try:
                                if use_structured:
                                    # Use structured outputs with Pydantic models
                                    resp = judge_client.send_message(
                                        sys_prompt_single, usr_prompt_single, 
                                        stream=False,
                                        seed=self.judge_seed,
                                        response_model=CitationJudgment
                                    )
                                    
                                    # Parse and convert to legacy format
                                    judgment = JudgeResponseAdapter.parse_structured_response(resp, CitationJudgment)
                                    result = JudgeResponseAdapter.to_legacy_format(judgment)
                                    
                                    v_str = result["verdict"]
                                    reason_str = result["reason"]
                                    
                                else:
                                    # Use existing regex parsing (backward compatible)
                                    model_name_lower = model_name.lower()
                                    judge_supports_logprobs = (
                                        not model_name_lower.startswith("gpt-5") and
                                        not model_name_lower.startswith("claude") and
                                        "claude" not in model_name_lower
                                    )
                                    if judge_supports_logprobs:
                                        resp = judge_client.send_message(
                                            sys_prompt_single, usr_prompt_single, 
                                            stream=False, seed=self.judge_seed, extra_body={"logprobs": True, "top_logprobs": 5}
                                        )
                                    else:
                                        resp = judge_client.send_message(sys_prompt_single, usr_prompt_single, stream=False, seed=self.judge_seed)
                                    data = judge_client.parse_static_response(resp)
                                    llm_content = data.get("content", "")
                                    judge_log_data = data.get("logprobs") or {}
                                    if judge_log_data:
                                        judge_logprobs_list.append(judge_log_data)
                                    
                                    # Parse verdict and reason (with CoT support)
                                    # This handles both direct format and CoT format with reasoning steps
                                    v_match = re.search(
                                        r"VERDICT:\s*(SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|UNADDRESSED)",
                                        llm_content,
                                        re.IGNORECASE
                                    )
                                    r_match = re.search(r"REASON:\s*(.+?)(?:\n\n|$)", llm_content, re.DOTALL | re.IGNORECASE)
                                    if v_match:
                                        v_str = v_match.group(1).upper()
                                    else:
                                        v_str = "UNADDRESSED"
                                    reason_str = r_match.group(1).strip() if r_match else ""
                                
                                single_judge_outcomes.append({"verdict": v_str, "reason": reason_str})
                                
                            except Exception as e:
                                print(f"Error single-citation => Judge {j_idx+1} ({model_name}): {e}")
                                v_str = "UNADDRESSED"
                                reason_str = f"LLM error: {str(e)}"
                                single_judge_outcomes.append({"verdict": v_str, "reason": reason_str})

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

                        all_judge_logprobs.extend(judge_logprobs_list)

                if len(partial_results) == 0:
                    aggregate_score = 0.0
                else:
                    sum_scores = sum(r["score"] for r in partial_results)
                    aggregate_score = sum_scores / len(partial_results)
                
                judge_perplexity = None
                judge_min_prob = None
                judge_max_window_entropy = None
                judge_mean_token_prob = None
                judge_prob_variance = None
                judge_prob_std = None
                judge_max_prob_diff = None
                judge_token_prob_slope = None
                if all_judge_logprobs:
                        all_metrics = []
                        for logprob_dict in all_judge_logprobs:
                            metrics = _compute_and_trace_metrics_from_logprobs(logprob_dict)
                            if metrics and metrics.get("perplexity") is not None:
                                all_metrics.append(metrics)

                        if all_metrics:
                            # Core metrics (need special handling for exponential values)
                            avg_nlls = [m["avg_nll"] for m in all_metrics if m.get("avg_nll") is not None]
                            min_probs = [m["min_prob"] for m in all_metrics if m.get("min_prob") is not None]
                            max_entropies = [m["max_window_entropy"] for m in all_metrics if m.get("max_window_entropy") is not None]
                            
                            # Extended metrics (arXiv:2405.19648v1)
                            mean_token_probs = [m["mean_token_prob"] for m in all_metrics if m.get("mean_token_prob") is not None]
                            prob_variances = [m["prob_variance"] for m in all_metrics if m.get("prob_variance") is not None]
                            prob_stds = [m["prob_std"] for m in all_metrics if m.get("prob_std") is not None]
                            max_prob_diffs = [m["max_prob_diff"] for m in all_metrics if m.get("max_prob_diff") is not None]
                            token_prob_slopes = [m["token_prob_slope"] for m in all_metrics if m.get("token_prob_slope") is not None]

                            # Aggregate metrics with proper math:
                            # - Perplexity: average in log-space (avg_nll), then exp()
                            # - Min/Max metrics: take min/max across all citations
                            # - Mean metrics: standard averaging
                            # - Variance: average variances, then sqrt for std
                            if avg_nlls:
                                judge_avg_nll = sum(avg_nlls) / len(avg_nlls)
                                judge_perplexity = math.exp(judge_avg_nll)
                            if min_probs:
                                judge_min_prob = min(min_probs)  # CORRECTED: Take minimum of minimums
                            if max_entropies:
                                judge_max_window_entropy = max(max_entropies)  # CORRECTED: Take maximum of maximums
                            if mean_token_probs:
                                judge_mean_token_prob = sum(mean_token_probs) / len(mean_token_probs)
                            if prob_variances:
                                judge_prob_variance = sum(prob_variances) / len(prob_variances)
                                # CORRECTED: Compute std from averaged variance (cannot average std directly)
                                judge_prob_std = math.sqrt(judge_prob_variance)
                            if max_prob_diffs:
                                judge_max_prob_diff = max(max_prob_diffs)  # CORRECTED: Take maximum of maximums
                            if token_prob_slopes:
                                judge_token_prob_slope = sum(token_prob_slopes) / len(token_prob_slopes)

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
                        agg_resp = judge_client.send_message(sys_prompt_agg, usr_prompt_agg, stream=False, seed=self.judge_seed)
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

                self._update_edge_with_judge_verdict(
                    src, tgt, rel_type, 
                    final_agg_verdict, 
                    final_json, 
                    aggregate_score,
                    aggregator_message=None,
                    judge_perplexity=judge_perplexity,
                    judge_min_prob=judge_min_prob,
                    judge_max_window_entropy=judge_max_window_entropy,
                    judge_mean_token_prob=judge_mean_token_prob,
                    judge_prob_variance=judge_prob_variance,
                    judge_prob_std=judge_prob_std,
                    judge_max_prob_diff=judge_max_prob_diff,
                    judge_token_prob_slope=judge_token_prob_slope
                )

                return {
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "motivation": motivation_text[:100] + "..." if len(motivation_text) > 100 else motivation_text,
                    "judged_claim": claim_to_judge,
                    "aggregate_score": aggregate_score,
                    "aggregate_verdict": final_agg_verdict,
                    "aggregator_message": aggregator_reason_str,
                    "details_json": final_judge_data
                }

            elif approach == "one_causal_source_found":
                # ============================================================
                # APPROACH 4: one_causal_source_found (check for direct causal evidence)
                # ============================================================
                print(f"\nCITATION JUDGING (one_causal_source_found): Edge {idx}/{total_edges} => {src}-[{rel_type}]->{tgt}")
                
                citations = props.get("citations", []) or props.get("citation", [])
            
            # Build the claim first (needed even for NO_CITATION edges for consistency)
            if motivation_text and motivation_text.lower() not in ["expert-validated", "expert validated", "n/a", "na"]:
                one_causal_claim = motivation_text
            else:
                # Construct explicit direct causal claim from edge details
                if rel_type.upper() == "POSITIVE":
                    one_causal_claim = f"{src} directly causes an increase in {tgt}"
                elif rel_type.upper() == "NEGATIVE":
                    one_causal_claim = f"{src} directly causes a decrease in {tgt}"
                else:
                    one_causal_claim = f"{src} directly causes changes in {tgt}"
            
            # Check if we have citations
            if not citations:
                verdict = "NO_CITATION"
                reason = "Edge has no citations to judge."
                print(f"CITATION JUDGING: {verdict} => {reason}")
                self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
                return {
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "judged_claim": one_causal_claim,
                    "judge_verdict": verdict,
                    "judge_message": reason
                }
            
            print(f"CITATION JUDGING: Found {len(citations)} citations; motivation length={len(motivation_text)}")
            
            # Fetch + process citations
            try:
                if citation_fetcher.lower() == "jina":
                    # Use Jina AI Reader ONLY (no fallback)
                    import logging
                    logger = logging.getLogger(__name__)
                    print(f"CITATION JUDGING: Using Jina AI Reader ONLY (timeout={jina_timeout}s, retries={jina_retries}, no fallback)")
                    citation_text_map = fetch_citations_with_jina(
                        citations=citations,
                        timeout=jina_timeout,
                        max_retries=jina_retries,
                        fallback_to_scraper=False,  # No fallback - test Jina AI alone
                        logger=logger,
                        api_key=jina_api_key
                    )
                elif citation_fetcher.lower() == "scraper":
                    # Use ContentScraper directly (with whitelist: only trusted domains)
                    if ContentScraper is None:
                        raise ImportError("ContentScraper is not available. Cannot proceed with citation fetching.")
                    print(f"CITATION JUDGING: Using ContentScraper with whitelist")
                    processor = ContentScraper(timeout=15, max_workers=3, max_content_length=10_000_000, pdf_extraction_mode="local", disable_agentic_fallback=True, access_all_domains=False)
                    citation_meta = processor.process_citations(citations, fetch_content=True)
                    scraper_results = processor.prepare_for_llm(citation_meta, include_metadata=True)
                    
                    # Filter out None values, None keys, and error messages from ContentScraper
                    citation_text_map = {}
                    if scraper_results:
                        for url, content in scraper_results.items():
                            # Skip None keys, None values, string "None", and invalid content
                            if url is None or url == "None" or not isinstance(url, str):
                                continue
                            if not url.startswith(('http://', 'https://')):
                                print(f"⚠️ ContentScraper returned invalid URL format: {url}")
                                continue
                            if content is None or not isinstance(content, str):
                                continue
                            if len(content) <= 200 or "[ERROR]" in content[:100]:
                                print(f"⚠️ ContentScraper returned invalid content for: {url} (content length: {len(content)})")
                                continue
                            citation_text_map[url] = content
                    
                    # Final safety check: remove any None keys that might have slipped through
                    citation_text_map = {k: v for k, v in citation_text_map.items() if k is not None and k != "None" and isinstance(k, str) and k.startswith(('http://', 'https://'))}
                    
                    print(f"CITATION JUDGING: ContentScraper returned {len(citation_text_map)}/{len(citations)} valid citations")
                    
                    # If no valid citations, return early with NO_CITATION verdict
                    if not citation_text_map:
                        verdict = "NO_CITATION"
                        reason = "All citations failed to fetch valid content from ContentScraper."
                        print(f"CITATION JUDGING: {verdict} => {reason}")
                        self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
                        # Construct judged_claim for consistency
                        if rel_type.upper() == "POSITIVE":
                            judged_claim_text = f"{src} directly causes an increase in {tgt}"
                        elif rel_type.upper() == "NEGATIVE":
                            judged_claim_text = f"{src} directly causes a decrease in {tgt}"
                        else:
                            judged_claim_text = f"{src} directly causes changes in {tgt}"
                        return {
                            "source": src,
                            "target": tgt,
                            "type": rel_type,
                            "judged_claim": judged_claim_text,
                            "judge_verdict": verdict,
                            "judge_message": reason
                        }
                else:
                    raise ValueError(f"Unknown citation_fetcher: {citation_fetcher}. Must be 'jina' or 'scraper'")
            except Exception as e:
                verdict = "ERROR"
                reason = f"Error fetching citations: {e}"
                print(f"CITATION JUDGING ERROR: {reason}")
                self._update_edge_with_judge_verdict(src, tgt, rel_type, verdict, reason)
                return {
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                    "judge_verdict": verdict,
                    "judge_message": reason,
                }
            
            # Single big prompt: ephemeral judges => "CAUSAL" or "NOT_CAUSAL"
            # majority => final => store => aggregator_score = 1.0 or 0.0
            formatted_citations = ""
            for url, content in citation_text_map.items():
                # Double-check: skip None URLs (shouldn't happen after filtering, but be safe)
                if url is None or url == "None" or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                    print(f"⚠️ Skipping invalid URL in citation_text_map: {url}")
                    continue
                formatted_citations += f"\\n\\n--- CITATION: {url} ---\\n{content}\\n---"

            # Build the actual causal claim to judge
            # If motivation is just metadata (e.g., "Expert-validated"), construct the claim from edge details
            if motivation_text and motivation_text.lower() not in ["expert-validated", "expert validated", "n/a", "na"]:
                causal_claim = motivation_text
            else:
                # Construct explicit direct causal claim from edge details
                if rel_type.upper() == "POSITIVE":
                    causal_claim = f"{src} directly causes an increase in {tgt}"
                elif rel_type.upper() == "NEGATIVE":
                    causal_claim = f"{src} directly causes a decrease in {tgt}"
                else:
                    causal_claim = f"{src} directly causes changes in {tgt}"
            
            sys_prompt_causal = (
                "You are a precise judge checking if these citations provide direct causal evidence "
                "for the claim below. Only rely on the text below."
            )
            usr_prompt_causal = (
                f"CAUSAL CLAIM: {causal_claim}\\n\\n"
                f"ALL CITATIONS:\\n{formatted_citations}\\n\\n"
                "Respond EXACTLY with one of:\\n"
                "CAUSAL\\n"
                "NOT_CAUSAL\\n"
            )

            judge_results = []
            for j_idx, judge_client in enumerate(ephemeral_judges):
                model_name = models[j_idx]
                print(f"[one_causal_source_found] => Judge {j_idx+1} ({model_name}) for {src}->{tgt}")
                try:
                    resp = judge_client.send_message(sys_prompt_causal, usr_prompt_causal, stream=False, seed=self.judge_seed)
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
            
            return {
                "source": src,
                "target": tgt,
                "type": rel_type,
                "motivation": motivation_text[:100] + "..." if len(motivation_text) > 100 else motivation_text,
                "judged_claim": causal_claim,
                "judge_verdict": final_verdict,
                "judge_message": detail_msg,
                "aggregate_score": aggregate_score
            }

        else:
            raise ValueError(f"Unknown approach: '{approach}'. Must be 'all_citations_at_once', 'per_citation_aggregate', 'one_causal_source_found', or 'correctness'")

    def judge_all_edges_with_citations_serial(
    self,
    judge_models=None,
    num_judges=None,
    approach: str = "per_citation_aggregate",
    judge_parallel: bool = False,
    judge_max_workers: int = 10,
    citation_fetcher: str = "scraper",
    jina_timeout: int = 60,
    jina_retries: int = 2,
    jina_api_key: str = None,
    judge_config: Optional[Dict[str, Any]] = None,  # ← Added for structured outputs support
    edge_filter: Optional[List[Dict[str, str]]] = None  # ← NEW: Filter to specific edges
) -> List[Dict]:
        """
        A single function that can do any of THREE modes, controlled by 'approach':

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

        3) "correctness" (correctness-focused, no citations):
        - Judge based solely on overall correctness of motivation text.
        - No citations used - evaluates causal reasoning correctness and plausibility.
        - Each LLM returns CORRECT/PARTIALLY_CORRECT/INCORRECT with score.
        - Majority voting across LLMs for final verdict and average score.

        Args:
            judge_models: List of model names to use for judging, or a single model name.
                        If None, uses the original judge_llm for all calls.
            num_judges: Number of judge calls to make. If None, uses len(judge_models).
                        If judge_models is None or a string, defaults to 1.
            approach: "all_citations_at_once" | "per_citation_aggregate" | "correctness"
            judge_parallel: If True, process edges in parallel using ThreadPoolExecutor
            judge_max_workers: Maximum number of parallel workers (default: 10)
            citation_fetcher: Citation fetcher to use - "jina" for Jina AI Reader (no fallback) or "scraper" for ContentScraper (default: "scraper")
            jina_timeout: Timeout for Jina AI requests in seconds (default: 60)
            jina_retries: Number of retries for failed Jina AI requests (default: 2)
            jina_api_key: Jina AI API key (optional, improves rate limits)

        Returns:
            A list of dicts describing the final judgment of each edge.
        """
        import os
        import logging
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger = logging.getLogger(__name__)
        judged_edges: List[Dict] = []

        # ---------------------------------------------------------------
        # 1) Determine which judge models to use (multi-model approach)
        # ---------------------------------------------------------------
        print(f"DEBUG: judge_models={judge_models}, num_judges={num_judges}, type={type(judge_models)}")
        
        if judge_models is None:
            raise ValueError(
                "No judge model selected for citation-based judging. "
                "Please provide judge_models parameter."
            )
        elif isinstance(judge_models, str):
            models = [judge_models] * (num_judges or 1)  # Default to 1 judge instead of 2
        else:
            # judge_models is a list - use it directly
            models = judge_models
            print(f"DEBUG: Using judge_models list directly, len={len(models)}")
            if num_judges is not None and num_judges != len(models):
                logger.warning(
                    f"num_judges parameter ({num_judges}) is ignored when judge_models is a list. "
                    f"Using {len(models)} judges based on list length."
                )

        print(f"\n{'='*50}")
        print(f"CITATION JUDGING (approach='{approach}'): Starting with {len(models)} judge model(s)")
        print(f"Models: {models}")
        print(f"Parallel processing: {'ENABLED' if judge_parallel else 'DISABLED'}")
        if judge_parallel:
            print(f"Max workers: {judge_max_workers}")
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
                        enable_web_search=getattr(self, "judge_enable_web_search", False),
                        logger=logging.getLogger(f"app.llm.claude.judge{i}")
                    )
                )
                print(f"CITATION JUDGING: Created ephemeral Judge {i+1}: Claude with model {model_name}")
            elif "gpt" in model_lower or "openai" in model_lower or self._is_openai_compatible_model(model_name):
                api_key = os.environ.get("OPENAI_API_KEY", "")
                api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
                ephemeral_judges.append(
                    OpenAIClient(
                        api_key=api_key,
                        api_url=api_url,
                        model=model_name,
                        dev_mode=getattr(self, "dev_mode", False),
                        enable_websearch=False,  # Disable web search for citation judging - we provide content directly
                        logger=logging.getLogger(f"app.llm.openai.judge{i}")
                    )
                )
                print(f"CITATION JUDGING: Created ephemeral Judge {i+1}: OpenAI-compatible with model {model_name}")
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

        try:
            # ---------------------------------------------------------------
            # 3) Query edges from the database for this session
            # ---------------------------------------------------------------
            if edge_filter:
                # Judge only specific edges (for re-judging corrected edges)
                print(f"CITATION JUDGING: Filtering to {len(edge_filter)} specific edges")
                records = []
                with self.graph_db._get_session() as session:
                    for edge_spec in edge_filter:
                        query = f"""
                        MATCH (s:variable {{name: $source}})-[r:{edge_spec['type']}]->(t:variable {{name: $target}})
                        WHERE s.session_id = $session_id AND t.session_id = $session_id
                        RETURN s.name AS source,
                            t.name AS target,
                            type(r) AS rel_type,
                            properties(r) AS props
                        """
                        params = {
                            "session_id": self.session_id,
                            "source": edge_spec['source'],
                            "target": edge_spec['target']
                        }
                        result = session.run(query, params).single()
                        if result:
                            records.append(result)
            else:
                # Judge all edges in session (default behavior)
                query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $session_id AND t.session_id = $session_id
                RETURN s.name AS source,
                    t.name AS target,
                    type(r) AS rel_type,
                    properties(r) AS props
                """
                params = {"session_id": self.session_id}
                with self.graph_db._get_session() as session:
                    records = list(session.run(query, params))

            if not records:
                print("CITATION JUDGING: No edges found to judge.")
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
            # 4) Process edges - either sequentially or in parallel
            # ------------------------------------------------------------
            if judge_parallel:
                # Parallel processing using ThreadPoolExecutor
                print(f"\nProcessing {len(records)} edges in PARALLEL with {judge_max_workers} workers...")
                
                with ThreadPoolExecutor(max_workers=judge_max_workers) as executor:
                    # Submit all edge processing tasks
                    future_to_edge = {}
                    for idx, record in enumerate(records, start=1):
                        future = executor.submit(
                            self._process_single_edge_judgment,
                            idx=idx,
                            total_edges=len(records),
                            record=record,
                            ephemeral_judges=ephemeral_judges,
                            models=models,
                            approach=approach,
                            citation_fetcher=citation_fetcher,
                            jina_timeout=jina_timeout,
                            jina_retries=jina_retries,
                            jina_api_key=jina_api_key,
                            judge_config=judge_config
                        )
                        future_to_edge[future] = (idx, record["source"], record["target"])
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_edge):
                        idx, src, tgt = future_to_edge[future]
                        try:
                            result = future.result()
                            if result:
                                judged_edges.append(result)
                            print(f"✓ Completed edge {idx}/{len(records)}: {src} -> {tgt}")
                        except Exception as e:
                            logger.error(f"❌ Error processing edge {idx}/{len(records)} ({src} -> {tgt}): {e}")
                            print(f"❌ Error processing edge {idx}/{len(records)} ({src} -> {tgt}): {e}")
            else:
                # Sequential processing
                print(f"\nProcessing {len(records)} edges SEQUENTIALLY...")
                
                for idx, record in enumerate(records, start=1):
                    try:
                        result = self._process_single_edge_judgment(
                            idx=idx,
                            total_edges=len(records),
                            record=record,
                            ephemeral_judges=ephemeral_judges,
                            models=models,
                            approach=approach,
                            citation_fetcher=citation_fetcher,
                            jina_timeout=jina_timeout,
                            jina_retries=jina_retries,
                            jina_api_key=jina_api_key,
                            judge_config=judge_config
                        )
                        if result:
                            judged_edges.append(result)
                    except Exception as e:
                        src = record.get("source", "?")
                        tgt = record.get("target", "?")
                        logger.error(f"❌ Error processing edge {idx}/{len(records)} ({src} -> {tgt}): {e}")
                        print(f"❌ Error processing edge {idx}/{len(records)} ({src} -> {tgt}): {e}")

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

    def _delete_edge(self, src: str, tgt: str, rel_type: str) -> bool:
        """
        Set a relationship edge to "None" type and mark as corrected.
        This is used for actual edge removal, not for change_type (which deletes directly).
        
        Returns True if updated, False otherwise.
        """
        import logging
        logger = logging.getLogger(__name__)
        try:
            with self.graph_db._get_session() as session:
                # First check if edge exists
                check_result = session.run(
                    f"""
                    MATCH (s:variable {{name:$src, session_id:$sid}})-[r:{rel_type}]->(t:variable {{name:$tgt, session_id:$sid}})
                    RETURN count(*) AS c
                    """,
                    {"src": src, "tgt": tgt, "sid": self.session_id},
                ).single()
                
                if not check_result or not check_result["c"] or int(check_result["c"]) == 0:
                    logger.warning(f"No relationship found to nullify: {src}-[{rel_type}]->{tgt}")
                    return False
                
                # Delete the existing relationship and create a new "None" relationship
                session.run(
                    f"""
                    MATCH (s:variable {{name:$src, session_id:$sid}})-[r:{rel_type}]->(t:variable {{name:$tgt, session_id:$sid}})
                    DELETE r
                    """,
                    {"src": src, "tgt": tgt, "sid": self.session_id},
                )
                
                # Create new "NONE" relationship with correction tracking (use NONE not None to avoid Python/Neo4j conflicts)
                # For removed edges, set corrected_motivation to the corrector_message so it appears in Excel
                session.run(
                    """
                    MATCH (s:variable {name:$src, session_id:$sid})
                    MATCH (t:variable {name:$tgt, session_id:$sid})
                    CREATE (s)-[r:NONE]->(t)
                    SET r.session_id = $sid,
                        r.corrected = true,
                        r.correction_action = 'remove',
                        r.original_relationship_type = $original_rel_type,
                        r.corrector_message = 'Edge relationship set to NONE due to correction',
                        r.corrected_motivation = 'Edge removed - see Corrector Message for reasoning'
                    """,
                    {"src": src, "tgt": tgt, "sid": self.session_id, "original_rel_type": rel_type},
                )
                
                logger.info(f"Set relationship to NONE: {src}-[{rel_type}]->{tgt} -> {src}-[NONE]->{tgt}")
                return True
        except Exception as e:
            logger.error(f"Failed to nullify relationship {src}-[{rel_type}]->{tgt}: {e}")
            return False

    def _build_corrector_client(self, model_name=None, temperature=0.2, top_p=1.0):
        """
        Build an LLM client for the corrector role. Uses the stored judge_provider and judge_model.
        """
        # Use the stored judge configuration (set during __init__)
        provider = getattr(self, "judge_provider", None)
        model = model_name or getattr(self, "judge_model", None)
        
        if not provider or not model:
            raise ValueError(
                "No corrector model configured. Please ensure judge_provider and judge_model are set during CausalDiscovery initialization, "
                "or provide model_name parameter to _build_corrector_client."
            )

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "None")
        OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "None")
        CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
        PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "None")
        PERPLEXITY_API_URL = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/v1/chat/completions")

        return self._build_llm_client(
            provider=provider,
            model=model,
            dev_mode=getattr(self, "dev_mode", False),
            temperature=temperature,
            top_p=top_p,
            openai_key=OPENAI_API_KEY,
            openai_url=OPENAI_API_URL,
            claude_key=CLAUDE_API_KEY,
            claude_url=CLAUDE_API_URL,
            perplexity_key=PERPLEXITY_API_KEY,
            perplexity_url=PERPLEXITY_API_URL,
            role_name="corrector",
        )

    def _revise_motivation_with_citations(
        self,
        corrector_llm,
        src,
        tgt,
        rel_type,
        current_motivation,
        citation_text_map,
        judge_verdict: Optional[str] = None,
        judge_message: Optional[str] = None,
        aggregate_verdict: Optional[str] = None,
        aggregate_score: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        Ask the corrector to produce a corrected motivation aligned with citations.
        Returns (corrected_motivation, used_citations, note).
        """
        import logging
        import json
        import re
        logger = logging.getLogger(__name__)
        try:
            joined_citations = "\n\n".join(
                [f"--- CITATION: {url} ---\n{content}" for url, content in (citation_text_map or {}).items()]
            )

            # Try to parse structured per-citation details, if judge_message contains JSON
            per_citation_block = ""
            jm = judge_message or ""
            try:
                jm_obj = json.loads(jm)
                if isinstance(jm_obj, dict) and jm_obj.get("citations"):
                    lines = []
                    for c in jm_obj.get("citations", []):
                        url = c.get("url", "")
                        v = c.get("verdict", "")
                        r = c.get("reason", "")
                        lines.append(f"- {url}: {v} :: {r}")
                    if lines:
                        per_citation_block = "\n".join(lines)
            except Exception:
                # fall back to raw text
                per_citation_block = jm[:800]

            judge_summary = (
                f"Judge verdict: {judge_verdict or 'n/a'}; "
                f"Aggregate verdict: {aggregate_verdict or 'n/a'}; Aggregate score: {aggregate_score if aggregate_score is not None else 'n/a'}\n"
                f"Judge details:\n{per_citation_block}"
            ).strip()

            sys_prompt = (
                "You are a careful scientific editor. Given a causal relationship, its current rationale, citation "
                "snippets, and the judge's criticisms, rewrite the rationale to be logically consistent, mechanistic, "
                "and aligned with the citations. If citations do not support the claim, propose a narrower claim that is "
                "supported. Respond with JSON: {corrected_motivation, used_citations: [URLs], note}."
            )
            usr_prompt = (
                f"Source: {src}\nTarget: {tgt}\nRelationship: {rel_type}\n\n"
                f"Current rationale:\n{current_motivation}\n\n"
                f"Judge feedback:\n{judge_summary}\n\n"
                f"CITATIONS:\n{joined_citations if joined_citations else '[NO CITATIONS AVAILABLE]'}\n\n"
                "Return strictly JSON."
            )
            resp = corrector_llm.send_message(sys_prompt, usr_prompt, stream=False, seed=seed)
            data = corrector_llm.parse_static_response(resp)
            content = (data.get("content") or "").strip()
            try:
                payload = json.loads(content)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", content)
                payload = json.loads(m.group(0)) if m else {"corrected_motivation": content}

            corrected = (payload.get("corrected_motivation") or "").strip()
            used = payload.get("used_citations") or []
            note = (payload.get("note") or "").strip()
            return corrected, used, note
        except Exception as e:
            logger.error(f"Corrector failed for {src}-[{rel_type}]->{tgt}: {e}")
            return "", [], f"error: {e}"

    # ==================== AGENT-BASED CORRECTOR SYSTEM ====================
    
    def _create_corrector_agent(self, corrector_llm, prompt_variant="original"):
        """
        Create a pydantic_ai agent with TOOLS (proper pydantic-ai 1.10.0+ approach)
        
        Args:
            corrector_llm: LLM client for correction
            prompt_variant: Which prompt to use from corrector_prompts.yaml
                           Options: "original", "improved", "aggressive_fix"
        
        Returns the agent instance.
        """
        from pydantic_ai import Agent, RunContext
        from pydantic import BaseModel
        import logging
        import yaml
        import os
        logger = logging.getLogger(__name__)
        
        # Extract model string
        provider = getattr(self, "judge_provider", "openai").lower()
        model_name = getattr(self, "judge_model", "gpt-4o")
        provider_map = {"openai": "openai", "anthropic": "anthropic", "perplexity": "openai"}
        pydantic_provider = provider_map.get(provider, "openai")
        model_string = f"{pydantic_provider}:{model_name}"
        
        # Load prompt from YAML
        prompt_file = os.path.join(os.path.dirname(__file__), "parameter_tuning_experiments/corrector_prompts.yaml")
        try:
            with open(prompt_file, 'r') as f:
                prompts = yaml.safe_load(f)
            prompt_key = f"{prompt_variant}_prompt"
            system_prompt = prompts.get(prompt_key)
            if not system_prompt:
                logger.warning(f"Prompt variant '{prompt_variant}' not found, using original")
                system_prompt = prompts.get("original_prompt")
            logger.info(f"Creating pydantic_ai agent with prompt variant: {prompt_variant}")
        except Exception as e:
            logger.warning(f"Could not load prompt from YAML: {e}, using default")
            system_prompt = (
                "You are an expert causal reasoning corrector. You have 2 tools:\\n\\n"
                "1. revise_motivation: Fix incorrect motivation text (keeps same edge type)\\n"
                "2. change_edge_type: Change edge type (POSITIVE/NEGATIVE/NONE) with new motivation\\n\\n"
                "IMPORTANT: Use edge type NONE to represent 'no causal relationship'.\\n"
                "Do NOT remove edges - instead, use change_edge_type to set them to NONE with proper justification.\\n\\n"
                "Examples:\\n"
                "- NONE edge that should be causal? Use change_edge_type(old_type='NONE', new_type='POSITIVE'/'NEGATIVE', new_motivation='...')\\n"
                "- Causal edge that should be NONE? Use change_edge_type(old_type='POSITIVE'/'NEGATIVE', new_type='NONE', new_motivation='Explain why no causal relationship exists')\\n"
                "- Wrong polarity (POSITIVE should be NEGATIVE)? Use change_edge_type(old_type='POSITIVE', new_type='NEGATIVE', new_motivation='...')\\n"
                "- Bad motivation but correct type? Use revise_motivation(new_motivation='...')\\n\\n"
                "Analyze the judge feedback and call the appropriate tool."
            )
        
        logger.info(f"Creating pydantic_ai agent with tools: {model_string}")
        
        # Create agent with str output (simpler for tool-based approach)
        # pydantic_ai will automatically use OPENAI_API_KEY from environment
        agent = Agent(
            model_string,
            system_prompt=system_prompt
        )
        
        # Define tools as module-level functions that close over self
        @agent.tool_plain
        def revise_motivation(source: str, target: str, rel_type: str, new_motivation: str) -> str:
            """Revise the motivation for an edge to provide better causal reasoning."""
            success = self.update_relationship_properties(
                source, target, rel_type,
                {
                    "motivation": new_motivation,
                    "corrected": True,
                    "correction_action": "revise"
                }
            )
            return "Success" if success else "Failed"
        
        @agent.tool_plain
        def change_edge_type(source: str, target: str, old_type: str, new_type: str, new_motivation: str) -> str:
            """Change edge relationship type (e.g., NONE to POSITIVE, NEGATIVE to NONE, etc.)."""
            # Get current motivation if not provided
            if not new_motivation or new_motivation.strip() == "":
                with self.graph_db._get_session() as session:
                    result = session.run(
                        f"MATCH (s:variable {{name:$s}})-[r:{old_type}]->(t:variable {{name:$t}}) "
                        f"WHERE s.session_id=$sid RETURN r.motivation as mot",
                        {"s": source, "t": target, "sid": self.session_id}
                    ).single()
                    new_motivation = result["mot"] if result and result["mot"] else f"Relationship between {source} and {target}"
            
            # For change_type, we directly delete and recreate without creating a NONE placeholder
            # This avoids having two corrected edges (NONE + new type) in the database
            with self.graph_db._get_session() as session:
                # Delete the old edge completely
                session.run(
                    f"""
                    MATCH (s:variable {{name:$src, session_id:$sid}})-[r:{old_type}]->(t:variable {{name:$tgt, session_id:$sid}})
                    DELETE r
                    """,
                    {"src": source, "tgt": target, "sid": self.session_id}
                )
            
            # Create new edge with new type
            success = self._create_edge(source, target, new_type, new_motivation)
            if success:
                self.update_relationship_properties(source, target, new_type, {
                    "corrected": True,
                    "correction_action": "change_type",
                    "original_type": old_type
                })
                return f"Changed {old_type}→{new_type}"
            else:
                return f"Failed to create new edge {new_type}"
        
        logger.info(f"Agent created with 2 tools: revise_motivation, change_edge_type")
        logger.info(f"Note: To mark an edge as non-causal, use change_edge_type with new_type='NONE'")
        return agent
    
    def _create_simple_deleter_agent(self, corrector_llm):
        """
        Create a pydantic_ai agent with ONLY remove_edge and create_edge tools.
        This tests whether sophisticated tools (revise, flip, change_type) add value 
        vs. simple add/remove operations.
        
        Returns the agent instance.
        """
        from pydantic_ai import Agent
        import logging
        logger = logging.getLogger(__name__)
        
        # Extract model string
        provider = getattr(self, "judge_provider", "openai").lower()
        model_name = getattr(self, "judge_model", "gpt-4o")
        provider_map = {"openai": "openai", "anthropic": "anthropic", "perplexity": "openai"}
        pydantic_provider = provider_map.get(provider, "openai")
        model_string = f"{pydantic_provider}:{model_name}"
        
        logger.info(f"Creating SIMPLE ADD/REMOVE agent: {model_string}")
        
        # Create agent with only add/remove capability
        agent = Agent(
            model_string,
            system_prompt=(
                "You are a causal reasoning corrector with two simple tools:\n\n"
                "1. remove_edge: Delete wrong or spurious edges\n"
                "2. create_edge: Add missing causal edges\n\n"
                "Your task: Decide whether to ADD, DELETE, or KEEP edges.\n\n"
                "- If an edge is fundamentally wrong or spurious, call remove_edge\n"
                "- If an important causal edge is missing, call create_edge\n"
                "- If the edge is OK or salvageable, do nothing\n\n"
                "You CANNOT fix motivations or flip polarities - only add/remove.\n"
                "Analyze the judge feedback and decide: add, remove, or keep?"
            )
        )
        
        # Define the two simple tools
        @agent.tool_plain
        def remove_edge(source: str, target: str, rel_type: str, reasoning: str) -> str:
            """Remove a spurious or fundamentally incorrect edge."""
            success = self._delete_edge(source, target, rel_type)
            if success:
                self._store_removal_reason(source, target, reasoning)
            return "Removed" if success else "Failed"
        
        @agent.tool_plain
        def create_edge(source: str, target: str, rel_type: str, motivation: str) -> str:
            """Create a new causal edge (POSITIVE or NEGATIVE)."""
            if rel_type not in ["POSITIVE", "NEGATIVE"]:
                return f"Failed: rel_type must be POSITIVE or NEGATIVE, got {rel_type}"
            
            success = self._create_edge(source, target, rel_type, motivation)
            if success:
                # No need to update properties - _create_edge already sets:
                # - corrected = true
                # - correction_action = 'add'
                # - added_by_corrector = true
                return f"Created {rel_type} edge"
            return "Failed to create edge"
        
        logger.info(f"Simple add/remove agent created with 2 tools: remove_edge, create_edge")
        return agent
    
    def _correct_edge_with_agent(
        self,
        agent,
        source: str,
        target: str,
        rel_type: str,
        props: dict,
        citation_text_map: dict,
        available_variables: list[str]
    ) -> tuple[bool, str, str]:
        """
        Use the corrector agent with tools to fix an edge.
        
        Returns:
            Tuple of (corrected: bool, action: str, note: str)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Prepare edge information
            current_mot = (props.get("spurious_motivation") or props.get("motivation") or "").strip()
            citations = props.get("citations") or props.get("citation") or []
            jv = props.get("judge_verdict")
            jm = props.get("judge_message")
            aggv = props.get("aggregate_verdict")
            aggs = props.get("aggregate_score")
            
            # Build prompt with edge details
            prompt = (
                f"Analyze this edge and use tools to correct it:\n\n"
                f"Edge: {source} -[{rel_type}]-> {target}\n"
                f"Judge Verdict: {jv}\n"
                f"Aggregate Score: {aggs}\n"
                f"Judge Reasoning: {jm}\n\n"
                f"Current Motivation:\n{current_mot}\n\n"
                f"Available actions:\n"
                f"- If edge type is wrong (wrong polarity or should be NONE/causal), call change_edge_type with new type and motivation\n"
                f"- If motivation is wrong but type is correct, call revise_motivation with better text\n"
                f"- If edge should be non-causal, use change_edge_type to set it to NONE with explanation"
            )
            
            # Run agent with tools
            # Create a new event loop for thread-safe execution (needed for parallel correction)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in a thread, create a new event loop
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = agent.run_sync(prompt)
                    finally:
                        new_loop.close()
                        asyncio.set_event_loop(loop)
                else:
                    result = agent.run_sync(prompt)
            except RuntimeError:
                # No event loop exists, create one
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = agent.run_sync(prompt)
                finally:
                    new_loop.close()
            
            # Check what tools were called by inspecting the result object
            action = "none"
            reasoning = str(result.output) if result.output else "No action taken"  # Don't truncate
            corrected = False
            new_type = None  # For change_type actions
            
            # Check if any tools were called (PydanticAI result has 'all_messages' attribute)
            if hasattr(result, 'all_messages'):
                for msg in result.all_messages():
                    # Check for tool calls in the message history
                    if hasattr(msg, 'parts'):
                        for part in msg.parts:
                            if hasattr(part, 'tool_name'):
                                tool_name = part.tool_name
                                corrected = True
                                # Map tool names to actions
                                if tool_name == 'flip_polarity':
                                    action = "flip_polarity"
                                elif tool_name == 'change_edge_type':
                                    action = "change_type"
                                    # Extract new_type from tool arguments
                                    try:
                                        if hasattr(part, 'args'):
                                            args_data = part.args
                                            # If args is a JSON string, parse it
                                            if isinstance(args_data, str):
                                                import json
                                                args_data = json.loads(args_data)
                                            # Now extract new_type
                                            if isinstance(args_data, dict) and 'new_type' in args_data:
                                                new_type = args_data['new_type']
                                                logger.info(f"  Extracted new_type: {new_type}")
                                    except Exception as e:
                                        logger.warning(f"  Failed to extract new_type: {e}")
                                elif tool_name == 'remove_edge':
                                    action = "remove"
                                elif tool_name == 'revise_motivation':
                                    action = "revise"
                                break
                        if corrected:
                            break
            
            # Fallback to output string parsing if no tool info found
            if not corrected:
                output_str = str(result.output)
                if result.output and ("Success" in output_str or "Flipped" in output_str or "Changed" in output_str or "Removed" in output_str or "Created" in output_str):
                    corrected = True
                    if "Flipped" in output_str or "flipped" in output_str.lower():
                        action = "flip_polarity"
                    elif ("Changed" in output_str and "→" in output_str) or "change" in output_str.lower():
                        action = "change_type"
                        # Try to extract new_type from output string (e.g., "Changed NONE→POSITIVE")
                        import re
                        match = re.search(r'→(\w+)', output_str)
                        if match:
                            new_type = match.group(1)
                            logger.info(f"  Extracted new_type from output: {new_type}")
                    elif "Removed" in output_str or "removed" in output_str.lower():
                        action = "remove"
                    elif "Success" in output_str or "revised" in output_str.lower():
                        action = "revise"
            
            return corrected, action, reasoning, new_type
                
        except Exception as e:
            logger.error(f"Agent correction failed for {source}->{target}: {e}")
            import traceback
            traceback.print_exc()
            return False, "error", str(e), None  # 4th value is new_type
    
    def _correct_edge_with_structured_outputs(
        self,
        corrector_llm,
        source: str,
        target: str,
        rel_type: str,
        props: dict,
        prompt_variant: str = "baseline"
    ) -> tuple[bool, str, str, str]:
        """
        Use OpenAI structured outputs to get a reliable correction decision.
        
        This replaces the complex agent-based approach with direct API calls
        using Pydantic models for guaranteed valid JSON responses.
        
        Args:
            corrector_llm: LLM client for correction
            source: Source variable name
            target: Target variable name
            rel_type: Current edge type (POSITIVE/NEGATIVE/NONE)
            props: Edge properties including judge feedback
            prompt_variant: Which prompt variant to use
            
        Returns:
            Tuple of (corrected: bool, action: str, reasoning: str, new_type: str|None)
        """
        import logging
        import json
        import time
        logger = logging.getLogger(__name__)
        
        if not CORRECTOR_STRUCTURED_OUTPUTS_AVAILABLE:
            logger.warning("Corrector structured outputs not available, falling back to agent")
            return False, "error", "Structured outputs not available", None
        
        try:
            # Prepare edge information
            current_mot = (props.get("spurious_motivation") or props.get("motivation") or "").strip()
            jv = props.get("judge_verdict")
            jm = props.get("judge_message")
            aggs = props.get("aggregate_score")
            
            # Build system prompt
            sys_prompt = """You are an expert causal reasoning corrector. Analyze the edge and judge feedback, then decide on the correction action.

Actions available:
- "revise": Fix the motivation text while keeping the same edge type. Use when the causal mechanism is poorly explained.
- "change_type": Change the edge type (POSITIVE/NEGATIVE/NONE) with new motivation. Use when the edge polarity is wrong or should be non-causal.
- "none": No correction needed. Use when the edge is already correct.

Respond with your reasoning and the appropriate action. For revise/change_type, provide new_motivation. For change_type, also provide new_type."""

            # Build user prompt
            usr_prompt = f"""Analyze this causal edge and determine the correction needed:

Edge: {source} -[{rel_type}]-> {target}

Current Motivation:
{current_mot}

Judge Verdict: {jv}
Aggregate Score: {aggs}
Judge Feedback: {jm}

What correction action should be taken?"""

            # Call LLM with structured output
            # Retry logic for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = corrector_llm.send_message(
                        sys_prompt, usr_prompt, stream=False,
                        response_model=CorrectorResponse
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise
            
            # Parse response
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            correction = CorrectorResponse(**json.loads(content))
            
            # Map to legacy format
            action = correction.action.value if hasattr(correction.action, 'value') else str(correction.action)
            reasoning = correction.reasoning
            new_type = correction.new_type.value if correction.new_type and hasattr(correction.new_type, 'value') else correction.new_type
            corrected = action != "none"
            
            # Execute the correction in Neo4j if needed
            if corrected and action == "revise" and correction.new_motivation:
                success = self.update_relationship_properties(
                    source, target, rel_type,
                    {"motivation": correction.new_motivation, "corrected": True, "correction_action": "revise"}
                )
                if not success:
                    return False, "error", "Failed to update motivation in Neo4j", None
                    
            elif corrected and action == "change_type" and correction.new_type and correction.new_motivation:
                # Change edge type
                success = self._change_edge_type(
                    source, target, rel_type, 
                    str(correction.new_type.value if hasattr(correction.new_type, 'value') else correction.new_type),
                    correction.new_motivation
                )
                if not success:
                    return False, "error", "Failed to change edge type in Neo4j", None
            
            logger.info(f"  Structured output: action={action}, corrected={corrected}")
            return corrected, action, reasoning, new_type
            
        except Exception as e:
            logger.error(f"Structured output correction failed for {source}->{target}: {e}")
            import traceback
            traceback.print_exc()
            return False, "error", str(e), None
    
    def _execute_corrector_decision(
        self,
        decision,
        source: str,
        target: str,
        rel_type: str,
        props: dict
    ) -> tuple[bool, str]:
        """
        Execute the corrector's decision by calling the appropriate helper methods.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            if decision.action == "revise" and decision.revised_motivation:
                # Revise the motivation
                success = self.update_relationship_properties(
                    source, target, rel_type,
                    {
                        "original_motivation": props.get("original_motivation") or props.get("motivation") or props.get("spurious_motivation"),
                        "motivation": decision.revised_motivation,
                        "corrected": True,
                        "correction_action": "revise",
                        "corrector_reasoning": decision.reasoning
                    }
                )
                return success, f"Revised: {decision.reasoning}"
            
            elif decision.action == "flip_polarity" and decision.new_polarity:
                # Flip the edge polarity by deleting and recreating with new polarity
                old_motivation = props.get("motivation") or props.get("spurious_motivation") or ""
                
                # Delete old edge
                success1 = self._delete_edge(source, target, rel_type)
                if not success1:
                    return False, "Failed to delete edge for polarity flip"
                
                # Create edge with new polarity
                success2 = self._create_edge(source, target, decision.new_polarity, old_motivation)
                if success2:
                    # Mark as corrected
                    self.update_relationship_properties(
                        source, target, decision.new_polarity,
                        {
                            "corrected": True,
                            "correction_action": "flip_polarity",
                            "corrector_reasoning": decision.reasoning,
                            "original_polarity": rel_type
                        }
                    )
                return success2, f"Flipped polarity {rel_type}→{decision.new_polarity}: {decision.reasoning}"
            
            elif decision.action == "remove":
                # Remove the edge
                success = self._delete_edge(source, target, rel_type)
                if success:
                    self._store_removal_reason(source, target, decision.reasoning)
                return success, f"Removed: {decision.reasoning}"
            
            else:
                # No action
                return False, f"No action: {decision.reasoning}"
        
        except Exception as e:
            logger.error(f"Error executing corrector decision: {e}")
            return False, f"Error: {e}"
        
    def _create_corrector_agent_OLD_WITH_TOOLS(self, corrector_llm):
        """
        OLD VERSION with @agent.tool decorators - kept for reference.
        Has pydantic compatibility issues with nested tool definitions.
        """
        from pydantic_ai import Agent, RunContext
        from pydantic import BaseModel, Field
        import logging
        logger = logging.getLogger(__name__)
        
        provider = getattr(self, "judge_provider", "openai").lower()
        model_name = getattr(self, "judge_model", "gpt-4o")
        provider_map = {"openai": "openai", "anthropic": "anthropic", "perplexity": "openai"}
        pydantic_provider = provider_map.get(provider, "openai")
        model_string = f"{pydantic_provider}:{model_name}"
        
        class CorrectorDecision(BaseModel):
            action: str
            reasoning: str
            revised_motivation: Optional[str] = None
            new_polarity: Optional[str] = None
        
        agent = Agent(
            model_string,
            output_type=CorrectorDecision,
            system_prompt=(
                "You are an expert causal reasoning corrector. Analyze judge feedback and decide:\n\n"
                "1. **revise**: If motivation has errors, provide better reasoning (keeps same edge type)\n"
                "2. **change_type**: If the edge type is wrong (POSITIVE↔NEGATIVE, NONE↔causal), "
                "use change_edge_type with new motivation. IMPORTANT: Use NONE type to represent 'no causal relationship' - do NOT remove edges.\n\n"
                "3. **add**: If you believe based on domain knowledge and the context that a causal edge is missing "
                "from the graph, you can add it. Be conservative - only add edges with strong causal reasoning.\n\n"
                "4. **none**: If the edge is acceptable as-is or no action is appropriate.\n\n"
                "Base your decision ONLY on:\n"
                "- Judge feedback and verdicts\n"
                "- Edge motivation quality and citation support\n"
                "- Logical consistency and causal reasoning\n\n"
                "Do NOT use any 'is_corrupted' or 'ground_truth' flags - those don't exist in real scenarios."
            )
        )
        
        # Define tools for the agent
        @agent.tool
        def revise_motivation(ctx: RunContext[CorrectorContext], new_motivation: str) -> dict:
            """
            Revise the motivation for the current edge to be more precise and evidence-based.
            
            Args:
                ctx: The context containing edge information
                new_motivation: The revised motivation text
                
            Returns:
                dict with status and message
            """
            try:
                context = ctx.deps
                self.update_relationship_properties(
                    context.source,
                    context.target,
                    context.relationship_type,
                    {
                        "original_motivation": context.current_motivation,
                        "motivation": new_motivation,
                        "corrected": True,
                        "correction_action": "revise",
                        "corrector_reasoning": "Agent revised motivation for clarity and evidence support"
                    }
                )
                logger.info(f"Agent revised motivation for {context.source}->{context.target}")
                return {"status": "success", "message": "Motivation revised"}
            except Exception as e:
                logger.error(f"Error revising motivation: {e}")
                return {"status": "error", "message": str(e)}
        
        @agent.tool
        def recite_motivation(ctx: RunContext[CorrectorContext], new_citations: list[str]) -> dict:
            """
            Add or update citations for the current edge.
            
            Args:
                ctx: The context containing edge information
                new_citations: List of citation URLs to add
                
            Returns:
                dict with status and message
            """
            try:
                context = ctx.deps
                current_citations = list(context.citations) if context.citations else []
                all_citations = list(set(current_citations + new_citations))
                
                self.update_relationship_properties(
                    context.source,
                    context.target,
                    context.relationship_type,
                    {
                        "citations": all_citations,
                        "corrected": True,
                        "correction_action": "recite",
                        "corrector_reasoning": "Agent added/updated citations"
                    }
                )
                logger.info(f"Agent updated citations for {context.source}->{context.target}: {len(all_citations)} citations")
                return {"status": "success", "message": f"Added {len(new_citations)} citations"}
            except Exception as e:
                logger.error(f"Error updating citations: {e}")
                return {"status": "error", "message": str(e)}
        
        @agent.tool
        def add_edge(
            ctx: RunContext[CorrectorContext],
            source: str,
            target: str,
            relationship_type: str,
            motivation: str
        ) -> dict:
            """
            Add a new edge to the graph that appears to be missing.
            
            Args:
                ctx: The context containing available variables
                source: Source variable name
                target: Target variable name
                relationship_type: Type of relationship (POSITIVE, NEGATIVE, or NONE)
                motivation: Explanation for why this edge exists
                
            Returns:
                dict with status and message
            """
            try:
                context = ctx.deps
                
                # Validate that both variables exist
                if source not in context.available_variables or target not in context.available_variables:
                    return {"status": "error", "message": f"Variables not found: {source}, {target}"}
                
                # Check if edge already exists
                existing = self._edge_exists(source, target)
                if existing:
                    return {"status": "error", "message": f"Edge already exists: {source}->{target}"}
                
                # Create the edge
                success = self._create_edge(source, target, relationship_type, motivation)
                if success:
                    logger.info(f"Agent added edge {source}-[{relationship_type}]->{target}: {motivation}")
                    return {"status": "success", "message": f"Edge added: {source}->{target}"}
                else:
                    return {"status": "error", "message": "Failed to create edge"}
            except Exception as e:
                logger.error(f"Error adding edge: {e}")
                return {"status": "error", "message": str(e)}
        
        return agent
    
    # ==================== HELPER METHODS FOR AGENT TOOLS ====================
    
    def _edge_exists(self, source: str, target: str) -> bool:
        """
        Check if an edge exists between source and target (any relationship type).
        
        Args:
            source: Source variable name
            target: Target variable name
            
        Returns:
            True if any edge exists, False otherwise
        """
        try:
            with self.graph_db._get_session() as session:
                query = """
                MATCH (s:variable {name: $source, session_id: $session_id})-[r]->(t:variable {name: $target, session_id: $session_id})
                RETURN count(r) as edge_count
                """
                result = session.run(query, {
                    "source": source,
                    "target": target,
                    "session_id": self.session_id
                }).single()
                
                return result and result["edge_count"] > 0
        except Exception as e:
            logger.error(f"Error checking edge existence for {source}->{target}: {e}")
            return False
    
    def _create_edge(self, source: str, target: str, relationship_type: str, motivation: str) -> bool:
        """
        Create a new edge in the graph with the specified relationship type and motivation.
        
        Args:
            source: Source variable name
            target: Target variable name
            relationship_type: Type of relationship (POSITIVE, NEGATIVE, or NONE)
            motivation: Explanation for this edge
            
        Returns:
            True if edge was created successfully, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Verify both variables exist
            with self.graph_db._get_session() as session:
                check_query = """
                MATCH (s:variable {name: $source, session_id: $session_id})
                MATCH (t:variable {name: $target, session_id: $session_id})
                RETURN count(*) as node_count
                """
                check_result = session.run(check_query, {
                    "source": source,
                    "target": target,
                    "session_id": self.session_id
                }).single()
                
                if not check_result or check_result["node_count"] < 1:
                    logger.error(f"Cannot create edge: one or both variables not found ({source}, {target})")
                    return False
                
                # Create the edge
                create_query = f"""
                MATCH (s:variable {{name: $source, session_id: $session_id}})
                MATCH (t:variable {{name: $target, session_id: $session_id}})
                CREATE (s)-[r:{relationship_type}]->(t)
                SET r.motivation = $motivation,
                    r.session_id = $session_id,
                    r.added_by_corrector = true,
                    r.corrected = true,
                    r.correction_action = 'add'
                RETURN r
                """
                result = session.run(create_query, {
                    "source": source,
                    "target": target,
                    "session_id": self.session_id,
                    "motivation": motivation
                })
                
                if result.single():
                    logger.info(f"Created new edge: {source}-[{relationship_type}]->{target}")
                    return True
                else:
                    logger.error(f"Failed to create edge: {source}-[{relationship_type}]->{target}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating edge {source}-[{relationship_type}]->{target}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _store_removal_reason(self, source: str, target: str, removal_reason: str) -> bool:
        """
        Store the removal reasoning on a removed edge (now marked as NONE type).
        
        Args:
            source: Source variable name
            target: Target variable name
            removal_reason: Explanation of why the edge was removed
            
        Returns:
            True if successful, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            with self.graph_db._get_session() as session:
                # Find the NONE relationship that was created by _delete_edge
                # Also set corrected_motivation so it appears in the Excel export
                query = """
                MATCH (s:variable {name: $source, session_id: $session_id})-[r:NONE]->(t:variable {name: $target, session_id: $session_id})
                SET r.removal_reasoning = $removal_reason,
                    r.removed_by_corrector = true,
                    r.corrected_motivation = $removal_reason
                RETURN count(r) as updated_count
                """
                result = session.run(query, {
                    "source": source,
                    "target": target,
                    "session_id": self.session_id,
                    "removal_reason": removal_reason
                }).single()
                
                if result and result["updated_count"] > 0:
                    logger.info(f"Stored removal reason for {source}->{target}")
                    return True
                else:
                    logger.warning(f"No NONE edge found to update for {source}->{target}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error storing removal reason for {source}->{target}: {e}")
            return False

    def _correct_single_edge(self, agent, rec, available_variables, edge_idx=None, total_edges=None, corrector_llm=None, use_structured_outputs=True):
        """
        Process a single edge correction (for parallel execution).
        
        Args:
            agent: The corrector agent to use (legacy approach)
            rec: The edge record from Neo4j query
            available_variables: List of available variable names for context
            edge_idx: Optional edge index for logging (used in sequential mode)
            total_edges: Optional total edges for logging (used in sequential mode)
            corrector_llm: Optional LLM client for structured outputs approach
            use_structured_outputs: Whether to use structured outputs (default True)
            
        Returns:
            dict with correction outcome
        """
        import logging
        logger = logging.getLogger(__name__)
        
        src = rec["src"]
        tgt = rec["tgt"]
        rel_type = rec["rel_type"]
        props = rec.get("props") or {}
        
        if edge_idx and total_edges:
            logger.info(f"\n[{edge_idx}/{total_edges}] Processing edge: {src}-[{rel_type}]->{tgt}")
        else:
            logger.info(f"Processing edge: {src}-[{rel_type}]->{tgt}")
        
        # Get edge information
        jv = props.get("judge_verdict")
        aggv = props.get("aggregate_verdict")
        aggs = props.get("aggregate_score")
        
        logger.info(f"  Judge verdict: {jv}, Aggregate: {aggv}, Score: {aggs}")

        # Try structured outputs first if available and enabled
        if use_structured_outputs and CORRECTOR_STRUCTURED_OUTPUTS_AVAILABLE and corrector_llm:
            logger.info("  Using structured outputs for correction")
            corrected, action, reasoning, new_type = self._correct_edge_with_structured_outputs(
                corrector_llm=corrector_llm,
                source=src,
                target=tgt,
                rel_type=rel_type,
                props=props
            )
        else:
            # Fallback to agent-based approach
            logger.info("  Using agent-based correction")
            corrected, action, reasoning, new_type = self._correct_edge_with_agent(
                agent=agent,
                source=src,
                target=tgt,
                rel_type=rel_type,
                props=props,
                citation_text_map={},  # Don't pass citations to corrector
                available_variables=available_variables
            )
        
        logger.info(f"  Agent decision: action={action}, corrected={corrected}")
        logger.info(f"  Reasoning: {reasoning}")
        if new_type:
            logger.info(f"  New type: {new_type}")
        
        # Save the corrector reasoning to Neo4j as corrector_message
        if corrected and reasoning:
            try:
                with self.graph_db._get_session() as neo_session:
                    update_query = """
                    MATCH (s:variable {name: $src, session_id: $sid})-[r]->(t:variable {name: $tgt, session_id: $sid})
                    WHERE r.corrected = true
                    SET r.corrector_message = $message
                    RETURN count(r) as updated
                    """
                    result = neo_session.run(update_query, {
                        'src': src,
                        'tgt': tgt,
                        'sid': self.session_id,
                        'message': reasoning
                    })
                    updated = result.single()
                    if updated and updated['updated'] > 0:
                        logger.info(f"  Saved corrector message to {updated['updated']} edge(s)")
            except Exception as e:
                logger.warning(f"  Failed to save corrector message to Neo4j: {e}")
        
        # Build and return the outcome
        outcome = {
            "source": src,
            "target": tgt,
            "type": rel_type,
            "action": action,
            "corrected": corrected,
            "note": reasoning
        }
        if new_type:
            outcome["new_type"] = new_type
            
        return outcome

    def correct_edges_serial(
        self,
        corrector_models=None,
        max_rounds=1,
        action_order=("revise", "recite", "remove"),
        rejudge_after=True,
        rejudge_approach="correctness",  # NEW: Approach for re-judging after correction
        rejudge_parallel=False,  # NEW: Enable parallel re-judging
        rejudge_max_workers=10,  # NEW: Max workers for parallel re-judging
        judge_models=None,
        num_judges=None,
        use_simple_corrector=False,  # NEW: Use simple add/remove only (no revise/flip/change)
        corrector_prompt_variant="original",  # NEW: Which prompt to use from corrector_prompts.yaml
        correction_parallel=False,  # NEW: Enable parallel correction
        correction_max_workers=5,  # NEW: Max workers for parallel correction
    ):
        """
        LLM-as-a-corrector pipeline.
        - Select candidate edges (is_corrupted OR judged inconsistent/not supported/no evidence)
        - Actions in order: revise rationale, recite citations (if missing), remove edge
        - Optionally re-judge after corrections
        
        Args:
            corrector_models: Model(s) to use for correction
            use_simple_corrector: If True, use only remove_edge and create_edge tools (no revise/flip/change)
            corrector_prompt_variant: Which prompt to use ("original", "improved", "aggressive_fix")
            max_rounds: Maximum correction rounds per edge
            action_order: Order of correction actions to try
            rejudge_after: Whether to re-judge all edges after corrections
            judge_models: Model(s) to use for re-judging (if rejudge_after=True)
            num_judges: Number of judges for re-judging (if rejudge_after=True)
            
        Returns list of outcomes per edge.
        """
        import logging
        logger = logging.getLogger(__name__)
        outcomes = []

        q = (
            """
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $sid AND t.session_id = $sid
            AND (
                r.judge_verdict IN ['INCONSISTENT','Contradicted','Not supported','NO_MOTIVATION','NO_CITATION','INCORRECT','ERROR','PARTIALLY_CORRECT'] OR
                coalesce(r.aggregate_verdict,'') IN ['Not supported', 'INCORRECT', 'PARTIALLY_CORRECT'] OR
                (r.aggregate_score IS NOT NULL AND r.aggregate_score < 1.0)
            )
            RETURN s.name AS src, t.name AS tgt, type(r) AS rel_type, properties(r) AS props
            """
        )
        with self.graph_db._get_session() as session:
            recs = list(session.run(q, {"sid": self.session_id}))
        if not recs:
            logger.info("No candidate edges found for correction.")
            return outcomes

        if corrector_models is None:
            raise ValueError(
                "No corrector model selected. Please provide corrector_models parameter."
            )
        elif isinstance(corrector_models, str):
            models = [corrector_models]
        else:
            models = list(corrector_models)
        
        # Build corrector LLM clients
        correctors = [self._build_corrector_client(m) for m in models]
        
        # Create agent instances for each corrector
        if use_simple_corrector:
            logger.info(f"Creating {len(correctors)} SIMPLE ADD/REMOVE agents...")
            agents = [self._create_simple_deleter_agent(corr) for corr in correctors]
        else:
            logger.info(f"Creating {len(correctors)} SOPHISTICATED corrector agents (prompt: {corrector_prompt_variant})...")
            try:
                agents = [self._create_corrector_agent(corr, prompt_variant=corrector_prompt_variant) for corr in correctors]
            except Exception as e:
                logger.error(f"Failed to create corrector agents: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Get all available variables in the graph for context
        all_variables = self.get_variables()
        available_variables = [var.get("name") for var in all_variables if var.get("name")]
        logger.info(f"Available variables for corrector context: {len(available_variables)}")

        # Process candidate edges with the agent (parallel or sequential)
        if correction_parallel:
            # PARALLEL CORRECTION MODE
            from concurrent.futures import ThreadPoolExecutor, as_completed
            logger.info(f"\nStarting PARALLEL correction for {len(recs)} edges with {correction_max_workers} workers...")
            
            agent = agents[0]
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=correction_max_workers) as executor:
                # Submit all edge corrections as futures
                futures = {
                    executor.submit(self._correct_single_edge, agent, rec, available_variables): rec
                    for rec in recs
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    rec = futures[future]
                    try:
                        outcome = future.result()
                        outcomes.append(outcome)
                        completed_count += 1
                        logger.info(f"[{completed_count}/{len(recs)}] Completed: {outcome['source']}->{outcome['target']} ({outcome['action']})")
                    except Exception as e:
                        src = rec.get("src", "unknown")
                        tgt = rec.get("tgt", "unknown")
                        logger.error(f"Error correcting edge {src}->{tgt}: {e}")
                        # Record failed outcome
                        outcomes.append({
                            "source": src,
                            "target": tgt,
                            "type": rec.get("rel_type", "unknown"),
                            "action": "error",
                            "corrected": False,
                            "note": str(e)
                        })
                        completed_count += 1
            
            logger.info(f"Parallel correction complete: {len(outcomes)} edges processed")
        else:
            # SEQUENTIAL CORRECTION MODE (original behavior)
            logger.info(f"\nStarting SEQUENTIAL correction for {len(recs)} candidate edges...")
            agent = agents[0]
            
            # Get the LLM client for structured outputs
            corrector_llm = correctors[0] if correctors else None
            
            for edge_idx, rec in enumerate(recs, 1):
                outcome = self._correct_single_edge(
                    agent=agent,
                    rec=rec,
                    available_variables=available_variables,
                    edge_idx=edge_idx,
                    total_edges=len(recs),
                    corrector_llm=corrector_llm,
                    use_structured_outputs=True  # Use structured outputs when available
                )
                outcomes.append(outcome)

        if rejudge_after:
            try:
                _ = self.judge_all_edges_with_citations_serial(
                    judge_models=judge_models,
                    num_judges=num_judges,
                    approach=rejudge_approach,  # Use configurable approach for re-judging after correction
                    judge_parallel=rejudge_parallel,  # Enable parallel re-judging if requested
                    judge_max_workers=rejudge_max_workers  # Use specified max workers
                )
            except Exception as e:
                logger.warning(f"Re-judge after corrections failed: {e}")

        return outcomes

    

    

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
        
        Supports both direct format and Chain-of-Thought format where reasoning
        steps appear before the final verdict.
        """
        verdict_match = re.search(
            r"VERDICT:\s*(SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|UNADDRESSED)",
            llm_content,
            re.IGNORECASE
        )
        # Extract reason up to double newline or end of string (handles CoT format)
        reason_match = re.search(r"REASON:\s*(.+?)(?:\n\n|$)", llm_content, re.DOTALL | re.IGNORECASE)

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


        
    def _update_edge_citation_info_with_relationship(self,
            source: str,
            target: str,
            rel_type: str,
            citation: str,
            relevance: str,
            relationship_text: str,  # Added missing parameter
        ):
        """
        Minimal helper: sets 
        r.citations                 = [citation],
        r.citation_relevance        = relevance,
        r.citation_relationship_motivation = relationship_text

        on the edge (source)-[rel_type]->(target).
        """
        query = f"""
        MATCH (s:variable {{name:$src, session_id:$sid}})-[r:{rel_type}]->(t:variable {{name:$tgt, session_id:$sid}})
        SET r.citations = $cites,
            r.citation_relevance = $relevance,
            r.citation_relationship_motivation = $relationship_text
        RETURN r
        """
        params = {
            "src": source,
            "tgt": target,
            "sid": self.session_id,
            "cites": [citation] if citation else [],
            "relevance": relevance,
            "relationship_text": relationship_text
        }
        with self.graph_db._get_session() as session:
            session.run(query, params)
# Configuration parameters
# Edit these parameters as needed
# num_variables = 5
# output_file = 'causal_graph.json'
# target_variable = 'Lower Back Pain'
# temporal_scale = 'Years'
# spatial_scale = 'Community Dwelling Elderly'
# parallel_processing = False
# max_workers = 7
# yaml_path = "../backend/configs/prompts.yaml"
# dev_mode = False
# corruption_rate = 1

# Environment-specific configurations
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "your-api-key")
PERPLEXITY_API_URL = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/chat/completions")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
# Get Claude API credentials from environment
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "your-claude-api-key")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")


# Get OpenAI API credentials from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


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
        generator_config=os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219"),
        corruptor_config=os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219"),
        judge_config=os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
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
        query += " AND r.citations IS NOT NULL AND size(r.citations) > 0"
    if only_consistent_edges:
        query += " AND r.judge_verdict = 'OK'"
    
    query += """
    RETURN s.name AS source, t.name AS target
    """
    
    # For each session ID, fetch nodes and edges
    for session_id in session_ids:
        params = {"session_id": session_id}
        
        print(f"🔍 EXTRACT_NETWORKX DEBUG: Session ID = {session_id}")
        print(f"   Query filters: only_cited_edges={only_cited_edges}, only_consistent_edges={only_consistent_edges}")
        print(f"   Full query: {query}")
        
        # Execute query with current session ID
        with discovery.graph_db._get_session() as session:
            results = list(session.run(query, params))
            
        print(f"   Edges found: {len(results)}")
        for record in results:
            print(f"     - {record['source']} → {record['target']}")
        
        # Add nodes and edges to the graph
        for record in results:
            src = record["source"]
            tgt = record["target"]
            
            G.add_node(src)
            G.add_node(tgt)
            G.add_edge(src, tgt)
        
        # If no edges found, also query nodes directly (for node_generation_only mode)
        if len(results) == 0:
            print(f"   No edges found - querying nodes directly")
            node_query = """
            MATCH (n:variable)
            WHERE n.session_id = $session_id AND n.deleted = false
            RETURN n.name AS name
            """
            with discovery.graph_db._get_session() as session:
                node_results = list(session.run(node_query, params))
            
            print(f"   Nodes found: {len(node_results)}")
            for record in node_results:
                G.add_node(record["name"])
                print(f"     - {record['name']}")
    
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


def _build_node_text_map_for_embedding(G, nodes):
    """Return dict name->text using node attributes if available."""
    text_map = {}
    for n in nodes:
        attrs = G.nodes.get(n, {}) if hasattr(G, "nodes") else {}
        desc = attrs.get("description") or attrs.get("definition") or ""
        unit = attrs.get("units") or ""
        pieces = [str(n)]
        if desc:
            pieces.append(str(desc))
        if unit:
            pieces.append(f"units: {unit}")
        text_map[n] = " | ".join(pieces)
    return text_map


def _compute_node_cosine_candidates(session_nodes, validation_nodes, G_session, G_validation, percentile=85.0, *, model="text-embedding-3-small", dimensions=None):
    """
    Compute cosine similarity between session and validation node texts, return
    candidate pairs above the given percentile threshold.
    Returns (candidates, score_map) where candidates is List[(s, v, sim)].
    """
    import os
    import numpy as np
    from data_science.logit_metrics import openai_embed_batch

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return [], {}

    session_nodes_list = list(session_nodes)
    validation_nodes_list = list(validation_nodes)

    s_texts_map = _build_node_text_map_for_embedding(G_session, session_nodes_list)
    v_texts_map = _build_node_text_map_for_embedding(G_validation, validation_nodes_list)

    s_texts = [s_texts_map[n] for n in session_nodes_list]
    v_texts = [v_texts_map[n] for n in validation_nodes_list]

    # Embed
    s_vecs = openai_embed_batch(s_texts, openai_key, model=model, dimensions=dimensions)
    v_vecs = openai_embed_batch(v_texts, openai_key, model=model, dimensions=dimensions)
    if not s_vecs or not v_vecs:
        return [], {}

    # Normalize for cosine via dot
    s_mat = np.vstack(s_vecs)
    v_mat = np.vstack(v_vecs)
    s_norm = s_mat / (np.linalg.norm(s_mat, axis=1, keepdims=True) + 1e-12)
    v_norm = v_mat / (np.linalg.norm(v_mat, axis=1, keepdims=True) + 1e-12)
    sims = s_norm @ v_norm.T  # shape [S, V]

    # Threshold by percentile over all scores
    all_scores = sims.flatten()
    thr = float(np.percentile(all_scores, percentile)) if all_scores.size else 1.0

    candidates = []
    score_map = {}
    for i, s in enumerate(session_nodes_list):
        for j, v in enumerate(validation_nodes_list):
            score = float(sims[i, j])
            score_map[(s, v)] = score
            if score >= thr:
                candidates.append((s, v, score))
    # Sort high->low
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates, score_map


def _maximum_weight_bipartite_matching(pairs_with_weight):
    """Greedy fallback maximum matching on weighted pairs (s, v, w)."""
    matched_s = set()
    matched_v = set()
    mapping = {}
    for s, v, w in sorted(pairs_with_weight, key=lambda t: t[2], reverse=True):
        if s in matched_s or v in matched_v:
            continue
        mapping[s] = v
        matched_s.add(s)
        matched_v.add(v)
    return mapping


def infer_context_from_validation_nodes(validation_nodes: list, generator_llm) -> dict:
    """
    Infer temporal scale, spatial scale, and context description from validation variable names.
    Uses the generator LLM directly (not pydantic_ai) to ensure we use the configured model.
    
    Args:
        validation_nodes: List of validation variable names
        generator_llm: LLM client to use for inference
    
    Returns:
        dict with keys: temporal_scale, spatial_scale, context_description, context_list
    """
    import json
    import re
    
    CONTEXT_PROMPT = f"""You are an expert in causal systems analysis.

Given these variables from a causal loop diagram, infer the context of the system:

VARIABLES:
{chr(10).join(f"- {v}" for v in validation_nodes)}

Analyze these variables and infer:
1. **Temporal Scale**: How long do causal effects take? (e.g., "days to weeks", "2-5 years", "decades")
2. **Spatial Scale**: What level is this system operating at? (e.g., "individual level", "community level", "national level", "global")
3. **Domain**: What research field or domain? (e.g., "public health", "environmental science", "economics")
4. **Context Description**: A brief 1-2 sentence description of what this causal system is studying

Be specific and evidence-based from the variable names provided.

Return your answer in JSON format:
{{"temporal_scale": "...", "spatial_scale": "...", "domain": "...", "context_description": "..."}}
"""
    
    try:
        # Use the generator_llm directly to ensure we use the configured model (e.g., gpt-4.1)
        response = generator_llm.send_message(
            sys_prompt="You are an expert in causal systems analysis.",
            usr_prompt=CONTEXT_PROMPT,
            stream=False
        )
        data = generator_llm.parse_static_response(response)
        content = data.get('content', '').strip()
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            inferred = json.loads(json_match.group(0))
        else:
            inferred = json.loads(content)
        
        context_list = [
            f"Domain: {inferred.get('domain', 'unknown')}",
            f"Context: {inferred.get('context_description', 'Not specified')}",
            f"Temporal Scale: {inferred.get('temporal_scale', 'unknown')}",
            f"Spatial Scale: {inferred.get('spatial_scale', 'unknown')}"
        ]
        
        return {
            "temporal_scale": inferred.get("temporal_scale"),
            "spatial_scale": inferred.get("spatial_scale"),
            "domain": inferred.get("domain"),
            "context_description": inferred.get("context_description"),
            "context_list": context_list
        }
    except Exception as e:
        print(f"⚠️  ERROR inferring context from validation nodes: {e}")
        traceback.print_exc()
        return {
            "temporal_scale": None,
            "spatial_scale": None,
            "domain": None,
            "context_description": None,
            "context_list": []
        }


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
    node_match_mode: str = "llm_batch",  # "llm_batch" (single call), "llm" (pairwise), "cosine", "hybrid"
    cosine_percentile: float = 85.0,
    cosine_model: str = "text-embedding-3-small",
    cosine_dimensions: Optional[int] = None,
    # Context-insensitive metrics options
    embedding_enable: bool = True,
    ci_fetch_reuse: bool = True,
    ci_chunk_chars_override: Optional[int] = None,
    ci_overlap_ratio: float = 0.2,
    # Seeded target variable (if used, compute adjusted metrics excluding it)
    seeded_target_variable: str = None,
    # Skip node comparison (use when node metrics are already computed)
    skip_node_comparison: bool = False,
    # Pre-computed node mapping (when skip_node_comparison=True, use this for hybrid metrics)
    precomputed_node_mapping: dict = None,
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
            - node_mapping (dict) if computed
            - node_precision, node_recall, node_f1 (if mapping computed)
            - tp, fp, fn, tn
            - precision, recall, f1, accuracy
            - avg_perplexity, avg_min_prob, avg_max_window_entropy, avg_cosine_similarity
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

    # Compare sameness of variables (configurable modes)
    node_mapping = {}
    node_precision = node_recall = node_f1 = None
    ratio_vars_correct = 'n/a'
    total_comparisons = 'n/a'
    correct = 'n/a'
    incorrect = 'n/a'
    score_map = {}  # Initialize to avoid UnboundLocalError
    candidates = []

    # Always compute cosine similarity scores for cosine/hybrid metrics
    # (even when skipping LLM-based node matching)
    if session_nodes and validation_nodes:
            candidates, score_map = _compute_node_cosine_candidates(
                session_nodes,
                validation_nodes,
                G_session,
                G_validation,
                percentile=cosine_percentile,
                model=cosine_model,
                dimensions=cosine_dimensions,
            )
    
    if skip_node_comparison:
        # Skip LLM-based node matching - use pre-computed node_mapping
        # But we still have score_map for cosine/hybrid calculations
        print(f"\n⏭️  SKIPPING LLM NODE MATCHING (using pre-computed node_mapping)")
        if precomputed_node_mapping:
            node_mapping = precomputed_node_mapping
            matched = len(node_mapping)
            node_precision = matched / len(session_nodes) if session_nodes else 0.0
            node_recall = matched / len(validation_nodes) if validation_nodes else 0.0
            node_f1 = (2 * node_precision * node_recall / (node_precision + node_recall)) if (node_precision + node_recall) > 0 else 0.0
            print(f"   ✅ Using precomputed mapping: {matched} matches (P={node_precision:.3f}, R={node_recall:.3f}, F1={node_f1:.3f})")
        else:
            print(f"   ⚠️  Warning: skip_node_comparison=True but no precomputed_node_mapping provided")
    elif compare_variable_overlap or node_match_mode in ("llm", "llm_batch", "cosine", "hybrid"):
        print(f"\n🔍 STARTING NODE COMPARISON PHASE")
        print(f"Session nodes ({len(session_nodes)}): {list(session_nodes)}")
        print(f"Validation nodes ({len(validation_nodes)}): {list(validation_nodes)}")
        mode = node_match_mode or "llm"
        print(f"Comparison mode: {mode}")

        # Note: cosine similarity scores (score_map) already computed above

        if mode == "cosine":
            # Build a conflict-free mapping using greedy max weight matching
            node_mapping = _maximum_weight_bipartite_matching(candidates)
            matched = len(node_mapping)
            node_precision = matched / len(session_nodes) if session_nodes else 0.0
            node_recall = matched / len(validation_nodes) if validation_nodes else 0.0
            node_f1 = (2 * node_precision * node_recall / (node_precision + node_recall)) if (node_precision and node_recall and (node_precision + node_recall)) else 0.0
            ratio_vars_correct = node_f1
            correct = matched
            incorrect = 0
            total_comparisons = len(candidates)

        elif mode == "llm_batch":
            # Single LLM call to match all variables at once (much faster than pairwise)
            print(f"\n{'='*60}")
            print(f"LLM BATCH MATCHING: Single call for all {len(session_nodes)}×{len(validation_nodes)} pairs")
            print(f"{'='*60}\n")
            
            # Note: score_map is already computed above for cosine/hybrid metrics
            
            from pydantic_ai import Agent
            from pydantic import BaseModel, Field
            
            class VariableMatch(BaseModel):
                generated_variable: str = Field(..., description="Name of the generated variable")
                validation_variable: str = Field(..., description="Name of the validation variable it matches, or 'NO_MATCH' if no match")
                confidence: float = Field(..., description="Confidence score 0.0-1.0 for this match")
                reasoning: str = Field(..., description="Brief explanation for the match or non-match")
            
            class BatchMatchingResult(BaseModel):
                matches: list[VariableMatch] = Field(..., description="List of matches for each generated variable")
            
            BATCH_PROMPT = f"""You are an expert in variable semantics for causal discovery.

Given two lists of variables, find the best semantic match for each GENERATED variable from the VALIDATION list.

GENERATED VARIABLES:
{chr(10).join(f"- {s}" for s in session_nodes)}

VALIDATION VARIABLES:
{chr(10).join(f"- {v}" for v in validation_nodes)}

For each generated variable, identify its best match from the validation list. If no good semantic match exists, set validation_variable to "NO_MATCH".

Consider:
- Semantic equivalence (e.g., "Physical Activity Level" ≈ "Physical_Activity")
- Conceptual overlap (e.g., "Community Sugar Intake" is related to but not the same as "Social_Norms")
- Each generated variable should match at most ONE validation variable (1:1 or 1:0 mapping)

Return confidence scores:
- 1.0 = Exact/very strong match
- 0.7-0.9 = Good semantic match
- 0.4-0.6 = Partial/weak match
- 0.0-0.3 = No real match (use NO_MATCH)
"""
            
            try:
                batch_agent = Agent(
                    "openai:gpt-4o",
                    output_type=BatchMatchingResult,
                    system_prompt=BATCH_PROMPT
                )
                result = batch_agent.run_sync("")
                
                # Process results
                node_mapping = {}
                positives = []
                correct = 0
                incorrect = 0
                
                for match in result.output.matches:
                    gen_var = match.generated_variable
                    val_var = match.validation_variable
                    confidence = match.confidence
                    reasoning = match.reasoning
                    
                    print(f"[{gen_var}] → [{val_var}] (confidence: {confidence:.2f})")
                    print(f"  Reasoning: {reasoning}")
                    
                    # Hybrid strategy: Use LLM if confidence >= 0.7, otherwise use cosine similarity
                    if val_var != "NO_MATCH" and confidence >= 0.7:
                        # High confidence LLM match - trust it
                        node_mapping[gen_var] = val_var
                        positives.append((gen_var, val_var, confidence))
                        correct += 1
                        if confidence >= 1.0:
                            print(f"  ✅ PERFECT MATCH (LLM confidence: 1.0)")
                        else:
                            print(f"  ✅ STRONG MATCH (LLM confidence: {confidence:.2f})")
                    else:
                        # Low confidence or NO_MATCH - use cosine similarity fallback
                        if score_map:
                            best_cosine_match = None
                            best_cosine_score = 0.0
                            for v in validation_nodes:
                                cosine_score = score_map.get((gen_var, v), 0.0)
                                if cosine_score > best_cosine_score:
                                    best_cosine_score = cosine_score
                                    best_cosine_match = v
                            
                            if best_cosine_score >= 0.7:
                                # High cosine similarity - use it
                                node_mapping[gen_var] = best_cosine_match
                                positives.append((gen_var, best_cosine_match, best_cosine_score))
                                correct += 1
                                print(f"  🔄 COSINE MATCH: [{gen_var}] → [{best_cosine_match}] (cosine: {best_cosine_score:.2f})")
                                print(f"     LLM confidence too low ({confidence:.2f}), using cosine similarity")
                            else:
                                incorrect += 1
                                print(f"  ❌ NO MATCH (LLM conf: {confidence:.2f}, best cosine: {best_cosine_score:.2f})")
                        else:
                            incorrect += 1
                            print(f"  ❌ NO MATCH (LLM conf: {confidence:.2f}, no cosine data)")
                
                matched = len(node_mapping)
                node_precision = matched / len(session_nodes) if session_nodes else 0.0
                node_recall = matched / len(validation_nodes) if validation_nodes else 0.0
                node_f1 = (2 * node_precision * node_recall / (node_precision + node_recall)) if (node_precision + node_recall) > 0 else 0.0
                ratio_vars_correct = node_f1
                total_comparisons = len(session_nodes)
                
                print(f"\n{'='*60}")
                print(f"BATCH MATCHING COMPLETE: {matched} matches found in 1 LLM call")
                print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"⚠️  ERROR in batch matching: {e}")
                traceback.print_exc()
                # Fallback to empty mapping
                node_mapping = {}
                matched = 0
                node_precision = node_recall = node_f1 = 0.0
                ratio_vars_correct = 0.0
                correct = incorrect = 0
                total_comparisons = 0

        elif mode == "hybrid" or mode == "llm":
            # LLM verification: all-pairs if llm, else only cosine candidates
            to_check = None
            if mode == "llm":
                to_check = [(s, v, 1.0) for s in session_nodes for v in validation_nodes]
            else:
                to_check = candidates

            PROMPT = """
You are an expert in variable semantics.
Decide if two variable names refer to the same underlying concept.
Return a boolean and a one-sentence rationale.
"""

            # Import locally to avoid module import errors when node comparison is disabled
            from pydantic_ai import Agent, RunContext  # type: ignore
            from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField

            class VariableComparison(PydanticBaseModel):
                target: str = PydanticField(..., description="First variable name")
                query: str = PydanticField(..., description="Second variable name to compare")

            class OutputEquivalent(PydanticBaseModel):
                MeansEquivalent: bool = PydanticField(..., description="True if the two variables are semantically equivalent")
                Reasoning: str = PydanticField(..., description="Short explanation of the decision")
            equivalency_checker_agent = Agent(
                'openai:gpt-4o',
                deps_type=VariableComparison,
                output_type=OutputEquivalent,
                system_prompt="You are an expert in variable semantics.",
            )

            @equivalency_checker_agent.system_prompt
            def system_prompt_with_vars(ctx: RunContext[VariableComparison]) -> str:
                return (
                    f"{PROMPT}\n\n"
                    f"Compare:\n"
                    f"- Target: {ctx.deps.target}\n"
                    f"- Query: {ctx.deps.query}"
                )

            positives = []  # (s, v, score)
            correct = 0
            incorrect = 0
            total_comparisons = 0

            print(f"\n{'='*60}")
            print(f"VARIABLE COMPARISON PHASE: Comparing {len(to_check)} variable pairs using LLM")
            print(f"Mode: {mode}")
            print(f"{'='*60}")

            for i, (s, v, score) in enumerate(to_check, 1):
                try:
                    print(f"[{i}/{len(to_check)}] Comparing session '{s}' ↔ validation '{v}' (score={score:.3f})")
                    comp = VariableComparison(target=str(s), query=str(v))
                    result = equivalency_checker_agent.run_sync(
                        "Are these variables semantically equivalent?",
                        deps=comp,
                    )
                    is_eq = bool(result.output.MeansEquivalent)
                    reasoning = str(result.output.Reasoning)
                    
                    if is_eq:
                        positives.append((s, v, score_map.get((s, v), score)))
                        correct += 1
                        print(f"  ✅ EQUIVALENT: {reasoning}")
                    else:
                        incorrect += 1
                        print(f"  ❌ DIFFERENT: {reasoning}")
                    total_comparisons += 1
                except Exception as e:
                    print(f"  ⚠️  ERROR comparing '{s}' and '{v}': {e}")
                    traceback.print_exc()

            print(f"\n{'='*60}")
            print(f"VARIABLE COMPARISON COMPLETE: {correct} equivalent, {incorrect} different, {len(to_check)} total")
            print(f"{'='*60}\n")

            # Build mapping from positives with highest scores, conflict-free
            node_mapping = _maximum_weight_bipartite_matching(positives)
            matched = len(node_mapping)
            node_precision = matched / len(session_nodes) if session_nodes else 0.0
            node_recall = matched / len(validation_nodes) if validation_nodes else 0.0
            node_f1 = (2 * node_precision * node_recall / (node_precision + node_recall)) if (node_precision and node_recall and (node_precision + node_recall)) else 0.0
            ratio_vars_correct = node_f1
        
        print(f"\n✅ NODE COMPARISON PHASE COMPLETE")
        print(f"Final node mapping: {node_mapping}")
        print(f"Node precision: {node_precision:.3f}, recall: {node_recall:.3f}, F1: {node_f1:.3f}")
        print(f"Matched {len(node_mapping)} variable pairs")

    
    if filter_validation_nodes_to_session_nodes and not node_mapping:
        G_filtered_validation = filter_validation_graph_to_session_nodes(G_validation, session_nodes, plot_graph=True)
        validation_edges = set(G_filtered_validation.edges())
        validation_nodes = set(G_filtered_validation.nodes())

    # 3) Directly compare edge sets (optionally after node mapping)
    #    -------------------------------------
    #    True Positive (TP) = edges in both
    #    False Positive (FP) = edges only in session graph
    #    False Negative (FN) = edges only in validation graph
    #    True Negative (TN) = pairs of nodes that are in neither set
    
    # If we computed a node mapping, remap validation nodes to session label space
    effective_validation_edges = validation_edges
    effective_validation_nodes = validation_nodes
    if node_mapping:
        inv_map = {v: s for s, v in node_mapping.items()}
        remapped_edges = set()
        for src, tgt in validation_edges:
            s2 = inv_map.get(src, src)
            t2 = inv_map.get(tgt, tgt)
            remapped_edges.add((s2, t2))
        effective_validation_edges = remapped_edges
        effective_validation_nodes = {inv_map.get(n, n) for n in validation_nodes}

    # Intersection of edges: found in both
    overlap_set = session_edges & effective_validation_edges
    tp = len(overlap_set)

    # Edges that session has but validation doesn't
    fp = len(session_edges - effective_validation_edges)

    # Edges that validation has but session doesn't
    fn = len(effective_validation_edges - session_edges)

    # 4) Compute TN:
    #    The universe is all possible directed pairs among the union of *all* nodes
    #    (since we're not filtering/discarding anything).
    union_nodes = session_nodes.union(effective_validation_nodes)
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

    # Context-insensitive metrics (averages across session edges)
    if embedding_enable:
        try:
            ci_avgs = _compute_session_context_insensitive_metrics(
                discovery=discovery,
                session_ids=session_ids if session_ids else discovery.session_id,
                ci_fetch_reuse=ci_fetch_reuse,
                ci_chunk_chars_override=ci_chunk_chars_override,
                ci_overlap_ratio=ci_overlap_ratio,
            )
        except Exception:
            ci_avgs = {
                "avg_perplexity": None,
                "avg_min_prob": None,
                "avg_max_window_entropy": None,
                "avg_cosine_similarity": None,
            }
    else:
        ci_avgs = {
            "avg_perplexity": None,
            "avg_min_prob": None,
            "avg_max_window_entropy": None,
            "avg_cosine_similarity": None,
        }

    # Compute pure cosine similarity metrics (all nodes use their max cosine similarity)
    cosine_node_precision = None
    cosine_node_recall = None
    cosine_node_f1 = None
    avg_cosine_gen_to_val = None
    avg_cosine_val_to_gen = None
    
    # Compute hybrid metrics (matched nodes=1.0, non-matched=cosine similarity)
    # This is the recommended approach that ensures hybrid >= binary
    avg_similarity_gen_to_val = None
    avg_similarity_val_to_gen = None
    hybrid_node_precision = None
    hybrid_node_recall = None
    hybrid_node_f1 = None
    
    if score_map and session_nodes and validation_nodes:
        # FIRST: Pure cosine similarity (all nodes use their max similarity)
        cosine_gen_sims = []
        for s in session_nodes:
            max_sim = max([score_map.get((s, v), 0.0) for v in validation_nodes], default=0.0)
            cosine_gen_sims.append(max_sim)
        
        cosine_val_sims = []
        for v in validation_nodes:
            max_sim = max([score_map.get((s, v), 0.0) for s in session_nodes], default=0.0)
            cosine_val_sims.append(max_sim)
        
        avg_cosine_gen_to_val = sum(cosine_gen_sims) / len(cosine_gen_sims) if cosine_gen_sims else 0.0
        avg_cosine_val_to_gen = sum(cosine_val_sims) / len(cosine_val_sims) if cosine_val_sims else 0.0
        
        cosine_node_precision = avg_cosine_gen_to_val
        cosine_node_recall = avg_cosine_val_to_gen
        cosine_node_f1 = (
            2 * cosine_node_precision * cosine_node_recall / (cosine_node_precision + cosine_node_recall)
            if (cosine_node_precision + cosine_node_recall) > 0 else 0.0
        )
        
        print(f"\n📊 COSINE SIMILARITY METRICS (pure cosine, all nodes):")
        print(f"   Avg Cosine (Generated → Validation): {avg_cosine_gen_to_val:.3f}")
        print(f"   Avg Cosine (Validation → Generated): {avg_cosine_val_to_gen:.3f}")
        print(f"   Cosine Node Precision: {cosine_node_precision:.3f}")
        print(f"   Cosine Node Recall: {cosine_node_recall:.3f}")
        print(f"   Cosine Node F1: {cosine_node_f1:.3f}")
        
        # SECOND: Hybrid approach with TWO-STAGE GREEDY BIPARTITE MATCHING
        # Stage 1: LLM matching (already done in node_mapping)
        # Stage 2: Greedy cosine matching on remaining nodes (prevents double-counting)
        
        matched_validation_nodes = set(node_mapping.values()) if node_mapping else set()
        matched_session_nodes = set(node_mapping.keys()) if node_mapping else set()
        
        # Identify unmatched nodes
        unmatched_session = [s for s in session_nodes if s not in matched_session_nodes]
        unmatched_validation = [v for v in validation_nodes if v not in matched_validation_nodes]
        
        # Stage 2: Greedy bipartite matching on remaining nodes using cosine similarity
        cosine_mapping = {}  # Maps unmatched session → (validation, similarity)
        
        if unmatched_session and unmatched_validation:
            # Build candidates: (session_node, validation_node, similarity)
            remaining_candidates = []
            for s in unmatched_session:
                for v in unmatched_validation:
                    sim = score_map.get((s, v), 0.0)
                    remaining_candidates.append((s, v, sim))
            
            # Sort by similarity (highest first) - GREEDY approach
            remaining_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Greedy matching: assign best available pairs
            assigned_session = set()
            assigned_validation = set()
            
            for s, v, sim in remaining_candidates:
                if s not in assigned_session and v not in assigned_validation:
                    cosine_mapping[s] = (v, sim)
                    assigned_session.add(s)
                    assigned_validation.add(v)
            
            # Count how many validation nodes were actually matched in Stage 2
            stage2_val_matched = len(set(v for v, sim in cosine_mapping.values()))
            stage2_gen_matched = len(cosine_mapping)
            
            print(f"\n🔗 TWO-STAGE BIPARTITE MATCHING:")
            print(f"   Stage 1 (LLM): {len(node_mapping)} matches")
            print(f"   Stage 2 (Cosine): {stage2_gen_matched} gen matched, {stage2_val_matched} val matched")
            print(f"   Unmatched generated: {len(unmatched_session) - stage2_gen_matched}")
            print(f"   Unmatched validation: {len(unmatched_validation) - stage2_val_matched}")
        
        # Calculate hybrid scores using the two-stage mapping
        hybrid_gen_sims = []
        for s in session_nodes:
            if s in node_mapping:
                # Stage 1: LLM matched → 1.0
                hybrid_gen_sims.append(1.0)
            elif s in cosine_mapping:
                # Stage 2: Cosine matched → use similarity score
                _, sim = cosine_mapping[s]
                hybrid_gen_sims.append(sim)
            else:
                # Unmatched → 0.0 (no validation node available)
                hybrid_gen_sims.append(0.0)
        
        hybrid_val_sims = []
        # Build reverse mapping for validation nodes
        cosine_reverse = {v: sim for s, (v, sim) in cosine_mapping.items()}
        
        for v in validation_nodes:
            if v in matched_validation_nodes:
                # Stage 1: LLM matched → 1.0
                hybrid_val_sims.append(1.0)
            elif v in cosine_reverse:
                # Stage 2: Cosine matched → use similarity score
                hybrid_val_sims.append(cosine_reverse[v])
            else:
                # Unmatched → 0.0 (no generated node available)
                hybrid_val_sims.append(0.0)
        
        # Compute hybrid averages
        avg_similarity_gen_to_val = sum(hybrid_gen_sims) / len(hybrid_gen_sims) if hybrid_gen_sims else 0.0
        avg_similarity_val_to_gen = sum(hybrid_val_sims) / len(hybrid_val_sims) if hybrid_val_sims else 0.0
        
        # DEBUG: Print detailed breakdown
        print(f"\n🔍 DEBUG HYBRID CALCULATION:")
        print(f"   Generated nodes: {len(session_nodes)}")
        print(f"   Validation nodes: {len(validation_nodes)}")
        print(f"   hybrid_gen_sims: {hybrid_gen_sims}")
        print(f"   hybrid_val_sims: {hybrid_val_sims}")
        print(f"   Sum gen: {sum(hybrid_gen_sims):.3f}, Avg: {avg_similarity_gen_to_val:.3f}")
        print(f"   Sum val: {sum(hybrid_val_sims):.3f}, Avg: {avg_similarity_val_to_gen:.3f}")
        
        # Hybrid metrics (matched=1.0, cosine-matched=similarity, unmatched=0.0)
        hybrid_node_precision = avg_similarity_gen_to_val
        hybrid_node_recall = avg_similarity_val_to_gen
        hybrid_node_f1 = (
            2 * hybrid_node_precision * hybrid_node_recall / (hybrid_node_precision + hybrid_node_recall)
            if (hybrid_node_precision + hybrid_node_recall) > 0 else 0.0
        )
        
        print(f"\n📊 HYBRID NODE METRICS (Two-Stage Bipartite Matching):")
        print(f"   Total matches: {len(node_mapping) + len(cosine_mapping)}")
        print(f"   Avg Hybrid (Generated → Validation): {avg_similarity_gen_to_val:.3f}")
        print(f"   Avg Hybrid (Validation → Generated): {avg_similarity_val_to_gen:.3f}")
        print(f"   Hybrid Node Precision: {hybrid_node_precision:.3f}")
        print(f"   Hybrid Node Recall: {hybrid_node_recall:.3f}")
        print(f"   Hybrid Node F1: {hybrid_node_f1:.3f}")
        print(f"   Verification: Binary ({node_f1:.3f}) ≤ Hybrid ({hybrid_node_f1:.3f})")
    
    # OLD DUPLICATE CODE - DISABLED (hybrid metrics already calculated above)
    if False and score_map and session_nodes and validation_nodes and node_mapping:
        # For each generated node: if it matched, use 1.0, else use its max similarity
        hybrid_gen_scores = []
        for s in session_nodes:
            if s in node_mapping:
                # This node matched - count as 1.0
                hybrid_gen_scores.append(1.0)
            else:
                # This node didn't match - use its max similarity
                max_sim = max([score_map.get((s, v), 0.0) for v in validation_nodes], default=0.0)
                hybrid_gen_scores.append(max_sim)
        
        # For each validation node: if it was matched to, use 1.0, else use its max similarity
        hybrid_val_scores = []
        matched_validation = set(node_mapping.values())
        for v in validation_nodes:
            if v in matched_validation:
                # This node was matched to - count as 1.0
                hybrid_val_scores.append(1.0)
            else:
                # This node wasn't matched - use its max similarity
                max_sim = max([score_map.get((s, v), 0.0) for s in session_nodes], default=0.0)
                hybrid_val_scores.append(max_sim)
        
        # Compute hybrid precision/recall
        hybrid_node_precision = sum(hybrid_gen_scores) / len(hybrid_gen_scores) if hybrid_gen_scores else 0.0
        hybrid_node_recall = sum(hybrid_val_scores) / len(hybrid_val_scores) if hybrid_val_scores else 0.0
        hybrid_node_f1 = (
            2 * hybrid_node_precision * hybrid_node_recall / (hybrid_node_precision + hybrid_node_recall)
            if (hybrid_node_precision + hybrid_node_recall) > 0 else 0.0
        )
        
        print(f"\n📊 HYBRID NODE METRICS (Binary matches=1.0 + Similarity for non-matches):")
        print(f"   Hybrid Node Precision: {hybrid_node_precision:.3f}")
        print(f"   Hybrid Node Recall: {hybrid_node_recall:.3f}")
        print(f"   Hybrid Node F1: {hybrid_node_f1:.3f}")
        print(f"   Verification: Binary ({node_f1:.3f}) ≤ Hybrid ({hybrid_node_f1:.3f})")
    
    # Compute adjusted metrics if a seeded target variable was used
    adjusted_node_precision = adjusted_node_recall = adjusted_node_f1 = None
    adjusted_matched_pairs = None
    adjusted_hybrid_precision = adjusted_hybrid_recall = adjusted_hybrid_f1 = None
    
    if seeded_target_variable and node_mapping:
        # Remove the seeded variable from both sides for adjusted metrics
        adjusted_mapping = {s: v for s, v in node_mapping.items() 
                           if s != seeded_target_variable and v != seeded_target_variable}
        adjusted_session_nodes = [n for n in session_nodes if n != seeded_target_variable]
        adjusted_validation_nodes = [n for n in validation_nodes if n != seeded_target_variable]
        
        adjusted_matched_pairs = len(adjusted_mapping)
        adjusted_node_precision = adjusted_matched_pairs / len(adjusted_session_nodes) if adjusted_session_nodes else 0.0
        adjusted_node_recall = adjusted_matched_pairs / len(adjusted_validation_nodes) if adjusted_validation_nodes else 0.0
        adjusted_node_f1 = (
            2 * adjusted_node_precision * adjusted_node_recall / (adjusted_node_precision + adjusted_node_recall)
            if (adjusted_node_precision + adjusted_node_recall) > 0 else 0.0
        )
        
        # Also compute adjusted hybrid metrics (excluding seed)
        if score_map:
            adj_matched_validation = set(adjusted_mapping.values())
            adj_hybrid_gen_sims = []
            for s in adjusted_session_nodes:
                if s in adjusted_mapping:
                    adj_hybrid_gen_sims.append(1.0)
                else:
                    max_sim = max([score_map.get((s, v), 0.0) for v in adjusted_validation_nodes], default=0.0)
                    adj_hybrid_gen_sims.append(max_sim)
            
            adj_hybrid_val_sims = []
            for v in adjusted_validation_nodes:
                if v in adj_matched_validation:
                    adj_hybrid_val_sims.append(1.0)
                else:
                    max_sim = max([score_map.get((s, v), 0.0) for s in adjusted_session_nodes], default=0.0)
                    adj_hybrid_val_sims.append(max_sim)
            
            adjusted_hybrid_precision = sum(adj_hybrid_gen_sims) / len(adj_hybrid_gen_sims) if adj_hybrid_gen_sims else 0.0
            adjusted_hybrid_recall = sum(adj_hybrid_val_sims) / len(adj_hybrid_val_sims) if adj_hybrid_val_sims else 0.0
            adjusted_hybrid_f1 = (
                2 * adjusted_hybrid_precision * adjusted_hybrid_recall / (adjusted_hybrid_precision + adjusted_hybrid_recall)
                if (adjusted_hybrid_precision + adjusted_hybrid_recall) > 0 else 0.0
            )
        
        print(f"\n📊 ADJUSTED NODE METRICS (excluding seeded target '{seeded_target_variable}'):")
        print(f"   Binary - Adjusted Matched Pairs: {adjusted_matched_pairs}/{len(adjusted_session_nodes)} generated, {adjusted_matched_pairs}/{len(adjusted_validation_nodes)} validation")
        print(f"   Binary - Adjusted Node Precision: {adjusted_node_precision:.3f}")
        print(f"   Binary - Adjusted Node Recall: {adjusted_node_recall:.3f}")
        print(f"   Binary - Adjusted Node F1: {adjusted_node_f1:.3f}")
        if adjusted_hybrid_precision is not None:
            print(f"   Hybrid - Adjusted Precision: {adjusted_hybrid_precision:.3f}")
            print(f"   Hybrid - Adjusted Recall: {adjusted_hybrid_recall:.3f}")
            print(f"   Hybrid - Adjusted F1: {adjusted_hybrid_f1:.3f}")

    metrics = {
        "session_graph_edges": session_edges,
        "validation_graph_edges": validation_edges,
        "session_nodes": session_nodes,
        "validation_nodes": validation_nodes,
        "node_mapping": node_mapping,
        # Binary node metrics
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        # Cosine similarity node metrics (pure cosine)
        "cosine_node_precision": cosine_node_precision,
        "cosine_node_recall": cosine_node_recall,
        "cosine_node_f1": cosine_node_f1,
        "avg_cosine_gen_to_val": avg_cosine_gen_to_val,
        "avg_cosine_val_to_gen": avg_cosine_val_to_gen,
        # Hybrid node metrics (binary matches=1.0 + similarity for non-matches)
        "avg_similarity_gen_to_val": avg_similarity_gen_to_val,
        "avg_similarity_val_to_gen": avg_similarity_val_to_gen,
        # Hybrid node metrics (binary matches=1.0 + similarity for non-matches)
        "hybrid_node_precision": hybrid_node_precision,
        "hybrid_node_recall": hybrid_node_recall,
        "hybrid_node_f1": hybrid_node_f1,
        # Adjusted binary metrics (excluding seed)
        "adjusted_node_precision": adjusted_node_precision,
        "adjusted_node_recall": adjusted_node_recall,
        "adjusted_node_f1": adjusted_node_f1,
        "adjusted_matched_pairs": adjusted_matched_pairs,
        # Adjusted hybrid metrics (excluding seed)
        "adjusted_hybrid_precision": adjusted_hybrid_precision,
        "adjusted_hybrid_recall": adjusted_hybrid_recall,
        "adjusted_hybrid_f1": adjusted_hybrid_f1,
        "seeded_target_variable": seeded_target_variable,
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
        "accuracy": accuracy,
        **ci_avgs,
    }
    
    # DEBUG: Print what's actually in the metrics dict for hybrid
    print(f"\n🔍 DEBUG METRICS DICT:")
    print(f"   hybrid_node_precision: {metrics.get('hybrid_node_precision')}")
    print(f"   hybrid_node_recall: {metrics.get('hybrid_node_recall')}")
    print(f"   hybrid_node_f1: {metrics.get('hybrid_node_f1')}")

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
    import os
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
                rels = discovery.discover_relationships(generation_parallel=generation_parallel, generation_max_workers=generation_max_workers)
                
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





def _build_edge_row(e, classification, session_edges, validation_edges, edge_data):
    """
    Helper to build a dictionary row for a single edge in the final Excel.
    classification is one of "TP","FP","FN".
    """
    (src, tgt) = e
    row = {
        "Source": src,
        "Target": tgt,
        "Relationship Type": "N/A",
        "Classification": classification,
        "In Session Graph": "Yes" if e in session_edges else "No",
        "In Validation Graph": "Yes" if e in validation_edges else "No",
        "Judge Verdict": "N/A",
        "Motivation": "",
        "Citation": "",
        "Judge Message": "",
        "Aggregate Score": "N/A",
        "Corrected": False,
        "Correction Action": "",
        "Corrected Motivation": "",
        "Corrector Message": "",
        "Is Corrupted": False,
        "Spurious Motivation": ""
    }
    if e in edge_data:
        row["Relationship Type"] = edge_data[e].get("relationship_type", "N/A")
        row["Judge Verdict"] = edge_data[e]["judge_verdict"]
        row["Motivation"]    = edge_data[e]["motivation"]
        row["Citation"]      = edge_data[e]["citation"]
        row["Judge Message"] = edge_data[e]["judge_message"]
        row["Aggregate Score"] = edge_data[e]["aggregate_score"]
        row["Corrected"] = edge_data[e]["corrected"]
        row["Correction Action"] = edge_data[e]["correction_action"]
        row["Corrected Motivation"] = edge_data[e]["corrected_motivation"]
        row["Corrector Message"] = edge_data[e]["corrector_message"]
        row["Is Corrupted"] = edge_data[e]["is_corrupted"]
        row["Spurious Motivation"] = edge_data[e]["spurious_motivation"]
    return row


def export_edges_comparison_to_excel(
    discovery,
    session_ids=None,
    validation_vars_json_path=None,
    validation_edges_json_path=None,
    output_filename=None,
    lm_stats: dict = None,
    # Context-insensitive metrics options
    embedding_enable: bool = True,
    ci_fetch_reuse: bool = True,
    ci_chunk_chars_override: Optional[int] = None,
    ci_overlap_ratio: float = 0.2,
    ci_compute_embeddings: bool = True,  # NEW: Control whether to compute expensive embeddings
    ci_parallel: bool = True,  # NEW: Parallel computation for CI metrics
    ci_max_workers: int = 10,  # NEW: Max workers for parallel CI metrics
    embedding_provider: str = "local",  # NEW: Embedding provider for CI metrics ("local" or "openai")
    embedding_device: str = "cuda",  # NEW: Device for local embeddings
    # Optional: include experiment parameters in a dedicated sheet
    experiment_params=None,
    # Optional: node comparison data for Node Analysis sheet
    node_comparison_data: dict = None,
    # Node comparison mode to use for scenario comparisons
    node_match_mode: str = "llm_batch",
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
    import json
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
            "Relationship Type": data.get("relationship_type", "N/A"),
            "Classification": classification,
            "In Session Graph": "Yes" if classification in ("TP","FP") else "No",
            "In Validation Graph": "Yes" if edge in validation_edges else "No",
            "Judge Verdict": data.get("judge_verdict", "N/A"),
            "Motivation": data.get("motivation", ""),
            "Citation": data.get("citation", ""),
            "Judge Message": data.get("judge_message", ""),
            "Aggregate Score": data.get("aggregate_score", "N/A"),
            "Corrected": data.get("corrected", False),
            "Correction Action": data.get("correction_action", ""),
            "Corrected Motivation": data.get("corrected_motivation", ""),
            "Corrector Message": data.get("corrector_message", ""),
            "Is Corrupted": data.get("is_corrupted", False),
            "Corruption Type": data.get("corruption_type", ""),
            "Corruption Subtype": data.get("corruption_subtype", ""),
            "Spurious Motivation": data.get("spurious_motivation", ""),
            "Original Motivation": data.get("original_motivation", "")
        }

    logger.info("📊 EXPORT TO EXCEL - STARTING")
    logger.info(f"  Session ID: {session_ids or discovery.session_id}")
    logger.info(f"  Context-Insensitive metrics: {embedding_enable}")
    
    # 1) Prepare the output filename (always append timestamp for uniqueness)
    ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not output_filename:
        output_filename = f"edges_evaluation_{ts_str}.xlsx"
    else:
        if not output_filename.endswith('.xlsx'):
            output_filename += ".xlsx"
        base, ext = os.path.splitext(output_filename)
        output_filename = f"{base}_{ts_str}{ext}"
    
    logger.info(f"  Output file: {output_filename}")

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
    # Pre-compute context-insensitive metrics for all edges (to attach to detail sheets)
    if embedding_enable:
        logger.info("🧮 PHASE: Computing context-insensitive metrics...")
        try:
            edge_ci_metrics = _compute_edge_level_context_insensitive_metrics(
                discovery=discovery,
                session_id=session_ids or discovery.session_id,
                ci_fetch_reuse=ci_fetch_reuse,
                ci_chunk_chars_override=ci_chunk_chars_override,
                ci_overlap_ratio=ci_overlap_ratio,
                ci_compute_embeddings=ci_compute_embeddings,
                ci_parallel=ci_parallel,
                ci_max_workers=ci_max_workers,
                embedding_provider=embedding_provider,
                embedding_device=embedding_device,
            )
            logger.info(f"✅ CI metrics computed for {len(edge_ci_metrics)} edges")
        except Exception as e:
            logger.error(f"❌ CI metrics computation failed: {e}")
            logger.exception("Full traceback for CI metrics failure:")
            edge_ci_metrics = {}
    else:
        logger.info("⏭️  PHASE: Skipping context-insensitive metrics (embedding_enable=False)")
        edge_ci_metrics = {}

    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:

        # 2) For each scenario, build "detail" & "summary" data frames
        logger.info(f"📑 PHASE: Processing {len(scenarios)} scenario sheets...")
        for idx, scenario in enumerate(scenarios, 1):
            scenario_name = scenario["name"]
            logger.info(f"  [{idx}/{len(scenarios)}] SCENARIO: {scenario_name}")
            try:
                # === (A) Call compare_session_graph_to_validation 
                # BUT: If node_comparison_data is provided, we can skip redundant node calculations
                # by only fetching edge data for this scenario
                logger.info(f"     Getting edges from Neo4j...")
                if node_comparison_data:
                    # Use pre-computed node comparison data
                    # Reconstruct node_mapping from matched_pairs for hybrid metric calculation
                    precomputed_mapping = {}
                    for s, v in node_comparison_data.get("matched_pairs", []):
                        precomputed_mapping[s] = v
                    
                    # Only call compare_session_graph_to_validation to get edge metrics
                    metrics = compare_session_graph_to_validation(
                        discovery=discovery,
                        session_ids=session_ids,
                        validation_vars_json_path=validation_vars_json_path,
                        validation_edges_json_path=validation_edges_json_path,
                        only_cited_edges=scenario["only_cited_edges"],
                        only_consistent_edges=scenario["only_consistent_edges"],
                        plot_session_graph=False,
                        plot_validation_graph=False,
                        node_match_mode=node_match_mode,  # Use the same mode as the main comparison
                        skip_node_comparison=True,  # Skip LLM matching since we already have it
                        precomputed_node_mapping=precomputed_mapping,  # Pass the mapping for hybrid metrics
                        embedding_enable=False  # Skip CI metrics - already computed once before scenario loop
                    )
                    # Override with pre-computed node data
                    session_nodes = node_comparison_data["session_nodes"]
                    validation_nodes = node_comparison_data["validation_nodes"]
                else:
                    # No pre-computed data, do full comparison
                    metrics = compare_session_graph_to_validation(
                        discovery=discovery,
                        session_ids=session_ids,
                        validation_vars_json_path=validation_vars_json_path,
                        validation_edges_json_path=validation_edges_json_path,
                        only_cited_edges=scenario["only_cited_edges"],
                        only_consistent_edges=scenario["only_consistent_edges"],
                        plot_session_graph=False,
                        plot_validation_graph=False,
                        node_match_mode=node_match_mode,
                        embedding_enable=False  # Skip CI metrics - already computed once before scenario loop
                    )
                    session_nodes = metrics["session_nodes"]
                    validation_nodes = metrics["validation_nodes"]
                
                logger.info(f"     Found {len(metrics['session_graph_edges'])} session edges, {len(metrics['validation_graph_edges'])} validation edges")
                
                # We'll get:
                # metrics["session_graph_edges"]   # a set of (src,tgt) from the DB
                # metrics["validation_graph_edges"]# a set from the validation JSON
                session_edges = metrics["session_graph_edges"]
                validation_edges = metrics["validation_graph_edges"]

                # === (B) If scenario has a verdict_filter, do local subsetting
                # We also need "edge_data" from Neo4j to check judge_verdict in Python.
                edge_data = {}
                query = """
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $session_id AND t.session_id = $session_id
                """
                # If scenario says only_cited_edges, add that condition
                if scenario["only_cited_edges"]:
                    query += " AND r.citations IS NOT NULL AND size(r.citations) > 0"

                # If scenario originally had "only_consistent_edges"
                if scenario["only_consistent_edges"]:
                    query += " AND r.judge_verdict = 'OK'"

                # IMPORTANT: now also fetch r.aggregate_score, correction fields, and corruption tracking
                query += """
                RETURN s.name AS source,
                       t.name AS target,
                       type(r) AS relationship_type,
                       r.judge_verdict AS judge_verdict,
                       r.motivation AS motivation,
                       r.original_motivation AS original_motivation,
                       r.citations AS citations,
                       r.judge_message AS judge_message,
                       r.aggregate_score AS aggregate_score,
                       r.corrected AS corrected,
                       r.correction_action AS correction_action,
                       r.corrected_motivation AS corrected_motivation,
                       r.corrector_message AS corrector_message,
                       r.is_corrupted AS is_corrupted,
                       r.spurious_motivation AS spurious_motivation,
                       r.corruption_type AS corruption_type,
                       r.corruption_subtype AS corruption_subtype
                """

                params = {"session_id": session_ids or discovery.session_id}
                
                with discovery.graph_db._get_session() as s:
                    results = s.run(query, params)
                    for rec in results:
                        src = rec["source"]
                        tgt = rec["target"]
                        # Use spurious_motivation if corrupted (what was judged), otherwise use motivation
                        # This matches the logic used during judging
                        motivation_text = rec["spurious_motivation"] or rec["motivation"] or ""
                        # Handle citations - support both list and string formats
                        citations = rec["citations"]
                        if isinstance(citations, list):
                            citation_text = ", ".join(citations)
                        else:
                            citation_text = str(citations) if citations else ""
                        
                        # For corrected edges, use the current motivation as the corrected_motivation
                        # (since revise/change_type update the motivation field)
                        # For removed edges, use corrected_motivation if set, otherwise use corrector_message
                        corrected_mot = rec["corrected_motivation"] or ""
                        if rec["corrected"] and not corrected_mot:
                            # If edge is corrected but corrected_motivation is not set, use current motivation
                            corrected_mot = motivation_text
                        
                        edge_data[(src,tgt)] = {
                            "relationship_type": rec["relationship_type"] or "N/A",
                            "judge_verdict": rec["judge_verdict"] or "N/A",
                            "motivation": motivation_text,
                            "citation": citation_text,
                            "judge_message": rec["judge_message"] or "",
                            "aggregate_score": rec["aggregate_score"] if rec["aggregate_score"] is not None else "N/A",
                            "corrected": rec["corrected"] if rec["corrected"] is not None else False,
                            "correction_action": rec["correction_action"] or "",
                            "corrected_motivation": corrected_mot,
                            "corrector_message": rec["corrector_message"] or "",
                            "is_corrupted": rec["is_corrupted"] if rec["is_corrupted"] is not None else False,
                            "spurious_motivation": rec["spurious_motivation"] or "",
                            "corruption_type": rec["corruption_type"] or "",
                            "corruption_subtype": rec["corruption_subtype"] or "",
                            "original_motivation": rec["original_motivation"] or ""
                        }

                # If scenario has a verdict_filter, refine session_edges (case-insensitive, trim)
                if scenario["verdict_filter"] is not None:
                    def _norm_verdict(x: Optional[str]) -> str:
                        if not x:
                            return ""
                        return str(x).strip().lower()

                    allowed = {_norm_verdict(x) for x in scenario["verdict_filter"]}
                    new_session_edges = set()
                    for (src, tgt) in session_edges:
                        data = edge_data.get((src, tgt))
                        v = _norm_verdict(data.get("judge_verdict") if data else None)
                        if v in allowed:
                            new_session_edges.add((src, tgt))
                    # Replace session_edges with the filtered version
                    session_edges = new_session_edges

                # === (B2) Filter out edges with relationship type "NONE" for accuracy metrics
                # (but they are still shown in the Excel with their relationship type)
                logger.info(f"     Filtering NONE relationships...")
                filtered_session_edges = set()
                for (src, tgt) in session_edges:
                    data = edge_data.get((src, tgt))
                    rel_type = data.get("relationship_type", "") if data else ""
                    if rel_type and rel_type.upper() not in ("NONE", "N/A", "NO RELATIONSHIP"):
                        filtered_session_edges.add((src, tgt))

                # Extract NONE edges for explicit tracking
                none_edges_set = session_edges - filtered_session_edges
                logger.info(f"     Removed {len(none_edges_set)} NONE relationships, keeping {len(filtered_session_edges)} edges")
                
                print(f"\n🔍 FILTERING NONE RELATIONSHIPS:")
                print(f"   Before filtering: {len(session_edges)} edges")
                print(f"   After filtering: {len(filtered_session_edges)} edges")
                print(f"   Removed {len(none_edges_set)} NONE relationships")

                # Use filtered edges for accuracy metrics computation
                session_edges_for_metrics = filtered_session_edges

                # === (C) Recompute confusion matrix with this final session_edges
                logger.info(f"     Computing confusion matrix...")
                tp_edges = session_edges_for_metrics.intersection(validation_edges)
                fp_edges = session_edges_for_metrics - validation_edges
                fn_edges = validation_edges - session_edges_for_metrics
                logger.info(f"     TP={len(tp_edges)}, FP={len(fp_edges)}, FN={len(fn_edges)}")

                # TN => among union of nodes
                union_nodes = session_nodes.union(validation_nodes)
                all_pairs = {
                    (a,b) for a in union_nodes for b in union_nodes if a != b
                }
                combined_edges = session_edges_for_metrics.union(validation_edges)
                tn_edges = all_pairs - combined_edges
                
                # === (C2) Verify NONE edges are correctly classified
                # NONE edges in validation → should be in FN
                # NONE edges not in validation → should be in TN
                none_edges_in_validation = none_edges_set.intersection(validation_edges)
                none_edges_not_in_validation = none_edges_set - validation_edges
                
                print(f"\n📊 NONE EDGE CLASSIFICATION:")
                print(f"   NONE edges in validation (→ FN): {len(none_edges_in_validation)}")
                print(f"   NONE edges not in validation (→ TN): {len(none_edges_not_in_validation)}")
                print(f"   Total NONE edges: {len(none_edges_in_validation) + len(none_edges_not_in_validation)}")
                
                # Verify these are included in fn_edges and tn_edges
                none_fn_overlap = none_edges_in_validation.intersection(fn_edges)
                none_tn_overlap = none_edges_not_in_validation.intersection(tn_edges)
                
                print(f"\n✅ VERIFICATION:")
                print(f"   NONE→FN overlap: {len(none_fn_overlap)}/{len(none_edges_in_validation)} ✓")
                print(f"   NONE→TN overlap: {len(none_tn_overlap)}/{len(none_edges_not_in_validation)} ✓")

                # Build local metrics (tp,fp,fn,tn) => precision, recall, f1, accuracy
                logger.info(f"     Calculating accuracy metrics...")
                tp = len(tp_edges)
                fp = len(fp_edges)
                fn = len(fn_edges)
                tn = len(tn_edges)

                precision = tp / (tp + fp) if (tp+fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp+fn) > 0 else 0.0
                f1 = (2 * precision * recall)/(precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn)/(tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                logger.info(f"     Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Accuracy={accuracy:.3f}")

                variable_correct_score = metrics.get("ratio_vars_correct","N/A")
                correct_vars = metrics.get("correct vars","N/A")
                incorrect_vars = metrics.get("incorrect vars","N/A")
                total_comparisons = metrics.get("total_comparisons","N/A")

                # === (D) Build up the edges_data rows for Excel (attach CI metrics when available)
                logger.info(f"     Building Excel data rows...")
                all_rows = []
                # 1) True Positives
                logger.info(f"     Adding {len(tp_edges)} TP edges to Excel...")
                for e in tp_edges:
                    row = _build_edge_row(e, "TP", session_edges, validation_edges, edge_data)
                    ci = edge_ci_metrics.get(e)
                    if ci:
                        cos = ci.get("cosine_similarity")
                        if cos is None:
                            # Fallbacks if consolidated cosine is missing
                            cos = ci.get("gen_cosine_similarity") or ci.get("judge_cosine_similarity")
                        row.update({
                            "Gen Perplexity": ci.get("gen_perplexity"),
                            "Gen Min Prob": ci.get("gen_min_prob"),
                            "Gen Max Window Entropy": ci.get("gen_max_window_entropy"),
                            "Gen Mean Token Prob": ci.get("gen_mean_token_prob"),
                            "Gen Prob Variance": ci.get("gen_prob_variance"),
                            "Gen Prob Std": ci.get("gen_prob_std"),
                            "Gen Max Prob Diff": ci.get("gen_max_prob_diff"),
                            "Gen Token Prob Slope": ci.get("gen_token_prob_slope"),
                            "Gen Cosine Similarity": ci.get("gen_cosine_similarity"),
                            "Judge Perplexity": ci.get("judge_perplexity"),
                            "Judge Min Prob": ci.get("judge_min_prob"),
                            "Judge Max Window Entropy": ci.get("judge_max_window_entropy"),
                            "Judge Mean Token Prob": ci.get("judge_mean_token_prob"),
                            "Judge Prob Variance": ci.get("judge_prob_variance"),
                            "Judge Prob Std": ci.get("judge_prob_std"),
                            "Judge Max Prob Diff": ci.get("judge_max_prob_diff"),
                            "Judge Token Prob Slope": ci.get("judge_token_prob_slope"),
                            "Judge Cosine Similarity": ci.get("judge_cosine_similarity"),
                        })
                    all_rows.append(row)
                # 2) False Positives
                logger.info(f"     Adding {len(fp_edges)} FP edges to Excel...")
                for e in fp_edges:
                    row = _build_edge_row(e, "FP", session_edges, validation_edges, edge_data)
                    ci = edge_ci_metrics.get(e)
                    if ci:
                        cos = ci.get("cosine_similarity")
                        if cos is None:
                            cos = ci.get("gen_cosine_similarity") or ci.get("judge_cosine_similarity")
                        row.update({
                            "Gen Perplexity": ci.get("gen_perplexity"),
                            "Gen Min Prob": ci.get("gen_min_prob"),
                            "Gen Max Window Entropy": ci.get("gen_max_window_entropy"),
                            "Gen Mean Token Prob": ci.get("gen_mean_token_prob"),
                            "Gen Prob Variance": ci.get("gen_prob_variance"),
                            "Gen Prob Std": ci.get("gen_prob_std"),
                            "Gen Max Prob Diff": ci.get("gen_max_prob_diff"),
                            "Gen Token Prob Slope": ci.get("gen_token_prob_slope"),
                            "Gen Cosine Similarity": ci.get("gen_cosine_similarity"),
                            "Judge Perplexity": ci.get("judge_perplexity"),
                            "Judge Min Prob": ci.get("judge_min_prob"),
                            "Judge Max Window Entropy": ci.get("judge_max_window_entropy"),
                            "Judge Mean Token Prob": ci.get("judge_mean_token_prob"),
                            "Judge Prob Variance": ci.get("judge_prob_variance"),
                            "Judge Prob Std": ci.get("judge_prob_std"),
                            "Judge Max Prob Diff": ci.get("judge_max_prob_diff"),
                            "Judge Token Prob Slope": ci.get("judge_token_prob_slope"),
                            "Judge Cosine Similarity": ci.get("judge_cosine_similarity"),
                        })
                    all_rows.append(row)
                # 3) False Negatives (only edges NOT in session graph at all, not even as NONE)
                # Note: FN edges that were proposed as NONE will be added in section 4
                fn_edges_not_in_session = fn_edges - session_edges  # Exclude edges that are in session as NONE
                logger.info(f"     Adding {len(fn_edges_not_in_session)} FN edges to Excel...")
                for e in fn_edges_not_in_session:
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
                        "Relationship Type": "N/A",  # These edges were never proposed
                    }
                    all_rows.append(row)
                
                # 4) NONE relationships (edges with relationship type NONE)
                # These were filtered out from accuracy metrics, but should still appear in the Excel
                # Classification logic:
                #   - If validation has this edge (non-NONE) → FN (missed the real relationship)
                #   - If validation doesn't have this edge → TN (correctly identified no relationship)
                none_edges = []
                for (src, tgt) in session_edges:
                    data = edge_data.get((src, tgt))
                    rel_type = data.get("relationship_type", "") if data else ""
                    if rel_type and rel_type.upper() in ("NONE", "None", "NO RELATIONSHIP"):
                        none_edges.append((src, tgt))
                
                logger.info(f"     Adding {len(none_edges)} NONE edges to Excel...")
                for e in none_edges:
                    # Determine classification: FN if in validation, TN otherwise
                    classification = "FN" if e in validation_edges else "TN"
                    row = _build_edge_row(e, classification, session_edges, validation_edges, edge_data)
                    ci = edge_ci_metrics.get(e)
                    if ci:
                        row.update({
                            "Gen Perplexity": ci.get("gen_perplexity"),
                            "Gen Min Prob": ci.get("gen_min_prob"),
                            "Gen Max Window Entropy": ci.get("gen_max_window_entropy"),
                            "Gen Mean Token Prob": ci.get("gen_mean_token_prob"),
                            "Gen Prob Variance": ci.get("gen_prob_variance"),
                            "Gen Prob Std": ci.get("gen_prob_std"),
                            "Gen Max Prob Diff": ci.get("gen_max_prob_diff"),
                            "Gen Token Prob Slope": ci.get("gen_token_prob_slope"),
                            "Gen Cosine Similarity": ci.get("gen_cosine_similarity"),
                            "Judge Perplexity": ci.get("judge_perplexity"),
                            "Judge Min Prob": ci.get("judge_min_prob"),
                            "Judge Max Window Entropy": ci.get("judge_max_window_entropy"),
                            "Judge Mean Token Prob": ci.get("judge_mean_token_prob"),
                            "Judge Prob Variance": ci.get("judge_prob_variance"),
                            "Judge Prob Std": ci.get("judge_prob_std"),
                            "Judge Max Prob Diff": ci.get("judge_max_prob_diff"),
                            "Judge Token Prob Slope": ci.get("judge_token_prob_slope"),
                            "Judge Cosine Similarity": ci.get("judge_cosine_similarity"),
                        })
                    all_rows.append(row)

                # Convert to DataFrame
                logger.info(f"     Converting {len(all_rows)} rows to DataFrame...")
                all_edges_df = pd.DataFrame(all_rows)
                logger.info(f"     DataFrame created with shape: {all_edges_df.shape}")

                # === (E) Build summary
                logger.info(f"     Building summary sheet...")
                # Compute averages of CI metrics for summary
                def _avg_safe(series):
                    s = pd.to_numeric(series, errors="coerce").dropna()
                    return float(s.mean()) if len(s) else None

                if embedding_enable:
                    avg_gen_cos = _avg_safe(all_edges_df.get("Gen Cosine Similarity", pd.Series(dtype=float)))
                    avg_judge_cos = _avg_safe(all_edges_df.get("Judge Cosine Similarity", pd.Series(dtype=float)))
                else:
                    avg_gen_cos = avg_judge_cos = None

                # Calculate RQ1-specific metrics from edge_data
                total_corrupted = sum(1 for data in edge_data.values() if data.get("is_corrupted", False))
                total_corrected = sum(1 for data in edge_data.values() if data.get("corrected", False))
                
                # Judge performance on corruption detection
                corrupted_detected = sum(1 for data in edge_data.values() 
                                       if data.get("is_corrupted", False) and data.get("judge_verdict", "").lower() in ["false", "reject", "no"])
                corrupted_missed = sum(1 for data in edge_data.values() 
                                     if data.get("is_corrupted", False) and data.get("judge_verdict", "").lower() in ["true", "accept", "yes"])
                
                judge_sensitivity = (corrupted_detected / total_corrupted) if total_corrupted > 0 else 0.0

                summary_data = {
                    "Metric": [
                        "True Positives (TP)",
                        "False Positives (FP)",
                        "False Negatives (FN)",
                        "True Negatives (TN)",
                        "Total Discovered Edges",
                        "Total Validation Edges",
                        "Node Precision (Binary)",
                        "Node Recall (Binary)", 
                        "Node F1 Score (Binary)",
                        "Hybrid Node Precision (Similarity)",
                        "Hybrid Node Recall (Similarity)",
                        "Hybrid Node F1 (Similarity)",
                        "Avg Similarity (Gen→Val)",
                        "Avg Similarity (Val→Gen)",
                        "Adjusted Node Precision (Binary, excl. seed)",
                        "Adjusted Node Recall (Binary, excl. seed)",
                        "Adjusted Node F1 (Binary, excl. seed)",
                        "Adjusted Hybrid Precision (Similarity, excl. seed)",
                        "Adjusted Hybrid Recall (Similarity, excl. seed)",
                        "Adjusted Hybrid F1 (Similarity, excl. seed)",
                        "Seeded Target Variable",
                        "Variables Correct Score",
                        "correct variables",
                        "incorrect variables",
                        "Total Comparisons",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "Accuracy",
                        "Avg Gen Cosine Similarity",
                        "Avg Judge Cosine Similarity",
                        "Total Corrupted Edges",
                        "Total Corrected Edges",
                        "Judge Sensitivity (Corruption Detection)",
                        "Corruption Rate",
                        "Correction Rate",
                    ],
                    "Value": [
                        tp, fp, fn, tn,
                        len(session_edges_for_metrics),
                        len(validation_edges),
                        metrics.get("node_precision", 0.0),
                        metrics.get("node_recall", 0.0),
                        metrics.get("node_f1", 0.0),
                        metrics.get("hybrid_node_precision") if metrics.get("hybrid_node_precision") is not None else "n/a",
                        metrics.get("hybrid_node_recall") if metrics.get("hybrid_node_recall") is not None else "n/a",
                        metrics.get("hybrid_node_f1") if metrics.get("hybrid_node_f1") is not None else "n/a",
                        metrics.get("avg_similarity_gen_to_val") if metrics.get("avg_similarity_gen_to_val") is not None else "n/a",
                        metrics.get("avg_similarity_val_to_gen") if metrics.get("avg_similarity_val_to_gen") is not None else "n/a",
                        metrics.get("adjusted_node_precision") if metrics.get("adjusted_node_precision") is not None else "n/a",
                        metrics.get("adjusted_node_recall") if metrics.get("adjusted_node_recall") is not None else "n/a",
                        metrics.get("adjusted_node_f1") if metrics.get("adjusted_node_f1") is not None else "n/a",
                        metrics.get("adjusted_hybrid_precision") if metrics.get("adjusted_hybrid_precision") is not None else "n/a",
                        metrics.get("adjusted_hybrid_recall") if metrics.get("adjusted_hybrid_recall") is not None else "n/a",
                        metrics.get("adjusted_hybrid_f1") if metrics.get("adjusted_hybrid_f1") is not None else "n/a",
                        metrics.get("seeded_target_variable", "None"),
                        variable_correct_score,
                        correct_vars,
                        incorrect_vars,
                        total_comparisons,
                        precision,
                        recall,
                        f1,
                        accuracy,
                        avg_gen_cos,
                        avg_judge_cos,
                        total_corrupted,
                        total_corrected,
                        judge_sensitivity,
                        (total_corrupted / len(session_edges)) if len(session_edges) > 0 else 0.0,
                        (total_corrected / len(session_edges)) if len(session_edges) > 0 else 0.0,
                    ]
                }
                summary_df = pd.DataFrame(summary_data)

                # === (F.1) Per-class breakdown computed pre-write using edge sets (TP/FP only)
                if scenario_name == "All Edges":
                    def _mean_from_edges(edges_set, col_name):
                        vals = []
                        if col_name == "Aggregate Score":
                            for e in edges_set:
                                m = edge_data.get(e, {})
                                v = m.get("aggregate_score", None)
                                if v is not None:
                                    try:
                                        vals.append(float(v))
                                    except Exception:
                                        pass
                        else:
                            for e in edges_set:
                                m = edge_ci_metrics.get(e, {})
                                key_map = {
                                    "Gen Perplexity": "gen_perplexity",
                                    "Gen Min Prob": "gen_min_prob",
                                    "Gen Max Window Entropy": "gen_max_window_entropy",
                                    "Gen Mean Token Prob": "gen_mean_token_prob",
                                    "Gen Prob Variance": "gen_prob_variance",
                                    "Gen Prob Std": "gen_prob_std",
                                    "Gen Max Prob Diff": "gen_max_prob_diff",
                                    "Gen Token Prob Slope": "gen_token_prob_slope",
                                    "Gen Cosine Similarity": "gen_cosine_similarity",
                                    "Judge Perplexity": "judge_perplexity",
                                    "Judge Min Prob": "judge_min_prob",
                                    "Judge Max Window Entropy": "judge_max_window_entropy",
                                    "Judge Mean Token Prob": "judge_mean_token_prob",
                                    "Judge Prob Variance": "judge_prob_variance",
                                    "Judge Prob Std": "judge_prob_std",
                                    "Judge Max Prob Diff": "judge_max_prob_diff",
                                    "Judge Token Prob Slope": "judge_token_prob_slope",
                                    "Judge Cosine Similarity": "judge_cosine_similarity",
                                }
                                v = m.get(key_map.get(col_name, ""), None)
                                if v is not None:
                                    try:
                                        vals.append(float(v))
                                    except Exception:
                                        pass
                        return (sum(vals) / len(vals)) if vals else None

                    per_class_rows = []
                    for cls, edge_set in (
                        ("TP", tp_edges),
                        ("FP", fp_edges),
                        ("FN", set()),
                        ("TN", set()),
                    ):
                        for metric_label in [
                            "Aggregate Score",
                            "Gen Cosine Similarity",
                            "Judge Cosine Similarity",
                        ]:
                            value = _mean_from_edges(edge_set, metric_label) if cls in ("TP", "FP") else None
                            if value is None and cls in ("FN", "TN"):
                                value = "n/a"
                            per_class_rows.append({
                                "Metric": f"Avg {metric_label} ({cls})",
                                "Value": value,
                            })

                    summary_df = pd.concat([summary_df, pd.DataFrame(per_class_rows)], ignore_index=True)

                # === (F) Write these two DataFrames (detail & summary) to new sheets
                detail_sheet = scenario_name
                summary_sheet = f"{scenario_name} Summary"

                logger.info(f"     Writing '{detail_sheet}' sheet with {len(all_edges_df)} rows...")
                all_edges_df.to_excel(writer, sheet_name=detail_sheet, index=False)
                logger.info(f"     ✓ Wrote '{detail_sheet}' sheet")
                
                logger.info(f"     Writing '{summary_sheet}' sheet with {len(summary_df)} rows...")
                summary_df.to_excel(writer, sheet_name=summary_sheet, index=False)
                logger.info(f"     ✓ Wrote '{summary_sheet}' sheet")

            except Exception as exc:
                print(f"Error processing scenario '{scenario_name}': {exc}")
                traceback.print_exc()

        # 3) Optionally add a Node Analysis sheet with node comparison data
        if node_comparison_data:
            try:
                # Extract node lists and metrics
                session_nodes = node_comparison_data.get("session_nodes", [])
                validation_nodes = node_comparison_data.get("validation_nodes", [])
                matched_pairs = node_comparison_data.get("matched_pairs", [])
                node_metrics = node_comparison_data.get("metrics", {})
                
                # Create Generated Nodes sheet
                generated_rows = []
                for node in session_nodes:
                    generated_rows.append({
                        "Generated Variable": node,
                        "Matched To": next((v for s, v in matched_pairs if s == node), "No match")
                    })
                if generated_rows:
                    generated_df = pd.DataFrame(generated_rows)
                    generated_df.to_excel(writer, sheet_name="Generated Nodes", index=False)
                
                # Create Validation Nodes sheet
                validation_rows = []
                for node in validation_nodes:
                    validation_rows.append({
                        "Validation Variable": node,
                        "Matched By": next((s for s, v in matched_pairs if v == node), "No match")
                    })
                if validation_rows:
                    validation_df = pd.DataFrame(validation_rows)
                    validation_df.to_excel(writer, sheet_name="Validation Nodes", index=False)
                
                # Create Node Comparison Summary sheet
                summary_rows = [
                    {"Metric": "Total Generated Nodes", "Value": len(session_nodes)},
                    {"Metric": "Total Validation Nodes", "Value": len(validation_nodes)},
                    {"Metric": "Matched Pairs (Binary)", "Value": len(matched_pairs)},
                    {"Metric": "--- Binary Metrics (Exact Match) ---", "Value": ""},
                    {"Metric": "Node Precision (Binary)", "Value": node_metrics.get("node_precision", 0.0)},
                    {"Metric": "Node Recall (Binary)", "Value": node_metrics.get("node_recall", 0.0)},
                    {"Metric": "Node F1 Score (Binary)", "Value": node_metrics.get("node_f1", 0.0)},
                    {"Metric": "--- Cosine Similarity Metrics ---", "Value": ""},
                    {"Metric": "Cosine Node Precision", "Value": node_metrics.get("cosine_node_precision", "n/a")},
                    {"Metric": "Cosine Node Recall", "Value": node_metrics.get("cosine_node_recall", "n/a")},
                    {"Metric": "Cosine Node F1 Score", "Value": node_metrics.get("cosine_node_f1", "n/a")},
                    {"Metric": "Avg Cosine (Gen→Val)", "Value": node_metrics.get("avg_cosine_gen_to_val", "n/a")},
                    {"Metric": "Avg Cosine (Val→Gen)", "Value": node_metrics.get("avg_cosine_val_to_gen", "n/a")},
                    {"Metric": "--- Hybrid Metrics (Binary=1.0 + Similarity) ---", "Value": ""},
                    {"Metric": "Hybrid Node Precision", "Value": node_metrics.get("hybrid_node_precision", "n/a")},
                    {"Metric": "Hybrid Node Recall", "Value": node_metrics.get("hybrid_node_recall", "n/a")},
                    {"Metric": "Hybrid Node F1 Score", "Value": node_metrics.get("hybrid_node_f1", "n/a")},
                ]
                
                # Add adjusted metrics if a seeded target was used
                seeded_target = node_comparison_data.get("metrics", {}).get("seeded_target_variable")
                if seeded_target:
                    summary_rows.append({"Metric": "--- Adjusted Metrics (excluding seed) ---", "Value": ""})
                    summary_rows.append({"Metric": "Seeded Target Variable", "Value": seeded_target})
                    adj_prec = node_metrics.get("adjusted_node_precision")
                    adj_rec = node_metrics.get("adjusted_node_recall")
                    adj_f1 = node_metrics.get("adjusted_node_f1")
                    adj_matched = node_metrics.get("adjusted_matched_pairs")
                    if adj_prec is not None:
                        summary_rows.append({"Metric": "Adjusted Matched Pairs (Binary, excl. seed)", "Value": adj_matched})
                        summary_rows.append({"Metric": "Adjusted Node Precision (Binary)", "Value": adj_prec})
                        summary_rows.append({"Metric": "Adjusted Node Recall (Binary)", "Value": adj_rec})
                        summary_rows.append({"Metric": "Adjusted Node F1 Score (Binary)", "Value": adj_f1})
                    
                    # Add adjusted hybrid metrics
                    adj_h_prec = node_metrics.get("adjusted_hybrid_precision")
                    adj_h_rec = node_metrics.get("adjusted_hybrid_recall")
                    adj_h_f1 = node_metrics.get("adjusted_hybrid_f1")
                    if adj_h_prec is not None:
                        summary_rows.append({"Metric": "Adjusted Hybrid Precision (Similarity)", "Value": adj_h_prec})
                        summary_rows.append({"Metric": "Adjusted Hybrid Recall (Similarity)", "Value": adj_h_rec})
                        summary_rows.append({"Metric": "Adjusted Hybrid F1 (Similarity)", "Value": adj_h_f1})
                
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_excel(writer, sheet_name="Node Comparison Summary", index=False)
                
                print("✅ Added Node Analysis sheets (Generated Nodes, Validation Nodes, Node Comparison Summary)")
            except Exception as exc:
                print(f"Error creating Node Analysis sheets: {exc}")
                traceback.print_exc()

        # 4) Optionally add a Params sheet with experiment parameters
        if experiment_params:
            try:
                # Normalize parameters to two-column table
                rows = []
                for key, value in experiment_params.items():
                    if isinstance(value, (dict, list)):
                        try:
                            value = json.dumps(value)
                        except Exception:
                            value = str(value)
                    rows.append({"Parameter": str(key), "Value": value})
                params_df = pd.DataFrame(rows)
                params_df.to_excel(writer, sheet_name="Params", index=False)
            except Exception:
                pass

        # 5) If we have usage stats, create an extra "LLM Usage Stats" sheet
        if lm_stats:
            # Ensure embeddings stats reflect work done in this function
            try:
                from data_science.logit_metrics import get_embedding_stats as _get_embed_stats
                lm_stats["embeddings"] = _get_embed_stats()
            except Exception:
                pass
            usage_rows = []
            # We expect keys like "generator", "corruptor", "judge", "embeddings"
            for role_name in ["generator", "corruptor", "judge", "embeddings"]:
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
          "  6) (Optional) Params\n"
          "  7) (Optional) LLM Usage Stats\n")

    return output_filename


# ------------------------------
# Context-insensitive metrics helpers
# ------------------------------
def _fetch_text_from_url(url: str, timeout: int = 15) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return None


def _ci_cache_dir() -> str:
    import os
    d = os.path.join(os.path.expanduser("~"), ".cache", "causalix_ci")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _sha1_hex(text: str) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _cached_text_path(url: str) -> str:
    import os
    return os.path.join(_ci_cache_dir(), f"text_{_sha1_hex(url)}.txt")


def _load_cached_text(url: str) -> Optional[str]:
    import os
    p = _cached_text_path(url)
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return None
    return None


def _save_cached_text(url: str, text: str) -> None:
    try:
        p = _cached_text_path(url)
        with open(p, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text or "")
    except Exception:
        pass


def _fetch_text_from_url_cached(url: str, timeout: int = 15) -> Optional[str]:
    cached = _load_cached_text(url)
    if cached:
        return cached
    txt = _fetch_text_from_url(url, timeout=timeout)
    if txt:
        _save_cached_text(url, txt)
    return txt


def _fetch_citation_texts_cached_via_scraper(urls: List[str], timeout: int = 15) -> List[str]:
    """Fetch citation texts using the platform ContentScraper when available (handles PDF),
    with per-URL text caching. Falls back to simple fetch in parallel.
    """
    logger.debug(f"[CI-FETCH] Begin fetch for {len(urls)} URLs (timeout={timeout}s)")
    texts: List[str] = []
    missing: List[str] = []
    # Use cache first
    for u in urls:
        t = _load_cached_text(u)
        if t:
            texts.append(t)
        else:
            missing.append(u)

    if not missing:
        return texts

    logger.debug(f"[CI-FETCH] cache hits={len(texts)}; misses={len(missing)}")

    # Try to import platform scraper dynamically
    ContentScraper = None
    try:
        from platform.backend.websearch_clients.content_scraper import ContentScraper as _CS  # type: ignore
        ContentScraper = _CS
    except Exception:
        try:
            # Try relative path if running from this repo structure
            import os, sys
            here = os.path.dirname(__file__)
            root = os.path.abspath(os.path.join(here, "..", ".."))
            plat_backend = os.path.join(root, "platform", "backend")
            if plat_backend not in sys.path:
                sys.path.append(plat_backend)
            from websearch_clients.content_scraper import ContentScraper as _CS2  # type: ignore
            ContentScraper = _CS2
        except Exception:
            ContentScraper = None

    if ContentScraper is not None:
        try:
            scraper = ContentScraper(timeout=timeout, max_workers=min(8, max(3, len(missing))), pdf_extraction_mode="local", disable_agentic_fallback=True)
            # The scraper exposes process_citations(fetch_content=True) returning metadata with .content
            results = scraper.process_citations(missing, fetch_content=True)
            for meta in results or []:
                try:
                    content = getattr(meta, "content", None)
                    url = getattr(meta, "url", None)
                    if isinstance(url, str) and content:
                        _save_cached_text(url, content)
                        texts.append(content)
                except Exception:
                    continue
            logger.debug(f"[CI-FETCH] scraper fetched new={max(0, len(texts) - (len(urls) - len(missing)))}; total_texts={len(texts)}")
        except Exception:
            ContentScraper = None  # fall through to simple fetch

    if ContentScraper is None:
        # Fallback: parallel simple fetch with cache
        from concurrent.futures import ThreadPoolExecutor, as_completed
        try:
            with ThreadPoolExecutor(max_workers=min(16, max(4, len(missing)))) as ex:
                fut_to_url = {ex.submit(_fetch_text_from_url_cached, u, timeout): u for u in missing}
                for fut in as_completed(fut_to_url):
                    try:
                        txt = fut.result()
                        if txt:
                            texts.append(txt)
                    except Exception:
                        continue
            logger.debug(f"[CI-FETCH] simple-fetch fetched total_texts={len(texts)} (including cache)")
        except Exception:
            # Last-resort sequential
            for u in missing:
                t = _fetch_text_from_url_cached(u, timeout)
                if t:
                    texts.append(t)
            logger.debug(f"[CI-FETCH] sequential fetched total_texts={len(texts)} (including cache)")

    return texts

def _embed_texts_with_cache(texts, openai_key: str, model: str, dimensions: Optional[int], embedding_provider: str = "openai", embedding_device: str = "cuda"):
    """Embed texts with per-text cache to minimize API calls.
    Returns list of numpy arrays aligned to input order.
    
    Args:
        texts: List of texts to embed
        openai_key: OpenAI API key (used if provider is "openai")
        model: Model name (for openai) or HuggingFace model ID (for local)
        dimensions: Embedding dimensions (for openai provider)
        embedding_provider: "openai" or "local" (default: "openai")
        embedding_device: Device for local embeddings (default: "cuda")
    """
    import os
    import json
    import numpy as np
    
    if embedding_provider == "local":
        from data_science.logit_metrics import get_embedding_local
    else:
        from data_science.logit_metrics import openai_embed_batch

    if not texts:
        return []

    cache_dir = _ci_cache_dir()

    def emb_path(t: str) -> str:
        key = f"emb_{embedding_provider}_{model}_{str(dimensions)}_{_sha1_hex(t)}"
        return os.path.join(cache_dir, key + ".npy")

    results: List[Optional[np.ndarray]] = [None] * len(texts)
    hits = 0
    misses = 0
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    # Load cached where available
    for i, t in enumerate(texts):
        try:
            p = emb_path(t)
            if os.path.exists(p):
                arr = np.load(p)
                if isinstance(arr, np.ndarray):
                    results[i] = arr.astype(float)
                    hits += 1
                    continue
        except Exception:
            pass
        missing_indices.append(i)
        missing_texts.append(t)
        misses += 1

    # Compute missing in reasonably sized batches
    if missing_texts:
        # Conservative batch sizes (revert from optimization attempt)
        # For OpenAI API, keep smaller batches to avoid rate limits
        B = 128  # outer batch size (API chunking)
        inner_batch_size = B  # SentenceTransformer batch size
        
        logger.debug(
            f"[CI-EMBED] provider={embedding_provider} model={model} dims={dimensions} to_embed={len(missing_texts)} (hits={hits}, misses={misses}) batch={B} inner_batch={inner_batch_size}"
        )
        for start in range(0, len(missing_texts), B):
            batch = missing_texts[start:start+B]
            try:
                if embedding_provider == "local":
                    vecs = get_embedding_local(batch, model=model, batch_size=inner_batch_size, device=embedding_device)
                else:
                    vecs = openai_embed_batch(batch, openai_key, model=model, dimensions=dimensions)
            except Exception as e:
                logger.warning(f"[CI-EMBED] Embedding batch failed, skipping {len(batch)} texts: {e}")
                # Fill with None for this batch and continue
                vecs = [None] * len(batch)
            # Write back to results and cache
            for j, v in enumerate(vecs):
                idx = missing_indices[start + j]
                results[idx] = v
                if v is not None:
                    try:
                        np.save(emb_path(texts[idx]), v)
                    except Exception:
                        pass
        logger.debug("[CI-EMBED] completed embedding batches")

    # Return possibly sparse list with None entries; callers must handle None
    return results

def _compute_edge_level_context_insensitive_metrics(
    discovery,
    session_id: str,
    *,
    ci_fetch_reuse: bool = True,
    ci_chunk_chars_override: Optional[int] = None,
    ci_overlap_ratio: float = 0.2,
    ci_compute_embeddings: bool = True,
    ci_parallel: bool = True,
    ci_max_workers: int = 10,
    embedding_provider: str = "local",
    embedding_device: str = "cuda",
) -> Dict[Tuple[str, str], Dict[str, Optional[float]]]:
    """Compute context-insensitive metrics per edge for a given session.
    
    Args:
        discovery: CausalDiscovery instance
        session_id: Session ID to process
        ci_fetch_reuse: Whether to use cached citation texts
        ci_chunk_chars_override: Override for chunk size
        ci_overlap_ratio: Overlap ratio for chunking
        ci_compute_embeddings: Whether to compute expensive embedding-based metrics (default: True)
        ci_parallel: Whether to process edges in parallel (default: True)
        ci_max_workers: Maximum number of parallel workers (default: 10)

    Returns a dict keyed by (src, tgt) with values containing:
      - perplexity, min_prob, max_window_entropy, cosine_similarity
    """
    logger.info(f"🔬 CI METRICS - STARTING COMPUTATION (device: {embedding_device})")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Note: Do NOT parse or reuse URLs from judge_message; use only r.citations/r.citation

    with discovery.graph_db._get_session() as sess:
        query = """
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id = $sid AND t.session_id = $sid
        RETURN s.name AS src, t.name AS tgt,
               coalesce(r.spurious_motivation, r.motivation, '') AS motivation,
               r.judge_message AS judge_message,
               r.citations AS citations,
               r.citation  AS citation,
               r.avg_nll AS avg_nll,
               r.perplexity AS perplexity,
               r.min_prob AS min_prob,
               r.max_window_entropy AS max_window_entropy,
               r.mean_token_prob AS mean_token_prob,
               r.prob_variance AS prob_variance,
               r.prob_std AS prob_std,
               r.max_prob_diff AS max_prob_diff,
               r.token_prob_slope AS token_prob_slope,
               r.judge_avg_nll AS judge_avg_nll,
               r.judge_perplexity AS judge_perplexity,
               r.judge_min_prob AS judge_min_prob,
               r.judge_max_window_entropy AS judge_max_window_entropy,
               r.judge_mean_token_prob AS judge_mean_token_prob,
               r.judge_prob_variance AS judge_prob_variance,
               r.judge_prob_std AS judge_prob_std,
               r.judge_max_prob_diff AS judge_max_prob_diff,
               r.judge_token_prob_slope AS judge_token_prob_slope
        """
        records = list(sess.run(query, {"sid": session_id}))
    
    total_edges = len(records)
    logger.info(f"  Total edges to process: {total_edges}")

    def _chunk_text(text: str, chunk_chars: int, overlap_chars: int = 0) -> List[str]:
        if chunk_chars <= 0:
            return [text]
        chunks: List[str] = []
        i = 0
        n = len(text)
        step = max(1, chunk_chars - overlap_chars)
        while i < n:
            chunks.append(text[i:i + chunk_chars])
            i += step
        return chunks

    def _prepare_edge_data_for_embedding(edge_idx, total_edges, rec, ci_chunk_chars_override, ci_overlap_ratio):
        """Stage 1: Prepare edge data (CPU-bound: fetch citations, chunk text).
        
        Returns edge_data dict with prepared texts but NO embeddings yet.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        src = rec["src"]
        tgt = rec["tgt"]
        
        motivation = (rec["motivation"] or "").strip()
        citations = rec.get("citations")
        citation_single = rec.get("citation")
        citation_list: List[str] = []
        if isinstance(citations, list):
            citation_list = [c for c in citations if isinstance(c, str) and c]
        if not citation_list and isinstance(citation_single, str) and citation_single:
            citation_list = [citation_single]
        
        edge_data = {
            "src": src,
            "tgt": tgt,
            "motivation": motivation,
            "citation_list": citation_list,
            "unique_chunks": [],
            "judge_message": rec.get("judge_message", "")
        }
        
        # Fetch and chunk citation texts (CPU I/O work)
        if citation_list:
            fetched_texts: List[str] = _fetch_citation_texts_cached_via_scraper(citation_list, timeout=15)
            
            if fetched_texts:
                # Determine chunk size
                if ci_chunk_chars_override is not None and ci_chunk_chars_override > 0:
                    target_chars = ci_chunk_chars_override
                else:
                    base_len = len(motivation) if motivation else len(edge_data["judge_message"])
                    target_chars = max(256, min(4000, base_len)) if base_len > 0 else 512
                overlap = int(max(0.0, ci_overlap_ratio) * target_chars)
                
                # Chunk texts in parallel
                all_chunks: List[str] = []
                with ThreadPoolExecutor(max_workers=min(16, max(4, len(fetched_texts)))) as ex:
                    chunk_lists = list(ex.map(lambda full: _chunk_text(full, target_chars, overlap), fetched_texts))
                for cl in chunk_lists:
                    if cl:
                        all_chunks.extend(cl)
                
                # De-duplicate chunks
                unique_chunks: List[str] = []
                seen = set()
                for c in all_chunks:
                    if c and c not in seen:
                        seen.add(c)
                        unique_chunks.append(c)
                
                edge_data["unique_chunks"] = unique_chunks
        
        return edge_data
    
    def _finalize_edge_ci_with_embeddings(edge_idx, total_edges, rec, edge_data, openai_key, ci_compute_embeddings, embedding_provider, embedding_device):
        """Stage 2: Compute embeddings and finalize CI metrics (GPU-bound).
        
        Takes prepared edge_data and computes embeddings, then calculates final metrics.
        """
        import numpy as np
        
        src = edge_data["src"]
        tgt = edge_data["tgt"]
        motivation = edge_data["motivation"]
        unique_chunks = edge_data["unique_chunks"]
        judge_message = edge_data["judge_message"]
        
        gen_cosine_sim: Optional[float] = None
        judge_cosine_sim: Optional[float] = None
        
        # Compute embeddings if we have chunks
        if unique_chunks and ci_compute_embeddings:
            # Select model
            if embedding_provider == "local":
                model = "sentence-transformers/all-mpnet-base-v2"
                dims = None
            else:
                model = "text-embedding-3-small"
                dims = None
            
            # Embed narratives and chunks
            if motivation:
                v1_list = _embed_texts_with_cache([motivation], openai_key, model=model, dimensions=dims, 
                                                   embedding_provider=embedding_provider, embedding_device=embedding_device)
                v1 = v1_list[0] if v1_list else None
                if v1 is not None:
                    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
            else:
                v1 = None
            
            # Embed all citation chunks in one batch
            v2_list = _embed_texts_with_cache(unique_chunks, openai_key, model=model, dimensions=dims,
                                               embedding_provider=embedding_provider, embedding_device=embedding_device)
            
            # Compute max cosine similarity for generator
            if v1 is not None and v2_list:
                v2_arrays = [v for v in v2_list if v is not None]
                if v2_arrays:
                    # Normalize all chunk vectors
                    v2_normed = [v / (np.linalg.norm(v) + 1e-12) for v in v2_arrays]
                    # Compute cosine similarities
                    sims = [float(np.dot(v1, v2)) for v2 in v2_normed]
                    gen_cosine_sim = max(sims) if sims else None
            
            # Same for judge message
            if judge_message:
                v3_list = _embed_texts_with_cache([judge_message], openai_key, model=model, dimensions=dims,
                                                   embedding_provider=embedding_provider, embedding_device=embedding_device)
                v3 = v3_list[0] if v3_list else None
                if v3 is not None:
                    v3 = v3 / (np.linalg.norm(v3) + 1e-12)
                    if v2_list:
                        v2_arrays = [v for v in v2_list if v is not None]
                        if v2_arrays:
                            v2_normed = [v / (np.linalg.norm(v) + 1e-12) for v in v2_arrays]
                            sims = [float(np.dot(v3, v2)) for v2 in v2_normed]
                            judge_cosine_sim = max(sims) if sims else None
        
        # Build final result
        edge_key = (src, tgt)
        edge_metrics = {
            # Perplexity metrics (legacy keys for backwards compatibility)
            "perplexity": rec.get("perplexity"),
            "min_prob": rec.get("min_prob"),
            "max_window_entropy": rec.get("max_window_entropy"),
            # Generator metrics (map from unprefixed Neo4j properties to gen_ keys)
            "gen_perplexity": rec.get("perplexity"),  # Maps to generator perplexity
            "gen_min_prob": rec.get("min_prob"),  # Maps to generator min_prob
            "gen_max_window_entropy": rec.get("max_window_entropy"),  # Maps to generator max_window_entropy
            # Judge metrics (already have judge_ prefix in Neo4j)
            "judge_perplexity": rec.get("judge_perplexity"),
            "judge_min_prob": rec.get("judge_min_prob"),
            "judge_max_window_entropy": rec.get("judge_max_window_entropy"),
            # Cosine similarities
            "cosine_similarity": None,  # Overall (if needed)
            "gen_cosine_similarity": gen_cosine_sim,
            "judge_cosine_similarity": judge_cosine_sim,
            # Token-level metrics (legacy keys)
            "mean_token_prob": rec.get("mean_token_prob"),
            "token_prob_slope": rec.get("token_prob_slope"),
            "prob_variance": rec.get("prob_variance"),
            "prob_std": rec.get("prob_std"),
            "max_prob_diff": rec.get("max_prob_diff"),
            # Generator token-level metrics (map from unprefixed Neo4j properties)
            "gen_mean_token_prob": rec.get("mean_token_prob"),  # Maps to generator mean_token_prob
            "gen_token_prob_slope": rec.get("token_prob_slope"),  # Maps to generator token_prob_slope
            "gen_prob_variance": rec.get("prob_variance"),  # Maps to generator prob_variance
            "gen_prob_std": rec.get("prob_std"),  # Maps to generator prob_std
            "gen_max_prob_diff": rec.get("max_prob_diff"),  # Maps to generator max_prob_diff
            # Judge token-level metrics (already have judge_ prefix in Neo4j)
            "judge_mean_token_prob": rec.get("judge_mean_token_prob"),
            "judge_token_prob_slope": rec.get("judge_token_prob_slope"),
            "judge_prob_variance": rec.get("judge_prob_variance"),
            "judge_prob_std": rec.get("judge_prob_std"),
            "judge_max_prob_diff": rec.get("judge_max_prob_diff"),
        }
        
        return edge_key, edge_metrics
    
    def _process_single_edge_ci(edge_idx, total_edges, rec, openai_key, ci_chunk_chars_override, ci_overlap_ratio, ci_compute_embeddings, embedding_provider):
        """Process CI metrics for a single edge (for parallelization)."""
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        
        src = rec["src"]
        tgt = rec["tgt"]
        logger.info(f"  📊 Processing edge {edge_idx}/{total_edges}: {src} → {tgt}")
        
        motivation = (rec["motivation"] or "").strip()
        citations = rec.get("citations")
        citation_single = rec.get("citation")
        citation_list: List[str] = []
        if isinstance(citations, list):
            citation_list = [c for c in citations if isinstance(c, str) and c]
        if not citation_list and isinstance(citation_single, str) and citation_single:
            citation_list = [citation_single]

        # Compute cosine similarity between narratives and citation text chunks
        gen_cosine_sim: Optional[float] = None
        judge_cosine_sim: Optional[float] = None
        # Check if we should compute embeddings: either openai_key is set OR we're using local provider
        should_compute_embeddings = citation_list and (openai_key or embedding_provider == "local")
        if should_compute_embeddings:
            # Fetch citation texts via platform scraper (PDF-capable) with cache
            fetched_texts: List[str] = _fetch_citation_texts_cached_via_scraper(citation_list, timeout=15)
            # Chunk each fetched text roughly to motivation size with 20% overlap
            if fetched_texts:
                if ci_chunk_chars_override is not None and ci_chunk_chars_override > 0:
                    target_chars = ci_chunk_chars_override
                else:
                    # Choose a reasonable default based on available narrative text
                    base_len = len(motivation) if motivation else (len(rec.get("judge_message") or ""))
                    target_chars = max(256, min(4000, base_len)) if base_len > 0 else 512
                overlap = int(max(0.0, ci_overlap_ratio) * target_chars)
                try:
                    # Build chunks in parallel across fetched texts
                    all_chunks: List[str] = []
                    with ThreadPoolExecutor(max_workers=min(16, max(4, len(fetched_texts)))) as ex:
                        chunk_lists = list(ex.map(lambda full: _chunk_text(full, target_chars, overlap), fetched_texts))
                    for cl in chunk_lists:
                        if cl:
                            all_chunks.extend(cl)
                    # De-duplicate identical chunks to reduce embedding calls
                    # Keep index map to compute max later
                    unique_chunks: List[str] = []
                    seen = set()
                    for c in all_chunks:
                        if c and c not in seen:
                            seen.add(c)
                            unique_chunks.append(c)

                    # Only compute embeddings if enabled (this is the expensive part)
                    if ci_compute_embeddings:
                        # Select model based on embedding provider
                        if embedding_provider == "local":
                            model = "sentence-transformers/all-mpnet-base-v2"  # Fast, high-quality model (768-dim)
                            dims = None  # Not used for local models
                        else:
                            model = "text-embedding-3-small"  # OpenAI default
                            dims = None

                        # Embed narratives once (with cache)
                        if motivation:
                            v1_list = _embed_texts_with_cache([motivation], openai_key, model=model, dimensions=dims, embedding_provider=embedding_provider, embedding_device=embedding_device)
                            v1 = v1_list[0] if v1_list else None
                            if v1 is not None:
                                v1 = v1 / (np.linalg.norm(v1) + 1e-12)
                        else:
                            v1 = None
                        judge_msg = rec.get("judge_message") or ""
                        if judge_msg:
                            vj_list = _embed_texts_with_cache([judge_msg], openai_key, model=model, dimensions=dims, embedding_provider=embedding_provider, embedding_device=embedding_device)
                            vj = vj_list[0] if vj_list else None
                            if vj is not None:
                                vj = vj / (np.linalg.norm(vj) + 1e-12)
                        else:
                            vj = None

                        # Embed unique chunks once (with cache) and normalize
                        v_chunks = _embed_texts_with_cache(unique_chunks, openai_key, model=model, dimensions=dims, embedding_provider=embedding_provider, embedding_device=embedding_device)
                        v_chunks_norm = []
                        for vc in v_chunks:
                            if vc is None:
                                continue
                            n = np.linalg.norm(vc)
                            v_chunks_norm.append(vc / (n + 1e-12))

                        # Compute max cosine for generator narrative
                        if v1 is not None and v_chunks_norm:
                            # dot with all chunks
                            sims = [float(np.dot(v1, vc)) for vc in v_chunks_norm]
                            gen_cosine_sim = max(sims) if sims else None
                        elif v1 is None:
                            logger.warning(f"⚠️  Skipping generator cosine for edge {src}->{tgt}: motivation embedding failed")
                        elif not v_chunks_norm:
                            logger.warning(f"⚠️  Skipping generator cosine for edge {src}->{tgt}: citation chunk embeddings failed")
                        
                        # Compute max cosine for judge narrative
                        if vj is not None and v_chunks_norm:
                            sims_j = [float(np.dot(vj, vc)) for vc in v_chunks_norm]
                            judge_cosine_sim = max(sims_j) if sims_j else None
                        elif vj is None and judge_msg:
                            logger.warning(f"⚠️  Skipping judge cosine for edge {src}->{tgt}: judge message embedding failed")
                        elif not v_chunks_norm and judge_msg:
                            logger.warning(f"⚠️  Skipping judge cosine for edge {src}->{tgt}: citation chunk embeddings failed")
                    else:
                        # Embeddings disabled - skip this expensive computation
                        logger.info(f"⏭️  Skipping embeddings for edge {src}->{tgt} (ci_compute_embeddings=False)")
                        gen_cosine_sim = None
                        judge_cosine_sim = None
                except Exception as e:
                    logger.error(f"❌ Cosine similarity computation failed for edge {src}->{tgt}: {e}")
                    logger.exception("Full traceback for cosine similarity failure:")
                    gen_cosine_sim = None
                    judge_cosine_sim = None

        # Logit-based metrics from stored OpenAI-style logprobs
        perplexity = None
        min_prob = None
        max_window_entropy = None
        # Prefer generator aggregates; fall back to judge aggregates
        try:
            gen_perplexity = rec.get("perplexity")
            gen_min_prob = rec.get("min_prob")
            gen_max_window_entropy = rec.get("max_window_entropy")
            judge_perplexity = rec.get("judge_perplexity")
            judge_min_prob = rec.get("judge_min_prob")
            judge_max_window_entropy = rec.get("judge_max_window_entropy")
            # Back-compat combined fields (prefer generator)
            perplexity = gen_perplexity if gen_perplexity is not None else judge_perplexity
            min_prob = gen_min_prob if gen_min_prob is not None else judge_min_prob
            max_window_entropy = (
                gen_max_window_entropy if gen_max_window_entropy is not None else judge_max_window_entropy
            )
        except Exception:
            gen_perplexity = gen_min_prob = gen_max_window_entropy = None
            judge_perplexity = judge_min_prob = judge_max_window_entropy = None

        # Note: We only return the differentiated "Gen" and "Judge" prefixed metrics
        # The back-compat combined fields are no longer needed
        
        return (src, tgt), {
            # Differentiated metrics only
            "gen_perplexity": gen_perplexity,
            "gen_min_prob": gen_min_prob,
            "gen_max_window_entropy": gen_max_window_entropy,
            "gen_mean_token_prob": rec.get("mean_token_prob"),
            "gen_prob_variance": rec.get("prob_variance"),
            "gen_prob_std": rec.get("prob_std"),
            "gen_max_prob_diff": rec.get("max_prob_diff"),
            "gen_token_prob_slope": rec.get("token_prob_slope"),
            "gen_cosine_similarity": gen_cosine_sim,
            "judge_perplexity": judge_perplexity,
            "judge_min_prob": judge_min_prob,
            "judge_max_window_entropy": judge_max_window_entropy,
            "judge_mean_token_prob": rec.get("judge_mean_token_prob"),
            "judge_prob_variance": rec.get("judge_prob_variance"),
            "judge_prob_std": rec.get("judge_prob_std"),
            "judge_max_prob_diff": rec.get("judge_max_prob_diff"),
            "judge_token_prob_slope": rec.get("judge_token_prob_slope"),
            "judge_cosine_similarity": judge_cosine_sim,
        }

    # Process edges in parallel or sequentially
    results: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    
    if ci_parallel and len(records) > 1:
        logger.info(f"  🚀 Processing {total_edges} edges in PARALLEL with {ci_max_workers} workers")
        logger.info(f"  💡 Using PIPELINED approach: CPU fetching → GPU embedding in stages")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # OPTIMIZATION: Two-stage pipeline to maximize GPU utilization
        # Stage 1: Fetch all citations in parallel (CPU-bound I/O)
        # Stage 2: Batch embed all texts together (GPU-bound compute)
        
        # Check if we should use pipelined approach (requires embedding computation)
        use_pipeline = ci_compute_embeddings and total_edges >= 10
        
        if use_pipeline:
            logger.info(f"  📦 Stage 1/2: Fetching citations for {total_edges} edges in parallel...")
            
            # Pre-fetch all citations for all edges
            edge_data_list = []
            with ThreadPoolExecutor(max_workers=ci_max_workers) as executor:
                fetch_futures = {}
                for edge_idx, rec in enumerate(records, 1):
                    future = executor.submit(
                        _prepare_edge_data_for_embedding,
                        edge_idx, total_edges, rec, ci_chunk_chars_override, ci_overlap_ratio
                    )
                    fetch_futures[future] = (edge_idx, rec)
                
                for future in as_completed(fetch_futures):
                    edge_idx, rec = fetch_futures[future]
                    try:
                        edge_data = future.result()
                        edge_data_list.append((edge_idx, rec, edge_data))
                    except Exception as e:
                        logger.error(f"  ❌ Error preparing edge {edge_idx}: {e}")
            
            logger.info(f"  ✓ Stage 1/2 complete: Prepared {len(edge_data_list)} edges")
            logger.info(f"  🚀 Stage 2/2: Computing embeddings in large batches (GPU)...")
            
            # Now compute all embeddings together in batches
            # This keeps GPU busy continuously
            for edge_idx, rec, edge_data in edge_data_list:
                try:
                    edge_key, edge_metrics = _finalize_edge_ci_with_embeddings(
                        edge_idx, total_edges, rec, edge_data, openai_key,
                        ci_compute_embeddings, embedding_provider, embedding_device
                    )
                    results[edge_key] = edge_metrics
                    logger.info(f"  ✓ Completed CI metrics for edge {edge_idx}/{total_edges}: {rec['src']} → {rec['tgt']}")
                except Exception as e:
                    logger.error(f"  ❌ Error computing CI metrics for edge {edge_idx}: {e}")
        else:
            # Fall back to original per-edge parallel approach for small batches or non-embedding work
            logger.info(f"  📝 Using per-edge parallel processing (small batch or no embeddings)")
            with ThreadPoolExecutor(max_workers=ci_max_workers) as executor:
                # Submit all edge processing tasks
                future_to_edge = {}
                for edge_idx, rec in enumerate(records, 1):
                    future = executor.submit(
                        _process_single_edge_ci,
                        edge_idx, total_edges, rec, openai_key,
                        ci_chunk_chars_override, ci_overlap_ratio, ci_compute_embeddings, embedding_provider
                    )
                    future_to_edge[future] = (edge_idx, rec["src"], rec["tgt"])
                
                # Collect results as they complete
                for future in as_completed(future_to_edge):
                    edge_idx, src, tgt = future_to_edge[future]
                    try:
                        edge_key, edge_metrics = future.result()
                        results[edge_key] = edge_metrics
                        logger.info(f"  ✓ Completed CI metrics for edge {edge_idx}/{total_edges}: {src} → {tgt}")
                    except Exception as e:
                        logger.error(f"  ❌ Error computing CI metrics for edge {edge_idx}/{total_edges} ({src} → {tgt}): {e}")
    else:
        logger.info(f"  📝 Processing {total_edges} edges SEQUENTIALLY")
        for edge_idx, rec in enumerate(records, 1):
            try:
                edge_key, edge_metrics = _process_single_edge_ci(
                    edge_idx, total_edges, rec, openai_key,
                    ci_chunk_chars_override, ci_overlap_ratio, ci_compute_embeddings, embedding_provider
                )
                results[edge_key] = edge_metrics
            except Exception as e:
                src = rec.get("src", "?")
                tgt = rec.get("tgt", "?")
                logger.error(f"  ❌ Error computing CI metrics for edge {edge_idx}/{total_edges} ({src} → {tgt}): {e}")

    logger.info(f"  ✅ CI METRICS COMPUTATION COMPLETE: {len(results)} edges processed")
    return results


def _compute_session_context_insensitive_metrics(
    discovery,
    session_ids: Union[str, List[str]],
    *,
    ci_fetch_reuse: bool = True,
    ci_chunk_chars_override: Optional[int] = None,
    ci_overlap_ratio: float = 0.2,
) -> Dict[str, Optional[float]]:
    if isinstance(session_ids, str):
        session_ids = [session_ids]

    agg: Dict[str, List[float]] = {
        "avg_nll": [],  # CORRECTED: collect avg_nll instead of perplexity for proper averaging
        "min_prob": [],
        "max_window_entropy": [],
        "cosine_similarity": [],
    }
    for sid in session_ids:
        per_edge = _compute_edge_level_context_insensitive_metrics(
            discovery,
            sid,
            ci_fetch_reuse=ci_fetch_reuse,
            ci_chunk_chars_override=ci_chunk_chars_override,
            ci_overlap_ratio=ci_overlap_ratio,
            ci_parallel=False,  # Sequential for session aggregation
            ci_max_workers=10,
        )
        for vals in per_edge.values():
            # CORRECTED: collect avg_nll for perplexity averaging
            avg_nll_val = vals.get("avg_nll")
            if avg_nll_val is not None:
                agg["avg_nll"].append(avg_nll_val)
            # Other metrics can be averaged directly
            for k in ["min_prob", "max_window_entropy", "cosine_similarity"]:
                v = vals.get(k)
                if v is not None:
                    agg[k].append(v)

    def mean_or_none(arr: List[float]) -> Optional[float]:
        return (sum(arr) / len(arr)) if arr else None

    # CORRECTED: compute perplexity from averaged avg_nll (exponential quantity)
    avg_nll_mean = mean_or_none(agg["avg_nll"])
    avg_perplexity = math.exp(avg_nll_mean) if avg_nll_mean is not None else None

    return {
        "avg_perplexity": avg_perplexity,
        "avg_min_prob": mean_or_none(agg["min_prob"]),
        "avg_max_window_entropy": mean_or_none(agg["max_window_entropy"]),
        "avg_cosine_similarity": mean_or_none(agg["cosine_similarity"]),
    }




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





@profile(stream=open('memory_profile_detailed.log', 'w+'))
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
    generator_config: Union[Dict[str, str], str] = "PERPLEXITY_MODEL",
    corruptor_config: Union[Dict[str, str], str] = "PERPLEXITY_MODEL",
    judge_config: Union[Dict[str, str], str] = "CLAUDE_MODEL",
    corruption_rate: float = 0.0,
    corruption_seed: int = None,
    judge_edges: bool = False,
    judge_approach: str = "per_citation_aggregate",  # "all_citations_at_once" | "per_citation_aggregate" | "correctness"
    judge_enable_web_search: bool = False,  # Enable web search for judges (disable for correctness mode)
    judge_parallel: bool = False,  # Enable parallel edge judging
    judge_max_workers: int = 10,  # Max workers for parallel judging
    run_correction: bool = False,
    rejudge_approach: str = "correctness",  # Approach for re-judging after correction
    rejudge_parallel: bool = False,  # Enable parallel re-judging after correction
    rejudge_max_workers: int = 10,  # Max workers for parallel re-judging
    corrector_models: list = None,
    correction_rounds: int = 1,
    judge_models: list = None,
    num_judges: int = 3,
    use_simple_corrector: bool = False,  # NEW: Use simple add/remove corrector (vs sophisticated revise/flip/change)
    corrector_prompt_variant: str = "original",  # NEW: Which prompt variant to use for corrector
    generator_temperature: float = 0.7,
    generator_top_p: float = 1.0,
    corruptor_temperature: float = 0.7,
    corruptor_top_p: float = 1.0,
    judge_temperature: float = 0.7,
    judge_top_p: float = 1.0,
    judge_seed: Optional[int] = None,
    corruptor_seed: Optional[int] = None,
    corrector_seed: Optional[int] = None,
    corrector_temperature: float = 0.1,
    generation_parallel: bool = True,
    generation_max_workers: int = 3,
    corruption_parallel: bool = False,
    corruption_max_workers: int = 10,
    load_env: bool = True,
    plot_session_graph: bool = True,
    plot_validation_graph: bool = True,
    only_cited_edges: bool = False,
    only_consistent_edges: bool = False,
    filter_validation_nodes_to_session_nodes: bool = True,
    experiment_description: str = None,  # Will be derived from Excel filename if not provided
    output_json_prefix: str = None,  # Will be derived from Excel filename if not provided
            compare_variable_overlap: bool = False,
        result_excel_path: str = "result_excel_path",
        overide_target_variable: str = None,
        citation_search_provider: str = None,  # NEW: Control search provider for citations
    # Node generation only mode (skip edge discovery)
    node_generation_only: bool = False,
    # Context-insensitive metrics options
    embedding_enable: bool = True,
    ci_fetch_reuse: bool = True,
    ci_chunk_chars_override: Optional[int] = None,
    ci_overlap_ratio: float = 0.2,
    ci_parallel: bool = True,  # Enable parallel CI metrics computation
    ci_max_workers: int = 10,  # Max workers for parallel CI metrics
    ci_compute_embeddings: bool = True,  # Enable expensive embedding computations in CI metrics
    embedding_provider: str = "local",  # Embedding provider for CI metrics ("local" or "openai")
    embedding_device: str = "cuda",  # Device for local embeddings: "cpu", "cuda" (default), "cuda:0", etc.
    # Citation filling parallelization
    citation_fill_parallel: bool = True,  # Enable parallel citation filling
    citation_fill_max_workers: int = 5,  # Max workers for parallel citation filling
    # Where to write outputs (JSON/XLSX). If None, current directory is used or inferred
    output_dir: Optional[str] = None,
    # Node comparison controls
    node_comparison_enable: bool = True,
    node_comparison_mode: str = "llm_batch",
    node_comparison_cosine_percentile: float = 85.0,
    node_comparison_cosine_model: str = "text-embedding-3-small",
    node_comparison_cosine_dimensions: Optional[int] = None,
    # Variable discovery mode
    preload_validation_variables: bool = True,  # If True, load variables from Excel (node metrics become meaningless)
    num_variables_to_generate: Optional[int] = None,  # If preload=False, how many variables should LLM generate? (None = use validation count)
    # Context inference mode
    infer_context_from_validation: bool = False,  # If True, infer temporal/spatial/context from validation nodes instead of using Excel context
    ):
    """
    Runs a discovery experiment for causal relationships among variables.

    Args:
        retrieved_session_id (str): Reuse an existing Neo4j session ID if not 'False'.
        excel_path (str): Path to the Excel file for CLD data.
        export_json (bool): Whether to export CLD data to JSON.
        plot_graph (bool): Whether to plot the CLD graph upon loading.
        target_variable_key (str): The key in CLD context dict that identifies the target variable.
        temporal_scale_key (str): The key in CLD context dict that identifies the temporal scale.
        spatial_scale_key (str): The key in CLD context dict that identifies the spatial scale.
        context (str): Additional context passed to CausalDiscovery.
        yaml_path (str): Path to the prompts/configuration YAML for CausalDiscovery.
        dev_mode (bool): Whether to enable dev_mode in CausalDiscovery.
        generator_config (Union[dict, str]): Config (or placeholder string) for the generator LLM. 
        corruptor_config (Union[dict, str]): Config (or placeholder string) for the corruptor LLM.
        judge_config (Union[dict, str]): Config (or placeholder string) for the judge LLM.
        corruption_rate (float): Rate of spurious relationship corruption (0.0 to 1.0).
        judge_edges (bool): Whether to judge edges.
        judge_models (list): List of model names for multi-judge logic.
        num_judges (int): Number of judges to use in multi-judge logic.
        generator_temperature (float): Generator LLM temperature.
        generator_top_p (float): Generator LLM top-p.
        citation_search_provider (str): Search provider for citations ('brave', 'perplexity', or None for LLM-only).
        corruptor_temperature (float): Corruptor LLM temperature.
        corruptor_top_p (float): Corruptor LLM top-p.
        judge_temperature (float): Judge LLM temperature.
        judge_top_p (float): Judge LLM top-p.
        generation_parallel (bool): Whether to discover relationships in parallel.
        generation_max_workers (int): Number of workers for parallel relationship discovery.
        corruption_parallel (bool): Whether to corrupt relationships in parallel (future use).
        corruption_max_workers (int): Number of workers for parallel corruption (future use).
        load_env (bool): Whether to load environment variables via dotenv.
        plot_session_graph (bool): Whether to plot the generated session graph in validation step.
        plot_validation_graph (bool): Whether to plot the validation/ground-truth graph.
        only_cited_edges (bool): If True, only include edges that have citations in the final comparison.
        only_consistent_edges (bool): If True, only consider consistent edges in the final comparison.
        filter_validation_nodes_to_session_nodes (bool): Only compare validation nodes that appear in session.
        experiment_description (str): Description to include in the final experiment info.
        output_json_prefix (str): Prefix for the final output JSON filename.
        compare_variable_overlap (bool): Compare overlap of discovered variables with the validation variables.
        result_excel_path (str): Path where final Excel with edges comparison will be saved.
        node_comparison_enable (bool): Whether to run the node/edge comparison step.
        node_comparison_mode (str): Node matching mode: "llm" (default), "cosine", or "hybrid".
        node_comparison_cosine_percentile (float): Percentile threshold for cosine candidate pruning.
        node_comparison_cosine_model (str): Embedding model to use for cosine matching.
        node_comparison_cosine_dimensions (Optional[int]): Override output dimensions for embeddings.

    Returns:
        dict: A dictionary with experiment metadata, including variables, session ID, stats, etc.
    """
    
    if load_env:
        from dotenv import load_dotenv
        # Do not override already-set environment variables; honor entrypoint overrides
        load_dotenv(".env.dev", override=False)

    # Enforce explicit judge_models configuration
    if judge_models is None and judge_edges:
        raise ValueError(
            "No judge model selected. Please provide judge_models parameter when judge_edges=True."
        )

    relationship_count = 0
    corrupted_count = 0

    # Derive output_json_prefix and experiment_description from Excel filename if not provided
    if output_json_prefix is None or experiment_description is None:
        base_filename = os.path.splitext(os.path.basename(excel_path))[0]
        if output_json_prefix is None:
            output_json_prefix = base_filename
            print(f"Using Excel filename as output_json_prefix: {output_json_prefix}")
        if experiment_description is None:
            experiment_description = base_filename
            print(f"Using Excel filename as experiment_description: {experiment_description}")

    # 1) Convert Excel -> JSON (ephemeral in a temp dir)
    with tempfile.TemporaryDirectory() as tmpdir:
        cld_data = load_cld_from_excel(
            excel_path,
            plot_graph=plot_graph,
            export_json=True,
            output_dir=tmpdir
        )
        variables_json_path = cld_data["json_files"].get("variables", None)
        edges_json_path = cld_data["json_files"].get("edges", None)

        # Persist copies of the temp JSONs beyond the TemporaryDirectory lifetime
        # so later steps can safely read them after the 'with' block exits.
        try:
            import shutil
            vars_copy = tempfile.NamedTemporaryFile(delete=False, suffix="_vars_data.json")
            vars_copy_path = vars_copy.name
            vars_copy.close()
            shutil.copyfile(variables_json_path, vars_copy_path)
            variables_json_path = vars_copy_path

            edges_copy = tempfile.NamedTemporaryFile(delete=False, suffix="_edges_data.json")
            edges_copy_path = edges_copy.name
            edges_copy.close()
            shutil.copyfile(edges_json_path, edges_copy_path)
            edges_json_path = edges_copy_path
        except Exception:
            pass

        # 2) Load variables from ephemeral JSON
        with open(variables_json_path, "r") as f:
            excel_vars_data = json.load(f)  # list of { "name": ..., "definition": ... }

        # 3) Extract context from Excel
        original_target = cld_data['context'].get(target_variable_key, None)
        if isinstance(original_target, dict) and "name" in original_target:
            # Some Excel exports store target as a dict with "name"
            original_target = original_target["name"]

        temporal_scale = cld_data['context'].get(temporal_scale_key, None)
        spatial_scale = cld_data['context'].get(spatial_scale_key, None)

        # Decide final target_variable based on override logic
        if overide_target_variable is not None:
            # If user specifically passes a string
            if overide_target_variable.strip() == "":
                # Empty => skip target entirely
                target_variable = None
            else:
                # Non-empty => override
                target_variable = overide_target_variable.strip()
        else:
            # None => use the Excel target if present
            target_variable = original_target

        # Build a cleaned list of variables:
        # If we are overriding the Excel target with a new name, remove the old one from the list
        # so it does not get re-added as a non-target.
        if overide_target_variable is not None and original_target is not None:
            # Remove the "old" Excel target from the final set
            filtered_vars_data = [v for v in excel_vars_data if v["name"] != original_target]
        else:
            filtered_vars_data = excel_vars_data

        # If the override was an empty string => no target => do nothing else,
        # or if it's a new target => we create that ourselves
        # So we won't add any "Excel" node for the old target in that scenario.

        print(f"target_variable in run_discovery_experiment: {target_variable}")
        print(f"DEBUG: Before CausalDiscovery init:")
        print(f"  temporal_scale={temporal_scale}")
        print(f"  spatial_scale={spatial_scale}")

        # 4) Instantiate the discovery
        discovery = CausalDiscovery(
            target_variable=target_variable,  # can be None if empty override
            temporal_scale=temporal_scale,
            spatial_scale=spatial_scale,
            yaml_path=yaml_path,
            dev_mode=dev_mode,
            generator_config=generator_config,
            corruptor_config=corruptor_config,
            judge_config=judge_config,
            context=context,
            generator_temperature=generator_temperature,
            generator_top_p=generator_top_p,
            corruptor_temperature=corruptor_temperature,
            corruptor_top_p=corruptor_top_p,
            judge_temperature=judge_temperature,
            judge_top_p=judge_top_p,
            judge_seed=judge_seed,
            corruptor_seed=corruptor_seed,
            corrector_seed=corrector_seed,
            judge_enable_web_search=judge_enable_web_search
        )
        
        # Configure citation filling parallelization
        discovery.citation_fill_parallel = citation_fill_parallel
        discovery.citation_fill_max_workers = citation_fill_max_workers

        # 5) Setup session
        if retrieved_session_id == 'False':
            session_id = discovery.session_id
            # Create nodes in the new session
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

                # (a) If we DO have a real target_variable, create it
                if target_variable:
                    session.run(
                        create_query,
                        {
                            "name": target_variable,
                            "description": f"Target variable: {target_variable}",
                            "session_id": session_id,
                            "is_target": True
                        }
                    )
                    discovery.variables.append(target_variable)

                # (b) Add the rest from the filtered Excel list
                # ONLY if preload_validation_variables is True
                if preload_validation_variables:
                    print(f"⚠️  PRE-LOADING {len(filtered_vars_data)} validation variables (node overlap metrics will be meaningless)")
                    for var_info in filtered_vars_data:
                        var_name = var_info["name"]
                        var_def = var_info.get("definition", "")
                        # If this name is our new override target name, skip it
                        # (since we just created it as target).
                        if target_variable and var_name == target_variable:
                            continue

                        session.run(
                            create_query,
                            {
                                "name": var_name,
                                "description": f"definition: {var_def}",
                                "session_id": session_id,
                                "is_target": False
                            }
                        )
                        discovery.variables.append(var_name)
                else:
                    print(f"✅ TRUE DISCOVERY MODE: LLM will generate variables (node overlap metrics will be calculated)")
                    # Don't pre-load variables - let the LLM discover them
                    # Determine how many to generate
                    target_var_count = num_variables_to_generate if num_variables_to_generate is not None else len(excel_vars_data)
                    print(f"   Target: Generate {target_var_count} variables (validation has {len(excel_vars_data)})")
                    
                    # Extract target info from Excel data if available
                    target_units = None
                    target_definition = None
                    if target_variable and filtered_vars_data:
                        # Try to find target variable in the Excel data
                        for var_info in excel_vars_data:
                            if var_info.get("name") == target_variable:
                                target_definition = var_info.get("definition", "")
                                target_units = var_info.get("units", "")
                                break
                    
                    # Log warnings if we're missing data that will use defaults
                    if not target_units:
                        print(f"   ⚠️  WARNING: No target_units found in Excel - will use default")
                    if not target_definition:
                        print(f"   ⚠️  WARNING: No target_definition found in Excel - will use default")
                    if not temporal_scale:
                        print(f"   ⚠️  WARNING: No temporal_scale found in Excel - will use default")
                    if not spatial_scale:
                        print(f"   ⚠️  WARNING: No spatial_scale found in Excel - will use default")
                    
                    # Optionally infer context from validation nodes
                    if infer_context_from_validation:
                        print(f"\n{'='*60}")
                        print(f"INFERRING CONTEXT FROM VALIDATION NODES")
                        print(f"{'='*60}")
                        validation_node_names = [v.get("name") for v in excel_vars_data]
                        print(f"Validation nodes: {validation_node_names}")
                        
                        inferred_context = infer_context_from_validation_nodes(
                            validation_nodes=validation_node_names,
                            generator_llm=discovery.generator_llm
                        )
                        
                        # Override temporal/spatial/context with inferred values
                        if inferred_context.get("temporal_scale"):
                            temporal_scale = inferred_context["temporal_scale"]
                            discovery.temporal_scale = temporal_scale
                            print(f"   Inferred Temporal Scale: {temporal_scale}")
                        
                        if inferred_context.get("spatial_scale"):
                            spatial_scale = inferred_context["spatial_scale"]
                            discovery.spatial_scale = spatial_scale
                            print(f"   Inferred Spatial Scale: {spatial_scale}")
                        
                        if inferred_context.get("domain"):
                            print(f"   Inferred Domain: {inferred_context['domain']}")
                        
                        if inferred_context.get("context_description"):
                            print(f"   Inferred Context: {inferred_context['context_description']}")
                        
                        context_list = inferred_context.get("context_list", [])
                        print(f"{'='*60}\n")
                    else:
                        # Build context list from Excel context data
                        context_list = []
                        for key, value in cld_data.get('context', {}).items():
                            if key not in [target_variable_key, temporal_scale_key, spatial_scale_key] and value:
                                context_list.append(f"{key}: {value}")
                    
                    # Debug: Show what we're passing to generate_variables
                    print(f"   DEBUG: Calling generate_variables with:")
                    print(f"     num_variables={target_var_count}")
                    print(f"     target_units={target_units}")
                    print(f"     target_definition={target_definition}")
                    print(f"     context_list={context_list}")
                    print(f"     discovery.target_variable={discovery.target_variable}")
                    print(f"     discovery.temporal_scale={discovery.temporal_scale}")
                    print(f"     discovery.spatial_scale={discovery.spatial_scale}")
                    
                    # Generate variables using LLM with complete context
                    generated_vars = discovery.generate_variables(
                        num_variables=target_var_count,
                        target_units=target_units,
                        target_definition=target_definition,
                        context_list=context_list
                    )
                    print(f"   Generated {len(generated_vars)} variables: {generated_vars}")
                    
                    # DEBUG: Check what's actually in Neo4j for this session
                    with discovery.graph_db._get_session() as session:
                        debug_nodes = session.run(
                            "MATCH (n:variable) WHERE n.session_id=$sid RETURN n.name as name",
                            {"sid": session_id}
                        )
                        debug_node_list = [rec["name"] for rec in debug_nodes]
                        print(f"   DEBUG: Nodes in Neo4j for session {session_id}: {debug_node_list}")
                        print(f"   DEBUG: Expected {len(generated_vars)}, found {len(debug_node_list)}")

            # 6) Discover relationships (skip if node_generation_only mode)
            if node_generation_only:
                print(f"\n{'='*60}")
                print(f"NODE GENERATION ONLY MODE - Skipping edge discovery")
                print(f"{'='*60}\n")
                relationships = []
                relationship_count = 0
                corrupted_count = 0
            else:
                relationships = discovery.discover_relationships(
                    generation_parallel=generation_parallel,
                    generation_max_workers=generation_max_workers,
                    corruption_rate=corruption_rate
                )
                relationship_count = len(relationships)
                corrupted_count = getattr(discovery, "corrupted_count", 0)

        else:
            # Reuse an existing session
            session_id = retrieved_session_id
            discovery.set_session_id(session_id)

            # Just fetch how many relationships exist already
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

            # Also rebuild discovery.variables from DB
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
            
            # Apply corruption to existing edges if requested
            corrupted_count = 0
            if corruption_rate > 0:
                logger.info(f"Applying {corruption_rate*100}% corruption to existing session {session_id}")
                corrupted_count = discovery._corrupt_relationships(corruption_rate, seed=corruption_seed)
                logger.info(f"Corrupted {corrupted_count} edges in existing session")

        # 7) Fill missing citations (skip if node_generation_only mode)
        if node_generation_only:
            print(f"\n{'='*60}")
            print(f"NODE GENERATION ONLY MODE - Skipping citation filling")
            print(f"{'='*60}\n")
            citation_fill_time = 0
        else:
            #    This tries to add real references if not present
            # Skip citation filling if citation_search_provider is None (e.g., correctness mode)
            if citation_search_provider is None:
                print(f"\n{'='*50}")
                print("SKIPPING CITATION FILLING (citation_search_provider=None)")
                print(f"{'='*50}\n")
                citation_fill_time = 0
            else:
                print(f"\n{'='*50}")
                print("STARTING CITATION FILLING PHASE")
                print(f"{'='*50}")
                print(f"Citation search provider: {citation_search_provider}")
            
                citation_fill_start_time = time.time()
                # When Brave Search is specified, skip LLM and go straight to search
                # Otherwise use LLM first with search as fallback
                if citation_search_provider == "brave":
                    print("Using Brave Search for direct citation filling...")
                    discovery.fill_in_missing_citations(
                        llm_provider=None,  # Skip LLM entirely for Brave Search
                        llm_model=None,
                        search_provider=citation_search_provider
                    )
                else:
                    print("Using LLM first, then search fallback for citation filling...")
                    # Use LLM first, then fallback to search (if any search provider specified)
                    discovery.fill_in_missing_citations(
                        llm_provider="anthropic", 
                        llm_model="claude-4-sonnet-20250514",
                        search_provider=citation_search_provider
                    )

                citation_fill_time = time.time() - citation_fill_start_time
                print(f"\n{'='*50}")
                print(f"CITATION FILLING PHASE COMPLETED in {citation_fill_time:.2f} seconds")
                print(f"{'='*50}")

        # 8) Possibly judge (skip if node_generation_only mode)
        judged_count = 0
        if node_generation_only and judge_edges:
            print(f"\n{'='*60}")
            print(f"NODE GENERATION ONLY MODE - Skipping judging phase")
            print(f"{'='*60}\n")
        elif judge_edges:
            print(f"\n{'='*50}")
            print(f"STARTING JUDGING PHASE")
            print(f"Judge models: {judge_models}")
            print(f"Num judges: {num_judges}")
            print(f"{'='*50}\n")
            try:
                # Citation-based approach with parallel support
                judged_edges = discovery.judge_all_edges_with_citations_serial(
                    judge_models=judge_models,
                    num_judges=num_judges,
                    approach=judge_approach,
                    judge_parallel=judge_parallel,
                    judge_max_workers=judge_max_workers
                )
                judged_count = len(judged_edges)
                print(f"\n{'='*50}")
                print(f"JUDGING PHASE COMPLETED")
                print(f"Judged {judged_count} edges")
                print(f"{'='*50}\n")
            except AttributeError as e:
                # fallback to basic judging if citation-based judging is not available
                print(f"Citation-based judging not available ({e}), falling back to basic judging...")
                try:
                    judged_edges = discovery.judge_all_edges_serial(
                        judge_models=judge_models,
                        num_judges=num_judges
                    )
                    judged_count = len(judged_edges)
                    print(f"\n{'='*50}")
                    print(f"JUDGING PHASE COMPLETED (fallback)")
                    print(f"Judged {judged_count} edges")
                    print(f"{'='*50}\n")
                except Exception as fallback_error:
                    print(f"ERROR: Fallback judging also failed: {fallback_error}")
                    import traceback
                    traceback.print_exc()
                    judged_count = 0
            except Exception as e:
                print(f"ERROR: Judging phase failed with exception: {e}")
                import traceback
                traceback.print_exc()
                judged_count = 0

        # 9) Optionally run LLM-as-a-corrector (skip if node_generation_only mode)
        correction_outcomes = []
        corrected_count = 0
        if node_generation_only and run_correction:
            print(f"\n{'='*60}")
            print(f"NODE GENERATION ONLY MODE - Skipping correction phase")
            print(f"{'='*60}\n")
        elif run_correction:
            print(f"\n{'='*50}")
            print("STARTING CORRECTION PHASE")
            print(f"{'='*50}")
            print(f"Corrector models: {corrector_models}")
            print(f"Max rounds: {correction_rounds}")
            print(f"Action order: ('revise', 'recite', 'remove')")
            try:
                correction_start_time = time.time()
                correction_outcomes = discovery.correct_edges_serial(
                    corrector_models=corrector_models,
                    max_rounds=correction_rounds,
                    action_order=("revise", "recite", "remove"),
                    rejudge_after=True,
                    rejudge_approach=rejudge_approach,
                    judge_models=judge_models,
                    num_judges=num_judges,
                    use_simple_corrector=use_simple_corrector,  # Pass through simple corrector flag
                    corrector_prompt_variant=corrector_prompt_variant,  # Pass through prompt variant
                )
                correction_time = time.time() - correction_start_time
                corrected_count = sum(1 for o in correction_outcomes if o.get("action") in ("revise_or_recite", "remove"))
                print(f"\n{'='*50}")
                print(f"CORRECTION PHASE COMPLETED in {correction_time:.2f} seconds")
                print(f"Corrected {corrected_count} edges")
                print(f"Correction outcomes: {len(correction_outcomes)} total actions")
                print(f"{'='*50}")
            except Exception as e:
                print(f"Correction stage failed: {e}")
        else:
            print("\nSkipping correction phase (run_correction=False)")

        # Gather LLM usage stats
        def safe_get_stats(llm_client):
            if hasattr(llm_client, 'get_stats'):
                return llm_client.get_stats()
            return {
                "status_counts": {},
                "token_totals": {},
                "inference_time_total": 0.0,
                "inference_call_count": 0
            }

        gen_stats = safe_get_stats(discovery.generator_llm)
        corr_stats = safe_get_stats(discovery.corruptor_llm)
        judge_stats = safe_get_stats(discovery.judge_llm)
        
    # Include embedding stats (cosine CI metrics) if available
    try:
        from data_science.logit_metrics import get_embedding_stats
        embed_stats = get_embedding_stats()
    except Exception:
        embed_stats = {
            "status_counts": {},
            "token_totals": {},
            "inference_time_total": 0.0,
            "inference_call_count": 0,
        }

    # 10) Summarize
    result = {
        "session_id": session_id,
        "stats": {
            "variable_count": len(discovery.variables),
            "relationship_count": relationship_count,
            "corrupted_count": corrupted_count,
            "judged_count": judged_count,
            "corrected_count": corrected_count
        },
        "lm_stats": {
            "generator": gen_stats,
            "corruptor": corr_stats,
            "judge": judge_stats,
            "embeddings": embed_stats,
        }
    }

    # Debug: Inspect
    discovery.inspect_relationships()
    print(
        f"\nGenerated {result['stats']['variable_count']} variables; "
        f"{result['stats']['relationship_count']} relationships. "
        f"Corrupted: {result['stats']['corrupted_count']}; "
        f"Judged: {result['stats']['judged_count']}; "
        f"Corrected: {result['stats']['corrected_count']}"
    )
    print(f"\nSample Cypher:\nMATCH (n)-[r]->(m) WHERE n.session_id='{session_id}' RETURN n,r,m")

    # 10) Save experiment info to JSON
    ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Prepare output directory
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            pass
    out_fname = f"{output_json_prefix}_{ts_str}.json"
    if output_dir:
        out_fname = os.path.join(output_dir, out_fname)
    experiment_info = {
        "timestamp": ts_str,
        "description": experiment_description,
        "variables": discovery.variables,
        "target_variable": target_variable,
        "session_id": session_id,
        "stats": result["stats"],
        "temporal_scale": temporal_scale,
        "spatial_scale": spatial_scale,
        "lm_stats": result["lm_stats"]
    }
    with open(out_fname, "w") as f:
        json.dump(experiment_info, f, indent=2)
    print(f"Saved session info to {out_fname}")

    # 11) Compare with the same JSON we used from Excel (unless you have separate ground-truth files).
    # IMPORTANT: Skip node comparison if variables were pre-loaded (metrics would be meaningless)
    node_comparison_results = None
    if node_comparison_enable and not preload_validation_variables:
        print(f'variables_json_path: {variables_json_path}')
        print(f'edges_json_path: {edges_json_path}')
        print("✅ Running node comparison (TRUE DISCOVERY MODE - variables were not pre-loaded)")
        # Pass target_variable if it was used as a seed from validation
        seeded_target = target_variable if (target_variable and not preload_validation_variables) else None
        if seeded_target:
            print(f"   Note: Using '{seeded_target}' as seeded target (will compute adjusted metrics)")
        node_comparison_results = compare_session_graph_to_validation(
            discovery,
            session_ids=session_id,
            validation_vars_json_path=variables_json_path,
            validation_edges_json_path=edges_json_path,
        plot_session_graph=plot_session_graph,
        plot_validation_graph=plot_validation_graph,
        only_cited_edges=only_cited_edges,
        only_consistent_edges=only_consistent_edges,
        filter_validation_nodes_to_session_nodes=filter_validation_nodes_to_session_nodes,
        compare_variable_overlap=compare_variable_overlap,
            # Node matching controls
            node_match_mode=node_comparison_mode,
            cosine_percentile=node_comparison_cosine_percentile,
            cosine_model=node_comparison_cosine_model,
            cosine_dimensions=node_comparison_cosine_dimensions,
            # CI metrics
        embedding_enable=embedding_enable,
        ci_fetch_reuse=ci_fetch_reuse,
        ci_chunk_chars_override=ci_chunk_chars_override,
        ci_overlap_ratio=ci_overlap_ratio,
            # Seeded target for adjusted metrics
            seeded_target_variable=seeded_target,
        )
    elif node_comparison_enable and preload_validation_variables:
        print("⚠️  SKIPPING NODE COMPARISON: Variables were pre-loaded from validation Excel")
        print("   Node overlap metrics would be meaningless (100% by construction)")
        print("   Set preload_validation_variables=False for true discovery mode")
    else:
        print("Skipping node/edge comparison (node_comparison_enable=False)")

    # 12) Export scenario-based edges to Excel (including embedding usage stats)
    # Build experiment parameters for the "Params" sheet
    experiment_params = {
        "excel_path": excel_path,
        "yaml_path": yaml_path,
        "dev_mode": dev_mode,
        "generator_config": generator_config,
        "corruptor_config": corruptor_config,
        "judge_config": judge_config,
        "corruption_rate": corruption_rate,
        "judge_edges": judge_edges,
        "judge_models": judge_models,
        "num_judges": num_judges,
        "generator_temperature": generator_temperature,
        "generator_top_p": generator_top_p,
        "corruptor_temperature": corruptor_temperature,
        "corruptor_top_p": corruptor_top_p,
        "judge_temperature": judge_temperature,
        "judge_top_p": judge_top_p,
        "judge_seed": judge_seed,
        "corruptor_seed": corruptor_seed,
        "corrector_seed": corrector_seed,
        "generation_parallel": generation_parallel,
        "generation_max_workers": generation_max_workers,
        "corruption_parallel": corruption_parallel,
        "corruption_max_workers": corruption_max_workers,
        "plot_session_graph": plot_session_graph,
        "plot_validation_graph": plot_validation_graph,
        "only_cited_edges": only_cited_edges,
        "only_consistent_edges": only_consistent_edges,
        "filter_validation_nodes_to_session_nodes": filter_validation_nodes_to_session_nodes,
        "compare_variable_overlap": compare_variable_overlap,
        "node_comparison_enable": node_comparison_enable,
        "node_comparison_mode": node_comparison_mode,
        "node_comparison_cosine_percentile": node_comparison_cosine_percentile,
        "node_comparison_cosine_model": node_comparison_cosine_model,
        "node_comparison_cosine_dimensions": node_comparison_cosine_dimensions,
        "result_excel_path": result_excel_path,
        "overide_target_variable": overide_target_variable,
        "citation_search_provider": citation_search_provider,
        "embedding_enable": embedding_enable,
        "ci_fetch_reuse": ci_fetch_reuse,
        "ci_chunk_chars_override": ci_chunk_chars_override,
        "ci_overlap_ratio": ci_overlap_ratio,
        "session_id": session_id,
        "timestamp": ts_str,
        "temporal_scale": temporal_scale,
        "spatial_scale": spatial_scale,
        "target_variable": target_variable,
        "preload_validation_variables": preload_validation_variables,
        "num_variables_to_generate": num_variables_to_generate,
        "node_generation_only": node_generation_only,
        "infer_context_from_validation": infer_context_from_validation,
    }

    print(f"\n{'='*50}")
    print("STARTING EXCEL EXPORT PHASE")
    print(f"{'='*50}")
    print(f"Creating Excel file with multiple scenario sheets...")
    print(f"Output path: {(os.path.join(output_dir, os.path.basename(result_excel_path)) if output_dir else result_excel_path)}")
    
    # Prepare node comparison data for Excel export
    node_comparison_data = None
    if node_comparison_results:
        # Extract matched pairs from node_mapping
        matched_pairs = []
        node_mapping = node_comparison_results.get("node_mapping", {})
        for session_node, validation_node in node_mapping.items():
            matched_pairs.append((session_node, validation_node))
        
        node_comparison_data = {
            "session_nodes": node_comparison_results.get("session_nodes", []),
            "validation_nodes": node_comparison_results.get("validation_nodes", []),
            "matched_pairs": matched_pairs,
            "metrics": {
                # Binary metrics
                "node_precision": node_comparison_results.get("node_precision", 0.0),
                "node_recall": node_comparison_results.get("node_recall", 0.0),
                "node_f1": node_comparison_results.get("node_f1", 0.0),
                # Cosine similarity metrics
                "cosine_node_precision": node_comparison_results.get("cosine_node_precision"),
                "cosine_node_recall": node_comparison_results.get("cosine_node_recall"),
                "cosine_node_f1": node_comparison_results.get("cosine_node_f1"),
                "avg_cosine_gen_to_val": node_comparison_results.get("avg_cosine_gen_to_val"),
                "avg_cosine_val_to_gen": node_comparison_results.get("avg_cosine_val_to_gen"),
                # Hybrid metrics
                "hybrid_node_precision": node_comparison_results.get("hybrid_node_precision"),
                "hybrid_node_recall": node_comparison_results.get("hybrid_node_recall"),
                "hybrid_node_f1": node_comparison_results.get("hybrid_node_f1"),
                "avg_similarity_gen_to_val": node_comparison_results.get("avg_similarity_gen_to_val"),
                "avg_similarity_val_to_gen": node_comparison_results.get("avg_similarity_val_to_gen"),
                # Adjusted binary metrics (excluding seed)
                "adjusted_node_precision": node_comparison_results.get("adjusted_node_precision"),
                "adjusted_node_recall": node_comparison_results.get("adjusted_node_recall"),
                "adjusted_node_f1": node_comparison_results.get("adjusted_node_f1"),
                "adjusted_matched_pairs": node_comparison_results.get("adjusted_matched_pairs"),
                # Adjusted hybrid metrics (excluding seed)
                "adjusted_hybrid_precision": node_comparison_results.get("adjusted_hybrid_precision"),
                "adjusted_hybrid_recall": node_comparison_results.get("adjusted_hybrid_recall"),
                "adjusted_hybrid_f1": node_comparison_results.get("adjusted_hybrid_f1"),
                "seeded_target_variable": node_comparison_results.get("seeded_target_variable"),
            }
        }
    
    excel_export_start_time = time.time()
    excel_path_final = export_edges_comparison_to_excel(
        discovery=discovery,
        session_ids=session_id,
        validation_vars_json_path=variables_json_path,
        validation_edges_json_path=edges_json_path,
        output_filename=(os.path.join(output_dir, os.path.basename(result_excel_path)) if output_dir else result_excel_path),
        lm_stats=experiment_info["lm_stats"],
        experiment_params=experiment_params,
        node_comparison_data=node_comparison_data,
        node_match_mode=node_comparison_mode,  # Pass the mode to avoid redundant comparisons
        embedding_enable=embedding_enable,
        ci_fetch_reuse=ci_fetch_reuse,
        ci_chunk_chars_override=ci_chunk_chars_override,
        ci_overlap_ratio=ci_overlap_ratio,
        ci_compute_embeddings=ci_compute_embeddings,  # Pass embedding computation flag
        ci_parallel=ci_parallel,  # Pass parallel flag for CI metrics
        ci_max_workers=ci_max_workers,  # Pass max workers for CI metrics
        embedding_provider=embedding_provider,  # Pass embedding provider
        embedding_device=embedding_device,  # Pass embedding device
    )
    excel_export_time = time.time() - excel_export_start_time
    print(f"\n{'='*50}")
    print(f"EXCEL EXPORT PHASE COMPLETED in {excel_export_time:.2f} seconds")
    print(f"Final Excel file: {excel_path_final}")
    print(f"{'='*50}")
    experiment_info["result_excel_path"] = excel_path_final
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

    # If user didn't provide connected_subsets, we find them:
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

                # Insert the subgraph's variables into the DB
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
                rels = discovery.discover_relationships(generation_parallel=generation_parallel, generation_max_workers=generation_max_workers)
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

    # 1) Collect relevant .xlsx files (support nested per-run directories)
    pattern = os.path.join(input_folder, experiment_id, "**", "*.xlsx")
    all_xlsx = glob.glob(pattern, recursive=True)
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
            # Optionally skip if we can't map it
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

    # 4) For each prompt, we'll write blocks side-by-side (one block per param_combo)
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

        # --- Next: Each metric repeated across each param combo's scenario block
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



def multirun_parameter_experiments(
    PROMPTS_FOLDER="parameter_tuning_experiments/alternative_prompts",
    CONFIG_FILE="parameter_tuning_experiments/configs/experiment_28_04_2025_config.yaml",
    CLD_FOLDER="parameter_tuning_experiments/ground_truth_clds_for_experiments",
    RESULTS_FOLDER="parameter_tuning_experiments/results",
    FORCE_RERUN=False
):
    """
    Main function:
      1) Generate a unique 'experiment_id' for all runs of this entire experiment.
      2) Find all prompt YAMLs that don't have results yet (or skip logic if not needed).
      3) Load the single experiment config (param grid & runs).
      4) Gather ground-truth CLDs.
      5) For each prompt => for each param combo => for each CLD => for each run:
           - Call `run_discovery_experiment(...)`
           - Produce a unique Excel filename (with combo_number).
           - Log each result/parameter combo in a separate CSV file.
    """
    # 1) Create a single experiment ID for *all* runs of this script
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_uuid = str(uuid.uuid4())[:8]  # short random
    experiment_id = f"exp_{timestamp_str}_{experiment_uuid}"
    print(f"Experiment ID: {experiment_id}\n")

    # 2) Determine which prompts to process
    # Load the full config to check for prompt_files specification
    with open(CONFIG_FILE, "r") as f:
        full_config = yaml.safe_load(f)
    specified_prompt_files = full_config.get("prompt_files", None)
    
    if specified_prompt_files:
        # Use only the specified prompt files from config (ignores FORCE_RERUN and existing results)
        print(f"Config specifies prompt_files: {specified_prompt_files}")
        unprocessed_prompts = []
        for prompt_file in specified_prompt_files:
            full_path = os.path.join(PROMPTS_FOLDER, prompt_file)
            if os.path.exists(full_path):
                unprocessed_prompts.append(full_path)
                print(f"✅ Found specified prompt: {prompt_file}")
            else:
                print(f"❌ Specified prompt not found: {prompt_file}")
        if not unprocessed_prompts:
            print("No valid specified prompt files found. Exiting.")
            return
    else:
        # Default behavior: process all prompts (with optional result skipping)
        if FORCE_RERUN:
            print("FORCE_RERUN=True: Processing all prompts regardless of existing results.")
            yaml_files = glob.glob(os.path.join(PROMPTS_FOLDER, "*.yaml"))
            unprocessed_prompts = yaml_files
        else:
            unprocessed_prompts = find_unprocessed_prompts(PROMPTS_FOLDER, RESULTS_FOLDER)
            if not unprocessed_prompts:
                print("All prompts already have corresponding results. Nothing to do.")
                print("Use --force flag to re-run experiments even when results exist.")
                return

    # 3) Load config
    param_grid, runs = load_experiment_config(CONFIG_FILE)
    param_names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in param_names]
    combos      = list(itertools.product(*value_lists))  # each combo is a tuple

    # 4) Gather ground-truth CLDs (with optional filtering)
    # Load the full config to check for excel_files filter
    with open(CONFIG_FILE, "r") as f:
        full_config = yaml.safe_load(f)
    excel_files_filter = full_config.get("excel_files", None)
    
    cld_sets = find_cld_sets(CLD_FOLDER, excel_files_filter)
    if not cld_sets:
        print("No valid CLDs found. Exiting.")
        return

    # Prepare a CSV file to map each result Excel to its parameter combo
    csv_mapping_path = os.path.join(RESULTS_FOLDER, f"param_combo_mapping_{experiment_id}.csv")
    with open(csv_mapping_path, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "experiment_id",
            "prompt",
            "cld_prefix",
            "run_idx",
            "excel_filename",
            "combo_number",
            "combo_str",
            "parameters"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # If the file is new/empty, write the header first
        if csvfile.tell() == 0:
            writer.writeheader()

        # 5) For each prompt
        for prompt_file in unprocessed_prompts:
            prompt_base = os.path.splitext(os.path.basename(prompt_file))[0]  # e.g. "chain_of_thought1"
            print(f"\n=== Prompt: {prompt_file} ===")

            # Enumerate over combos to get combo_number
            for combo_idx, combo in enumerate(combos, start=1):
                combo_dict = dict(zip(param_names, combo))
                combo_str = "_".join(f"{k}{v}" for k,v in combo_dict.items())
                combo_number = combo_idx  # This will be used in filename and CSV mapping

                for (cld_prefix, cld_excel_path) in cld_sets:

                    for run_idx in range(1, runs + 1):
                        print(f" -> Running {prompt_base}, CLD '{cld_prefix}', combo={combo_dict}, run={run_idx}/{runs}")

                        # Create a unique per-run directory structure
                        run_uuid = str(uuid.uuid4())[:8]
                        run_dir = os.path.join(
                            RESULTS_FOLDER,
                            experiment_id,
                            f"combo_{combo_number}",
                            prompt_base,
                            cld_prefix,
                            f"run{run_idx}_{run_uuid}"
                        )
                        
                        print(f"\n📁 OUTPUT DIRECTORY (START): {run_dir}")
                        print(f"   Experiment: {experiment_id}")
                        print(f"   Combo: {combo_number} | Prompt: {prompt_base} | CLD: {cld_prefix} | Run: {run_idx}")
                        
                        try:
                            os.makedirs(run_dir, exist_ok=True)
                        except Exception:
                            pass

                        # Construct result path inside the run directory
                        excel_filename = (
                            f"results_{experiment_id}_"
                            f"param_combo_nr_{combo_number}_"
                            f"{prompt_base}_"
                            f"run{run_idx}_"
                            f"{cld_prefix}.xlsx"
                        )
                        result_excel_path = os.path.join(run_dir, excel_filename)

                        # Actually call your discovery experiment
                        # Only set overide_target_variable if not already in combo_dict
                        if 'overide_target_variable' not in combo_dict:
                            combo_dict['overide_target_variable'] = ""
                        
                        exp_info = run_discovery_experiment(
                            # retrieved_session_id="797c9efb-6ef3-48ab-8897-aac10cac6f25",
                            excel_path=cld_excel_path,
                            yaml_path=prompt_file,
                            result_excel_path=result_excel_path,
                            output_dir=run_dir,
                            **combo_dict
                        )
                        
                        print(f"\n✅ RUN COMPLETED SUCCESSFULLY!")
                        print(f"📁 OUTPUT DIRECTORY (END): {run_dir}")
                        print(f"📊 Results saved to: {result_excel_path}")
                        print(f"🔧 Parameters used: {combo_dict}")
                        if exp_info and 'session_id' in exp_info:
                            print(f"🆔 Session ID: {exp_info['session_id']}")
                        print(f"{'='*80}\n")
                        # Write one row to CSV for each run
                        writer.writerow({
                            "experiment_id": experiment_id,
                            "prompt": prompt_base,
                            "cld_prefix": cld_prefix,
                            "run_idx": run_idx,
                            "excel_filename": os.path.relpath(result_excel_path, RESULTS_FOLDER),
                            "combo_number": combo_number,
                            "combo_str": combo_str,
                            # Save the parameter dict as JSON or any format you prefer
                            "parameters": json.dumps(combo_dict)
                        })

    print("\nAll done! You can now aggregate the .xlsx files by experiment ID.\n")


def find_unprocessed_prompts(prompts_folder, results_folder):
    """
    Return a list of .yaml prompt files in `prompts_folder` that do *not*
    yet have a corresponding .xlsx in `results_folder`.
    (This logic can be adapted or removed if you prefer not to skip anything.)
    """
    yaml_files = glob.glob(os.path.join(prompts_folder, "*.yaml"))
    unprocessed = []
    for yf in yaml_files:
        base_name = os.path.splitext(os.path.basename(yf))[0]
        # Consider processed if any XLSX containing the prompt base exists anywhere under results
        any_match = glob.glob(os.path.join(results_folder, "**", f"*{base_name}*.xlsx"), recursive=True)
        if not any_match:
            unprocessed.append(yf)
    return unprocessed

def load_experiment_config(config_file):
    """
    Load a single experiment config YAML that defines:
      - param_grid: dictionary of param_name -> list[values]
      - runs: how many times to repeat each combo
    Return (param_grid, runs).
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Could not find config file: {config_file}")
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    param_grid = cfg.get("param_grid", {})
    runs       = cfg.get("runs", 1)
    return param_grid, runs

def find_cld_sets(folder, excel_files_filter=None):
    """
    Return a list of (prefix, excel_path) for each .xlsx file in `folder`.
    If excel_files_filter is provided, only include files in that list.
    For example: [("CLD1", ".../CLD1.xlsx"), ("CLD2", ".../CLD2.xlsx"), ...]
    """
    xlsx_files = glob.glob(os.path.join(folder, "*.xlsx"))
    cld_sets = []
    
    for xfile in xlsx_files:
        basename = os.path.basename(xfile)        # e.g. "CLD1.xlsx"
        
        # Apply filter if specified
        if excel_files_filter is not None:
            if basename not in excel_files_filter:
                print(f"Skipping {basename} (not in excel_files filter)")
                continue
            else:
                print(f"Including {basename} (found in excel_files filter)")
        
        prefix, _ = os.path.splitext(basename)    # e.g. "CLD1"
        cld_sets.append((prefix, xfile))
    
    print(f"Found {len(cld_sets)} CLD files to process")
    return cld_sets

def aggregate_experiment_excels(
    experiment_id, 
    results_folder="parameter_tuning_experiments/results", 
    aggregate_runs=True, 
    aggregate_clds=True,
    aggregate_parameters=True,
    output_excel_name=None,
    parameter_focus=None  # Optional parameter name to highlight in reporting
):
    """
    Aggregate edge evaluation metrics (confusion matrix and derived statistics) from experiment Excel files.
    
    This function reads the parameter mapping CSV and aggregates metrics across different dimensions.
    
    Aggregation options:
    - aggregate_runs=True: Combine metrics across different runs (run1, run2, etc.)
    - aggregate_clds=True: Combine metrics across different CLD datasets
    - aggregate_parameters=True: Combine metrics across different parameter combinations
    
    Args:
        experiment_id: The experiment ID to filter by
        results_folder: Folder containing result files
        aggregate_runs: Whether to aggregate across different runs
        aggregate_clds: Whether to aggregate across different CLDs
        aggregate_parameters: Whether to aggregate across different parameter combinations
        output_excel_name: Custom name for the output Excel file. If None, uses default naming.
        parameter_focus: Optional parameter name to highlight in grouping and reporting
        
    Returns:
        Dict containing the aggregated metrics with mean, min, max, and std values.
    """
    # 1) Find parameter mapping CSV file
    mapping_pattern = os.path.join(results_folder, f"param_combo_mapping_{experiment_id}.csv")
    mapping_files = glob.glob(mapping_pattern)
    
    if not mapping_files:
        print(f"No parameter mapping file found for experiment_id={experiment_id} in {results_folder}")
        print("Falling back to filename-based detection...")
        # Continue with filename-based detection as fallback
    else:
        # Use the first mapping file found
        mapping_file = mapping_files[0]
        print(f"Using parameter mapping from: {mapping_file}")
        
        # Read parameter mapping
        param_mapping_df = pd.read_csv(mapping_file)
        
        # Check if files exist
        file_info = []
        for _, row in param_mapping_df.iterrows():
            excel_path = os.path.join(results_folder, row['excel_filename'])
            if os.path.exists(excel_path):
                # Extract parameters as dictionary
                try:
                    params_dict = json.loads(row['parameters'])
                except:
                    params_dict = {}  # Use empty dict if JSON parsing fails
                
                file_info.append({
                    "path": excel_path,
                    "param_combo": row['combo_number'],
                    "run": row['run_idx'],
                    "cld": row['cld_prefix'],
                    "parameters": params_dict
                })
            else:
                print(f"Warning: Excel file not found: {excel_path}")
    
    if not file_info:
        print(f"No valid Excel files found for experiment_id={experiment_id} in {results_folder}")
        return {}

    # Extract parameter keys from the first file that has parameters
    param_keys = []
    for info in file_info:
        if "parameters" in info and info["parameters"]:
            param_keys = list(info["parameters"].keys())
            break
    
    # If parameter_focus was specified, check if it's valid
    if parameter_focus and param_keys and parameter_focus not in param_keys:
        print(f"Warning: Specified parameter_focus '{parameter_focus}' not found in parameter keys: {param_keys}")
        parameter_focus = None

    # Describe aggregation mode for output
    agg_descriptions = []
    if aggregate_parameters:
        agg_descriptions.append("ALL parameter combinations")
    else:
        if parameter_focus:
            agg_descriptions.append(f"EACH value of '{parameter_focus}'")
        else:
            agg_descriptions.append("EACH parameter combination")
    
    if aggregate_runs:
        agg_descriptions.append("ALL runs")
    else:
        agg_descriptions.append("EACH run")
        
    if aggregate_clds:
        agg_descriptions.append("ALL CLDs")
    else:
        agg_descriptions.append("EACH CLD")
    
    agg_mode = f"Aggregating across {', '.join(agg_descriptions)}"
    
    print(f"\n=== EDGE STATISTICS AGGREGATION ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Aggregation mode: {agg_mode}")
    print(f"Found {len(file_info)} Excel files to process")
    print("=" * 40)

    # 2) Group files based on aggregation settings
    groups = []
    
    if aggregate_parameters and aggregate_runs and aggregate_clds:
        # Aggregate everything
        groups = [{"name": "all", "files": [f["path"] for f in file_info]}]
    else:
        # Need to group by various combinations
        groupings = {}
        
        for info in file_info:
            # Create a key based on which dimensions we're NOT aggregating
            key_parts = []
            
            if not aggregate_parameters:
                if parameter_focus and "parameters" in info and info["parameters"]:
                    # Use only the focused parameter for grouping
                    param_value = info["parameters"].get(parameter_focus, "unknown")
                    key_parts.append(f"{parameter_focus}={param_value}")
                else:
                    # Use parameter combo number
                    key_parts.append(f"param{info['param_combo']}")
            
            if not aggregate_runs:
                key_parts.append(f"run{info['run']}")
                
            if not aggregate_clds:
                key_parts.append(f"cld_{info['cld']}")
            
            group_key = "_".join(key_parts) if key_parts else "all"
            
            if group_key not in groupings:
                groupings[group_key] = []
            
            groupings[group_key].append(info["path"])
        
        groups = [{"name": name, "files": files} for name, files in groupings.items()]

    # 3) Process each group
    all_results = {}
    all_raw_data = []
    
    for group in groups:
        group_name = group["name"]
        group_files = group["files"]
        
        print(f"\nProcessing group: {group_name} ({len(group_files)} files)")
        
        # Prepare accumulators
        metrics_list = []

        for fpath in group_files:
            try:
                # Read the "All Edges Summary" sheet
                df_summary = pd.read_excel(fpath, sheet_name="All Edges Summary", usecols=[0,1])
                
                # It's typically 2 columns: "Metric" | "Value"
                # Convert to a dictionary
                metrics_dict = dict(zip(df_summary.iloc[:, 0], df_summary.iloc[:, 1]))
                
                # Get key metrics
                one_run = {
                    "TP": float(metrics_dict.get("True Positives (TP)", 0)),
                    "FP": float(metrics_dict.get("False Positives (FP)", 0)),
                    "FN": float(metrics_dict.get("False Negatives (FN)", 0)),
                    "TN": float(metrics_dict.get("True Negatives (TN)", 0)),
                    "precision": float(metrics_dict.get("Precision", 0)),
                    "recall": float(metrics_dict.get("Recall", 0)),
                    "f1": float(metrics_dict.get("F1 Score", 0)),
                    "accuracy": float(metrics_dict.get("Accuracy", 0)),
                    "file": os.path.basename(fpath),
                    "group": group_name
                }
                
                # Extract parameter information if available
                for info in file_info:
                    if info["path"] == fpath and "parameters" in info:
                        if parameter_focus and parameter_focus in info["parameters"]:
                            one_run[parameter_focus] = info["parameters"][parameter_focus]
                        else:
                            # Add all parameters
                            for param_key, param_value in info["parameters"].items():
                                one_run[f"param_{param_key}"] = param_value
                        
                        # Add other metadata
                        one_run["param_combo"] = info["param_combo"]
                        one_run["run"] = info["run"]
                        one_run["cld"] = info["cld"]
                        break
                
                metrics_list.append(one_run)
                all_raw_data.append(one_run)
            except Exception as e:
                print(f"[WARN] Skipping file {fpath} due to error: {e}")

        if not metrics_list:
            print(f"No valid summary rows found in group '{group_name}'.")
            continue
            
        # Convert to DataFrame to compute statistics
        df_all = pd.DataFrame(metrics_list)
        
        # Calculate mean, min, max, and std for each metric
        df_mean = df_all.mean(numeric_only=True)
        df_min = df_all.min(numeric_only=True)
        df_max = df_all.max(numeric_only=True)
        df_std = df_all.std(numeric_only=True)

        # Store group results
        all_results[group_name] = {
            "file_count": len(metrics_list),
            # Mean values
            "TP_mean": df_mean["TP"],
            "FP_mean": df_mean["FP"],
            "FN_mean": df_mean["FN"],
            "TN_mean": df_mean["TN"],
            "precision_mean": df_mean["precision"],
            "recall_mean": df_mean["recall"],
            "f1_mean": df_mean["f1"],
            "accuracy_mean": df_mean["accuracy"],
            # Min values
            "TP_min": df_min["TP"],
            "FP_min": df_min["FP"],
            "FN_min": df_min["FN"],
            "TN_min": df_min["TN"],
            "precision_min": df_min["precision"],
            "recall_min": df_min["recall"],
            "f1_min": df_min["f1"],
            "accuracy_min": df_min["accuracy"],
            # Max values
            "TP_max": df_max["TP"],
            "FP_max": df_max["FP"],
            "FN_max": df_max["FN"],
            "TN_max": df_max["TN"],
            "precision_max": df_max["precision"],
            "recall_max": df_max["recall"],
            "f1_max": df_max["f1"],
            "accuracy_max": df_max["accuracy"],
            # Standard deviation values
            "TP_std": df_std["TP"],
            "FP_std": df_std["FP"],
            "FN_std": df_std["FN"],
            "TN_std": df_std["TN"],
            "precision_std": df_std["precision"],
            "recall_std": df_std["recall"],
            "f1_std": df_std["f1"],
            "accuracy_std": df_std["accuracy"],
        }
        
        # Print the min, max, mean, and std values for each group
        print(f"\nEdge statistics for group: {group_name} (from {len(metrics_list)} files)")
        print("-" * 80)
        metrics = ["TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]
        for metric in metrics:
            print(f"{metric.ljust(10)}: min = {df_min[metric]:.4f}, max = {df_max[metric]:.4f}, " +
                  f"mean = {df_mean[metric]:.4f}, std = {df_std[metric]:.4f}")
        print("-" * 80)
        
        # If focusing on a specific parameter, show its effect
        if parameter_focus and parameter_focus in df_all.columns:
            try:
                print(f"\nPerformance by {parameter_focus} value:")
                param_groups = df_all.groupby(parameter_focus)
                for param_value, group_df in param_groups:
                    print(f"  {parameter_focus} = {param_value}:")
                    for metric in ["precision", "recall", "f1", "accuracy"]:
                        mean_val = group_df[metric].mean()
                        std_val = group_df[metric].std()
                        print(f"    {metric.ljust(10)}: {mean_val:.4f} ± {std_val:.4f}")
            except Exception as e:
                print(f"Error analyzing parameter effect: {e}")
    
    # Add experiment metadata
    result = {
        "experiment_id": experiment_id,
        "total_file_count": sum(all_results[group_name]["file_count"] for group_name in all_results),
        "groups": all_results
    }
    
    # Save aggregated statistics to Excel
    if output_excel_name is None:
        # Create default filename with aggregation mode
        agg_mode_parts = []
        if aggregate_parameters:
            agg_mode_parts.append("param")
        if aggregate_runs:
            agg_mode_parts.append("run")
        if aggregate_clds:
            agg_mode_parts.append("cld")
            
        agg_mode_suffix = "all" if len(agg_mode_parts) == 3 else f"by_{'_'.join(agg_mode_parts)}"
        output_excel_name = f"agg_stats_{experiment_id}_{agg_mode_suffix}.xlsx"
    
    # Ensure it has .xlsx extension
    if not output_excel_name.endswith('.xlsx'):
        output_excel_name += '.xlsx'
    
    output_path = os.path.join(results_folder, output_excel_name)
    
    # Create a DataFrame for each group's statistics
    group_stats = []
    for group_name, stats in all_results.items():
        row = {
            "Group": group_name,
            "Files": stats["file_count"]
        }
        
        # Add all metrics
        metrics = ["TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]
        for metric in metrics:
            row[f"{metric}_mean"] = stats[f"{metric}_mean"]
            row[f"{metric}_min"] = stats[f"{metric}_min"]
            row[f"{metric}_max"] = stats[f"{metric}_max"]
            row[f"{metric}_std"] = stats[f"{metric}_std"]
        
        group_stats.append(row)
    
    # Convert to DataFrame
    df_group_stats = pd.DataFrame(group_stats)
    
    # Create raw data DataFrame
    df_raw_data = pd.DataFrame(all_raw_data)
    
    # Write to Excel
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Add group statistics sheet
            df_group_stats.to_excel(writer, sheet_name="Group Statistics", index=False)
            
            # Add raw data sheet
            if not df_raw_data.empty:
                df_raw_data.to_excel(writer, sheet_name="Raw Data", index=False)
            
            # Add experiment metadata sheet
            pd.DataFrame([{
                "Experiment ID": experiment_id,
                "Total Files": result["total_file_count"],
                "Aggregation Mode": agg_mode,
                "Parameter Focus": parameter_focus or "None",
                "Aggregate Parameters": str(aggregate_parameters),
                "Aggregate Runs": str(aggregate_runs),
                "Aggregate CLDs": str(aggregate_clds),
                "Generation Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }]).to_excel(writer, sheet_name="Metadata", index=False)
            
            # If parameter focus is specified, add a parameter analysis sheet
            if parameter_focus and parameter_focus in df_raw_data.columns:
                # Create the aggregation with multiple metrics
                param_analysis = df_raw_data.groupby(parameter_focus).agg({
                    'precision': ['mean', 'std', 'min', 'max'],
                    'recall': ['mean', 'std', 'min', 'max'],
                    'f1': ['mean', 'std', 'min', 'max'],
                    'accuracy': ['mean', 'std', 'min', 'max']
                })
                
                # Fix MultiIndex columns by flattening them
                param_analysis.columns = [
                    f"{col[0]}_{col[1]}" for col in param_analysis.columns
                ]
                
                # Reset index to make parameter values a column
                param_analysis = param_analysis.reset_index()
                
                # Now write to Excel (no MultiIndex issues)
                param_analysis.to_excel(writer, sheet_name=f"{parameter_focus} Analysis", index=False)
        
        print(f"\nAggregated statistics saved to: {output_path}")
    except Exception as e:
        print(f"Error saving aggregated statistics to Excel: {e}")
    
    return result

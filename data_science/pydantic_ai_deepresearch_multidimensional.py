"""
PydanticAI Deep Research Orchestration System with Multi-Dimensional Evidence Assessment

This module contains a set of Pydantic models, agent definitions, and a manager class 
(DeepResearchManager) for conducting a multi-step scientific literature investigation 
on a causal claim. It uses a sequence of PydanticAI agents to plan a search, retrieve 
relevant scientific evidence, score the evidence quality, judge whether the evidence 
supports the claim, and synthesize a final verdict.

ENHANCED VERSION: Includes multi-dimensional evidence quality assessment using LLM-as-judge
to score evidence on 5 dimensions: Causal Strength, Mechanistic Specificity, Construct Validity,
Scope Relevance, and Theoretical Coherence.

The workflow is orchestrated by the DeepResearchManager class using PydanticAI framework.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Tuple
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass, field
import json
import asyncio
from rich.console import Console
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logfire
import logging
import re
from pathlib import Path
import sys

# Add backend directory to path to import OpenAI client for multidimensional judge
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

# Load environment variables
load_dotenv()

# Configure Logfire for observability
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
ENABLE_LOGFIRE = os.getenv("ENABLE_LOGFIRE", "false").lower() in ("true", "1", "yes")

if ENABLE_LOGFIRE and LOGFIRE_TOKEN:
    # Enable Logfire with token from environment
    logfire.configure(token=LOGFIRE_TOKEN)
    # Instrument PydanticAI for automatic tracing of agent calls
    logfire.instrument_pydantic_ai()
    print("✅ Logfire enabled - sending telemetry to causalix-agentobs project")
else:
    # Disable logfire (local-only mode)
    logfire.configure(send_to_logfire=False)
    if ENABLE_LOGFIRE and not LOGFIRE_TOKEN:
        print("⚠️  ENABLE_LOGFIRE=true but LOGFIRE_TOKEN not set. Running without Logfire.")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see API calls
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable detailed API logging
logging.getLogger('pydantic_ai').setLevel(logging.DEBUG)
logging.getLogger('anthropic').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.DEBUG)  # To see actual HTTP requests

# Set API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "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"
BRAVE_SEARCH_API_KEY = "BSAkUYW2uG3mxguq4HI_oszbpc1Dc8j"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if BRAVE_SEARCH_API_KEY:
    os.environ["BRAVE_SEARCH_API_KEY"] = BRAVE_SEARCH_API_KEY

# Configure local/remote model selections via environment
# Defaults keep existing behavior unless overridden
HEAVY_MODEL = os.getenv("LOCAL_LLM_HEAVY_MODEL", "openai:gpt-5")
MINI_MODEL = os.getenv("LOCAL_LLM_MINI_MODEL", "openai:gpt-5-mini")

# Multi-Dimensional Evidence Quality Assessment Constants
DIMENSION_WEIGHTS = {
    'evidence_quality': 0.30,
    'mechanistic_specificity': 0.25,
    'construct_validity': 0.20,
    'scope_relevance': 0.15,
    'theoretical_coherence': 0.10
}

MULTIDIMENSIONAL_JUDGE_SYSTEM_PROMPT = """You are an expert methodologist evaluating the quality of causal evidence in scientific research.

Your task is to score a proposed causal relationship on 5 dimensions based on the evidence gathered through systematic literature review.
\
# SCORING DIMENSIONS

## 1. EVIDENCE QUALITY (Study Design Quality) - Weight: 30%
Score the quality of the causal evidence based on study design:

- 9-10: Randomized Controlled Trial (RCT) or meta-analysis of RCTs with consistent findings
- 7-8: High-quality quasi-experimental design, natural experiment, or longitudinal study with temporal precedence
- 5-6: Cross-sectional studies with mediation analysis or strong observational evidence
- 3-4: Cross-sectional correlational studies without temporal information
- 1-2: Theoretical reasoning only, no empirical support
- 0: No evidence or contradictory evidence

## 2. MECHANISTIC SPECIFICITY (Direct vs Mediated) - Weight: 25%
Score how directly the evidence supports the proposed causal mechanism:

- 9-10: Evidence shows a DIRECT causal mechanism matching the exact pathway in the claim
- 7-8: Evidence for a plausible direct mechanism with minor gaps
- 5-6: Evidence shows a general association but mechanism is unclear or complex
- 3-4: Evidence shows the relationship is MEDIATED through other variables (X→Z→Y, not X→Y)
- 1-2: No mechanism specified or mechanism contradicts the claim
- 0: Claim is tautological, circular, or conceptually incoherent

**CRITICAL**: If evidence shows "X affects Y ONLY THROUGH Z" or "effect is fully mediated by Z", this is NOT a direct effect and should score 2-4, NOT 7-10.

## 3. CONSTRUCT VALIDITY (Right Variables Measured) - Weight: 20%
Score how well the evidence measures the EXACT constructs named in the causal claim:

- 9-10: Exact match - same measurement instruments/operationalizations
- 7-8: Close proxy - validated alternative measure of the same construct
- 5-6: Related concept (e.g., BMI used as proxy for obesity, silhouettes for ideal BMI)
- 3-4: Distant relative - same domain but different aspect
- 1-2: Tangentially related - weak construct overlap
- 0: Wrong construct - evidence is for different variables entirely

**CRITICAL**: If studies use figure ratings/silhouettes instead of numeric BMI, or different scales than specified, score 5-6 max.

## 4. SCOPE RELEVANCE (Population & Context Match) - Weight: 15%
Score how well the evidence population/context matches the intended scope:

- 9-10: Exact match - same population, age group, cultural context
- 7-8: Close match - similar population with good generalizability
- 5-6: Generalizable - different population but findings likely transfer
- 3-4: Questionable transfer - different life stage, culture, or setting
- 1-2: Poor match - animal models, wrong age group, incompatible context
- 0: Completely wrong scope

## 5. THEORETICAL COHERENCE (Fits Conceptual Framework) - Weight: 10%
Score how well the evidence aligns with theoretical frameworks:

- 9-10: Theory-driven hypothesis with strong theoretical support
- 7-8: Consistent with established theoretical frameworks
- 5-6: Compatible with theory but uses different frameworks
- 3-4: Atheoretical empirical finding without theoretical grounding
- 1-2: Weak or unclear theoretical connection
- 0: Contradicts theory, or claim itself is tautological/circular

**CRITICAL**: If the reviewer states the claim is "tautological", "circular", "ill-posed", or "conceptually incorrect", score 0.

# OUTPUT FORMAT

You must output your scores in this EXACT format:

EVIDENCE_QUALITY: [score 0-10]
MECHANISTIC_SPECIFICITY: [score 0-10]
CONSTRUCT_VALIDITY: [score 0-10]
SCOPE_RELEVANCE: [score 0-10]
THEORETICAL_COHERENCE: [score 0-10]

REASONING: [2-3 sentences explaining your scores, focusing on the key strengths and weaknesses]

# IMPORTANT GUIDELINES

1. Read the JUDGE REASONING carefully - it often contains critical phrases like "mediated by", "tautological", "construct mismatch"
2. The MODIFIED CLAIM shows what the evidence ACTUALLY supports vs the original claim
3. High confidence and "supported" verdicts do NOT automatically mean high scores on all dimensions
4. An edge can have excellent causal evidence (high Causal Strength) but still score low if it's for the wrong mechanism or constructs
5. Be critical but fair - score based on what's written, not assumptions
"""


# Telemetry tracking class
@dataclass
class DeepResearchTelemetry:
    """Track telemetry for deep research operations."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    agent_call_details: List[Dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    def add_call(self, agent_name: str, success: bool, result=None, error=None):
        """Record an agent call with telemetry data."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        call_detail = {
            "agent": agent_name,
            "success": success,
            "timestamp": time.time()
        }
        
        if result and hasattr(result, 'usage'):
            # Extract token usage from Pydantic AI RunResult
            try:
                usage = result.usage()
                call_detail["tokens"] = {
                    "total": getattr(usage, 'total_tokens', 0),
                    "prompt": getattr(usage, 'prompt_tokens', 0),
                    "completion": getattr(usage, 'completion_tokens', 0)
                }
                self.total_tokens += call_detail["tokens"]["total"]
                self.total_prompt_tokens += call_detail["tokens"]["prompt"]
                self.total_completion_tokens += call_detail["tokens"]["completion"]
            except Exception as e:
                call_detail["tokens_error"] = str(e)
        
        if error:
            call_detail["error"] = str(error)
        
        self.agent_call_details.append(call_detail)
    
    def get_summary(self) -> Dict:
        """Get summary of telemetry data."""
        duration = self.end_time - self.start_time if self.end_time > 0 else 0
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / max(self.total_calls, 1),
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "duration_seconds": duration,
            "agent_call_details": self.agent_call_details
        }

# Dependency classes
@dataclass
class ResearchContext:
    """Context passed between agents containing claim and iteration info"""
    claim: str
    iteration: int = 1
    max_iterations: int = 3
    console: Console = None
    elapsed_seconds: float = 0.0
    time_limit_seconds: float = 120.0  # 2 minutes

@dataclass 
class SearchContext:
    """Context for search operations"""
    claim: str
    query: str
    max_sources: int = 5
    searches_made: int = 0  # Track how many web searches have been made in this thread

@dataclass
class EvidenceContext:
    """Context for evidence evaluation"""
    claim: str
    evidence_content: str
    source_url: str = ""


# Pydantic Models (same as original)
class SearchQuery(BaseModel):
    """Represents a single search query to investigate a causal claim."""
    query: str = Field(description="The search query to use")
    rationale: str = Field(description="Explanation for why this query is relevant to testing the causal claim")


class SearchPlan(BaseModel):
    """Search plan with supporting, counterfactual, and alternative mechanism queries."""
    
    supporting_queries: List[SearchQuery] = Field(
        description="2-3 queries to find SUPPORTING evidence for the claim"
    )
    
    counterfactual_queries: List[SearchQuery] = Field(
        description="1-2 queries to find CONTRADICTORY evidence (null results, opposite effects)"
    )
    
    alternative_mechanism_queries: List[SearchQuery] = Field(
        description="1-2 queries to find ALTERNATIVE mechanisms (mediation, confounding)"
    )
    
    keywords: List[str] = Field(description="Key terms and synonyms for the causal relationship")
    confounders: List[str] = Field(description="Potential confounding variables to check for")
    
    @property
    def all_queries(self) -> List[SearchQuery]:
        """Get all queries combined."""
        return self.supporting_queries + self.counterfactual_queries + self.alternative_mechanism_queries


class EvidenceScore(BaseModel):
    """Represents a scored evaluation of evidence quality and relevance."""
    relevance_score: int = Field(description="How directly the evidence addresses the causal relationship (1-10)")
    quality_score: int = Field(description="Methodological rigor of the evidence (1-10)")
    justification: str = Field(description="Reasoning for these scores")
    source_url: Optional[str] = Field(default=None, description="URL to the original source if available")
    
    # Intermediate judgment (support level) for snippets
    support_level: Optional[str] = Field(
        default=None,
        description="Support level: 'fully_supported' (1.0), 'partially_supported' (0.5), 'unsupported' (0.0), or 'contradicting' (-0.5)"
    )
    support_score: Optional[float] = Field(
        default=None,
        description="Numeric support score: 1.0 (fully), 0.5 (partial), 0.0 (unsupported), -0.5 (contradicting)"
    )


class EvidenceJudgment(BaseModel):
    """Represents a judgment about whether a causal claim is supported by evidence."""
    supported: bool = Field(description="Whether the claim is supported by the evidence")
    confidence: int = Field(description="Confidence level in this judgment (1-10)")
    reasoning: str = Field(description="Detailed reasoning for the judgment")
    suggested_modification: Optional[str] = Field(default=None, description="Suggested modification to the original claim if needed")


class MultidimensionalScores(BaseModel):
    """Multi-dimensional evidence quality scores from LLM-as-judge."""
    evidence_quality: float = Field(description="Study design quality score (0-10)")
    mechanistic_specificity: float = Field(description="Direct vs mediated mechanism score (0-10)")
    construct_validity: float = Field(description="Measures exact constructs score (0-10)")
    scope_relevance: float = Field(description="Population/context match score (0-10)")
    theoretical_coherence: float = Field(description="Fits theory score (0-10)")
    reasoning: str = Field(description="2-3 sentences explaining the scores")


class CausalEvidenceEvaluation(BaseModel):
    """Per-article GRADE assessment with 7 dimensions."""
    
    # Binary judgment
    direct_evidence_found: bool = Field(
        description="Does article explicitly discuss causal relationship X→Y?"
    )
    confidence: int = Field(description="Confidence in finding (0-10)")
    
    # Intermediate judgment (support level)
    support_level: Optional[str] = Field(
        default=None,
        description="Support level: 'fully_supported' (1.0), 'partially_supported' (0.5), 'unsupported' (0.0), or 'contradicting' (-0.5)"
    )
    support_score: Optional[float] = Field(
        default=None,
        description="Numeric support score: 1.0 (fully), 0.5 (partial), 0.0 (unsupported), -0.5 (contradicting)"
    )
    
    # ===== 7 GRADE DIMENSIONS =====
    
    risk_of_bias: float = Field(
        description="GRADE: Study design quality (0-10). 10=RCT low bias, 0=anecdotal"
    )
    
    indirectness_mechanism: float = Field(
        description="GRADE: Mechanism directness (0-10). 10=direct proven, 0=fully mediated"
    )
    
    indirectness_population: float = Field(
        description="GRADE: Population applicability (0-10). 10=exact match, 0=wrong population"
    )
    
    indirectness_constructs: float = Field(
        description="GRADE: Construct match (0-10). 10=exact variables, 0=wrong constructs"
    )
    
    imprecision: float = Field(
        description="GRADE: Precision of estimate (0-10). 10=large n + narrow CI, 0=small n + wide CI"
    )
    
    magnitude_of_effect: float = Field(
        description="GRADE: Effect size (0-10). 10=very large (d>0.8, RR>5), 0=none"
    )
    
    dose_response_gradient: float = Field(
        description="GRADE: Dose-response present? (0-10). 10=strong gradient, 0=none"
    )
    
    # Evidence details
    effect_size_reported: Optional[str] = Field(
        default=None,
        description="Reported effect size (e.g., 'Cohen's d=0.65', 'RR=2.3')"
    )
    cited_passage: str
    study_type: str
    causal_evidence_summary: str
    source_url: Optional[str] = None


class ArticleSelection(BaseModel):
    """Selection of articles to fetch full text for based on likelihood of containing causal evidence."""
    urls_to_fetch: List[str] = Field(description="URLs of 2-3 articles most likely to contain direct causal evidence (maximum 3 articles)", max_length=3)
    rationale: str = Field(description="Explanation for why these articles were selected")


class IterationHistory(BaseModel):
    """History of a single research iteration."""
    iteration_number: int = Field(description="Iteration number (1, 2, 3...)")
    claim_tested: str = Field(description="The claim being tested in this iteration")
    evidence_count: int = Field(description="Number of evidence sources found")
    high_quality_count: int = Field(description="Number of high-quality sources (rel>=8, qual>=7)")
    judgment: Optional[str] = Field(description="Judgment at end of iteration (supported/not supported)")
    judgment_confidence: Optional[int] = Field(description="Confidence in judgment (1-10)")
    suggested_modification: Optional[str] = Field(description="Suggested claim modification for next iteration")


class FinalVerdict(BaseModel):
    """Final verdict on a causal claim after research."""
    original_claim: str
    modified_claim: Optional[str] = None
    verdict: str = Field(description="One of: supported, partially_supported, unsupported, or inconclusive")
    confidence: int = Field(description="Confidence in the verdict (1-10)")
    judge_reasoning: str
    key_evidence_labels: List[str]
    key_evidence_summaries: List[str]
    key_evidence_rationales: List[str]
    evidence_urls: List[str]
    limitations: List[str]
    future_research: List[str]
    # Fields populated by verdict_synthesizer based on consensus
    strongest_causal_quote: Optional[str] = Field(default=None, description="Direct quote from the strongest piece of evidence supporting the direct causal mechanism")
    strongest_evidence_url: Optional[str] = Field(default=None, description="URL to the strongest piece of evidence where the quote is from")
    
    # ===== SNIPPET CONSENSUS (LLM assessment of search results) =====
    snippet_consensus_support: Optional[str] = Field(
        default=None,
        description="LLM assessment of support consensus across search snippets (how many support vs oppose)"
    )
    snippet_consensus_quality: Optional[str] = Field(
        default=None,
        description="LLM assessment of quality pattern across search snippets"
    )
    counter_evidence_found: Optional[bool] = Field(
        default=None,
        description="Whether counterfactual evidence (null results, opposite effects) was found"
    )
    counter_evidence_summary: Optional[str] = Field(
        default=None,
        description="Summary of counterfactual evidence found, if any"
    )
    alternative_mechanisms_found: Optional[bool] = Field(
        default=None,
        description="Whether evidence for alternative/mediated mechanisms was found"
    )
    alternative_mechanisms_summary: Optional[str] = Field(
        default=None,
        description="Summary of alternative mechanisms found, if any"
    )
    
    # ===== ARTICLE CONSENSUS (LLM judgment, not average) =====
    consensus_mechanism: Optional[str] = Field(
        default=None,
        description="LLM assessment of mechanism consensus across full articles"
    )
    consensus_effect: Optional[str] = Field(
        default=None,
        description="LLM assessment of effect magnitude consensus across full articles"
    )
    consensus_quality: Optional[str] = Field(
        default=None,
        description="LLM assessment of study quality consensus across full articles"
    )
    
    # ===== BINARY JUDGMENTS (Holistic LLM assessment, NOT based on consensus voting) =====
    
    # Direct causal mechanism judgment with reasoning
    direct_causal_reasoning: Optional[str] = Field(
        default=None,
        description="LLM reasoning for direct causal judgment (2-3 sentences)"
    )
    direct_causal_confidence: Optional[str] = Field(
        default=None,
        description="Confidence in direct causal judgment: high/medium/low"
    )
    
    # ===== FINAL BINARY JUDGMENT: DIRECT CAUSAL EDGE FOR CLD =====
    is_direct_causal: Optional[int] = Field(
        default=None,
        description="Binary numeric (0 or 1): 1 if relationship is BOTH causal AND direct (not mediated), 0 otherwise. Only 1 means suitable for CLD inclusion."
    )
    
    # ===== SUPPORT SCORES (Programmatic averages, not from LLM) =====
    average_snippet_support_score: Optional[float] = Field(
        default=None,
        description="Average support score across all snippet evaluations (1.0=fully, 0.5=partial, 0.0=unsupported, -0.5=contradicting)"
    )
    average_article_support_score: Optional[float] = Field(
        default=None,
        description="Average support score across all full article evaluations (1.0=fully, 0.5=partial, 0.0=unsupported, -0.5=contradicting)"
    )
    average_combined_support_score: Optional[float] = Field(
        default=None,
        description="Average support score combining both snippets and articles (1.0=fully, 0.5=partial, 0.0=unsupported, -0.5=contradicting)"
    )
    snippet_support_scores: Optional[List[float]] = Field(
        default=None,
        description="Individual support scores from each snippet evaluation"
    )
    article_support_scores: Optional[List[float]] = Field(
        default=None,
        description="Individual support scores from each article evaluation"
    )
    
    iteration_history: Optional[List[IterationHistory]] = Field(default=None, description="History of claim refinements and evidence across iterations")
    # Multi-dimensional evidence quality scores (0-10 each)
    evidence_quality: Optional[float] = Field(default=None, description="Study design quality (0-10)")
    mechanistic_specificity: Optional[float] = Field(default=None, description="Direct vs mediated mechanism (0-10)")
    construct_validity: Optional[float] = Field(default=None, description="Measures exact constructs (0-10)")
    scope_relevance: Optional[float] = Field(default=None, description="Population/context match (0-10)")
    theoretical_coherence: Optional[float] = Field(default=None, description="Fits theory (0-10)")
    overall_quality_score: Optional[float] = Field(default=None, description="Weighted average quality score (0-10)")
    multidimensional_reasoning: Optional[str] = Field(default=None, description="Explanation of multidimensional scores")
    
    # ===== TELEMETRY =====
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed across all API calls")
    total_api_calls: Optional[int] = Field(default=None, description="Total number of LLM API calls made")
    elapsed_time_seconds: Optional[float] = Field(default=None, description="Total elapsed time in seconds")
    iterations_completed: Optional[int] = Field(default=None, description="Number of research iterations completed")


class SearchResult(BaseModel):
    """Result from web search with content and URL."""
    content: str
    url: str
    query_type: Optional[str] = Field(
        default="supporting",
        description="Type of query: 'supporting', 'counterfactual', or 'alternative_mechanism'"
    )


# Brave Search API Implementation for Scientific Literature
class BraveSearchEngine:
    """Brave Search Engine for scientific literature with academic domain filtering."""
    
    def __init__(self, api_key: str = None, logger=None):
        """Initialize the Brave searcher with API key"""
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        self.logger = logger
        
        # Scientific and academic trusted domains
        self.trusted_domains = [
            "pubmed.ncbi.nlm.nih.gov", "arxiv.org", "scholar.google.com",
            "nature.com", "science.org", "cell.com", "nejm.org", "bmj.com",
            "plos.org", "frontiersin.org", "springer.com", "wiley.com",
            "sciencedirect.com", "tandfonline.com", "sage.com", "jstor.org",
            "cochranelibrary.com", "nih.gov", "who.int", "cdc.gov",
            "academic.oup.com", "cambridge.org", "elsevier.com"
        ]
        
        # Create session with retry strategy and connection pooling
        self.session = self._create_session_with_retries()
        
        if self.logger:
            self.logger.info(f"BraveSearchEngine initialized with {len(self.trusted_domains)} trusted domains")
        
    def _create_session_with_retries(self):
        """Create a requests session with retry strategy and connection pooling."""
        session = requests.Session()
        
        # Retry strategy for transient failures
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Perform search and return SearchResult objects for scientific literature."""
        
        # Fallback to mock data if no API key
        if not self.api_key:
            if self.logger:
                self.logger.warning("No Brave Search API key found, using mock data")
            return [
                SearchResult(
                    content=f"[No Brave API Key] Mock search result for query: {query}. This would contain actual scientific literature from Brave Search API.",
                    url=f"https://example.com/search/{query.replace(' ', '-')}"
                )
            ]
        
        # Enhance query for scientific content
        enhanced_query = f"{query} scientific study research evidence"
        
        if self.logger:
            self.logger.debug(f"Searching for: {enhanced_query}")
        
        try:
            # Make API request with timeout and retry handling
            response = self.session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json", 
                    "X-Subscription-Token": self.api_key
                },
                params={
                    "q": enhanced_query,
                    "count": max_results * 2,  # Get more results to filter for scientific content
                    "search_lang": "en",
                    "extra_snippets": True
                },
                timeout=(10, 40)  # (connect_timeout, read_timeout)
            )
            
            # Process response
            if response.status_code == 200:
                if self.logger:
                    self.logger.info(f"Successfully received search results for query: {query}")
                return self._process_results(response.json(), max_results)
            else:
                if self.logger:
                    self.logger.error(f"Search API request failed with status code: {response.status_code}")
                    self.logger.error(f"Response text: {response.text}")
                
                # Fallback to mock data on API error
                return [
                    SearchResult(
                        content=f"[API Error {response.status_code}] Mock search result for query: {query}. Brave Search API returned error.",
                        url=f"https://example.com/search/{query.replace(' ', '-')}"
                    )
                ]
                
        except requests.exceptions.Timeout as e:
            if self.logger:
                self.logger.error(f"Search API request timed out: {e}")
            return [
                SearchResult(
                    content=f"[Timeout Error] Mock search result for query: {query}. API request timed out.",
                    url=f"https://example.com/search/{query.replace(' ', '-')}"
                )
            ]
        except requests.exceptions.ConnectionError as e:
            if self.logger:
                self.logger.error(f"Connection error to Brave Search API: {e}")
            return [
                SearchResult(
                    content=f"[Connection Error] Mock search result for query: {query}. Failed to connect to API.",
                    url=f"https://example.com/search/{query.replace(' ', '-')}"
                )
            ]
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during search: {e}")
            return [
                SearchResult(
                    content=f"[Search Error] Mock search result for query: {query}. Error: {str(e)}",
                    url=f"https://example.com/search/{query.replace(' ', '-')}"
                )
            ]
        
    def _parse_description(self, description):
        """Parse HTML description and return clean text."""
        try:
            if description:
                soup = BeautifulSoup(description, 'html.parser')
                return soup.get_text()
            return ""
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error parsing description: {e}")
            return description or ""

    def _process_results(self, data, max_results):
        """Extract and format scientific citations from search results."""
        search_results = []
        
        try:
            web_results = data.get("web", {}).get("results", [])
            if self.logger:
                self.logger.debug(f"Processing {len(web_results)} search results")
            
            trusted_results = []
            general_results = []
            
            for result in web_results:
                url = result.get("url", "")
                title = result.get("title", "")
                description = result.get("description", "")
                extra_snippets = result.get("extra_snippets", [])
                
                # Clean HTML from description
                clean_description = self._parse_description(description)
                
                # Combine title, description, and snippets for content
                content_parts = []
                if title:
                    content_parts.append(f"Title: {title}")
                if clean_description:
                    content_parts.append(f"Description: {clean_description}")
                
                # Add extra snippets for more context
                for i, snippet in enumerate(extra_snippets[:3]):  # Limit to 3 snippets
                    content_parts.append(f"Snippet {i+1}: {snippet}")
                
                content = " | ".join(content_parts)
                
                # Check if from trusted scientific domain
                is_trusted = any(domain in url.lower() for domain in self.trusted_domains)
                
                search_result = SearchResult(content=content, url=url)
                
                if is_trusted:
                    trusted_results.append(search_result)
                else:
                    general_results.append(search_result)
            
            # Prioritize trusted scientific sources
            search_results.extend(trusted_results[:max_results])
            
            # Fill remaining slots with general results if needed
            remaining_slots = max_results - len(search_results)
            if remaining_slots > 0:
                search_results.extend(general_results[:remaining_slots])
            
            if self.logger:
                trusted_count = len(trusted_results)
                general_count = len(search_results) - trusted_count
                self.logger.info(f"Found {len(search_results)} total results: {trusted_count} from trusted scientific domains, {general_count} general results")
            
            return search_results[:max_results]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing search results: {e}")
            return []


# Agent Definitions using PydanticAI
hypothesis_evaluator = Agent(
    HEAVY_MODEL,
    deps_type=ResearchContext,
    output_type=SearchPlan,
    retries=5,  # Increase retries for structured output validation
    system_prompt=(
        "You are an expert research librarian planning a systematic search to TEST a causal claim.\n\n"
        "**CRITICAL**: To avoid confirmation bias, generate THREE types of queries:\n\n"
        "# 1. SUPPORTING QUERIES (2-3 queries)\n"
        "Find evidence that X causes Y (the proposed claim)\n\n"
        "Guidelines:\n"
        "- Use affirmative language: 'X increases Y', 'X affects Y'\n"
        "- Include study design terms: 'RCT', 'meta-analysis', 'longitudinal'\n"
        "- Be specific with variable names\n\n"
        "Examples for 'Physical Activity → TDEI':\n"
        "- 'physical activity increases total daily energy intake RCT'\n"
        "- 'exercise effect on caloric intake longitudinal study'\n"
        "- 'physical activity causal effect on energy consumption'\n\n"
        "# 2. COUNTERFACTUAL QUERIES (1-2 queries)\n"
        "Find evidence that X does NOT affect Y, or has OPPOSITE effect\n\n"
        "Guidelines:\n"
        "- Use negation: 'no effect', 'does not', 'null result', 'no association'\n"
        "- Search for opposite effects\n"
        "- Look for failed replications\n\n"
        "Examples:\n"
        "- 'physical activity no effect on energy intake'\n"
        "- 'exercise does not increase caloric intake null results'\n"
        "- 'physical activity no relationship with daily calories'\n\n"
        "# 3. ALTERNATIVE MECHANISM QUERIES (1-2 queries)\n"
        "Find evidence that effect is MEDIATED or has DIFFERENT mechanism\n\n"
        "Guidelines:\n"
        "- Use mediation terms: 'mediated by', 'indirect effect', 'through', 'via'\n"
        "- Search for intervening variables\n"
        "- Look for mechanism studies\n\n"
        "Examples:\n"
        "- 'physical activity appetite mediation energy intake'\n"
        "- 'exercise affects caloric intake through hunger signals'\n"
        "- 'physical activity indirect effect on eating behavior'\n\n"
        "# OUTPUT REQUIREMENTS\n\n"
        "Generate 5-7 total queries distributed as:\n"
        "- 2-3 supporting queries\n"
        "- 1-2 counterfactual queries\n"
        "- 1-2 alternative mechanism queries\n\n"
        "Be specific, use scientific terminology, avoid generic terms.\n\n"
        "⏱️ IMPORTANT: You have a 2-minute time limit for the entire research process."
    )
)


# Conversation History Summarizer for managing token limits
class ConversationSummary(BaseModel):
    """Compressed summary of search conversation history."""
    key_findings: List[str] = Field(description="Critical findings from searches (with URLs)")
    search_themes: List[str] = Field(description="Main topics explored")
    sources_found: List[str] = Field(description="All URLs found so far")
    next_focus: str = Field(description="What the next search should explore based on gaps")

history_summarizer = Agent(
    MINI_MODEL,
    output_type=ConversationSummary,
    system_prompt=(
        "You are an expert at compressing search conversation histories. "
        "Given a long conversation of web searches and results, extract: "
        "1) Key findings with URLs (preserve exact URLs and critical stats) "
        "2) Main themes already explored "
        "3) All source URLs found "
        "4) What areas still need investigation. "
        "Be concise but preserve all actionable intelligence."
    )
)

# Web Search Agent with tool
search_agent = Agent(
    MINI_MODEL,
    deps_type=SearchContext,
    output_type=List[SearchResult],
    retries=5,  # Increase retries for structured output validation
    system_prompt=(
        "You are a research assistant specializing in scientific literature. "
        "Given a search query about a causal relationship, use web search to "
        "retrieve relevant scientific sources. Focus on peer-reviewed studies, "
        "meta-analyses, and authoritative research reviews. Extract key findings, "
        "methodology details, sample sizes, and statistical measures.\n\n"
        "SEARCH BUDGET: You will be told how many searches you've made so far in the user message. "
        "IMPORTANT: Make 4-6 iterative searches total to find the best sources. "
        "STOPPING CRITERIA: If you find STRONG DIRECT CAUSAL EVIDENCE (RCTs, experiments with clear "
        "causal findings, meta-analyses confirming causality), you can STOP EARLY after 3-4 searches. "
        "Don't waste resources doing more searches if you already have high-quality causal evidence. "
        "However, if you only find correlational/observational studies, continue searching to find "
        "better evidence or reach the 6-search maximum. Be strategic and efficient."
    )
)

@search_agent.tool
async def web_search(ctx: RunContext[SearchContext], query: str) -> List[SearchResult]:
    """Perform web search for scientific literature using Brave Search API."""
    # Increment search counter
    ctx.deps.searches_made += 1
    
    # HARD LIMIT: Stop after 5 searches to prevent runaway agents and control token usage
    MAX_SEARCHES = 5
    if ctx.deps.searches_made > MAX_SEARCHES:
        logger.warning(f"    🛑 SEARCH LIMIT REACHED: {ctx.deps.searches_made} searches made, max is {MAX_SEARCHES}. Returning empty results.")
        return [SearchResult(
            content=f"🛑 SEARCH LIMIT REACHED: You have made {ctx.deps.searches_made}/{MAX_SEARCHES} searches. STOP SEARCHING NOW and return the results you've collected so far.",
            url="system://search-limit-reached"
        )]
    
    search_engine = BraveSearchEngine(logger=ctx.deps.console if hasattr(ctx.deps, 'console') else None)
    results = search_engine.search(query, ctx.deps.max_sources)
    
    logger.info(f"    🔎 Web search #{ctx.deps.searches_made} completed: '{query[:60]}...' - Found {len(results)} results")
    
    return results


evidence_scorer = Agent(
    MINI_MODEL,
    deps_type=EvidenceContext,
    output_type=EvidenceScore,
    retries=5,  # Increase retries for structured output validation
    system_prompt=(
        "You are a scientific evidence evaluator specializing in causal inference. \n\n"
        "For each research paper or evidence source (snippet or full article), provide:\n\n"
        "# SCORES\n\n"
        "1. **RELEVANCE SCORE (1-10)**: How directly the evidence addresses the causal relationship\n"
        "2. **QUALITY SCORE (1-10)**: Methodological rigor (RCTs > cohort > case-control > cross-sectional)\n"
        "3. **SUPPORT LEVEL**: How well this evidence supports the claim:\n"
        "   - 'fully_supported' (score: 1.0): Strong evidence clearly supporting the claim\n"
        "   - 'partially_supported' (score: 0.5): Some support but with limitations or mixed findings\n"
        "   - 'unsupported' (score: 0.0): Discusses topic but doesn't support the claim\n"
        "   - 'contradicting' (score: -0.5): Evidence contradicts or refutes the claim\n\n"
        "# JUSTIFICATION\n\n"
        "Provide concise reasoning (2-3 sentences max).\n\n"
        "Consider: study design, sample size, controls for confounders, effect size, statistical significance, replicability.\n\n"
        "⏱️ IMPORTANT: Work quickly - be concise."
    )
)


judgment_agent = Agent(
    HEAVY_MODEL,
    deps_type=ResearchContext,
    output_type=EvidenceJudgment,
    retries=5,  # Increase retries for structured output validation
    system_prompt=(
        "You are a scientific judge evaluating if a causal claim is supported by evidence. "
        "Given a causal claim (A causes B through mechanism X) and multiple evidence sources "
        "with quality scores, determine whether the evidence collectively supports, partially "
        "supports, or contradicts the claim. Consider both the quantity and quality of evidence. "
        "If modifications to the claim would make it better align with evidence, suggest specific "
        "revisions. Be fair but critical, and acknowledge limitations in the evidence base."
    )
)


causal_evidence_detector = Agent(
    HEAVY_MODEL,
    deps_type=ResearchContext,
    output_type=CausalEvidenceEvaluation,
    retries=5,  # Increase retries for structured output validation
    system_prompt=(
        "You are evaluating a SINGLE research article using GRADE criteria.\n\n"
        "# BINARY JUDGMENT\n\n"
        "First, determine if article discusses the causal relationship:\n\n"
        "DIRECT_EVIDENCE_FOUND:\n"
        "- TRUE if article explicitly discusses causal relationship X→Y\n"
        "- FALSE if only correlational, no causal claim, or unrelated\n\n"
        "# INTERMEDIATE SUPPORT JUDGMENT\n\n"
        "If direct evidence is found, assess how well the evidence supports the claim:\n\n"
        "SUPPORT_LEVEL:\n"
        "- 'fully_supported' (score: 1.0): Strong evidence, clear causal mechanism, high quality study\n"
        "- 'partially_supported' (score: 0.5): Some evidence, but with limitations, gaps, or mixed findings\n"
        "- 'unsupported' (score: 0.0): Discusses the relationship but evidence is weak, inconclusive, or non-causal\n"
        "- 'contradicting' (score: -0.5): Evidence suggests opposite effect or refutes the claim\n\n"
        "# 7 GRADE DIMENSIONS (if direct evidence found)\n\n"
        "## 1. RISK OF BIAS (Study Quality) - Score 0-10\n\n"
        "**High quality (8-10):**\n"
        "- Randomized controlled trial (RCT)\n"
        "- Proper randomization + allocation concealment\n"
        "- Blinding of participants/assessors\n"
        "- Low attrition (<20%), intention-to-treat\n"
        "- Pre-registered\n\n"
        "**Moderate quality (5-7):**\n"
        "- RCT with limitations (no blinding, unclear allocation)\n"
        "- High-quality quasi-experimental design\n"
        "- Longitudinal/cohort with good controls\n\n"
        "**Low quality (3-4):**\n"
        "- Cross-sectional study\n"
        "- Moderate confounding risk\n\n"
        "**Very low (0-2):**\n"
        "- Case series, expert opinion\n"
        "- High bias risk, anecdotal\n\n"
        "## 2. INDIRECTNESS - MECHANISM - Score 0-10\n\n"
        "**CRITICAL FOR CLD: Is mechanism direct or mediated?**\n\n"
        "**Direct (8-10):**\n"
        "- Study tests X→Y mechanism directly\n"
        "- Controls for potential mediators\n"
        "- Mediation analysis shows PM < 0.3 (mostly direct)\n"
        "- States 'X directly affects Y'\n\n"
        "**Mixed (5-7):**\n"
        "- Tests X→Y but doesn't control mediators\n"
        "- No mediation analysis\n"
        "- Unclear if direct or mediated\n\n"
        "**Mediated (0-4):**\n"
        "- 'X affects Y through Z' (mediation found)\n"
        "- PM > 0.6 (primarily indirect)\n"
        "- Only tests part of path (X→Z or Z→Y)\n\n"
        "**If score < 6.0: Effect is likely mediated → NOT suitable for direct CLD edge**\n\n"
        "## 3. INDIRECTNESS - POPULATION - Score 0-10\n\n"
        "**Exact match (8-10):**\n"
        "- Same demographics, age, culture, setting\n\n"
        "**Close match (5-7):**\n"
        "- Similar population, good generalizability\n\n"
        "**Poor match (0-4):**\n"
        "- Different life stage, culture\n"
        "- Animal models, wrong demographics\n\n"
        "## 4. INDIRECTNESS - CONSTRUCTS - Score 0-10\n\n"
        "**Exact match (8-10):**\n"
        "- Same variable definitions and instruments\n\n"
        "**Close proxy (5-7):**\n"
        "- Validated alternative measures\n\n"
        "**Poor match (0-4):**\n"
        "- Different constructs, invalid measures\n\n"
        "## 5. IMPRECISION - Score 0-10\n\n"
        "**High precision (8-10):**\n"
        "- Large sample (n>500)\n"
        "- Narrow confidence intervals\n"
        "- p < 0.001\n\n"
        "**Moderate precision (5-7):**\n"
        "- Adequate sample (n=200-500)\n"
        "- Reasonable CI, p < 0.01\n\n"
        "**Low precision (0-4):**\n"
        "- Small sample (n<50)\n"
        "- Wide CI, p > 0.05\n"
        "- Underpowered\n\n"
        "## 6. MAGNITUDE OF EFFECT - Score 0-10\n\n"
        "**Very large (9-10):**\n"
        "- RR > 5 or < 0.2\n"
        "- Cohen's d > 0.8\n\n"
        "**Large (7-8):**\n"
        "- RR 2-5\n"
        "- Cohen's d = 0.5-0.8\n\n"
        "**Moderate (5-6):**\n"
        "- RR 1.5-2\n"
        "- Cohen's d = 0.3-0.5\n\n"
        "**Small (3-4):**\n"
        "- RR 1.1-1.5\n"
        "- Cohen's d = 0.2-0.3\n\n"
        "**Very small/none (0-2):**\n"
        "- Minimal or no effect\n\n"
        "## 7. DOSE-RESPONSE GRADIENT - Score 0-10\n\n"
        "**Strong (8-10):**\n"
        "- Dose-response tested and significant\n\n"
        "**Moderate (5-7):**\n"
        "- Trend present\n\n"
        "**Weak/none (0-4):**\n"
        "- Not tested or no gradient\n\n"
        "# OUTPUT FORMAT\n\n"
        "DIRECT_EVIDENCE_FOUND: [true/false]\n"
        "CONFIDENCE: [0-10]\n\n"
        "SUPPORT_LEVEL: ['fully_supported', 'partially_supported', 'unsupported', or 'contradicting']\n"
        "SUPPORT_SCORE: [1.0 for fully, 0.5 for partial, 0.0 for unsupported, -0.5 for contradicting]\n\n"
        "RISK_OF_BIAS: [0-10]\n"
        "INDIRECTNESS_MECHANISM: [0-10]\n"
        "INDIRECTNESS_POPULATION: [0-10]\n"
        "INDIRECTNESS_CONSTRUCTS: [0-10]\n"
        "IMPRECISION: [0-10]\n"
        "MAGNITUDE_OF_EFFECT: [0-10]\n"
        "DOSE_RESPONSE_GRADIENT: [0-10]\n\n"
        "EFFECT_SIZE_REPORTED: [if mentioned, e.g. 'd=0.65' or 'RR=2.3']\n"
        "CITED_PASSAGE: [extract key causal statement]\n"
        "STUDY_TYPE: [RCT/quasi-experiment/cohort/cross-sectional/meta-analysis]\n"
        "CAUSAL_EVIDENCE_SUMMARY: [2-3 sentences]\n\n"
        "# REFERENCE\n"
        "GRADE: https://gdt.gradepro.org/app/handbook/handbook.html"
    )
)


verdict_agent = Agent(
    HEAVY_MODEL,
    deps_type=ResearchContext,
    output_type=FinalVerdict,
    retries=5,  # Increase retries for structured output validation
    system_prompt=(
        "You are synthesizing final verdict from multiple evidence sources using GRADE criteria with multi-dimensional quality assessment.\\n\\n"
        "# INPUT\\n\\n"
        "You receive:\\n"
        "1. ALL search snippets (10-20) with relevance scores\\n"
        "2. FULL ARTICLE EVALUATIONS (2-3) with 7 GRADE dimensions each\\n\\n"
        "# YOUR TASK\\n\\n"
        "## 1. ASSESS CONSENSUS ACROSS SNIPPETS\\n\\n"
        "First, analyze the search snippets by support levels and query types:\\n\\n"
        "A. SNIPPET_CONSENSUS_SUPPORT: Count snippets by support_level\\n"
        "   - Count: fully_supported, partially_supported, unsupported, contradicting\\n"
        "   - Example: '12/20 snippets show support (8 fully, 4 partially), 3 unsupported, 5 contradicting'\\n\\n"
        "B. SNIPPET_CONSENSUS_QUALITY: Summarize relevance and quality scores\\n"
        "   - Example: 'Most snippets (15/20) have relevance ≥7 and quality ≥6'\\n\\n"
        "C. COUNTER_EVIDENCE: Check for counterfactual query results\\n"
        "   - COUNTER_EVIDENCE_FOUND: TRUE if snippets with query_type='counterfactual' exist\\n"
        "   - COUNTER_EVIDENCE_SUMMARY: Summarize null results or opposite effects found\\n\\n"
        "D. ALTERNATIVE_MECHANISMS: Check for alternative pathway results\\n"
        "   - ALTERNATIVE_MECHANISMS_FOUND: TRUE if snippets with query_type='alternative_mechanism' exist\\n"
        "   - ALTERNATIVE_MECHANISMS_SUMMARY: Summarize mediation or confounding pathways found\\n\\n"
        "## 2. ASSESS CONSENSUS ACROSS ARTICLES\\n\\n"
        "For each key GRADE dimension, assess the consensus pattern:\\n\\n"
        "Example:\\n"
        "Article 1: indirectness_mechanism = 8.0 (direct)\\n"
        "Article 2: indirectness_mechanism = 3.0 (mediated by appetite)\\n"
        "Article 3: indirectness_mechanism = 4.0 (mediated)\\n\\n"
        "Consensus: 'Majority evidence (2/3 articles) indicates MEDIATED mechanism. "
        "One study claims direct effect but does not control for mediators.'\\n\\n"
        "Do this for key dimensions:\\n"
        "- Mechanism directness (CRITICAL for CLD)\\n"
        "- Effect magnitude\\n"
        "- Study quality\\n\\n"
        "## 3. MULTI-DIMENSIONAL EVIDENCE QUALITY SCORING\\n\\n"
        "Score the evidence on 5 dimensions (0-10 each). These scores will inform your binary judgment.\\n\\n"
        "### 3.1 EVIDENCE QUALITY (Study Design) - Weight: 30%\\n"
        "- 9-10: RCT or meta-analysis of RCTs with consistent findings\\n"
        "- 7-8: High-quality quasi-experimental, natural experiment, or longitudinal study\\n"
        "- 5-6: Cross-sectional with mediation analysis or strong observational evidence\\n"
        "- 3-4: Cross-sectional correlational without temporal information\\n"
        "- 1-2: Theoretical reasoning only, no empirical support\\n"
        "- 0: No evidence or contradictory evidence\\n\\n"
        "### 3.2 MECHANISTIC SPECIFICITY (Direct vs Mediated) - Weight: 25%\\n"
        "- 9-10: Evidence shows DIRECT causal mechanism matching the exact pathway\\n"
        "- 7-8: Plausible direct mechanism with minor gaps\\n"
        "- 5-6: General association but mechanism unclear or complex\\n"
        "- 3-4: Relationship is MEDIATED through other variables (X→Z→Y, not X→Y)\\n"
        "- 1-2: No mechanism specified or mechanism contradicts claim\\n"
        "- 0: Tautological, circular, or conceptually incoherent\\n"
        "**CRITICAL**: If evidence shows 'X affects Y ONLY THROUGH Z' or 'fully mediated', score 2-4, NOT 7-10.\\n\\n"
        "### 3.3 CONSTRUCT VALIDITY (Right Variables) - Weight: 20%\\n"
        "- 9-10: Exact match - same measurement instruments/operationalizations\\n"
        "- 7-8: Close proxy - validated alternative measure of same construct\\n"
        "- 5-6: Related concept (e.g., BMI for obesity, silhouettes for ideal BMI)\\n"
        "- 3-4: Distant relative - same domain but different aspect\\n"
        "- 1-2: Tangentially related - weak construct overlap\\n"
        "- 0: Wrong construct entirely\\n"
        "**CRITICAL**: If studies use figure ratings/silhouettes instead of numeric BMI, score 5-6 max.\\n\\n"
        "### 3.4 SCOPE RELEVANCE (Population Match) - Weight: 15%\\n"
        "- 9-10: Exact match - same population, age group, cultural context\\n"
        "- 7-8: Close match - similar population with good generalizability\\n"
        "- 5-6: Generalizable - different population but findings likely transfer\\n"
        "- 3-4: Questionable transfer - different life stage, culture, or setting\\n"
        "- 1-2: Poor match - animal models, wrong age group, incompatible context\\n"
        "- 0: Completely wrong scope\\n\\n"
        "### 3.5 THEORETICAL COHERENCE (Fits Theory) - Weight: 10%\\n"
        "- 9-10: Theory-driven hypothesis with strong theoretical support\\n"
        "- 7-8: Consistent with established theoretical frameworks\\n"
        "- 5-6: Compatible with theory but uses different frameworks\\n"
        "- 3-4: Atheoretical empirical finding without theoretical grounding\\n"
        "- 1-2: Weak or unclear theoretical connection\\n"
        "- 0: Contradicts theory, or claim is tautological/circular\\n"
        "**CRITICAL**: If reviewer states claim is 'tautological', 'circular', 'ill-posed', or 'conceptually incorrect', score 0.\\n\\n"
        "## 4. BINARY JUDGMENT FOR CLD (USE SCORES FROM SECTION 3)\\n\\n"
        "**CRITICAL**: Make this judgment based on HOLISTIC assessment of ALL evidence AND the multi-dimensional scores.\\n\\n"
        "IS_DIRECT_CAUSAL:\\n"
        "- TRUE if evidence strongly suggests DIRECT mechanism\\n"
        "- FALSE if evidence shows MEDIATED mechanism or effect too weak\\n\\n"
        "Consider:\\n"
        "- Quality of studies (RCTs > observational)\\n"
        "- Consistency vs inconsistency across studies\\n"
        "- Strength of mechanistic evidence\\n"
        "- Whether high-quality studies control for mediators\\n"
        "- **KEY GUIDANCE**: If mechanistic_specificity ≤ 4, likely NOT a direct causal edge\\n"
        "- **KEY GUIDANCE**: If evidence_quality < 5, insufficient methodological rigor\\n"
        "- Consider ALL dimensions together, not just one threshold\\n\\n"
        "REASONING: 2-3 sentences explaining your HOLISTIC judgment\\n\\n"
        "CONFIDENCE: high/medium/low\\n\\n"
        "## 5. SYNTHESIZE FINAL VERDICT\\n\\n"
        "Provide overall verdict considering all evidence:\\n"
        "- supported/partially_supported/unsupported\\n"
        "- Key evidence summaries\\n"
        "- Limitations\\n\\n"
        "## 6. IDENTIFY STRONGEST EVIDENCE\\n\\n"
        "**CRITICAL REQUIREMENTS:**\\n"
        "1. Identify the STRONGEST piece of evidence that supports a direct causal mechanism\\n"
        "2. Extract a VERBATIM quote from that evidence demonstrating the causal relationship\\n"
        "3. Include the URL to that strongest piece of evidence\\n"
        "4. The quote should be the most compelling statement about the direct causal link\\n\\n"
        "# OUTPUT FORMAT\\n\\n"
        "You must populate ALL fields in the FinalVerdict model, including:\\n"
        "- Consensus fields (snippet_consensus_support, etc.)\\n"
        "- Multi-dimensional scores (evidence_quality, mechanistic_specificity, construct_validity, scope_relevance, theoretical_coherence)\\n"
        "- Binary judgment (is_direct_causal, direct_causal_reasoning, direct_causal_confidence)\\n"
        "- Verdict fields (verdict, confidence, key_evidence_summaries, limitations)\\n"
        "- Strongest evidence (strongest_causal_quote, strongest_evidence_url)\\n\\n"
        "SNIPPET_CONSENSUS_SUPPORT: [e.g., '12/20 support (8 fully, 4 partially), 3 unsupported, 5 contradicting']\\n"
        "SNIPPET_CONSENSUS_QUALITY: [e.g., 'Most snippets (15/20) relevance ≥7, quality ≥6']\\n"
        "COUNTER_EVIDENCE_FOUND: [true/false]\\n"
        "COUNTER_EVIDENCE_SUMMARY: [if true, summarize null/opposite findings]\\n"
        "ALTERNATIVE_MECHANISMS_FOUND: [true/false]\\n"
        "ALTERNATIVE_MECHANISMS_SUMMARY: [if true, summarize mediation/confounding]\\n\\n"
        "CONSENSUS_MECHANISM: [describe pattern across articles, e.g., '2/3 articles show mediation']\\n"
        "CONSENSUS_EFFECT: [describe pattern, e.g., 'all show medium-large effect (d=0.5-0.8)']\\n"
        "CONSENSUS_QUALITY: [describe pattern, e.g., '2 RCTs, 1 observational']\\n\\n"
        "EVIDENCE_QUALITY: [0-10]\\n"
        "MECHANISTIC_SPECIFICITY: [0-10]\\n"
        "CONSTRUCT_VALIDITY: [0-10]\\n"
        "SCOPE_RELEVANCE: [0-10]\\n"
        "THEORETICAL_COHERENCE: [0-10]\\n"
        "MULTIDIMENSIONAL_REASONING: [2-3 sentences explaining scores]\\n\\n"
        "IS_DIRECT_CAUSAL: [true/false]\\n"
        "DIRECT_CAUSAL_REASONING: [2-3 sentences explaining HOLISTIC assessment]\\n"
        "DIRECT_CAUSAL_CONFIDENCE: [high/medium/low]\\n\\n"
        "STRONGEST_CAUSAL_QUOTE: [verbatim quote from strongest evidence]\\n"
        "STRONGEST_EVIDENCE_URL: [URL to strongest evidence]\\n\\n"
        "VERDICT: [supported/partially_supported/unsupported]\\n"
        "CONFIDENCE: [0-10]\\n"
        "KEY_EVIDENCE_SUMMARIES: [list of 3-5 key findings]\\n"
        "LIMITATIONS: [list of 2-3 limitations]\\n\\n"
        "# EXAMPLES\\n\\n"
        "**Example 1: Consistent Direct Effect**\\n"
        "Article 1: mechanism=8.0, effect=7.0, quality=9.0 (RCT with mediation control)\\n"
        "Article 2: mechanism=7.0, effect=8.0, quality=8.0 (RCT)\\n"
        "Article 3: mechanism=7.5, effect=7.0, quality=7.0 (quasi-experiment)\\n\\n"
        "→ CONSENSUS_MECHANISM: 'All 3 articles consistently show DIRECT mechanism (7.0-8.0/10)'\\n"
        "→ EVIDENCE_QUALITY: 8.5\\n"
        "→ MECHANISTIC_SPECIFICITY: 7.5\\n"
        "→ IS_DIRECT_CAUSAL: true\\n"
        "→ REASONING: 'Unanimous evidence for direct causal pathway. All studies control for potential mediators and find independent direct effect. High-quality RCTs provide strong support.'\\n\\n"
        "**Example 2: Mediated Effect**\\n"
        "Article 1: mechanism=3.0, effect=8.0, quality=9.0 (finds mediation by appetite, PM=0.7)\\n"
        "Article 2: mechanism=4.0, effect=7.0, quality=8.0 (indirect through hunger)\\n"
        "Article 3: mechanism=7.0, effect=6.0, quality=6.0 (claims direct but no mediation test)\\n\\n"
        "→ CONSENSUS_MECHANISM: 'Majority (2/3) show MEDIATED mechanism. High-quality studies explicitly test and find mediation.'\\n"
        "→ EVIDENCE_QUALITY: 8.0\\n"
        "→ MECHANISTIC_SPECIFICITY: 3.5\\n"
        "→ IS_DIRECT_CAUSAL: false\\n"
        "→ REASONING: 'Best-quality studies show effect is mediated by appetite/hunger (mechanistic_specificity=3.5/10). NOT suitable for direct CLD edge due to mediation.'\\n\\n"
        "IMPORTANT: Return only valid JSON matching the FinalVerdict schema exactly."
    )
)


article_selector = Agent(
    HEAVY_MODEL,
    deps_type=ResearchContext,
    output_type=ArticleSelection,
    system_prompt=(
        "You are an expert at identifying which scientific articles are most likely to contain "
        "DIRECT CAUSAL EVIDENCE for a given causal claim. Given search snippets with quality scores, "
        "select 2-3 articles that should be fetched in full text based on:\n\n"
        "1. **Causal Language**: Look for snippets mentioning 'caused', 'causal effect', 'intervention', 'RCT', 'randomized', 'experimental manipulation'\n"
        "2. **Study Design**: Prioritize RCTs, experiments, meta-analyses, systematic reviews over observational studies\n"
        "3. **Quality Scores**: Higher relevance (≥7) and quality (≥7) scores indicate better sources\n"
        "4. **Mechanistic Detail**: Articles explaining HOW the causal relationship works are valuable\n\n"
        "IMPORTANT: Only select articles that appear to contain direct causal evidence based on their snippets. "
        "If no snippets suggest causal evidence, return an empty list."
    )
)


class DeepResearchManager:
    """
    Manages the end-to-end workflow of validating a causal claim through
    scientific literature search and evaluation using PydanticAI agents.
    """

    def __init__(self, max_iterations: int = 3, minimum_evidence_count: int = 5, max_sources_per_query: int = 5):
        """Initialize the DeepResearchManager."""
        self.console = Console()
        self.max_iterations = max_iterations
        self.minimum_evidence_count = minimum_evidence_count
        self.max_sources_per_query = max_sources_per_query
        self.iteration = 0
        self.search_results: List[SearchResult] = []
        self.evidence_scores: List[EvidenceScore] = []
        self.judgments: List[EvidenceJudgment] = []
        self.iteration_history: List[IterationHistory] = []  # Track iteration history
        self.current_claim = None
        self.total_tokens = 0
        self.agent_call_count = 0
    
    def _log_agent_call(self, agent_name: str, from_info: str, to_info: str, result):
        """Log detailed information about an agent call including token usage (Pydantic AI native)."""
        self.agent_call_count += 1
        
        # Extract token usage using Pydantic AI's native usage property
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        model_name = "unknown"
        
        try:
            if hasattr(result, 'usage'):
                usage = result.usage()
                # Pydantic AI native properties
                input_tokens = getattr(usage, 'request_tokens', getattr(usage, 'input_tokens', 0))
                output_tokens = getattr(usage, 'response_tokens', getattr(usage, 'output_tokens', 0))
                total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens)
                self.total_tokens += total_tokens
                
            # Try to extract model name
            if hasattr(result, 'model'):
                model_name = result.model
            elif hasattr(result, '_result') and hasattr(result._result, 'model'):
                model_name = result._result.model
        except Exception as e:
            logger.debug(f"Could not extract usage/model info: {e}")
        
        # Log detailed information with more visibility
        logger.info("="*100)
        logger.info(
            f"🤖 API CALL COMPLETE | AGENT: {agent_name} | MODEL: {model_name} | CALL #: {self.agent_call_count}"
        )
        logger.info(
            f"   📊 TOKENS: Input={input_tokens} | Output={output_tokens} | Call_Total={total_tokens} | Session_Total={self.total_tokens}"
        )
        logger.info(
            f"   📥 INPUT: {from_info[:100]}"
        )
        logger.info(
            f"   📤 OUTPUT: {to_info[:100]}"
        )
        logger.info("="*100)
        
        self.console.print(
            f"\n[bold cyan]💰 API CALL #{self.agent_call_count}:[/bold cyan] {agent_name} | "
            f"[yellow]Model: {model_name}[/yellow] | "
            f"[green]In: {input_tokens} | Out: {output_tokens}[/green] | "
            f"[magenta]Call: {total_tokens}[/magenta]"
        )
        self.console.print(
            f"[bold magenta]📊 CUMULATIVE SESSION TOKENS: {self.total_tokens:,}[/bold magenta]\n"
        )

    async def run(self, claim: str) -> FinalVerdict:
        """Run the deep research process on a causal claim."""
        self.start_time = time.time()
        time_limit = 120.0  # 2 minutes
        
        self.console.print(f"Starting deep research on causal claim: {claim}")
        self.console.print(f"⏱️  Time limit: {time_limit} seconds ({time_limit/60:.1f} minutes)")
        
        # Initialize state
        original_claim = claim
        current_claim = claim
        self.current_claim = claim
        self.search_results = []
        self.evidence_scores = []
        self.judgments = []
        self.grade_evaluations = []  # NEW: Collect GRADE evaluations for full articles
        
        causal_evidence_result = None

        # Main research iteration loop
        for self.iteration in range(1, self.max_iterations + 1):
            elapsed = time.time() - self.start_time
            remaining = time_limit - elapsed
            
            # Check time limit
            if elapsed >= time_limit:
                self.console.print(f"[yellow]⏱️  TIME LIMIT REACHED ({elapsed:.1f}s / {time_limit}s). Terminating search.[/yellow]")
                logger.warning(f"Time limit reached: {elapsed:.1f}s / {time_limit}s")
                break
            
            self.console.print(f"Research Iteration {self.iteration}/{self.max_iterations} (⏱️  {elapsed:.1f}s elapsed, {remaining:.1f}s remaining)")
            logger.info(f"Starting iteration {self.iteration}, elapsed: {elapsed:.1f}s, remaining: {remaining:.1f}s")
            
            # Create research context with time information
            research_ctx = ResearchContext(
                claim=current_claim,
                iteration=self.iteration,
                max_iterations=self.max_iterations,
                console=self.console,
                elapsed_seconds=elapsed,
                time_limit_seconds=time_limit
            )
            
            # 1. Generate search plan
            search_plan = await self._generate_search_plan(research_ctx)
            
            # 2. Execute searches
            new_results = await self._run_searches(search_plan, current_claim)
            if new_results:
                self.search_results.extend(new_results)
                
                # 3. Score evidence
                new_scores = await self._score_evidence(new_results, current_claim)
                self.evidence_scores.extend(new_scores)
                
                # 3.5. Smart AI-based article selection and full-text fetching
                await self._fetch_promising_articles(new_results, new_scores, current_claim)
                
                # 3.6. NEW: Evaluate full articles with GRADE dimensions
                grade_evals = await self._evaluate_full_articles_with_grade(research_ctx, self.search_results, current_claim)
                self.grade_evaluations.extend(grade_evals)
                
                # 4. Check for direct causal evidence (using quality scores to inform early stopping)
                if len(self.search_results) >= 1:
                    # Count high-quality sources that might contain causal evidence
                    high_quality_count = len([s for s in self.evidence_scores if s.relevance_score >= 8 and s.quality_score >= 7])
                    
                    # Check for direct causal evidence in ALL search results
                    causal_evidence_result = await self._check_for_direct_causal_evidence(
                        research_ctx, self.search_results
                    )
                    
                    # EARLY STOP: Only if we have BOTH high-quality sources AND direct causal evidence
                    if causal_evidence_result.direct_evidence_found and high_quality_count >= 3:
                        self.console.print(f"[bold green]✓ EARLY STOP: Direct causal evidence found (Confidence: {causal_evidence_result.confidence}/10) with {high_quality_count} high-quality sources backing it.[/bold green]")
                        logger.info(f"EARLY STOP: Direct causal evidence (conf={causal_evidence_result.confidence}/10) + {high_quality_count} high-quality sources (rel≥8, qual≥7)")
                        break
                    elif causal_evidence_result.direct_evidence_found:
                        # Found causal evidence but not enough high-quality sources - continue searching
                        self.console.print(f"[yellow]Direct causal evidence found (Confidence: {causal_evidence_result.confidence}/10), but only {high_quality_count}/3 high-quality sources. Continuing search...[/yellow]")
                        logger.info(f"Direct causal evidence found but insufficient quality: {high_quality_count}/3 high-quality sources")
                    elif high_quality_count >= 3:
                        # Have high-quality sources but no direct causal evidence - continue searching
                        self.console.print(f"[cyan]Found {high_quality_count} high-quality sources, but no direct causal evidence yet. Continuing search...[/cyan]")
                        logger.info(f"Quality threshold met ({high_quality_count} sources) but no direct causal evidence")
                
                # 5. Judge evidence
                judgment = await self._judge_evidence(research_ctx, self.search_results, self.evidence_scores)
                self.judgments.append(judgment)
                
                # 6. Update claim if needed
                if judgment.suggested_modification:
                    current_claim = judgment.suggested_modification
                    self.current_claim = current_claim
                
                # 6.5. Record iteration history
                iteration_record = IterationHistory(
                    iteration_number=self.iteration,
                    claim_tested=current_claim,
                    evidence_count=len(new_results),
                    high_quality_count=len([s for s in new_scores if s.relevance_score >= 8 and s.quality_score >= 7]),
                    judgment="supported" if judgment.supported else "not_supported",
                    judgment_confidence=judgment.confidence,
                    suggested_modification=judgment.suggested_modification
                )
                self.iteration_history.append(iteration_record)
                logger.info(f"  📝 Recorded iteration #{self.iteration} history: {judgment.supported}, conf={judgment.confidence}/10, evidence={len(new_results)}, high_qual={iteration_record.high_quality_count}")
                
                # Terminate conditions
                if judgment.supported and judgment.confidence >= 8:
                    self.console.print("[green]High confidence support found. Terminating search.[/green]")
                    break
                    
                if not judgment.supported and self.iteration >= self.max_iterations - 1:
                    self.console.print("[yellow]Maximum iterations reached.[/yellow]")
                    break
            else:
                self.console.print("[yellow]No new results found for this iteration.[/yellow]")

        # 7. Synthesize final verdict
        final_research_ctx = ResearchContext(
            claim=original_claim,
            iteration=self.iteration,
            max_iterations=self.max_iterations,
            console=self.console
        )
        
        verdict = await self._synthesize_verdict(
            final_research_ctx,
            original_claim,
            current_claim,
            self.search_results,
            self.evidence_scores,
            self.judgments[-1] if self.judgments else None,
            causal_evidence_result,
            self.grade_evaluations  # NEW: Pass GRADE evaluations for consensus assessment
        )
        
        self.console.print(f"\n[bold green]Research Complete: {verdict.verdict.upper()}[/bold green]")
        return verdict

    async def _generate_search_plan(self, ctx: ResearchContext) -> SearchPlan:
        """Generate a search plan using the hypothesis evaluator agent."""
        self.console.print("[bold]Generating search plan...[/bold]")
        
        logger.info(f"  🔄 HANDOFF → hypothesis_evaluator: Input='Causal claim: {ctx.claim[:100]}...'")
        
        result = await hypothesis_evaluator.run(
            f"Causal claim: {ctx.claim}",
            deps=ctx
        )
        
        search_plan = result.output
        
        # Add verbatim motivation as a supporting query (user requirement)
        verbatim_query = SearchQuery(
            query=ctx.claim,
            rationale="Verbatim motivation search - directly search for the exact claim to find related literature"
        )
        search_plan.supporting_queries.insert(0, verbatim_query)
        logger.info(f"  ➕ Added verbatim motivation as Supporting Query #1: '{ctx.claim[:100]}...'")
        
        logger.info(f"  ✅ HANDOFF ← hypothesis_evaluator: Generated {len(search_plan.all_queries)} total queries")
        logger.info(f"     - Supporting: {len(search_plan.supporting_queries)}")
        logger.info(f"     - Counterfactual: {len(search_plan.counterfactual_queries)}")
        logger.info(f"     - Alternative mechanisms: {len(search_plan.alternative_mechanism_queries)}")
        
        self._log_agent_call(
            "hypothesis_evaluator", 
            f"Claim: {ctx.claim}", 
            f"{len(search_plan.all_queries)} queries generated (S:{len(search_plan.supporting_queries)}, C:{len(search_plan.counterfactual_queries)}, A:{len(search_plan.alternative_mechanism_queries)})",
            result
        )
        
        logger.info(f"✅ Total queries: {len(search_plan.all_queries)}")
        for i, q in enumerate(search_plan.all_queries, 1):
            logger.info(f"    Query #{i}: {q.query[:60]}...")
        self.console.print(f"[bold]Generated {len(search_plan.all_queries)} search queries:[/bold]")
        self.console.print(f"  Supporting: {len(search_plan.supporting_queries)}")
        self.console.print(f"  Counterfactual: {len(search_plan.counterfactual_queries)}")
        self.console.print(f"  Alternative mechanisms: {len(search_plan.alternative_mechanism_queries)}")
            
        return search_plan

    async def _run_searches(self, search_plan: SearchPlan, claim: str) -> List[SearchResult]:
        """Perform literature searches based on the search plan (parallelized with Pydantic AI native async)."""
        self.console.print("[bold]Searching scientific literature...[/bold]")
        logger.info(f"🚀 Running {len(search_plan.all_queries)} searches IN PARALLEL using asyncio.gather()")
        
        # Log all queries upfront
        for i, query in enumerate(search_plan.all_queries, 1):
            logger.info(f"  🔍 SEARCH QUERY #{i}/{len(search_plan.all_queries)}: {query.query}")
            self.console.print(f"[dim]Query {i}: {query.query}[/dim]")
        
        # Determine query type for each query
        query_types = []
        for query in search_plan.supporting_queries:
            query_types.append("supporting")
        for query in search_plan.counterfactual_queries:
            query_types.append("counterfactual")
        for query in search_plan.alternative_mechanism_queries:
            query_types.append("alternative_mechanism")
        
        # Create search tasks for parallel execution (Pydantic AI native approach)
        async def run_single_search_with_history_management(query_obj: SearchQuery, query_num: int, query_type: str):
            """Run a single search with per-thread conversation history management."""
            search_ctx = SearchContext(
                claim=claim,
                query=query_obj.query,
                max_sources=self.max_sources_per_query
            )
            
            logger.info(f"  🔄 HANDOFF → search_agent (query #{query_num}, type={query_type}): Query='{query_obj.query[:50]}...'")
            
            # IMPORTANT: Each parallel search gets its own empty history to prevent contamination
            # But within this thread, history will accumulate across iterative tool calls
            # Pass search count in the user message for the agent to see
            result = await search_agent.run(
                f"[SEARCHES MADE SO FAR: {search_ctx.searches_made}] Search for: {query_obj.query}",
                deps=search_ctx,
                message_history=[]  # Fresh history for this thread
            )
            
            # Check token usage and compress if needed
            input_tokens = 0
            if hasattr(result, 'usage'):
                try:
                    usage = result.usage()
                    input_tokens = getattr(usage, 'request_tokens', getattr(usage, 'input_tokens', 0))
                except Exception as e:
                    logger.debug(f"Could not extract token usage: {e}")
            
            # Log if high token usage detected
            if input_tokens > 50000:
                logger.warning(
                    f"  ⚠️  High token usage in search_agent_q{query_num}: {input_tokens:,} input tokens. "
                    f"This search made many iterative web_search calls."
                )
                self.console.print(f"[yellow]⚠️  Query {query_num} used {input_tokens:,} tokens (high)[/yellow]")
            
            # Log the API call for this search
            self._log_agent_call(
                agent_name=f"search_agent_q{query_num}",
                from_info=f"Search query ({query_type}): {query_obj.query[:80]}",
                to_info=f"Found {len(result.output) if isinstance(result.output, list) else 0} results",
                result=result
            )
            
            search_results = result.output
            if isinstance(search_results, list):
                # Label all results with query type
                for sr in search_results:
                    sr.query_type = query_type
                logger.info(f"  ✅ HANDOFF ← search_agent (query #{query_num}): Found {len(search_results)} {query_type} results")
                return search_results
            
            logger.info(f"  ✅ HANDOFF ← search_agent (query #{query_num}): No results (empty or invalid response)")
            return []
        
        # Run all searches in parallel using asyncio.gather (Pydantic AI native pattern)
        tasks = [
            run_single_search_with_history_management(query, i+1, query_types[i]) 
            for i, query in enumerate(search_plan.all_queries)
        ]
        results_list = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for results in results_list:
            all_results.extend(results)
        
        logger.info(f"✅ Parallel search complete: {len(all_results)} total results from {len(search_plan.all_queries)} queries")
        self.console.print(f"[green]Found {len(all_results)} relevant sources (from {len(search_plan.all_queries)} parallel searches)[/green]")
        return all_results

    async def _score_evidence(self, evidence_list: List[SearchResult], claim: str) -> List[EvidenceScore]:
        """Score evidence quality and relevance (parallelized with Pydantic AI native async)."""
        self.console.print("[bold]Scoring evidence quality...[/bold]")
        logger.info(f"📊 SCORING {len(evidence_list)} EVIDENCE SOURCES IN PARALLEL")
        
        # Log all evidence upfront
        for i, evidence in enumerate(evidence_list, 1):
            logger.info(f"  📄 Evidence #{i}/{len(evidence_list)}: {evidence.url[:80]}...")
            self.console.print(f"[dim]Will score evidence {i}/{len(evidence_list)}: {evidence.url[:60]}...[/dim]")
        
        # Create scoring tasks for parallel execution (Pydantic AI native approach)
        async def score_single_evidence(evidence: SearchResult, evidence_num: int):
            """Score a single evidence source."""
            evidence_ctx = EvidenceContext(
                claim=claim,
                evidence_content=evidence.content,
                source_url=evidence.url
            )
            
            input_text = (
                f"Causal claim: {claim}\n\n"
                f"Evidence to evaluate:\n{evidence.content[:200]}...\n\n"
                f"Source: {evidence.url}"
            )
            
            logger.info(f"  🔄 HANDOFF → evidence_scorer (evidence #{evidence_num}): Input='{evidence.url[:50]}...'")
            
            result = await evidence_scorer.run(input_text, deps=evidence_ctx)
            score = result.output
            score.source_url = evidence.url
            
            logger.info(f"  ✅ HANDOFF ← evidence_scorer (evidence #{evidence_num}): Rel={score.relevance_score}/10, Qual={score.quality_score}/10")
            
            self._log_agent_call(
                "evidence_scorer",
                f"URL: {evidence.url[:50]}",
                f"Rel={score.relevance_score}/10, Qual={score.quality_score}/10",
                result
            )
            
            return score
        
        # Run all scoring in parallel using asyncio.gather (Pydantic AI native pattern)
        logger.info("🚀 Running evidence scoring IN PARALLEL using asyncio.gather()")
        tasks = [score_single_evidence(evidence, i+1) for i, evidence in enumerate(evidence_list)]
        scores = await asyncio.gather(*tasks)
        
        if scores:
            avg_relevance = sum(s.relevance_score for s in scores) / len(scores)
            avg_quality = sum(s.quality_score for s in scores) / len(scores)
            logger.info(f"📈 AVERAGE SCORES: Relevance={avg_relevance:.1f}/10, Quality={avg_quality:.1f}/10")
            self.console.print(f"[green]Parallel scoring complete: Avg relevance={avg_relevance:.1f}/10, Quality={avg_quality:.1f}/10[/green]")
        
        return scores

    async def _judge_evidence(
        self, 
        ctx: ResearchContext, 
        evidence_list: List[SearchResult], 
        scores: List[EvidenceScore]
    ) -> EvidenceJudgment:
        """Judge whether evidence supports the claim."""
        self.console.print("[bold]Judging evidence support...[/bold]")
        
        combined_evidence = "\n\n".join([
            f"EVIDENCE {i+1}:\n{ev.content}\n\nSCORE: Relevance {sc.relevance_score}/10, Quality {sc.quality_score}/10"
            for i, (ev, sc) in enumerate(zip(evidence_list, scores))
        ])

        input_text = f"CAUSAL CLAIM: {ctx.claim}\n\n{combined_evidence}"
        
        logger.info(f"  🔄 HANDOFF → judgment_agent: Judging {len(evidence_list)} pieces of evidence for claim: '{ctx.claim[:60]}...'")
        
        result = await judgment_agent.run(input_text, deps=ctx)
        judgment = result.output
        
        supported_text = "SUPPORTED" if judgment.supported else "NOT SUPPORTED"
        
        logger.info(f"  ✅ HANDOFF ← judgment_agent: {supported_text}, Confidence={judgment.confidence}/10")
        
        self._log_agent_call(
            "judgment_agent",
            f"{len(evidence_list)} evidence pieces",
            f"{supported_text}, Conf={judgment.confidence}/10",
            result
        )
        self.console.print(f"Judgment: {supported_text} (Confidence: {judgment.confidence}/10)")
        
        return judgment

    async def _fetch_promising_articles(
        self, 
        results: List[SearchResult], 
        scores: List[EvidenceScore],
        claim: str = None
    ) -> None:
        """Fetch full article text for promising evidence using smart AI-based selection.
        
        Uses article_selector agent to intelligently choose 2-3 articles most likely to contain
        direct causal evidence based on snippets + scores, then fetches their full text.
        """
        if not results or not scores:
            return
        
        # Build input for article_selector: snippets + scores
        evidence_summaries = []
        for i, (result, score) in enumerate(zip(results, scores), 1):
            evidence_summaries.append(
                f"ARTICLE #{i}\n"
                f"URL: {result.url}\n"
                f"RELEVANCE SCORE: {score.relevance_score}/10\n"
                f"QUALITY SCORE: {score.quality_score}/10\n"
                f"JUSTIFICATION: {score.justification}\n\n"
                f"SNIPPET:\n{result.content[:500]}...\n"
            )
        
        # Join summaries with separator (can't use backslash in f-string expression)
        separator = '\n' + '='*80 + '\n'
        summaries_text = separator.join(evidence_summaries)
        
        combined_input = (
            f"CAUSAL CLAIM: {claim or self.current_claim}\n\n"
            "AVAILABLE ARTICLES (with snippets and scores):\n\n"
            f"{summaries_text}\n\n"
            "Select 2-3 articles that are MOST LIKELY to contain direct causal evidence."
        )
        
        # Use article_selector agent to make smart selection
        logger.info(f"  🧠 Running article_selector AI to identify causal evidence candidates")
        self.console.print("[bold magenta]Using AI to select articles with causal evidence...[/bold magenta]")
        
        research_ctx = ResearchContext(claim=claim or self.current_claim)
        result = await article_selector.run(combined_input, deps=research_ctx)
        selection = result.output
        
        self._log_agent_call(
            "article_selector",
            f"{len(results)} snippets + scores",
            f"Selected {len(selection.urls_to_fetch)} articles",
            result
        )
        
        if not selection.urls_to_fetch:
            logger.info(f"  ❌ Article selector found NO articles likely to contain causal evidence")
            logger.info(f"     Rationale: {selection.rationale}")
            self.console.print(f"[yellow]No articles selected for full-text fetch (no causal evidence indicators)[/yellow]")
            return
        
        logger.info(f"  ✅ Article selector chose {len(selection.urls_to_fetch)} articles: {selection.urls_to_fetch}")
        logger.info(f"     Rationale: {selection.rationale}")
        self.console.print(f"[green]✓ Selected {len(selection.urls_to_fetch)} articles with causal evidence indicators[/green]")
        self.console.print(f"[dim]Rationale: {selection.rationale}[/dim]")
        
        # Find the selected articles from results
        selected_results = []
        for url in selection.urls_to_fetch:
            for result in results:
                if result.url == url:
                    selected_results.append(result)
                    break
        
        logger.info(f"  📚 Fetching full text for {len(selected_results)} AI-selected articles")
        self.console.print(f"[bold cyan]Fetching full text for {len(selected_results)} AI-selected articles...[/bold cyan]")
        
        # Fetch all in parallel (get corresponding scores for selected articles)
        async def fetch_and_replace(result: SearchResult):
            url = result.url
            if "system://" in url:
                return  # Skip system URLs
            
            # Find the score for this result
            result_score = None
            for r, s in zip(results, scores):
                if r.url == url:
                    result_score = s
                    break
            
            logger.info(f"    📄 Fetching {url}")
            
            full_text = await self._fetch_full_article(url)
            
            if full_text:
                # Replace the snippet with full article text
                original_len = len(result.content)
                result.content = f"[FULL ARTICLE TEXT]\\n\\n{full_text}"
                logger.info(f"    ✅ Replaced snippet ({original_len} chars) with full article ({len(full_text)} chars)")
                self.console.print(f"[green]✓ Fetched {url[:80]}...[/green]")
            else:
                logger.warning(f"    ⚠️  Failed to fetch {url}")
                self.console.print(f"[yellow]⚠️  Could not fetch {url[:80]}... (paywall/error)[/yellow]")
        
        # Fetch all selected articles in parallel
        tasks = [fetch_and_replace(result) for result in selected_results]
        await asyncio.gather(*tasks)
        
        logger.info(f"  ✅ Full article fetching complete")

    async def _fetch_full_article(self, url: str) -> Optional[str]:
        """Fetch full article text from a URL.
        
        Attempts to extract the main content of an academic article.
        Returns None if fetching fails (paywall, 404, etc.).
        """
        try:
            # Use session with timeout
            session = requests.Session()
            retry_strategy = Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            
            # Add headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            logger.info(f"    Fetching: {url}")
            response = session.get(url, headers=headers, timeout=15, allow_redirects=True)
            
            if response.status_code != 200:
                logger.warning(f"    Failed to fetch ({response.status_code}): {url}")
                return None
            
            # Parse HTML and extract main content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                element.decompose()
            
            # Try common article content selectors
            content = None
            article_selectors = [
                'article',
                '.article-body',
                '.article-content',
                '.article-text',
                '.content-body',
                '.full-text',
                '.article__body',
                'main',
                '#article-content',
                '.pmc-article',
                '.abstract',
                '.article-section'
            ]
            
            for selector in article_selectors:
                elements = soup.select(selector)
                if elements:
                    content = '\n\n'.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                    if len(content) > 500:  # Minimum content length
                        break
            
            # Fallback: extract all paragraph text
            if not content or len(content) < 500:
                paragraphs = soup.find_all('p')
                content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
            
            if not content or len(content) < 500:
                logger.warning(f"    Insufficient content extracted from {url}")
                return None
            
            # Limit to first 50K characters to avoid token overload
            if len(content) > 50000:
                content = content[:50000] + "...[content truncated]"
            
            logger.info(f"    ✓ Successfully extracted {len(content)} chars from {url}")
            return content
            
        except requests.exceptions.Timeout:
            logger.warning(f"    Timeout fetching {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"    Request error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"    Unexpected error fetching {url}: {e}")
            return None

    async def _evaluate_full_articles_with_grade(
        self,
        ctx: ResearchContext,
        all_results: List[SearchResult],
        claim: str
    ) -> List[CausalEvidenceEvaluation]:
        """Evaluate each full article with 7 GRADE dimensions.
        
        This is the critical missing step that evaluates FULL ARTICLES
        (not snippets) using the causal_evidence_detector agent.
        """
        # Find articles that have full text (marked with [FULL ARTICLE TEXT])
        full_articles = [r for r in all_results if "[FULL ARTICLE TEXT]" in r.content]
        
        if not full_articles:
            logger.info("  ⚠️  No full articles to evaluate with GRADE dimensions")
            return []
        
        logger.info(f"  🎓 Evaluating {len(full_articles)} full articles with 7 GRADE dimensions")
        self.console.print(f"[bold cyan]Evaluating {len(full_articles)} full articles with GRADE criteria...[/bold cyan]")
        
        grade_evaluations = []
        
        for i, article in enumerate(full_articles, 1):
            logger.info(f"    📊 GRADE evaluation {i}/{len(full_articles)}: {article.url[:60]}...")
            
            input_text = (
                f"CAUSAL CLAIM: {claim}\n\n"
                f"ARTICLE TO EVALUATE:\n{article.content}\n\n"
                f"Source: {article.url}\n\n"
                "Evaluate this article using the 7 GRADE dimensions."
            )
            
            logger.info(f"  🔄 HANDOFF → causal_evidence_detector (article #{i}): URL='{article.url[:50]}...'")
            
            result = await causal_evidence_detector.run(input_text, deps=ctx)
            evaluation = result.output
            evaluation.source_url = article.url
            
            logger.info(
                f"  ✅ HANDOFF ← causal_evidence_detector (article #{i}): "
                f"direct={evaluation.direct_evidence_found}, "
                f"mechanism={evaluation.indirectness_mechanism:.1f}/10, "
                f"quality={evaluation.risk_of_bias:.1f}/10"
            )
            
            self._log_agent_call(
                f"causal_evidence_detector_art{i}",
                f"Full article: {article.url[:50]}",
                f"Direct={evaluation.direct_evidence_found}, Mechanism={evaluation.indirectness_mechanism:.1f}/10",
                result
            )
            
            grade_evaluations.append(evaluation)
            
            # Display summary
            if evaluation.direct_evidence_found:
                self.console.print(f"  [green]✓ Article {i}: Direct evidence found (mechanism={evaluation.indirectness_mechanism:.1f}/10)[/green]")
            else:
                self.console.print(f"  [yellow]○ Article {i}: No direct evidence (mechanism={evaluation.indirectness_mechanism:.1f}/10)[/yellow]")
        
        logger.info(f"  ✅ GRADE evaluation complete: {len(grade_evaluations)} articles evaluated")
        return grade_evaluations

    async def _check_for_direct_causal_evidence(
        self, 
        ctx: ResearchContext, 
        search_results: List[SearchResult]
    ) -> CausalEvidenceEvaluation:
        """Check for direct causal evidence."""
        self.console.print("[bold]Checking for direct causal evidence...[/bold]")
        
        combined_results = "\n\n".join([
            f"SEARCH RESULT #{i+1}:\nContent: {result.content}\nSource: {result.url}"
            for i, result in enumerate(search_results)
        ])
        
        input_text = (
            f"CAUSAL CLAIM: {ctx.claim}\n\n"
            "SEARCH RESULTS TO EVALUATE:\n\n"
            f"{combined_results}\n"
            "Based on these, do we have DIRECT causal evidence?"
        )
        
        logger.info(f"  🔄 HANDOFF → causal_evidence_detector: Analyzing {len(search_results)} search results for direct causal evidence")
        
        result = await causal_evidence_detector.run(input_text, deps=ctx)
        evaluation = result.output
        
        evidence_found_text = "FOUND" if evaluation.direct_evidence_found else "NOT FOUND"
        
        logger.info(f"  ✅ HANDOFF ← causal_evidence_detector: Direct evidence {evidence_found_text}, Confidence={evaluation.confidence}/10")
        
        self._log_agent_call(
            "causal_evidence_detector",
            f"{len(search_results)} search results",
            f"Direct evidence {evidence_found_text}, Conf={evaluation.confidence}/10",
            result
        )
        
        if evaluation.direct_evidence_found:
            self.console.print(f"[green]Direct evidence found (Confidence: {evaluation.confidence}/10)[/green]")
            
            # Check if the article was already fetched by the selector (contains full text)
            article_already_fetched = False
            for result in search_results:
                if result.url == evaluation.source_url and "[FULL ARTICLE TEXT]" in result.content:
                    article_already_fetched = True
                    logger.info(f"  ✓ Article already fetched by selector: {evaluation.source_url}")
                    self.console.print(f"[green]✓ Article already fetched at selector step (using full text)[/green]")
                    break
            
            # Only fetch if NOT already fetched by selector
            if not article_already_fetched and evaluation.source_url and "system://" not in evaluation.source_url:
                logger.info(f"  ⚠️ Article not in selector results, but found causal evidence. Skipping re-fetch to save tokens.")
                self.console.print(f"[yellow]⚠️  Article with causal evidence not in selector's top 3. Using snippet only.[/yellow]")
        else:
            self.console.print(f"[yellow]No direct evidence (Confidence: {evaluation.confidence}/10)[/yellow]")
        
        return evaluation

    async def _assess_multidimensional_quality(
        self,
        ctx: ResearchContext,
        verdict: FinalVerdict
    ) -> MultidimensionalScores:
        """Assess evidence quality using multi-dimensional framework."""
        self.console.print("[bold magenta]Assessing multi-dimensional evidence quality...[/bold magenta]")
        
        # Build comprehensive prompt with all verdict information
        prompt = (
            f"PROPOSED CAUSAL EDGE:\n"
            f"  Source → Target: {verdict.original_claim}\n"
            f"  Modified Claim: {verdict.modified_claim or 'N/A'}\n\n"
            
            f"ORIGINAL VERDICT:\n"
            f"  Verdict: {verdict.verdict}\n"
            f"  Confidence: {verdict.confidence}/10\n\n"
            
            f"JUDGE REASONING:\n{verdict.judge_reasoning}\n\n"
            
            f"KEY EVIDENCE (Top 3):\n"
        )
        
        # Add key evidence summaries (limit to 3 for token efficiency)
        for i in range(min(3, len(verdict.key_evidence_summaries))):
            prompt += f"  {i+1}. {verdict.key_evidence_summaries[i]}\n"
            prompt += f"     Rationale: {verdict.key_evidence_rationales[i]}\n"
            if i < len(verdict.evidence_urls):
                prompt += f"     URL: {verdict.evidence_urls[i]}\n"
            prompt += "\n"
        
        # Add strongest causal evidence if found
        if verdict.direct_causal_evidence_found:
            prompt += f"STRONGEST CAUSAL EVIDENCE:\n"
            prompt += f"  Study Type: {verdict.strongest_causal_evidence_study_type}\n"
            prompt += f"  Confidence: {verdict.strongest_causal_evidence_confidence}/10\n"
            prompt += f"  Summary: {verdict.strongest_causal_evidence_summary}\n\n"
        
        # Add limitations
        if verdict.limitations:
            prompt += f"LIMITATIONS:\n"
            for lim in verdict.limitations[:3]:  # Limit to 3
                prompt += f"  - {lim}\n"
            prompt += "\n"
        
        prompt += "Based on ALL this information, score the edge on the 5 dimensions."
        
        logger.info(f"  🔄 HANDOFF → multidimensional_judge: Assessing 5D quality for claim")
        
        # Define agent locally to ensure types are available and avoid forward-ref issues
        multidimensional_judge = Agent(
            HEAVY_MODEL,
            deps_type=ResearchContext,
            output_type=MultidimensionalScores,
            retries=5,  # Increase retries for structured output validation
            system_prompt=MULTIDIMENSIONAL_JUDGE_SYSTEM_PROMPT,
        )
        
        result = await multidimensional_judge.run(prompt, deps=ctx)
        scores = result.output
        
        # Calculate overall score for logging
        overall = (
            scores.evidence_quality * 0.3 + 
            scores.mechanistic_specificity * 0.25 + 
            scores.construct_validity * 0.2 + 
            scores.scope_relevance * 0.15 + 
            scores.theoretical_coherence * 0.1
        )
        
        logger.info(f"  ✅ HANDOFF ← multidimensional_judge: Scores computed, overall={overall:.2f}/10")
        
        self._log_agent_call(
            "multidimensional_judge",
            f"Verdict + evidence",
            f"5D scores: EQ={scores.evidence_quality:.1f}, MS={scores.mechanistic_specificity:.1f}, CV={scores.construct_validity:.1f}, SR={scores.scope_relevance:.1f}, TC={scores.theoretical_coherence:.1f}",
            result
        )
        
        return scores

    async def _synthesize_verdict(
        self,
        ctx: ResearchContext,
        original_claim: str,
        current_claim: str,
        evidence_list: List[SearchResult],  # ALL snippets
        scores: List[EvidenceScore],         # ALL snippet scores
        judgment: Optional[EvidenceJudgment],
        causal_evidence_result: Optional[CausalEvidenceEvaluation] = None,
        grade_evaluations: List[CausalEvidenceEvaluation] = None  # NEW: GRADE evaluations for full articles
    ) -> FinalVerdict:
        """LLM synthesizes verdict and makes consensus judgment."""
        self.console.print("[bold]Synthesizing final verdict with LLM consensus assessment...[/bold]")
        
        if grade_evaluations is None:
            grade_evaluations = []

        # Build comprehensive prompt with ALL snippets + FULL ARTICLE GRADE EVALUATIONS
        # Only show the CURRENT claim (which is either the original or modified)
        prompt = f"CLAIM TO EVALUATE: {current_claim}\n\n"
        
        # Add all evidence snippets with scores
        prompt += f"# ALL EVIDENCE SNIPPETS ({len(evidence_list)} total)\n"
        for i, (ev, sc) in enumerate(zip(evidence_list, scores), 1):
            query_type_label = ev.query_type or "supporting"
            support_label = sc.support_level or "N/A"
            prompt += f"\nSnippet {i}: [query_type={query_type_label}, support_level={support_label}, relevance={sc.relevance_score}/10, quality={sc.quality_score}/10]\n"
            prompt += f"{ev.content[:200]}...\n"
        
        # Add full article evaluations with GRADE scores
        if grade_evaluations:
            prompt += f"\n\n# FULL ARTICLE EVALUATIONS ({len(grade_evaluations)} articles with GRADE scores)\n"
            for i, eval_result in enumerate(grade_evaluations, 1):
                prompt += f"\nARTICLE {i}:\n"
                prompt += f"  Study Type: {eval_result.study_type}\n"
                prompt += f"  Summary: {eval_result.causal_evidence_summary}\n"
                prompt += f"  \n"
                prompt += f"  GRADE Scores:\n"
                prompt += f"    Risk of Bias:             {eval_result.risk_of_bias:.1f}/10\n"
                prompt += f"    Indirectness - Mechanism: {eval_result.indirectness_mechanism:.1f}/10 ⚠️ KEY\n"
                prompt += f"    Indirectness - Population: {eval_result.indirectness_population:.1f}/10\n"
                prompt += f"    Indirectness - Constructs: {eval_result.indirectness_constructs:.1f}/10\n"
                prompt += f"    Imprecision:              {eval_result.imprecision:.1f}/10\n"
                prompt += f"    Magnitude of Effect:      {eval_result.magnitude_of_effect:.1f}/10 ⚠️ KEY\n"
                prompt += f"    Dose-Response:            {eval_result.dose_response_gradient:.1f}/10\n"
                prompt += f"  \n"
                prompt += f"  Cited Passage: \"{eval_result.cited_passage}\"\n"
        
        prompt += "\n\n# TASK\n\n"
        prompt += "1. Assess CONSENSUS across articles for key dimensions (mechanism, effect, quality)\n"
        prompt += "2. Make BINARY JUDGMENT for CLD inclusion based on holistic assessment\n"
        prompt += "3. Synthesize final verdict with key evidence and limitations\n"
        prompt += "4. Identify the direct quote and URL from the strongest piece of evidence.\n"

        logger.info(f"  🔄 HANDOFF → verdict_synthesizer: Synthesizing with LLM consensus judgment")

        result = await verdict_agent.run(prompt, deps=ctx)
        verdict = result.output
        
        # Convert is_direct_causal from boolean to numeric (0 or 1) if needed
        if verdict.is_direct_causal is not None and isinstance(verdict.is_direct_causal, bool):
            verdict.is_direct_causal = 1 if verdict.is_direct_causal else 0
        
        # Explicitly set original_claim and modified_claim to ensure they're in the output
        verdict.original_claim = original_claim
        if current_claim != original_claim:
            verdict.modified_claim = current_claim
        else:
            verdict.modified_claim = None
        
        logger.info(
            f"  ✅ HANDOFF ← verdict_synthesizer: "
            f"is_direct_causal={verdict.is_direct_causal}, "
            f"verdict={verdict.verdict.upper()}"
        )
        
        self._log_agent_call(
            "verdict_agent",
            f"{len(evidence_list)} evidence + judgment",
            f"Verdict: {verdict.verdict.upper()}, Conf={verdict.confidence}/10",
            result
        )
        
        # Display consensus + binary judgment
        self.console.print("\n[bold cyan]═══ CONSENSUS ASSESSMENT ═══[/bold cyan]")
        self.console.print(f"  Mechanism: {verdict.consensus_mechanism}")
        self.console.print(f"  Effect: {verdict.consensus_effect}")
        self.console.print(f"  Quality: {verdict.consensus_quality}")
        
        self.console.print("\n[bold cyan]═══ BINARY JUDGMENT FOR CLD ═══[/bold cyan]")
        if verdict.is_direct_causal:
            self.console.print(f"[bold green]✅ DIRECT CAUSAL LINK[/bold green]")
            self.console.print(f"[green]   Mechanistic Specificity: {verdict.mechanistic_specificity:.1f}/10[/green]")
            if verdict.direct_causal_confidence:
                self.console.print(f"[green]   Confidence: {verdict.direct_causal_confidence.upper()}[/green]")
        else:
            self.console.print(f"[bold red]❌ NOT DIRECT CAUSAL LINK[/bold red]")
            self.console.print(f"[yellow]   Mechanistic Specificity: {verdict.mechanistic_specificity:.1f}/10[/yellow]")
            if verdict.direct_causal_confidence:
                self.console.print(f"[yellow]   Confidence: {verdict.direct_causal_confidence.upper()}[/yellow]")
        
        if verdict.direct_causal_reasoning:
            self.console.print(f"[dim]   {verdict.direct_causal_reasoning}[/dim]")

        # Display strongest evidence quote and URL
        if verdict.strongest_causal_quote and verdict.strongest_evidence_url:
            self.console.print("\n[bold cyan]═══ STRONGEST CAUSAL EVIDENCE ═══[/bold cyan]")
            self.console.print(f"  Quote: \"{verdict.strongest_causal_quote}\"")
            self.console.print(f"  Source: {verdict.strongest_evidence_url}")

        # Add direct causal evidence information if found
        if causal_evidence_result and causal_evidence_result.direct_evidence_found:
            verdict.direct_causal_evidence_found = True
            verdict.strongest_causal_evidence_url = causal_evidence_result.source_url
            verdict.strongest_causal_evidence_passage = causal_evidence_result.cited_passage
            verdict.strongest_causal_evidence_study_type = causal_evidence_result.study_type
            verdict.strongest_causal_evidence_summary = causal_evidence_result.causal_evidence_summary
            verdict.strongest_causal_evidence_confidence = causal_evidence_result.confidence
        
        # Add iteration history for transparency of claim refinements
        verdict.iteration_history = self.iteration_history
        logger.info(f"  📝 Added {len(self.iteration_history)} iterations to verdict history")
        
        # ===== PROGRAMMATIC: Calculate average support scores from SNIPPETS + ARTICLES =====
        
        # Extract support scores from snippets (evidence_scores)
        snippet_support_scores = [s.support_score for s in scores if s.support_score is not None]
        
        # Extract support scores from full articles (grade_evaluations)
        article_support_scores = []
        if grade_evaluations:
            article_support_scores = [eval_result.support_score for eval_result in grade_evaluations if eval_result.support_score is not None]
        
        # Calculate snippet average
        if snippet_support_scores:
            avg_snippet_score = sum(snippet_support_scores) / len(snippet_support_scores)
            verdict.average_snippet_support_score = round(avg_snippet_score, 3)
            verdict.snippet_support_scores = snippet_support_scores
            logger.info(f"  📄 Snippet average support score: {avg_snippet_score:.3f} (from {len(snippet_support_scores)} snippets)")
        else:
            logger.info("  ⚠️  No snippet support scores available")
        
        # Calculate article average
        if article_support_scores:
            avg_article_score = sum(article_support_scores) / len(article_support_scores)
            verdict.average_article_support_score = round(avg_article_score, 3)
            verdict.article_support_scores = article_support_scores
            logger.info(f"  📚 Article average support score: {avg_article_score:.3f} (from {len(article_support_scores)} articles)")
        else:
            logger.info("  ⚠️  No article support scores available")
        
        # Calculate combined average
        all_support_scores = snippet_support_scores + article_support_scores
        if all_support_scores:
            avg_combined_score = sum(all_support_scores) / len(all_support_scores)
            verdict.average_combined_support_score = round(avg_combined_score, 3)
            
            logger.info(f"  📊 Combined average support score: {avg_combined_score:.3f} (from {len(all_support_scores)} total evaluations)")
            
            # Display detailed breakdown
            self.console.print(f"\n[bold cyan]═══ SUPPORT SCORES (SNIPPETS + ARTICLES) ═══[/bold cyan]")
            
            if snippet_support_scores:
                self.console.print(f"  Snippet scores ({len(snippet_support_scores)}): {[round(s, 2) for s in snippet_support_scores]}")
                self.console.print(f"  [bold]Snippet average: {avg_snippet_score:.3f}[/bold]")
            else:
                self.console.print(f"  Snippet scores: None")
            
            if article_support_scores:
                self.console.print(f"\n  Article scores ({len(article_support_scores)}): {[round(s, 2) for s in article_support_scores]}")
                self.console.print(f"  [bold]Article average: {avg_article_score:.3f}[/bold]")
            else:
                self.console.print(f"\n  Article scores: None")
            
            self.console.print(f"\n  [bold green]Combined average: {avg_combined_score:.3f}[/bold green] (from {len(all_support_scores)} total evaluations)")
        else:
            logger.info("  ⚠️  No support scores available (neither from snippets nor articles)")

        # Display strongest causal evidence quote if provided by verdict_agent
        if verdict.strongest_causal_quote and verdict.strongest_evidence_url:
            self.console.print("\n[bold cyan]═══ STRONGEST CAUSAL EVIDENCE (FROM VERDICT) ═══[/bold cyan]")
            self.console.print(f"[bold]Quote:[/bold] \"{verdict.strongest_causal_quote}\"")
            self.console.print(f"[bold]Source:[/bold] {verdict.strongest_evidence_url}")
            logger.info(f"  ✅ Verdict included strongest causal quote: {verdict.strongest_causal_quote[:50]}...")
            logger.info(f"     From: {verdict.strongest_evidence_url}")

        # ===== NEW: Multi-dimensional assessment =====
        try:
            self.console.print("\n[bold cyan]Running multi-dimensional evidence quality assessment...[/bold cyan]")
            md_scores = await self._assess_multidimensional_quality(ctx, verdict)
            
            # Calculate overall weighted score
            overall_score = (
                md_scores.evidence_quality * DIMENSION_WEIGHTS['evidence_quality'] +
                md_scores.mechanistic_specificity * DIMENSION_WEIGHTS['mechanistic_specificity'] +
                md_scores.construct_validity * DIMENSION_WEIGHTS['construct_validity'] +
                md_scores.scope_relevance * DIMENSION_WEIGHTS['scope_relevance'] +
                md_scores.theoretical_coherence * DIMENSION_WEIGHTS['theoretical_coherence']
            )
            
            # Populate verdict with multi-dimensional scores
            verdict.evidence_quality = md_scores.evidence_quality
            verdict.mechanistic_specificity = md_scores.mechanistic_specificity
            verdict.construct_validity = md_scores.construct_validity
            verdict.scope_relevance = md_scores.scope_relevance
            verdict.theoretical_coherence = md_scores.theoretical_coherence
            verdict.overall_quality_score = round(overall_score, 2)
            verdict.multidimensional_reasoning = md_scores.reasoning
            
            self.console.print(f"[green]✓ Multi-dimensional scores computed:[/green]")
            self.console.print(f"  Evidence Quality: {md_scores.evidence_quality:.1f}/10")
            self.console.print(f"  Mechanistic Specificity: {md_scores.mechanistic_specificity:.1f}/10")
            self.console.print(f"  Construct Validity: {md_scores.construct_validity:.1f}/10")
            self.console.print(f"  Scope Relevance: {md_scores.scope_relevance:.1f}/10")
            self.console.print(f"  Theoretical Coherence: {md_scores.theoretical_coherence:.1f}/10")
            self.console.print(f"  [bold]Overall Quality Score: {overall_score:.2f}/10[/bold]")
            
            logger.info(f"  ✅ Multi-dimensional assessment complete: Overall={overall_score:.2f}/10")
            
        except Exception as e:
            logger.error(f"  ❌ Error in multi-dimensional assessment: {e}")
            self.console.print(f"[yellow]⚠️  Multi-dimensional assessment failed: {e}[/yellow]")
            # Continue without multi-dimensional scores

        # ===== TELEMETRY: Populate tracking metrics =====
        end_time = time.time()
        total_elapsed = end_time - self.start_time
        
        verdict.total_tokens = self.total_tokens
        verdict.total_api_calls = self.agent_call_count
        verdict.elapsed_time_seconds = round(total_elapsed, 2)
        verdict.iterations_completed = self.iteration
        
        logger.info(f"\\n{'='*100}")
        logger.info(f"📊 DEEP RESEARCH TELEMETRY")
        logger.info(f"{'='*100}")
        logger.info(f"  Total API Calls:      {self.agent_call_count}")
        logger.info(f"  Total Tokens Used:    {self.total_tokens:,}")
        logger.info(f"  Iterations Completed: {self.iteration}/{self.max_iterations}")
        logger.info(f"  Total Time:           {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
        logger.info(f"{'='*100}\\n")
        
        self.console.print(f"\\n[bold magenta]{'='*80}[/bold magenta]")
        self.console.print(f"[bold magenta]📊 DEEP RESEARCH TELEMETRY[/bold magenta]")
        self.console.print(f"[bold magenta]{'='*80}[/bold magenta]")
        self.console.print(f"  Total API Calls:      {self.agent_call_count}")
        self.console.print(f"  Total Tokens Used:    [green]{self.total_tokens:,}[/green]")
        self.console.print(f"  Iterations Completed: {self.iteration}/{self.max_iterations}")
        self.console.print(f"  Total Time:           {total_elapsed:.2f}s ([cyan]{total_elapsed/60:.1f} minutes[/cyan])")
        self.console.print(f"[bold magenta]{'='*80}[/bold magenta]\\n")

        return verdict


async def main():
    """Main entrypoint for the PydanticAI deep research pipeline."""
    console = Console()
    console.print("[bold]PydanticAI Causal Claim Deep Research Pipeline[/bold]")
    console.print("This system will search scientific literature to verify causal claims using PydanticAI.")
    
    # Check API keys
    if not BRAVE_SEARCH_API_KEY:
        console.print("[yellow]Warning: BRAVE_SEARCH_API_KEY not found. Using mock search data.[/yellow]")
        console.print("To use real search, add BRAVE_SEARCH_API_KEY to your .env file.")
    else:
        console.print("[green]Brave Search API key found. Using live search data.[/green]")
    
    claim = input("\nEnter a causal claim (A causes B through mechanism X): ")
    
    manager = DeepResearchManager(max_iterations=3, minimum_evidence_count=5, max_sources_per_query=3)
    verdict = await manager.run(claim)
    
    console.print("\n[bold green]===== FINAL RESEARCH REPORT =====[/bold green]")
    console.print(f"[bold]Original claim:[/bold] {verdict.original_claim}")
    
    if verdict.modified_claim:
        console.print(f"[bold]Modified claim:[/bold] {verdict.modified_claim}")
        
    console.print(f"\n[bold]Verdict:[/bold] {verdict.verdict.upper()} (Confidence: {verdict.confidence}/10)")

    # Display direct causal evidence if found
    if verdict.direct_causal_evidence_found:
        console.print("\n[bold red]DIRECT CAUSAL EVIDENCE FOUND[/bold red]")
        if verdict.strongest_causal_evidence_url:
            console.print(f"[bold]Source:[/bold] [blue underline]{verdict.strongest_causal_evidence_url}[/blue underline]")
        if verdict.strongest_causal_evidence_study_type:
            console.print(f"[bold]Study Type:[/bold] {verdict.strongest_causal_evidence_study_type}")
        if verdict.strongest_causal_evidence_passage:
            console.print(f"[bold]Cited Passage:[/bold] \"{verdict.strongest_causal_evidence_passage}\"")
        if verdict.strongest_causal_evidence_summary:
            console.print(f"[bold]Evidence Summary:[/bold] {verdict.strongest_causal_evidence_summary}")
        console.print(f"[bold]Confidence:[/bold] {verdict.strongest_causal_evidence_confidence}/10")

    console.print("\n[bold]Judge Reasoning:[/bold]")
    console.print(verdict.judge_reasoning)

    console.print("\n[bold]Key Evidence:[/bold]")
    for i in range(len(verdict.key_evidence_labels)):
        label = verdict.key_evidence_labels[i]
        rationale = verdict.key_evidence_rationales[i]
        url = verdict.evidence_urls[i]
        
        if verdict.direct_causal_evidence_found:
            console.print(f"\n[bold red]{label} (DIRECT CAUSAL EVIDENCE)[/bold red]")
        else:
            console.print(f"\n[yellow]{label}[/yellow]")
        
        console.print(f"  [bold]Rationale:[/bold] {rationale}")
        
        if url and "no url" not in url.lower():
            console.print(f"    [blue underline]{url}[/blue underline]")
        else:
            console.print("    [dim]No URL available[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
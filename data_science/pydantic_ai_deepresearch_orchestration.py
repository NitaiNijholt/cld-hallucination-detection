"""
PydanticAI Deep Research Orchestration System

This module contains a set of Pydantic models, agent definitions, and a manager class 
(DeepResearchManager) for conducting a multi-step scientific literature investigation 
on a causal claim. It uses a sequence of PydanticAI agents to plan a search, retrieve 
relevant scientific evidence, score the evidence quality, judge whether the evidence 
supports the claim, and synthesize a final verdict.

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
    """Represents a structured plan for searching scientific literature."""
    queries: List[SearchQuery] = Field(description="List of search queries to perform")
    keywords: List[str] = Field(description="Key terms and synonyms for the causal relationship")
    confounders: List[str] = Field(description="Potential confounding variables to check for")


class EvidenceScore(BaseModel):
    """Represents a scored evaluation of evidence quality and relevance."""
    relevance_score: int = Field(description="How directly the evidence addresses the causal relationship (1-10)")
    quality_score: int = Field(description="Methodological rigor of the evidence (1-10)")
    justification: str = Field(description="Reasoning for these scores")
    source_url: Optional[str] = Field(default=None, description="URL to the original source if available")


class EvidenceJudgment(BaseModel):
    """Represents a judgment about whether a causal claim is supported by evidence."""
    supported: bool = Field(description="Whether the claim is supported by the evidence")
    confidence: int = Field(description="Confidence level in this judgment (1-10)")
    reasoning: str = Field(description="Detailed reasoning for the judgment")
    suggested_modification: Optional[str] = Field(default=None, description="Suggested modification to the original claim if needed")


class CausalEvidenceEvaluation(BaseModel):
    """Result of evaluating evidence for direct causal relationship."""
    direct_evidence_found: bool = Field(description="Whether direct evidence of causality was found")
    confidence: int = Field(description="Confidence in the assessment (1-10)")
    source_url: Optional[str] = Field(default=None, description="URL of the source containing the strongest causal evidence")
    cited_passage: Optional[str] = Field(default=None, description="VERBATIM quote copied exactly from the source text/abstract demonstrating causality. Must be actual text from the paper, not paraphrased or summarized.")
    study_type: Optional[str] = Field(default=None, description="Type of study (e.g., 'RCT', 'Meta-analysis', 'Experimental manipulation')")
    causal_evidence_summary: str = Field(description="Summary of the strongest direct causal evidence found, explicitly referencing the source and target variables from the original causal claim")
    explanation: str = Field(description="Explanation of why this is or isn't direct causal evidence, explicitly mentioning the source and target variables from the original claim")


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
    direct_causal_evidence_found: Optional[bool] = None
    strongest_causal_evidence_url: Optional[str] = None
    strongest_causal_evidence_passage: Optional[str] = None
    strongest_causal_evidence_study_type: Optional[str] = None
    strongest_causal_evidence_summary: Optional[str] = None
    strongest_causal_evidence_confidence: Optional[int] = None
    iteration_history: Optional[List[IterationHistory]] = Field(default=None, description="History of claim refinements and evidence across iterations")


class SearchResult(BaseModel):
    """Result from web search with content and URL."""
    content: str
    url: str


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
        "You are a methodologist helping to test a causal claim. "
        "Given a source variable, target variable, and explanatory "
        "mechanism you must: (1) restate the causal hypothesis in PICO "
        "form, (2) enumerate key search keywords & synonyms, (3) list "
        "potential confounders, (4) define inclusion/exclusion criteria "
        "for admissible evidence. Return a search plan with queries that "
        "would help verify this causal relationship. "
        "⏱️ IMPORTANT: You have a 2-minute time limit for the entire research process. "
        "Generate 3-4 focused, high-impact search queries to maximize efficiency."
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
    retries=5,  # Increase retries for structured output validation
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
        "You are a scientific evidence evaluator specializing in causal inference. "
        "For each research paper or evidence source, provide a numerical score "
        "and justification on relevance to the causal claim and methodological quality. "
        "Consider: study design (RCTs > cohort > case-control > cross-sectional), "
        "sample size, controls for confounders, effect size, statistical significance, "
        "and replicability. Be critical and methodologically rigorous. "
        "⏱️ IMPORTANT: Work quickly - provide concise justifications (2-3 sentences max)."
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
        "You are an expert in causal inference and scientific evidence evaluation. "
        "Your task is to determine if the provided search results contain DIRECT evidence supporting "
        "a causal relationship. Direct causal evidence typically includes: "
        "1. Experimental studies with randomized controlled trials "
        "2. Natural experiments with clear causal identification "
        "3. Research that explicitly states a causal relationship was found and explains the mechanism "
        "4. Meta-analyses or systematic reviews confirming causality. "
        "Be strict in your assessment - correlation, association, or hypothesized relationships "
        "do NOT count as direct causal evidence.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. cited_passage MUST be a VERBATIM quote copied word-for-word from the paper/abstract. "
        "   DO NOT paraphrase, summarize, or rewrite. Copy the exact text that demonstrates causality.\n"
        "2. Always explicitly reference the SOURCE and TARGET variables from the original causal claim "
        "   in your causal_evidence_summary and explanation fields.\n"
        "3. Include the exact URL of the source.\n"
        "4. Specify the study type (e.g., 'RCT', 'Meta-analysis', 'Experimental manipulation').\n\n"
        "Example of GOOD cited_passage: \"The randomized controlled trial demonstrated that increased physical activity "
        "caused significant reductions in BMI (p<0.001) compared to control group.\"\n\n"
        "Example of BAD cited_passage (don't do this): 'This study found that physical activity affects BMI through metabolic changes.'"
    )
)


verdict_agent = Agent(
    HEAVY_MODEL,
    deps_type=ResearchContext,
    output_type=FinalVerdict,
    retries=5,  # Increase retries for structured output validation
    system_prompt=(
        "You are a scientific synthesis expert creating final verdicts on causal claims. "
        "Based on all evidence and judgments, determine if the original claim is supported, "
        "partially supported, or not supported. Provide a modified claim if needed. "
        "IMPORTANT: Return only valid JSON matching the FinalVerdict schema exactly."
    )
)


article_selector = Agent(
    HEAVY_MODEL,
    deps_type=ResearchContext,
    output_type=ArticleSelection,
    retries=5,  # Increase retries for structured output validation
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
        start_time = time.time()
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
        
        causal_evidence_result = None

        # Main research iteration loop
        for self.iteration in range(1, self.max_iterations + 1):
            elapsed = time.time() - start_time
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
            causal_evidence_result
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
        
        logger.info(f"  ✅ HANDOFF ← hypothesis_evaluator: Generated {len(search_plan.queries)} queries")
        
        self._log_agent_call(
            "hypothesis_evaluator", 
            f"Claim: {ctx.claim}", 
            f"{len(search_plan.queries)} queries generated",
            result
        )
        
        # Add verbatim motivation as a search query (user requirement)
        verbatim_query = SearchQuery(
            query=ctx.claim,
            rationale="Verbatim motivation search - directly search for the exact claim to find related literature"
        )
        search_plan.queries.insert(0, verbatim_query)
        logger.info(f"  ➕ Added verbatim motivation as Query #1: '{ctx.claim[:100]}...'")
        
        logger.info(f"✅ Total queries (with verbatim): {len(search_plan.queries)}")
        for i, q in enumerate(search_plan.queries, 1):
            logger.info(f"    Query #{i}: {q.query[:60]}...")
        self.console.print(f"Generated {len(search_plan.queries)} search queries (including verbatim motivation)")
            
        return search_plan

    async def _run_searches(self, search_plan: SearchPlan, claim: str) -> List[SearchResult]:
        """Perform literature searches based on the search plan (parallelized with Pydantic AI native async)."""
        self.console.print("[bold]Searching scientific literature...[/bold]")
        logger.info(f"🚀 Running {len(search_plan.queries)} searches IN PARALLEL using asyncio.gather()")
        
        # Log all queries upfront
        for i, query in enumerate(search_plan.queries, 1):
            logger.info(f"  🔍 SEARCH QUERY #{i}/{len(search_plan.queries)}: {query.query}")
            self.console.print(f"[dim]Query {i}: {query.query}[/dim]")
        
        # Create search tasks for parallel execution (Pydantic AI native approach)
        async def run_single_search_with_history_management(query_obj: SearchQuery, query_num: int):
            """Run a single search with per-thread conversation history management."""
            search_ctx = SearchContext(
                claim=claim,
                query=query_obj.query,
                max_sources=self.max_sources_per_query
            )
            
            logger.info(f"  🔄 HANDOFF → search_agent (query #{query_num}): Query='{query_obj.query[:50]}...'")
            
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
                from_info=f"Search query: {query_obj.query[:80]}",
                to_info=f"Found {len(result.output) if isinstance(result.output, list) else 0} results",
                result=result
            )
            
            search_results = result.output
            if isinstance(search_results, list):
                logger.info(f"  ✅ HANDOFF ← search_agent (query #{query_num}): Found {len(search_results)} results")
                return search_results
            
            logger.info(f"  ✅ HANDOFF ← search_agent (query #{query_num}): No results (empty or invalid response)")
            return []
        
        # Run all searches in parallel using asyncio.gather (Pydantic AI native pattern)
        tasks = [run_single_search_with_history_management(query, i+1) for i, query in enumerate(search_plan.queries)]
        results_list = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for results in results_list:
            all_results.extend(results)
        
        logger.info(f"✅ Parallel search complete: {len(all_results)} total results from {len(search_plan.queries)} queries")
        self.console.print(f"[green]Found {len(all_results)} relevant sources (from {len(search_plan.queries)} parallel searches)[/green]")
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

    async def _synthesize_verdict(
        self,
        ctx: ResearchContext,
        original_claim: str,
        current_claim: str,
        evidence_list: List[SearchResult],
        scores: List[EvidenceScore],
        judgment: Optional[EvidenceJudgment],
        causal_evidence_result: Optional[CausalEvidenceEvaluation] = None
    ) -> FinalVerdict:
        """Synthesize final verdict."""
        self.console.print("[bold]Synthesizing final verdict...[/bold]")

        # Prepare evidence details
        evidence_details = []
        for i, (ev, sc) in enumerate(zip(evidence_list, scores)):
            evidence_details.append(
                f"EVIDENCE #{i+1}\n"
                f"URL: {ev.url}\n"
                f"RELEVANCE: {sc.relevance_score}/10\n"
                f"QUALITY: {sc.quality_score}/10\n\n"
                f"CONTENT:\n{ev.content}\n"
            )

        judgment_text = ""
        if judgment:
            judgment_text = f"JUDGMENT:\nSupported: {judgment.supported}\nConfidence: {judgment.confidence}/10\nReasoning: {judgment.reasoning}\n\n"

        prompt = (
            f"ORIGINAL CLAIM:\n{original_claim}\n\n"
            f"CURRENT CLAIM:\n{current_claim}\n\n"
            f"ALL EVIDENCE:\n{chr(10).join(evidence_details)}\n\n"
            f"{judgment_text}"
            "Synthesize a final verdict with all required fields."
        )

        logger.info(f"  🔄 HANDOFF → verdict_agent: Synthesizing final verdict from {len(evidence_list)} evidence pieces + judgment")

        result = await verdict_agent.run(prompt, deps=ctx)
        verdict = result.output
        
        logger.info(f"  ✅ HANDOFF ← verdict_agent: Verdict={verdict.verdict.upper()}, Confidence={verdict.confidence}/10")
        
        self._log_agent_call(
            "verdict_agent",
            f"{len(evidence_list)} evidence + judgment",
            f"Verdict: {verdict.verdict.upper()}, Conf={verdict.confidence}/10",
            result
        )

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
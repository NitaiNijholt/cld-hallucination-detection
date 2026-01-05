# auto_experimentation_framework.py
"""Scientific Relation Auto‑Experimentation Framework
---------------------------------------------------
A reference implementation that orchestrates a team of OpenAI Agents to evaluate a
causal relationship between a *source* variable and a *target* variable, given a
textual motivation of the mechanism.

The system instantiates four agent roles:

1. **OrchestratorAgent** – Controls the overall experiment: interprets the task,
   delegates to specialist agents, aggregates results, and iteratively
   refines queries until a stopping criterion is met.
2. **HypothesisEvaluatorAgent** – Converts the initial natural‑language task
   (source → target + mechanism) into structured research questions and search
   strategies; may suggest alternative mechanisms and confounders.
3. **WebSearchAgent** – Executes literature and data discovery using the SDK’s
   built‑in `web_search` tool (or any custom Tool) and returns JSON metadata of
   candidate evidence (title, URL, snippet, publication year, study type).
4. **EvidenceScorerAgent** – Grades each evidence item for *relevance* and
   *methodological quality* using a GRADE/ROB‑2 inspired rubric; outputs a
   numerical score and justification.

The framework exposes a single public entry point — `run_experiment()` — which
accepts the user‑provided **source**, **target**, and **mechanism** strings, plus
optional hyper‑parameters for exploration depth, max iterations, and model
selection. The orchestrator loops until the cumulative evidence quality plateaus
or the budget is exhausted.

Dependencies
------------
```bash
pip install openai==1.30.1 openai-agents-python tiktoken duckduckgo-search tenacity
```

Environment variable `OPENAI_API_KEY` must be set.

For a minimal *quick‑start* demonstration, execute:
```python
from auto_experimentation_framework import run_experiment
report = run_experiment(
    source="Coffee consumption",
    target="Risk of type 2 diabetes",
    mechanism=(
        "Caffeine and chlorogenic acids improve insulin sensitivity and reduce "
        "post‑prandial glucose peaks"
    ),
)
print(report.summary)
```

The code below is purposely terse; in production you should add robust logging,
vector‑store caching, and error monitoring.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Literal, Sequence

import openai  # Agents SDK piggy‑backs on the official client
from openai import agents as oa  # type: ignore – hypothetical import path
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

MODELS: dict[str, str] = {
    "reasoning": "gpt-4o-mini",  # fast reasoning
    "grading": "gpt-4o",         # more tokens for rubric
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def ddg_web_search(query: str, max_results: int = 10) -> List[dict]:
    """Lightweight wrapper around DuckDuckGo Search API."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title"),
                    "href": r.get("href"),
                    "snippet": r.get("body"),
                }
            )
    return results


@dataclass
class EvidenceItem:
    title: str
    url: str
    snippet: str
    relevance_score: float | None = None
    quality_score: float | None = None
    justification: str | None = None

    @property
    def total_score(self) -> float | None:
        if self.relevance_score is None or self.quality_score is None:
            return None
        return 0.6 * self.relevance_score + 0.4 * self.quality_score

    def to_json(self) -> dict:
        return self.__dict__


# ---------------------------------------------------------------------------
# Agent implementations
# ---------------------------------------------------------------------------

class HypothesisEvaluatorAgent(oa.Agent):
    """Transforms problem statement into search queries & evaluation plan."""

    def __init__(self, model: str = MODELS["reasoning"]):
        super().__init__(
            model=model,
            instructions=(
                "You are a methodologist helping to test a causal claim. "
                "Given a source variable, target variable, and explanatory "
                "mechanism you must: (1) restate the causal hypothesis in PICO "
                "form, (2) enumerate key search keywords & synonyms, (3) list "
                "potential confounders, (4) define inclusion/exclusion criteria "
                "for admissible evidence. Return JSON with keys: 'questions', "
                "'keywords', 'confounders', 'criteria'."
            ),
        )


class WebSearchAgent(oa.Agent):
    """Fetches candidate literature using DDG (or other) and returns metadata."""

    def __init__(self, model: str = MODELS["reasoning"]):
        super().__init__(model=model)

        # Register a custom python function as a *tool*.
        @oa.tool(name="ddg_search", description="DuckDuckGo web search")
        def _search_tool(query: str) -> str:  # noqa: D401
            return json.dumps(ddg_web_search(query))

        self.register_tool(_search_tool)

        self.prompt = (
            "You are a research assistant. Use the web search tool to retrieve "
            "up to 10 relevant scientific sources about the given query. Return "
            "the raw JSON list from the tool call."
        )


class EvidenceScorerAgent(oa.Agent):
    """Scores evidence items for relevance & methodological quality."""

    RUBRIC = (
        "You are a systematic‑review expert using the GRADE and ROB‑2 "
        "frameworks. For each evidence JSON dict provided, output a JSON with "
        "keys: 'relevance_score' (0‑100), 'quality_score' (0‑100), and "
        "'justification' (max 75 words). Relevance considers alignment with the "
        "PICO question; quality considers study design, bias, sample size, etc."
    )

    def __init__(self, model: str = MODELS["grading"]):
        super().__init__(model=model, instructions=self.RUBRIC)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ExperimentReport:
    source: str
    target: str
    mechanism: str
    evidence: List[EvidenceItem]
    generation_time: float = field(default_factory=time.time)

    @property
    def summary(self) -> str:
        top = sorted(
            [e for e in self.evidence if e.total_score is not None],
            key=lambda x: x.total_score,
            reverse=True,
        )[:5]
        lines = [
            f"Top evidence for '{self.source} → {self.target}':",
            *[
                f"• {e.title} (score {e.total_score:.1f}) – {e.url}" for e in top
            ],
        ]
        return "\n".join(lines)


class OrchestratorAgent:
    """High‑level controller that runs the auto‑experimentation loop."""

    def __init__(
        self,
        max_iterations: int = 4,
        exploration_k: int = 6,
        min_improvement: float = 5.0,
    ) -> None:
        self.hypothesis_agent = HypothesisEvaluatorAgent()
        self.search_agent = WebSearchAgent()
        self.score_agent = EvidenceScorerAgent()

        self.max_iterations = max_iterations
        self.exploration_k = exploration_k
        self.min_improvement = min_improvement

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def _call_llm(self, agent: oa.Agent, content: str) -> str:
        return agent.chat(content).content  # type: ignore

    def run(self, source: str, target: str, mechanism: str) -> ExperimentReport:
        evidence: List[EvidenceItem] = []
        prev_best: float = 0.0

        # ---- 1. Plan the search strategy ---------------------------------
        plan_json = json.loads(
            self._call_llm(
                self.hypothesis_agent,
                f"Source: {source}\nTarget: {target}\nMechanism: {mechanism}",
            )
        )
        queries: Sequence[str] = plan_json["keywords"][: self.exploration_k]

        # ---- 2. Iterative search & scoring -------------------------------
        for step, q in enumerate(queries, start=1):
            raw_json = json.loads(
                self._call_llm(self.search_agent, q)
            )
            # Limit to first N
            for item in raw_json[: self.exploration_k]:
                ev = EvidenceItem(title=item["title"], url=item["href"], snippet=item["snippet"])
                # Score
                score_payload = json.dumps(item)
                eval_json = json.loads(self._call_llm(self.score_agent, score_payload))
                ev.relevance_score = eval_json["relevance_score"]
                ev.quality_score = eval_json["quality_score"]
                ev.justification = eval_json["justification"]
                evidence.append(ev)

            # Check for convergence
            best_now = max(
                (e.total_score or 0 for e in evidence),
                default=0.0,
            )
            if best_now - prev_best < self.min_improvement:
                break  # Diminishing returns
            prev_best = best_now
            if step >= self.max_iterations:
                break

        return ExperimentReport(source, target, mechanism, evidence)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment(
    source: str,
    target: str,
    mechanism: str,
    **kwargs,
) -> ExperimentReport:
    """Convenience wrapper – runs the orchestrator synchronously."""
    orchestrator = OrchestratorAgent(**kwargs)
    return orchestrator.run(source, target, mechanism)


# If executed directly, run a tiny smoke test
if __name__ == "__main__":
    TEST_REPORT = run_experiment(
        source="UV‑B exposure",
        target="Vitamin D serum levels",
        mechanism="Cutaneous synthesis of vitamin D3 (cholecalciferol)",
        max_iterations=1,  # keep cheap for demo
        exploration_k=3,
    )
    print(TEST_REPORT.summary)

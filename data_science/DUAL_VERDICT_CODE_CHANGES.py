"""
DUAL VERDICT IMPLEMENTATION - CODE CHANGES
==========================================

Apply these changes to pydantic_ai_deepresearch_orchestration.py

STEP 1: Replace EvidenceJudgment model (around line 175)
"""

# REPLACE THIS:
"""
class EvidenceJudgment(BaseModel):
    '''Represents a judgment about whether a causal claim is supported by evidence.'''
    supported: bool = Field(description="Whether the claim is supported by the evidence")
    confidence: int = Field(description="Confidence level in this judgment (1-10)")
    reasoning: str = Field(description="Detailed reasoning for the judgment")
    suggested_modification: Optional[str] = Field(default=None, description="Suggested modification to the original claim if needed")
"""

# WITH THIS:
class EvidenceJudgment(BaseModel):
    """Enhanced judgment with separate causal and correlational verdicts."""
    
    # CAUSAL EVIDENCE VERDICT (experimental studies)
    causal_verdict: str = Field(
        description="Verdict based ONLY on causal evidence (RCTs, experiments, IVs, RDD, natural experiments): supported, partially_supported, or unsupported"
    )
    causal_confidence: int = Field(
        description="Confidence in causal verdict (1-10)",
        ge=1, le=10
    )
    causal_reasoning: str = Field(
        description="Reasoning for causal verdict - which experimental studies were found and what they showed"
    )
    
    # CORRELATIONAL EVIDENCE VERDICT (observational studies)
    correlational_verdict: str = Field(
        description="Verdict based on observational/correlational evidence (cohort, cross-sectional, case-control, surveys): supported, partially_supported, or unsupported"
    )
    correlational_confidence: int = Field(
        description="Confidence in correlational verdict (1-10)",
        ge=1, le=10
    )
    correlational_reasoning: str = Field(
        description="Reasoning for correlational verdict - which observational studies were found and what they showed"
    )
    
    # OVERALL SYNTHESIS
    overall_verdict: str = Field(
        description="Overall verdict synthesizing both causal and correlational evidence: supported, partially_supported, or unsupported"
    )
    overall_confidence: int = Field(
        description="Overall confidence (1-10)",
        ge=1, le=10
    )
    overall_reasoning: str = Field(
        description="Synthesis of both evidence types, explaining how they relate and which is more convincing"
    )
    
    # LEGACY FIELDS (for backward compatibility)
    supported: bool = Field(
        description="DEPRECATED: Use overall_verdict instead. Whether the claim is supported by the evidence"
    )
    confidence: int = Field(
        description="DEPRECATED: Use overall_confidence instead. Confidence level in this judgment (1-10)"
    )
    reasoning: str = Field(
        description="DEPRECATED: Use overall_reasoning instead. Detailed reasoning for the judgment"
    )
    
    suggested_modification: Optional[str] = Field(
        default=None, 
        description="Suggested modification to the original claim if needed"
    )


"""
STEP 2: Update FinalVerdict model (around line 205)
Add these fields to the existing class:
"""

# ADD TO FinalVerdict class:
    # DUAL VERDICT DETAILS
    causal_verdict: Optional[str] = Field(
        default=None,
        description="Verdict based on causal evidence: supported, partially_supported, or unsupported"
    )
    causal_confidence: Optional[int] = Field(
        default=None,
        description="Confidence in causal verdict (1-10)"
    )
    causal_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning for causal verdict"
    )
    
    correlational_verdict: Optional[str] = Field(
        default=None,
        description="Verdict based on correlational evidence: supported, partially_supported, or unsupported"
    )
    correlational_confidence: Optional[int] = Field(
        default=None,
        description="Confidence in correlational verdict (1-10)"
    )
    correlational_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning for correlational verdict"
    )


"""
STEP 3: Update IterationHistory model (around line 196)
Add these fields to the existing class:
"""

# ADD TO IterationHistory class:
    # Dual verdict details
    causal_verdict: Optional[str] = Field(default=None, description="Causal evidence verdict")
    causal_confidence: Optional[int] = Field(default=None, description="Confidence in causal verdict")
    correlational_verdict: Optional[str] = Field(default=None, description="Correlational evidence verdict")
    correlational_confidence: Optional[int] = Field(default=None, description="Confidence in correlational verdict")


"""
STEP 4: Replace judgment_agent system prompt (around line 534)
"""

# REPLACE THIS:
"""
judgment_agent = Agent(
    'openai:gpt-5',
    deps_type=ResearchContext,
    output_type=EvidenceJudgment,
    system_prompt=(
        "You are a scientific judge evaluating if a causal claim is supported by evidence. "
        "Given a causal claim (A causes B through mechanism X) and multiple evidence sources "
        "with quality scores, determine whether the evidence collectively supports, partially "
        "supports, or contradicts the claim. Consider both the quantity and quality of evidence. "
        "If modifications to the claim would make it better align with evidence, suggest specific "
        "revisions. Be fair but critical, and acknowledge limitations in the evidence base."
    )
)
"""

# WITH THIS:
judgment_agent = Agent(
    'openai:gpt-5',
    deps_type=ResearchContext,
    output_type=EvidenceJudgment,
    system_prompt=(
        "You are a scientific judge evaluating causal claims using a DUAL VERDICT approach.\n\n"
        
        "You must provide THREE SEPARATE VERDICTS:\n\n"
        
        "1. CAUSAL VERDICT (causal_verdict, causal_confidence, causal_reasoning):\n"
        "   - Based ONLY on EXPERIMENTAL/CAUSAL studies:\n"
        "     • Randomized Controlled Trials (RCTs)\n"
        "     • Natural experiments\n"
        "     • Instrumental variable analysis\n"
        "     • Regression discontinuity designs\n"
        "     • Experimental manipulations\n"
        "   - Verdict: 'supported', 'partially_supported', or 'unsupported'\n"
        "   - Explain: Which causal studies were found and what they demonstrated\n\n"
        
        "2. CORRELATIONAL VERDICT (correlational_verdict, correlational_confidence, correlational_reasoning):\n"
        "   - Based ONLY on OBSERVATIONAL studies:\n"
        "     • Cross-sectional studies\n"
        "     • Cohort studies (prospective or retrospective)\n"
        "     • Case-control studies\n"
        "     • Surveys and questionnaires\n"
        "     • Ecological studies\n"
        "   - Verdict: 'supported', 'partially_supported', or 'unsupported'\n"
        "   - Explain: Which observational studies were found and what patterns they showed\n\n"
        
        "3. OVERALL VERDICT (overall_verdict, overall_confidence, overall_reasoning):\n"
        "   - Synthesis of BOTH evidence types\n"
        "   - Weight experimental evidence more heavily than observational\n"
        "   - Consider: quality, quantity, consistency, and effect sizes\n"
        "   - Explain: How causal and correlational evidence relate, which is more convincing, and overall conclusion\n\n"
        
        "IMPORTANT RULES:\n"
        "• If NO experimental studies found → causal_verdict MUST be 'unsupported'\n"
        "• If NO observational studies found → correlational_verdict MUST be 'unsupported'\n"
        "• overall_verdict should prioritize causal evidence when available\n"
        "• Be explicit about which studies fall into which category\n"
        "• Consider study quality within each category (well-designed RCT > poorly-designed RCT)\n\n"
        
        "For backward compatibility, also populate:\n"
        "• supported: bool (True if overall_verdict is 'supported')\n"
        "• confidence: int (same as overall_confidence)\n"
        "• reasoning: str (same as overall_reasoning)\n\n"
        
        "If modifications to the claim would better align with evidence, suggest specific revisions in 'suggested_modification'."
    )
)


"""
STEP 5: Replace _judge_evidence method (around line 1159)
"""

# REPLACE THIS METHOD:
"""
async def _judge_evidence(
    self, 
    ctx: ResearchContext, 
    evidence_list: List[SearchResult], 
    scores: List[EvidenceScore]
) -> EvidenceJudgment:
    '''Judge whether evidence supports the claim.'''
    self.console.print("[bold]Judging evidence support...[/bold]")
    
    combined_evidence = "\\n\\n".join([
        f"EVIDENCE {i+1}:\\n{ev.content}\\n\\nSCORE: Relevance {sc.relevance_score}/10, Quality {sc.quality_score}/10"
        for i, (ev, sc) in enumerate(zip(evidence_list, scores))
    ])
    input_text = f"CAUSAL CLAIM: {ctx.claim}\\n\\n{combined_evidence}"
    
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
"""

# WITH THIS:
async def _judge_evidence(
    self, 
    ctx: ResearchContext, 
    evidence_list: List[SearchResult], 
    scores: List[EvidenceScore]
) -> EvidenceJudgment:
    """Judge whether evidence supports the claim using dual verdict approach."""
    self.console.print("[bold]Judging evidence support (dual verdict: causal + correlational)...[/bold]")
    
    combined_evidence = "\n\n".join([
        f"EVIDENCE {i+1}:\n{ev.content}\n\nSCORE: Relevance {sc.relevance_score}/10, Quality {sc.quality_score}/10"
        for i, (ev, sc) in enumerate(zip(evidence_list, scores))
    ])
    input_text = f"CAUSAL CLAIM: {ctx.claim}\n\n{combined_evidence}"
    
    logger.info(f"  🔄 HANDOFF → judgment_agent (DUAL VERDICT): Judging {len(evidence_list)} pieces of evidence for claim: '{ctx.claim[:60]}...'")
    
    result = await judgment_agent.run(input_text, deps=ctx)
    judgment = result.output
    
    # Log dual verdicts
    logger.info(f"  ✅ HANDOFF ← judgment_agent (DUAL VERDICT):")
    logger.info(f"      Causal: {judgment.causal_verdict.upper()}, Conf={judgment.causal_confidence}/10")
    logger.info(f"      Correlational: {judgment.correlational_verdict.upper()}, Conf={judgment.correlational_confidence}/10")
    logger.info(f"      Overall: {judgment.overall_verdict.upper()}, Conf={judgment.overall_confidence}/10")
    
    self._log_agent_call(
        "judgment_agent",
        f"{len(evidence_list)} evidence pieces",
        f"Causal={judgment.causal_verdict}, Corr={judgment.correlational_verdict}, Overall={judgment.overall_verdict}",
        result
    )
    
    # Display in console with formatting
    self.console.print(f"[cyan]═══════════════════════════════════════════════════[/cyan]")
    self.console.print(f"[bold]DUAL VERDICT RESULTS:[/bold]")
    self.console.print(f"  [bold cyan]Causal Evidence:[/bold cyan] {judgment.causal_verdict.upper()} (Confidence: {judgment.causal_confidence}/10)")
    self.console.print(f"    {judgment.causal_reasoning[:150]}...")
    self.console.print(f"  [bold yellow]Correlational Evidence:[/bold yellow] {judgment.correlational_verdict.upper()} (Confidence: {judgment.correlational_confidence}/10)")
    self.console.print(f"    {judgment.correlational_reasoning[:150]}...")
    self.console.print(f"  [bold green]Overall:[/bold green] {judgment.overall_verdict.upper()} (Confidence: {judgment.overall_confidence}/10)")
    self.console.print(f"[cyan]═══════════════════════════════════════════════════[/cyan]")
    
    return judgment


"""
STEP 6: Update _synthesize_verdict method (around line 1320)
Find this section and add after the verdict_agent call:
"""

# FIND THIS:
"""
result = await verdict_agent.run(prompt, deps=ctx)
verdict = result.output

logger.info(f"  ✅ HANDOFF ← verdict_agent: Verdict={verdict.verdict.upper()}, Confidence={verdict.confidence}/10")
"""

# ADD AFTER IT:
# Populate dual verdict fields from judgment
if judgment:
    verdict.causal_verdict = judgment.causal_verdict
    verdict.causal_confidence = judgment.causal_confidence
    verdict.causal_reasoning = judgment.causal_reasoning
    verdict.correlational_verdict = judgment.correlational_verdict
    verdict.correlational_confidence = judgment.correlational_confidence
    verdict.correlational_reasoning = judgment.correlational_reasoning


"""
STEP 7: Update iteration history recording in run() method (around line 934)
"""

# REPLACE THIS:
"""
iteration_record = IterationHistory(
    iteration_number=iteration,
    claim_tested=current_claim,
    evidence_count=len(all_evidence),
    high_quality_count=high_quality_count,
    judgment=judgment.overall_verdict if judgment else None,
    judgment_confidence=judgment.overall_confidence if judgment else None,
    suggested_modification=judgment.suggested_modification if judgment else None
)
"""

# WITH THIS:
iteration_record = IterationHistory(
    iteration_number=iteration,
    claim_tested=current_claim,
    evidence_count=len(all_evidence),
    high_quality_count=high_quality_count,
    judgment=judgment.overall_verdict if judgment else None,
    judgment_confidence=judgment.overall_confidence if judgment else None,
    causal_verdict=judgment.causal_verdict if judgment else None,
    causal_confidence=judgment.causal_confidence if judgment else None,
    correlational_verdict=judgment.correlational_verdict if judgment else None,
    correlational_confidence=judgment.correlational_confidence if judgment else None,
    suggested_modification=judgment.suggested_modification if judgment else None
)


"""
STEP 8: Update test_deep_research_ground_truth.py 
Add dual verdict fields to result dictionary (around line 90)
"""

# FIND THE result DICT AND ADD THESE FIELDS:
result = {
    "CLD": edge['CLD'],
    "classification": edge['Classification'],
    "source": edge['Source'],
    "target": edge['Target'],
    "verdict": verdict.verdict,
    "confidence": verdict.confidence,
    
    # ADD THESE:
    "causal_verdict": verdict.causal_verdict,
    "causal_confidence": verdict.causal_confidence,
    "causal_reasoning": verdict.causal_reasoning,
    "correlational_verdict": verdict.correlational_verdict,
    "correlational_confidence": verdict.correlational_confidence,
    "correlational_reasoning": verdict.correlational_reasoning,
    
    # ... rest of existing fields ...
    "direct_causal_found": verdict.direct_causal_evidence_found or False,
    # etc...
}


"""
TESTING COMMANDS
================
"""

# Test 1: Single edge smoke test
# cd /home/nitai/code/causalix.ai/data_science
# python test_deep_research_ground_truth.py

# Modify test_deep_research_ground_truth.py to add this at the top of run_all_tests():
"""
# SMOKE TEST MODE - Remove after testing
edges = edges[:1]  # Test only first edge
console.print(f"[yellow]⚠️  SMOKE TEST MODE: Testing only {len(edges)} edge[/yellow]")
"""

# Test 2: Check JSON output has dual verdicts
# cat deep_research_results_*.json | grep -A 5 "causal_verdict"

# Test 3: Full run (remove smoke test mode first)
# python test_deep_research_ground_truth.py

print("""
IMPLEMENTATION CHECKLIST:
========================
□ Step 1: Update EvidenceJudgment model
□ Step 2: Add fields to FinalVerdict model  
□ Step 3: Add fields to IterationHistory model
□ Step 4: Update judgment_agent prompt
□ Step 5: Replace _judge_evidence method
□ Step 6: Update _synthesize_verdict method
□ Step 7: Update iteration history recording
□ Step 8: Update test script result dict
□ Step 9: Run smoke test (1 edge)
□ Step 10: Verify JSON output
□ Step 11: Run full test (107 edges)
""")





















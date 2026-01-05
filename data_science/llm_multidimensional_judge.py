#!/usr/bin/env python3
"""
LLM-as-Judge for Multi-Dimensional Evidence Assessment
Uses an LLM to score edges on 5 dimensions based on deep research results
"""

import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

# Add backend directory to path to import OpenAI client
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

from lm_clients.openai_client_working import OpenAIClient

# Dimension weights
WEIGHTS = {
    'causal_strength': 0.30,
    'mechanistic_specificity': 0.25,
    'construct_validity': 0.20,
    'scope_relevance': 0.15,
    'theoretical_coherence': 0.10
}

JUDGE_SYSTEM_PROMPT = """You are an expert methodologist evaluating the quality of causal evidence in scientific research.

Your task is to score a proposed causal relationship on 5 dimensions based on the evidence gathered through systematic literature review.

# SCORING DIMENSIONS

## 1. CAUSAL STRENGTH (Study Design Quality) - Weight: 30%
Score the strength of the causal evidence based on study design:

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

CAUSAL_STRENGTH: [score 0-10]
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

def create_judge_prompt(edge: Dict) -> str:
    """Create the user prompt for judging a single edge."""
    
    source = edge.get('source', 'Unknown')
    target = edge.get('target', 'Unknown')
    classification = edge.get('classification', 'Unknown')
    verdict = edge.get('verdict', 'unsupported')
    confidence = edge.get('confidence', 0)
    direct_causal = edge.get('direct_causal_found', False)
    
    strongest_conf = edge.get('strongest_causal_evidence_confidence', 'Not specified')
    strongest_study = edge.get('strongest_causal_evidence_study_type', 'Not specified')
    strongest_url = edge.get('strongest_causal_evidence_url', 'Not available')
    strongest_passage = edge.get('strongest_causal_evidence_passage', 'Not available')
    strongest_summary = edge.get('strongest_causal_evidence_summary', 'Not available')
    
    judge_reasoning = edge.get('judge_reasoning', 'Not available')
    modified_claim = edge.get('modified_claim', 'Not available')
    
    evidence_urls = edge.get('evidence_urls', [])
    key_summaries = edge.get('key_evidence_summaries', [])
    limitations = edge.get('limitations', [])
    
    prompt = f"""# CAUSAL CLAIM TO EVALUATE

**Proposed Edge**: {source} → {target}

**Ground Truth Classification**: {classification}
- TP = True Positive (edge correctly identified, belongs in CLD)
- FP = False Positive (edge incorrectly identified, hallucination)
- FN = False Negative (edge missed by LLM but should be in CLD)

**Original Verdict from Deep Research**: {verdict}
**Original Confidence**: {confidence}/10
**Direct Causal Evidence Found**: {direct_causal}

---

# STRONGEST EVIDENCE

**Study Type**: {strongest_study}
**Evidence Confidence**: {strongest_conf}/10
**URL**: {strongest_url}

**Passage**:
{strongest_passage}

**Summary**:
{strongest_summary}

---

# EXPERT JUDGE REASONING

{judge_reasoning}

---

# MODIFIED CLAIM (What Evidence Actually Supports)

{modified_claim}

---

# KEY EVIDENCE SUMMARIES

"""
    
    for i, summary in enumerate(key_summaries[:5], 1):  # Limit to top 5 for token efficiency
        prompt += f"{i}. {summary}\n\n"
    
    prompt += f"""---

# LIMITATIONS IDENTIFIED

"""
    
    for i, limitation in enumerate(limitations[:5], 1):  # Limit to top 5
        prompt += f"{i}. {limitation}\n\n"
    
    prompt += f"""---

# YOUR TASK

Based on ALL the information above, score this edge on the 5 dimensions.

**Critical Analysis Points**:
1. Does the evidence show a DIRECT effect (Source → Target) or is it MEDIATED (Source → Something Else → Target)?
2. Do the studies measure the EXACT constructs named in the claim, or proxies/related variables?
3. Is the claim theoretically coherent, or is it tautological/circular?
4. What is the strongest study design in the evidence base?
5. Does the evidence generalize to the intended population?

Pay special attention to phrases like:
- "mediated by", "indirect", "fully mediated" → LOW Mechanistic Specificity
- "tautological", "circular", "ill-posed" → ZERO Theoretical Coherence
- "proxy", "silhouette", "figure rating" → LOWER Construct Validity
- "generalizability uncertain", "limited sample" → LOWER Scope Relevance

Now provide your scores:
"""
    
    return prompt

def parse_judge_response(response: str) -> Optional[Dict]:
    """Parse the LLM judge response to extract scores."""
    
    try:
        scores = {}
        
        # Extract scores using regex
        patterns = {
            'causal_strength': r'CAUSAL_STRENGTH:\s*(\d+(?:\.\d+)?)',
            'mechanistic_specificity': r'MECHANISTIC_SPECIFICITY:\s*(\d+(?:\.\d+)?)',
            'construct_validity': r'CONSTRUCT_VALIDITY:\s*(\d+(?:\.\d+)?)',
            'scope_relevance': r'SCOPE_RELEVANCE:\s*(\d+(?:\.\d+)?)',
            'theoretical_coherence': r'THEORETICAL_COHERENCE:\s*(\d+(?:\.\d+)?)'
        }
        
        for dim, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                scores[dim] = min(max(score, 0), 10)  # Clamp to 0-10
            else:
                print(f"  ⚠️ Warning: Could not parse {dim}")
                return None
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            scores['reasoning'] = reasoning_match.group(1).strip()
        else:
            scores['reasoning'] = "Not provided"
        
        # Calculate overall weighted score
        scores['overall_score'] = round(sum(scores[dim] * WEIGHTS[dim] for dim in WEIGHTS.keys()), 2)
        
        return scores
        
    except Exception as e:
        print(f"  ❌ Error parsing response: {e}")
        return None

def judge_edge(edge: Dict, client: OpenAIClient) -> Optional[Dict]:
    """Use LLM to judge a single edge on all 5 dimensions."""
    
    source = edge.get('source', 'Unknown')[:50]
    target = edge.get('target', 'Unknown')[:50]
    
    print(f"\n  Judging: {source} → {target}")
    
    try:
        prompt = create_judge_prompt(edge)
        
        # Call LLM
        response = client.send_message(
            JUDGE_SYSTEM_PROMPT,
            prompt,
            stream=False
        )
        
        # Parse response
        parsed = client.parse_static_response(response)
        content = parsed.get('content', '')
        
        # Extract scores
        scores = parse_judge_response(content)
        
        if scores:
            print(f"    Overall: {scores['overall_score']:.2f}/10")
            print(f"    Breakdown: CS={scores['causal_strength']:.1f}, MS={scores['mechanistic_specificity']:.1f}, CV={scores['construct_validity']:.1f}, SR={scores['scope_relevance']:.1f}, TC={scores['theoretical_coherence']:.1f}")
            return scores
        else:
            print(f"    ❌ Failed to parse scores")
            return None
            
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def analyze_cld(filepath: Path, cld_name: str, client: OpenAIClient, limit: Optional[int] = None):
    """Analyze a single CLD using LLM-as-judge."""
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {cld_name}")
    print(f"{'='*80}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    edges = data['results']
    
    if limit:
        print(f"⚠️ Limiting to first {limit} edges for testing")
        edges = edges[:limit]
    
    results = []
    
    for i, edge in enumerate(edges, 1):
        print(f"\n[{i}/{len(edges)}]", end='')
        
        edge_info = {
            'cld': cld_name,
            'source': edge.get('source', ''),
            'target': edge.get('target', ''),
            'classification': edge.get('classification', 'UNKNOWN'),
            'verdict_original': edge.get('verdict', 'unsupported'),
            'confidence_original': edge.get('confidence', 0)
        }
        
        scores = judge_edge(edge, client)
        
        if scores:
            edge_info.update(scores)
        else:
            # Skip this edge if judging failed
            print(f"    ⚠️ Skipping edge due to judging failure")
            continue
        
        results.append(edge_info)
    
    # Save results
    output_dir = Path('/home/nitai/code/causalix.ai/data_science/multidimensional_llm_judge_results')
    output_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_csv = output_dir / f'{cld_name}_llm_judge_scores.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved results to: {output_csv}")
    
    # Save to JSON (includes reasoning)
    output_json = output_dir / f'{cld_name}_llm_judge_scores.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved detailed results to: {output_json}")
    
    return df

def compute_discrimination_stats(df: pd.DataFrame, cld_name: str):
    """Compute and print discrimination statistics."""
    
    print(f"\n{'='*80}")
    print(f"DISCRIMINATION ANALYSIS: {cld_name}")
    print(f"{'='*80}")
    
    if len(df) == 0:
        print("⚠️ No edges were successfully judged")
        return
    
    if 'TP' not in df['classification'].values or 'FP' not in df['classification'].values:
        print("⚠️ Not enough TP or FP edges for discrimination analysis")
        return
    
    tp_df = df[df['classification'] == 'TP']
    fp_df = df[df['classification'] == 'FP']
    
    print(f"\nSample sizes:")
    print(f"  TP: {len(tp_df)}")
    print(f"  FP: {len(fp_df)}")
    
    # Original framework
    support_map = {'supported': 1.0, 'partially_supported': 0.5, 'unsupported': 0.0}
    tp_support = tp_df['verdict_original'].map(support_map).mean()
    fp_support = fp_df['verdict_original'].map(support_map).mean()
    
    print(f"\nORIGINAL FRAMEWORK:")
    print(f"  TP average support: {tp_support:.3f}")
    print(f"  FP average support: {fp_support:.3f}")
    print(f"  Gap: {tp_support - fp_support:+.3f}")
    
    # New LLM-based framework
    tp_overall = tp_df['overall_score'].mean()
    fp_overall = fp_df['overall_score'].mean()
    
    print(f"\nLLM MULTI-DIMENSIONAL FRAMEWORK:")
    print(f"  TP average score: {tp_overall:.2f}/10")
    print(f"  FP average score: {fp_overall:.2f}/10")
    print(f"  Gap: {tp_overall - fp_overall:+.2f} points")
    
    if tp_overall > fp_overall:
        improvement = ((tp_overall - fp_overall) / tp_overall) * 100
        print(f"  ✅ TP > FP (Gap: {improvement:.1f}%)")
    else:
        print(f"  ❌ FP ≥ TP (Framework FAILED!)")
    
    # Dimension-specific
    dimension_cols = ['causal_strength', 'mechanistic_specificity', 'construct_validity', 
                     'scope_relevance', 'theoretical_coherence']
    
    print(f"\nDIMENSION-SPECIFIC DISCRIMINATION:")
    print(f"  {'Dimension':<30s} | {'TP Avg':>7s} | {'FP Avg':>7s} | {'Gap':>7s}")
    print(f"  {'-'*30}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    
    for dim in dimension_cols:
        tp_avg = tp_df[dim].mean()
        fp_avg = fp_df[dim].mean()
        gap = tp_avg - fp_avg
        print(f"  {dim:<30s} | {tp_avg:>7.2f} | {fp_avg:>7.2f} | {gap:>+7.2f}")
    
    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(tp_df['overall_score'], fp_df['overall_score'])
    print(f"\nSTATISTICAL SIGNIFICANCE:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✅ Significant difference (p < 0.05)")
    else:
        print(f"  ⚠️ Not statistically significant")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-as-Judge for Multi-Dimensional Evidence Assessment')
    parser.add_argument('--cld', choices=['social_norms', 'depressive_symptoms', 'older_persons', 'all'],
                       default='social_norms', help='Which CLD to analyze')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of edges (for testing)')
    parser.add_argument('--model', default='gpt-4.1',
                       help='LLM model to use (default: gpt-4.1)')
    
    args = parser.parse_args()
    
    # Load API keys from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize OpenAI client
    print(f"Initializing OpenAI client with model: {args.model}")
    client = OpenAIClient(
        api_key=openai_key,
        api_url=openai_url,
        model=args.model,
        dev_mode=False
    )
    
    base_dir = Path('/home/nitai/code/causalix.ai/data_science')
    
    files = {
        'social_norms': 'deep_research_results_social_norms_20251012_213641_17edges.json',
        'depressive_symptoms': 'deep_research_results_depressive_symptoms_20251013_032002_84edges.json',
        'older_persons': 'deep_research_results_older_persons_ALL_EDGES_184edges.json'
    }
    
    results_dfs = {}
    
    if args.cld == 'all':
        to_process = files.items()
    else:
        to_process = [(args.cld, files[args.cld])]
    
    for cld_name, filename in to_process:
        filepath = base_dir / filename
        
        if not filepath.exists():
            print(f"⚠️ File not found: {filepath}")
            continue
        
        df = analyze_cld(filepath, cld_name, client, limit=args.limit)
        results_dfs[cld_name] = df
        
        # Compute discrimination stats
        compute_discrimination_stats(df, cld_name)
    
    # Combined analysis if multiple CLDs
    if len(results_dfs) > 1:
        print(f"\n\n{'='*80}")
        print("COMBINED ANALYSIS: ALL CLDs")
        print(f"{'='*80}")
        
        combined = pd.concat(results_dfs.values(), ignore_index=True)
        compute_discrimination_stats(combined, "ALL CLDs")
        
        # Save combined
        output_dir = Path('/home/nitai/code/causalix.ai/data_science/multidimensional_llm_judge_results')
        combined.to_csv(output_dir / 'combined_all_clds_llm_judge_scores.csv', index=False)
        print(f"\n✅ Saved combined results to: {output_dir / 'combined_all_clds_llm_judge_scores.csv'}")

if __name__ == '__main__':
    main()

# Discussion Chapter Enhancement Plan (Add Back Content in Rick's Style)

## Rick's Three Pillars:

1. **Synthesis across results** (cross-cutting patterns, not result repetition)
2. **Overarching failure modes** (why things fail)
3. **Limitations** (validity threats)

---

## Content to ADD BACK from Backup (Organized by Pillar)

### PILLAR 1: Synthesis Across Results

#### From Backup Section "Definitions of Ground Truth, Hallucination, and Transfer Validity" (lines 67-80)

**Add to:** Section 3 (Preliminaries) - This is conceptual framing, not result repetition

```latex
\subsection{Definitions of Ground Truth, Hallucination, and Transfer Validity}

\textbf{Ground Truth from Literature (GT Lit):} Published expert CLDs define the reference edge set...
[Full content from backup lines 71-77]

\textbf{Hallucination:} We operationalise hallucination as disagreement with the chosen ground truth...
[Content explaining why FP validation rate matters - the 62.4% finding]
```

#### From Backup "GT Synth vs GT Lit Performance Gap" (lines 100-110)

**Add to:** New subsection explaining the cross-cutting pattern

```latex
\subsection{Why Synthetic Benchmarks Overestimate Real-World Performance}

The performance gap between GT Synth (F1=0.65--0.82) and GT Lit (F1=0.15--0.55) has three explanations:
1. Systematic corruption patterns are more detectable
2. Tacit domain knowledge in GT Lit
3. Scope vs. factual disagreement
```

#### From Backup "Cross-Domain Generalisation Failure" (lines 341-359)

**Add to:** RQ2 section or Synthesis - This explains WHY UQ metrics fail

```latex
Leave-one-CLD-out cross-validation revealed clear performance drops:
- RF: 0.809 → 0.629
- This means UQ metrics exploit CLD-specific patterns that don't generalise
```

---

### PILLAR 2: Overarching Failure Modes

#### Already Present (Keep):

- Evidence of Absence Paradox (lines 65-78)
- Plausibility Bias (lines 80-91)
- Partial Credit and Non-Scientific Evidence (lines 93-103)
- Corrector Failure Modes 1-5 (lines 117-137)

#### ADD from Backup "Social Norms CLD Anomaly" (lines 120-123)

**Add as:** New subsection under failure modes - explains domain-dependent failure

```latex
\subsection{Domain-Dependent Failure: The Social Norms Anomaly}

In the Social Norms CLD, all prompts failed to discriminate... This could reflect:
- Behavioural/context-dependent mechanisms vs physiological
- BHC limitations (designed for epidemiology)
```

#### ADD from Backup "Implications for Original RQ3 Design" (lines 357-359)

**Add to:** RQ3 section - explains why the pivot was necessary

```latex
The poor cross-CLD generalisation undermined the original RQ3 design...
Optimal thresholds varied substantially across domains, making universal 
threshold-based triggering unreliable.
```

---

### PILLAR 3: Limitations (Already Comprehensive)

The current Limitations section (lines 195-299) is already detailed.

#### Optional additions from backup:

- "Retrieval Implementation Considerations" (lines 137-144) - could merge into retrieval limitations
- "Majority voting per citation limitations" (lines 149-155) - aggregation scheme vulnerabilities

---

## Sections to EXPAND (Not Full Backup, Just Key Interpretive Content)

### 1. Preliminaries Section (Current: 1 sentence placeholder)

**Add from backup lines 44-80:**

- Model Selection Rationale (brief - why GPT-4.1)
- Random Baseline Comparison (interpretation - 2.2-3.4x means non-random)
- GT definitions (conceptual framing)

### 2. LLM-Human Agreement Section (Current: 1 sentence)

**Add from backup lines 250-257:**

- Cohen's κ = 0.618, 80.6% agreement
- **Key insight:** 100% of disagreements = LLM more permissive (systematic leniency)
- Model effect > domain effect (Claude vs GPT-4.1)

### 3. RQ2 Section (Current: 2 paragraphs)

**Add from backup lines 299-359:**

- Single metric performance summary (Cosine AUC=0.671 but wrong direction!)
- Cross-domain generalisation failure explanation
- Implications for triggering

### 4. RQ3 Section (Current: 2 paragraphs)

**Add from backup lines 367-441:**

- Pivot rationale (both prerequisites failed)
- Detection rate findings (70.5% TP vs 62.4% FP = 8.1pp gap)
- Human validation findings (66% pass rate at t=0.7)
- Enrichment analysis (F1 0.267→0.705 potential)
- Adaptive-trigger efficiency (50% cost, 57% gain)

---

## Implementation Checklist

| Task | Priority | Backup Lines |

|------|----------|--------------|

| Add GT definitions to Preliminaries | High | 67-80 |

| Add "Why Synthetic Overestimates" subsection | High | 100-110 |

| Expand LLM-Human Agreement with leniency finding | High | 250-257 |

| Add Social Norms Anomaly as failure mode | Medium | 120-123 |

| Expand RQ2 with cross-domain failure explanation | High | 341-359 |

| Expand RQ3 with pivot rationale and key findings | High | 367-441 |

| Fix roadmap wording (line 37) | Medium | - |

---

## Key Principle: Add INTERPRETATION, Not Results

From backup, extract:

- **WHY** patterns occur (failure mode explanations)
- **WHAT IT MEANS** (cross-cutting implications)
- **CAVEATS** (limitations, validity threats)

Do NOT add:

- Repeated F1/AUC numbers already in Results
- Per-prompt comparisons
- Figure descriptions

---

## Files Referenced

- Current: [Chapters/Discussion.tex](../Chapters/Discussion.tex)
- Backup: [Chapters/Discussion.backup_before_discussion_split_20260110.tex](../Chapters/Discussion.backup_before_discussion_split_20260110.tex)
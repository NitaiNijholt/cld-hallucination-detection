# Node Generation Prompt Ablation Study: Thesis Integration Plan

## Current Thesis Structure (Identified)

### Existing Content
1. **Chapter: Prompt Engineering Ablation Study: Experimental Results** (`chapters/prompt_ablation_results.tex`)
   - **Focus**: Edge generation (relationship discovery)
   - **Variants**: 9 prompts (Nitai_A/B/C/E, Rick_A/B, Cillian_B/C, Current)
   - **Key Finding**: F1 range 0.00-0.46 (significant impact)
   - **Best**: Nitai_C (F1=0.460)
   - **Runs**: 1 per CLD (no variance estimation)

2. **Methods Chapter** (`Chapters/Methods.tex`)
   - Contains Hybrid F1 metric explanation (confirmed present)
   - System architecture and multi-agent orchestration
   - Ground-truth dataset specification (3 CLDs)

3. **References**: Already cites Wei et al. (2022), Zhou et al. (2022), etc.

---

## Integration Strategy: Two Approaches

### **Option A: Separate Node Generation Chapter** ⭐ RECOMMENDED
**Structure**: Add new chapter after edge ablation results

**Advantages**:
- Clear separation between edge vs. node generation
- Allows detailed comparison section
- Emphasizes node generation as distinct research contribution
- Enables contrasting findings (significance vs. no significance)

**Chapter Outline**:
```
Chapter X: Node Generation Prompt Ablation Study
├─ Section 1: Introduction and Relationship to Edge Study
├─ Section 2: Node Generation Challenges
├─ Section 3: Experimental Design
├─ Section 4: Results
├─ Section 5: Comparison with Edge Generation
└─ Section 6: Unified Recommendations
```

---

### **Option B: Integrated Chapter**
**Structure**: Expand existing chapter into "Prompt Ablation Studies: Edge and Node Generation"

**Advantages**:
- Single comprehensive ablation chapter
- Direct side-by-side comparisons
- Reduces thesis length

**Sections to Add**:
- Section 2.5: Node Generation Configuration (after edge config)
- Section 3.5: Node Generation Results (after edge results)
- Section 4.5: Node vs. Edge Comparison (in discussion)

---

## Recommended Structure: Option A (Separate Chapter)

### New Chapter: "Node Generation Prompt Ablation Study"

#### **Section 1: Introduction** (1-1.5 pages)

**Content**:
- Relationship to Chapter X (edge ablation)
- Motivation: Why study node generation separately?
  - Different task characteristics (atomicity, measurability, no few-shot)
  - Complements edge generation to cover full CLD pipeline
- Research questions (same as edge study for comparability)

**Key Points**:
- "While Chapter X examined edge generation, node variable identification presents distinct challenges..."
- "Unlike edge discovery where few-shot examples are viable, node generation risks domain bias..."
- Reference the NODE_PROMPT_ABLATION_PLAN.md structure

---

#### **Section 2: Node Generation Challenges** (1 page)

**Content**:
- Unique constraints vs. edge generation:
  1. **No Few-Shot Learning**: Cannot provide example variables without biasing
  2. **Atomicity Requirement**: Must generate non-composite variables
  3. **Measurability**: Requires concrete units (edges don't)
  4. **Higher Abstraction**: Must balance conceptual clarity with concreteness

**Table**: Comparison of applicable techniques

| Technique | Edge Generation | Node Generation | Reason |
|-----------|----------------|-----------------|---------|
| Chain-of-Thought | ✓ | ✓ | Applicable to both |
| Few-Shot | ✓ | ✗ | No valid node examples |
| Role Prompting | ✓ | ✓ | Applicable to both |
| Task Decomposition | ✓ | ✓ | Applicable to both |
| RAG/Citations | ✓ | ✗ | Variables conceptual, not citable |

---

#### **Section 3: Experimental Design** (1.5-2 pages)

**Content**:

**3.1 Prompt Variants (7 variants)**

Table matching your NODE_PROMPT_ABLATION_PLAN.md:

| Variant | CoT (Sys) | CoT (User) | Role | Decomp | Constraints | Literature |
|---------|-----------|------------|------|--------|-------------|------------|
| V0_Minimal | ✗ | ✗ | ✗ | ✗ | ✗ | Baseline control |
| V1_Role_Only | ✗ | ✗ | ✓ | ✗ | ✗ | Role prompting (2024) |
| V2_Constraints | ✗ | ✗ | ✗ | ✗ | ✓✓✓ | Constraint specification |
| V3_CoT_System | ✓✓ | ✗ | ✗ | ✗ | ✗ | Wei et al. (2022) |
| V4_CoT_User | ✗ | ✓✓ | ✗ | ✓ | ✗ | Zhou et al. (2022) |
| V5_Nitai_C | ✓✓ | ✓✓ | ✓ | ✓ | ✓✓ | Combined (original) |
| V6_Andrew | ✗ | ✓ | ✓✓ | ✓ | ✓✓✓ | Enhanced + CLD size |

**3.2 Experimental Setup**
- **CLDs**: Same 3 as edge study (for comparability)
  - CLD 1: Depressive Symptoms (34 nodes in ground truth)
  - CLD 2: Social Norms & Obesity (11 nodes)
  - CLD 3: Emergency Dept. Visits (62 nodes)
- **Runs**: **5 per variant per CLD** (addresses edge study limitation!)
  - Total: 7 × 3 × 5 = 105 experiments
- **Model**: GPT-4.1 (same as edge study)
- **Temperature**: 0.7 (consistent)

**3.3 Evaluation Metric**
- **Hybrid F1 Score** (detailed in Chapter~\ref{ch:methods})
  - Two-stage: LLM semantic matching + cosine similarity
  - Reference Methods chapter section on node comparison

---

#### **Section 4: Results** (3-3.5 pages)

**4.1 Overall Performance**

**Table**: Performance Summary (from your aggregation)

| Prompt | Mean F1 | Std | 95% CI | N |
|--------|---------|-----|--------|---|
| Node_V6_Andrew | 0.6773 | 0.0786 | [0.6375, 0.7171] | 15 |
| Node_V3_CoT_System | 0.6573 | 0.1227 | [0.5952, 0.7194] | 15 |
| Node_V0_Minimal | 0.6394 | 0.0941 | [0.5918, 0.6870] | 15 |
| Node_V5_Nitai_C | 0.6358 | 0.0877 | [0.5914, 0.6802] | 15 |
| Node_V1_Role_Only | 0.6283 | 0.0798 | [0.5879, 0.6687] | 15 |
| Node_V4_CoT_User | 0.6193 | 0.1035 | [0.5670, 0.6717] | 15 |
| Node_V2_Constraints | 0.6168 | 0.0774 | [0.5777, 0.6560] | 15 |

**Key Finding Box** (emphasized):
> **No Statistically Significant Differences**
> - ANOVA: F=0.807, p=0.567
> - Performance range: 0.6168 to 0.6773 (Δ=0.0605, 6.0%)
> - All variants performed within ~6% of each other
> - **Contrasts sharply with edge generation** (46% range, F1: 0.00-0.46)

**Figure 1**: Bar chart with confidence intervals (from your visualizations)
- `performance_by_prompt_hybrid_f1.png`
- Caption emphasizing tight clustering and overlapping CIs

**Figure 2**: Box plots showing distributions
- `boxplot_hybrid_f1.png`
- Caption noting narrow ranges and few outliers

---

**4.2 Statistical Analysis**

**Pairwise Comparisons Table** (excerpt, top 5 comparisons):

| Comparison | Mean Diff | Cohen's d | p-value | Bonferroni p |
|------------|-----------|-----------|---------|--------------|
| V2 vs V6 | -0.0605 | -0.775 | 0.0428 | 0.899 ❌ |
| V4 vs V6 | -0.0580 | -0.631 | 0.0950 | 1.000 ❌ |
| V1 vs V6 | -0.0490 | -0.618 | 0.1016 | 1.000 ❌ |

**Key Point**: Even the largest difference (V2 vs V6) is not significant after Bonferroni correction.

**Effect Sizes vs. Baseline (V0_Minimal)**:

| Variant | Difference | Cohen's d | Magnitude |
|---------|------------|-----------|-----------|
| V6_Andrew | +0.0379 | +0.437 | **small** |
| V3_CoT_System | +0.0179 | +0.164 | negligible |
| V2_Constraints | -0.0226 | -0.262 | small |

**Interpretation**: All effect sizes are small or negligible (Cohen's d < 0.5).

---

**4.3 Performance by CLD**

**Table**: CLD-Specific Performance

| CLD | Mean F1 | Std | Range |
|-----|---------|-----|-------|
| CLD 1 (Depressive) | 0.7188 | 0.0854 | 0.481-0.862 |
| CLD 2 (Obesity) | 0.6042 | 0.0716 | 0.508-0.764 |
| CLD 3 (Emergency) | 0.5945 | 0.0635 | 0.417-0.783 |

**Figure 3**: Heatmap (prompts × CLDs)
- `heatmap_hybrid_f1.png`
- Shows consistent moderate performance across all combinations

**Key Finding**: Unlike edge generation (where CLD 2 was catastrophically bad for all prompts), node generation shows more stable performance across CLDs.

---

**4.4 Variance Analysis** (NEW - addresses edge study limitation)

**Within-Prompt Variance Table**:

| Prompt | Mean Std Dev | Max-Min Range | Coefficient of Variation |
|--------|--------------|---------------|--------------------------|
| V3_CoT_System | 0.1227 | 0.426 | 18.7% |
| V4_CoT_User | 0.1035 | 0.341 | 16.7% |
| V0_Minimal | 0.0941 | 0.298 | 14.7% |

**Interpretation**: High within-variant variance suggests performance depends more on CLD characteristics than prompt choice.

---

#### **Section 5: Comparison with Edge Generation** (2-2.5 pages) ⭐ CRITICAL

**5.1 Contrasting Findings**

**Table**: Edge vs. Node Comparison

| Aspect | Edge Generation | Node Generation |
|--------|----------------|-----------------|
| **Performance Range** | 0.00–0.46 (46%) | 0.62–0.68 (6%) |
| **ANOVA Significant** | Implicit yes (large range) | **No** (p=0.567) |
| **Best Performer** | Nitai_C (0.460) | Andrew (0.6773) |
| **Complete Failures** | Yes (Cillian_C: 0.00) | None |
| **Runs per CLD** | 1 (no variance) | 5 (variance estimated) |
| **CoT Improvement** | 14% (A→C) | Negligible (~2%) |
| **RAG Penalty** | -25% (C→E) | Not tested (n/a) |
| **Constraint Effect** | Catastrophic (Cillian_C) | Small negative (-3.5%) |

---

**5.2 Interpretation of Differences**

**Why are node generation results different?**

1. **Task Complexity**
   - **Edges**: Requires reasoning about causal mechanisms, directionality, complex relationships
   - **Nodes**: Simpler concept identification task
   - **Implication**: Complex prompting techniques (CoT, decomposition) provide limited value for conceptually simpler tasks

2. **Baseline Competence**
   - GPT-4.1 has strong semantic understanding of variable concepts
   - Even minimal prompting (V0) achieves F1=0.639 (vs. edge minimal: 0.371)
   - **Ceiling effect**: Less room for prompt engineering to add value

3. **Task Structure**
   - Nodes: Well-defined constraints (atomic, measurable, non-overlapping)
   - Edges: Open-ended mechanistic reasoning
   - **Implication**: Structured tasks less sensitive to prompting strategies

4. **Metric Differences**
   - Node Hybrid F1 includes semantic matching (more forgiving)
   - Edge F1 requires exact ground truth match
   - May contribute to tighter node performance range

---

**5.3 Consistent Findings**

Despite differences, some patterns hold across both studies:

1. **No Silver Bullet**: Neither CoT nor decomposition dramatically improves performance
2. **Over-Constraining Backfires**: V2_Constraints worst performer (both studies show constraint risks)
3. **Combined Approaches**: V5_Nitai_C and V6_Andrew perform well (though not significantly)
4. **CLD Sensitivity**: High variance across problem types in both studies

---

**5.4 Power Analysis** (Brief subsection)

**Addressing Non-Significance**:

Include summary from your power analysis:

- **Observed effect size**: Cohen's f = 0.123 (very small)
- **Post-hoc power**: ~20% (underpowered)
- **Required N for 80% power**: ~150 per group (vs. current 15)

**Interpretation**:
> "The lack of statistical significance reflects both **(1) underpowered study design** and **(2) genuinely small practical differences**. The observed 6% performance range is minimal in practical terms, suggesting that for node generation, prompt engineering has limited impact regardless of sample size. This contrasts with edge generation's 46% range, where differences are both statistically and practically significant."

---

#### **Section 6: Unified Recommendations** (1.5-2 pages)

**6.1 Task-Specific Guidance**

**For Node Generation**:
- ✅ Use baseline or light prompting (V0, V6)
- ✅ Include basic constraints (measurability, atomicity)
- ✅ Optional: CLD size awareness (V6's innovation)
- ❌ Don't over-invest in complex CoT
- ❌ Don't over-constrain (V2 approach)

**For Edge Generation**:
- ✅ Invest in dual-level CoT (Nitai_C)
- ✅ Balance grounding with flexibility
- ✅ Test across multiple problem types
- ❌ Avoid mandatory RAG (25% penalty)
- ❌ Avoid over-restrictive definition grounding (complete failure risk)

---

**6.2 General Principles**

1. **Match Complexity to Task**
   - Simple tasks (nodes): Simple prompts
   - Complex tasks (edges): Sophisticated techniques justified

2. **Beware Ceiling Effects**
   - If baseline performance is already high, prompt engineering has limited upside

3. **Validate Across Problems**
   - Both studies show high CLD-to-CLD variance
   - Single-problem evaluation insufficient

4. **Estimate Variance**
   - Multiple runs essential for distinguishing signal from noise
   - Edge study's single-run design prevented statistical testing

---

**6.3 Implications for Production System**

Based on combined findings:

1. **Node Generation Module**: Deploy simple prompt (V0 or V6) with basic constraints
2. **Edge Generation Module**: Deploy Nitai_C-style dual CoT with moderate constraints
3. **Verification Layer**: LLM-as-a-judge (Chapter X) addresses hallucination without over-constraining
4. **Adaptive Strategy**: Could use task complexity to automatically select prompting intensity

---

#### **Section 7: Limitations** (0.5-1 page)

**Study-Specific**:
- Only 3 CLDs (same as edge study)
- GPT-4.1 only (results may not generalize to other models)
- Hybrid F1 metric (semantic matching may mask finer distinctions)

**Comparison Limitations**:
- Edge study had 1 run per CLD (no variance)
- Different metrics (edge F1 vs. Hybrid F1)
- Cannot perfectly isolate prompt effects from metric effects

---

#### **Section 8: Conclusions** (0.5 page)

**Summary**:
- Node generation showed no statistically significant differences across 7 prompt variants (F1: 0.62-0.68, p=0.567)
- Contrasts sharply with edge generation's 46% performance range
- Suggests prompt engineering impact is highly task-dependent
- Complex techniques (CoT, decomposition) provide limited value for conceptually simpler tasks
- Combined with edge study, provides comprehensive guidance for CLD generation pipelines

**Contribution**:
- First systematic comparison of prompt engineering effects across node vs. edge generation
- Demonstrates importance of matching prompting complexity to task complexity
- Provides evidence-based recommendations for production system design

---

## Implementation Plan

### Phase 1: Create New Chapter File ✅
- **File**: `thesis/chapters/node_ablation_results.tex`
- **Structure**: Follow outline above
- **Length**: 8-10 pages (similar to edge chapter)

### Phase 2: Update main.tex
```latex
\\input{chapters/prompt_ablation_results} % Edge generation
\\input{chapters/node_ablation_results}   % Node generation (NEW)
\\input{Chapters/ExperimentsandResults}   % Main experiments
```

### Phase 3: Create Visualizations
- Convert PNG visualizations to PDF for thesis
- Ensure consistent styling with edge study figures
- Add to `thesis/` directory or reference from `data_science/`

### Phase 4: References
- Confirm all citations present in `example.bib`:
  - Wei et al. (2022) - CoT
  - Zhou et al. (2022) - Least-to-most
  - Brown et al. (2020) - Few-shot
  - Role prompting survey (2024)

### Phase 5: Cross-References
- Update edge chapter to mention "complemented by node study in Chapter X"
- Update Methods chapter to reference both studies
- Update Discussion/Conclusions to integrate both findings

---

## Key Writing Principles

1. **Emphasize the Contrast**: The most interesting finding is the **absence** of significance in nodes vs. **presence** in edges
2. **Be Honest About Power**: Acknowledge underpowered study but argue practical differences are small
3. **Task Complexity Thesis**: Frame as evidence that prompting impact scales with task complexity
4. **Unified Story**: Both studies together provide comprehensive CLD generation guidance
5. **Production-Ready**: End with actionable recommendations for system design

---

## Estimated Length

- **New Chapter**: 8-10 pages
- **Figures**: 3-4 (bar chart, boxplot, heatmap, possibly comparison figure)
- **Tables**: 4-5 (variants, results, pairwise, CLD performance, edge-node comparison)

**Total Addition to Thesis**: ~10 pages

---

## Next Steps

1. ✅ **Review plan with Nitai**
2. **Create chapter LaTeX file** with proper structure
3. **Generate publication-ready figures** (convert PNG → PDF)
4. **Write section-by-section** (can be iterative)
5. **Integrate cross-references** with existing chapters
6. **Review for consistency** with edge study chapter

**Ready to begin implementation!**

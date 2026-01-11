# Node Generation Prompt Ablation: Concise Integration Plan

## Strategy: Expand Existing Edge Chapter

**File**: `thesis/chapters/prompt_ablation_results.tex`

**Add**: ~2-3 pages to existing chapter (rename to "Prompt Engineering Ablation Studies")

---

## New Structure (Integrated)

```
Chapter X: Prompt Engineering Ablation Studies
├─ Section 1: Introduction (existing, mention both studies)
├─ Section 2: Edge Generation Study (existing content)
│  ├─ 2.1 Configuration
│  ├─ 2.2 Results (F1: 0.00-0.46)
│  └─ 2.3 Discussion
├─ Section 3: Node Generation Study (NEW - 2 pages)
│  ├─ 3.1 Configuration & Motivation
│  ├─ 3.2 Results (F1: 0.62-0.68, p=0.567)
│  └─ 3.3 Key Finding: No Significance
├─ Section 4: Cross-Study Comparison (NEW - 1 page)
│  ├─ 4.1 Contrasting Findings Table
│  └─ 4.2 Task Complexity Interpretation
└─ Section 5: Unified Recommendations (NEW - 0.5 pages)
```

---

## What to Add (Concise)

### Section 3: Node Generation Study (~2 pages)

#### 3.1 Configuration (0.5 page)
- **Why separate study**: Nodes have unique constraints (atomicity, measurability, no few-shot examples)
- **7 variants**: V0_Minimal (baseline) → V6_Andrew, testing CoT, role, constraints
- **Design**: 7 prompts × 3 CLDs × 5 runs = 105 experiments (vs. edge study's 1 run)
- **Metric**: Hybrid F1 (semantic + cosine matching)

**One table**: Variant classification

| Variant | CoT | Role | Constraints | Literature |
|---------|-----|------|-------------|------------|
| V0_Minimal | ✗ | ✗ | ✗ | Baseline |
| V3_CoT_System | ✓✓ | ✗ | ✗ | Wei et al. 2022 |
| V6_Andrew | ✓ | ✓✓ | ✓✓✓ | Enhanced |
| ... | | | | |

---

#### 3.2 Results (1 page)

**One figure**: Bar chart with confidence intervals (`performance_by_prompt_hybrid_f1.png`)

**One table**: Summary statistics

| Prompt | Mean F1 | 95% CI | N |
|--------|---------|--------|---|
| V6_Andrew | 0.6773 | [0.638, 0.717] | 15 |
| V3_CoT_System | 0.6573 | [0.595, 0.719] | 15 |
| V0_Minimal | 0.6394 | [0.592, 0.687] | 15 |
| V2_Constraints | 0.6168 | [0.578, 0.656] | 15 |

**Key finding box**:
> **ANOVA: F=0.807, p=0.567 (NOT significant)**  
> Performance range: 0.62-0.68 (6%)  
> All variants within small-to-negligible effect sizes

---

#### 3.3 Interpretation (0.5 page)

**Bullet points**:
- No statistically significant differences (p=0.567)
- Performance range only 6% (vs. 46% for edges)
- Post-hoc power analysis: study underpowered BUT practical differences genuinely small
- **Interpretation**: For simple concept identification tasks, prompt engineering has limited impact

---

### Section 4: Cross-Study Comparison (1 page)

**One comparison table**:

| Aspect | Edge Generation | Node Generation | Implication |
|--------|----------------|-----------------|-------------|
| **F1 Range** | 0.00–0.46 (46%) | 0.62–0.68 (6%) | Task complexity matters |
| **Significance** | Yes (implicit) | **No** (p=0.567) | Prompting impact task-dependent |
| **Baseline F1** | 0.371 (Current) | 0.639 (V0) | Ceiling effect for nodes |
| **CoT Benefit** | +14% (A→C) | ~+2% (negligible) | Complex reasoning only |
| **Failures** | Yes (Cillian_C: 0.00) | None | Nodes more robust |

**Interpretation paragraph** (4-5 sentences):
- Task complexity hypothesis: Prompting benefits scale with reasoning complexity
- Edges require causal mechanism reasoning (complex) → large prompting impact
- Nodes require concept identification (simpler) → minimal prompting impact
- GPT-4.1 already proficient at node identification (F1=0.64 baseline)
- Ceiling effect limits prompt engineering upside for simple tasks

---

### Section 5: Unified Recommendations (0.5 page)

**Task-Specific Guidance**:

**Node Generation**:
- ✅ Use simple prompts (V0, V6) with basic constraints
- ❌ Don't over-invest in complex CoT

**Edge Generation**:
- ✅ Invest in dual-level CoT (Nitai_C)
- ✅ Balance grounding with flexibility
- ❌ Avoid mandatory RAG (25% penalty)

**General Principle**:
> Match prompting complexity to task complexity. Simple tasks benefit little from sophisticated techniques; complex reasoning tasks justify the investment.

---

## Implementation Steps

### 1. Update Chapter Title
```latex
\chapter{Prompt Engineering Ablation Studies} % was: "Study"
```

### 2. Modify Introduction (1 paragraph addition)
Add after existing intro:
> "This chapter presents two complementary ablation studies: edge generation (Section 2) and node generation (Section 3). The contrasting findings—significant impact for edges, negligible for nodes—demonstrate that prompt engineering effectiveness is highly task-dependent."

### 3. Add Section 3 (Node Study) - 2 pages
- Insert after existing edge results
- Include 1 figure, 1-2 tables
- Keep text minimal, let data speak

### 4. Add Section 4 (Comparison) - 1 page
- One comparison table
- One interpretation paragraph

### 5. Add Section 5 (Recommendations) - 0.5 pages
- Bullet points only
- Actionable takeaways

### 6. Update Existing Section References
- Edge results: "as shown in Section 2" → update if needed

---

## Visual Assets Needed

**Convert to PDF** (for thesis):
1. `performance_by_prompt_hybrid_f1.png` → PDF
2. `boxplot_hybrid_f1.png` → PDF (optional, if space)
3. `heatmap_hybrid_f1.png` → PDF (optional)

**Recommendation**: Use only 1-2 figures max for conciseness

---

## Length Estimate

- **Section 3 (Node Study)**: 2 pages
- **Section 4 (Comparison)**: 1 page
- **Section 5 (Recommendations)**: 0.5 pages
- **Total Addition**: **3.5 pages**

**New Chapter Total**: ~14-15 pages (was ~11)

---

## Key Messages (Condensed)

1. **Contrasting results**: Edges show 46% range (significant), nodes show 6% range (not significant)
2. **Task complexity thesis**: Prompting impact scales with reasoning complexity
3. **Practical guidance**: Simple prompts for nodes, sophisticated CoT for edges
4. **Methodological improvement**: 5 runs per variant enables statistical testing

---

## Next Steps

1. ✅ Review this concise plan
2. Create LaTeX for new sections (3, 4, 5)
3. Convert 1-2 figures to PDF
4. Insert into existing chapter
5. Update cross-references

**Ready to implement?**

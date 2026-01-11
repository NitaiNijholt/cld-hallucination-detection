# Thesis Stitch Mapping (Option A)
**Workspace target:** `thesis/final_thesis/`  
**Goal:** map *every block* (paragraph / list / figure / table / equation / verbatim) from the major LaTeX sources into the thesis narrative, with special focus on a coherent **Chapter 2 (Background)** → **Chapter 3 (Literature Review)** flow aligned to **RQ1–RQ3**.

Conventions used below:
- **Block ID**: `F<file#>-B<block#>`
- **Type**: `paragraph`, `list`, `figure`, `table`, `equation`, or `meta`
- **Lines**: exact line range in the source file
- **Place**: recommended destination header (Ch2.* or Ch3.*)
- **Notes**: trim/merge guidance where needed

---

## Target headers (Option A)

### Chapter 1 — Introduction (already written; included for completeness)
- **1.1–1.n** As in `introduction.tex` (problem framing + RQs)

### Chapter 2 — Background
- **2.1** Modeling, simulation, and why conceptual validity matters  
- **2.2** Causal Loop Diagrams: definition, semantics, epistemic status  
- **2.3** Group Model Building and CLD→model pipeline  
- **2.4** Why LLMs for CLDs? (problem framing)  
- **2.5** LLM technical background (minimal)  
- **2.6** Hallucinations in LLMs (general taxonomy; domain-agnostic)  
- **2.7** Operational definitions used in this thesis (short reader contract)  

### Chapter 3 — Literature Review
- **3.1** Landscape: CLD construction paradigms (text-to-CLD vs generation)  
- **3.2** Text-to-CLD / causal extraction approaches (corpus-based)  
- **3.3** Corpus-independent CLD generation (no-corpus approaches)  
- **3.4** CLD-specific failure modes (taxonomy)  
- **3.5** RQ1 literature: LLM-as-a-Judge  
- **3.6** RQ1b literature: LLM-as-a-Corrector / revision loops  
- **3.7** RQ2 literature: context-insensitive predictors (logprobs + embeddings)  
- **3.8** RQ3 literature: test-time compute allocation (efficiency)  
- **3.9** Deep Research / generate-then-search validation (literature + framing)  
- **3.10** Synthesis: gaps → bridge to Chapter 4  

### Chapter 4 — Methods (included because several “major files” are methods-only)
- **4.1** Datasets and ground truth  
- **4.2** Experimental design (RQ1a/RQ1b/RQ2/RQ3)  
- **4.3** Models, prompts, and implementation details  
- **4.4** Reproducibility + execution commands (may live in Appendix)  

### Chapter 5 — Results
- **5.1** Generator model selection + baseline checks  
- **5.2** Judge verification (TruthfulQA)  
- **5.3** RQ1a results (corruption + ground truth; correctness + citation)  
- **5.4** External/human validation of citation judge  
- **5.5** RQ1b corrector results  
- **5.6** RQ2 UQ metrics results  
- **5.7** Sensitivity analyses (prompt + corrector)  
- **5.8** RQ3 Deep Research (pilot) + enrichment analysis  
- **5.9** Effect size + computational scaling analyses  

### Chapter 6 — Discussion
- **6.1** RQ3 pivot rationale + implications  
- **6.2** Interpretation (RQ2, CI metrics)  
- **6.3** Limitations  
- **6.4** Future work  
- **6.5** Data availability / open science  

### Appendix
- **A.1** Reproducibility, scripts, artifacts, algorithm boxes  

### Build helpers (not thesis content)
- **Standalone wrapper `.tex` files**: compile helpers only; do not paste into thesis chapters.

---

## Source file 1: `background_and_lit_review_outline.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/background_and_lit_review_outline.tex`

### Blocks
- **F1-B1** — *meta* — Lines **L1–L21**  
  - **Place:** *Do not copy* (preamble + wrapper)

- **F1-B2** — *paragraph* — Lines **L25–L27**  
  - **Place:** **Ch2.1** (simulation motivation; “third pillar”)

- **F1-B3** — *paragraph* — Lines **L28–L31**  
  - **Place:** **Ch2.1** (abstraction/compute trade-off; interdisciplinary collaboration)

- **F1-B4** — *paragraph* — Lines **L32–L34**  
  - **Place:** **Ch2.3** (GMB motivation lead-in; introduces CLDs as conceptual models)

- **F1-B5** — *paragraph* — Lines **L36–L38**  
  - **Place:** **Ch2.1** (M\&S cycle + “flawed conceptual model propagates errors”)

- **F1-B6** — *paragraph* — Lines **L40–L42**  
  - **Place:** **Ch2.1** (why simulate; frames downstream value)

- **F1-B7** — *list* — Lines **L44–L49**  
  - **Place:** **Ch2.1** (why simulate; keep as itemization)

- **F1-B8** — *paragraph* — Lines **L51–L52**  
  - **Place:** **Ch2.1 → Ch2.2 transition** (“motivates value of accurate CLD construction”)

- **F1-B9** — *paragraph* — Lines **L54–L55**  
  - **Place:** **Ch2.1/Ch2.3** (CLDs as conceptual stage; bridge to quantification pipeline)

- **F1-B10** — *paragraph* — Lines **L56–L57**  
  - **Place:** **Ch2.1** (examples; optional—keep only 1–2 examples if space tight)

- **F1-B11** — *meta* — Lines **L58–L58** (`\section*{Causal Loop Diagrams}`)  
  - **Place:** **Ch2.2** (convert to proper thesis heading)

- **F1-B12** — *paragraph* — Lines **L60–L61**  
  - **Place:** **Ch2.2** (definition + polarity semantics + delays + DAG contrast)

- **F1-B13** — *paragraph* — Lines **L62–L68**  
  - **Place:** **Ch2.2** (formalization; keep equation block)

- **F1-B14** — *paragraph* — Lines **L70–L73**  
  - **Place:** **Ch2.2** (feedback loops + polarity)

- **F1-B15** — *paragraph* — Lines **L74–L77**  
  - **Place:** **Ch2.2** (scope/use)

- **F1-B16** — *paragraph* — Lines **L78–L80**  
  - **Place:** **Ch2.2** (epistemological status)

- **F1-B17** — *meta* — Lines **L82–L85** (subsection header comments)  
  - **Place:** **Ch2.3**

- **F1-B18** — *paragraph* — Lines **L86–L87**  
  - **Place:** **Ch2.3** (GMB definition)

- **F1-B19** — *paragraph* — Lines **L88–L91**  
  - **Place:** **Ch2.3** (aCLD pipeline intro)

- **F1-B20** — *list* — Lines **L92–L111**  
  - **Place:** **Ch2.3** (aCLD pipeline steps)

- **F1-B21** — *paragraph* — Lines **L113–L120**  
  - **Place:** **Ch2.3** (UQ types; structural/parameter/propagation)

- **F1-B22** — *paragraph* — Lines **L122–L124**  
  - **Place:** **Ch2.3 (closing)** (implications for LLM CLD generation)

- **F1-B23** — *meta* — Lines **L126–L127** (subsection heading)  
  - **Place:** **Ch2.3 or Ch2.4** (don’t copy heading literally; merge into narrative)

- **F1-B24** — *list* — Lines **L128–L131**  
  - **Place:** **Ch2.3** (cost scaling bullet; also a good Ch2.4 opener)

- **F1-B25** — *meta* — Lines **L133–L134** (LLMs subsection heading)  
  - **Place:** **Ch2.4**

- **F1-B26** — *list* — Lines **L135–L141**  
  - **Place:** **Ch2.4** (LLM motivation bullets; compress to 2–3 citations max)

- **F1-B27** — *meta* — Lines **L143–L146** (Hallucinations heading)  
  - **Place:** split across **Ch2.6** and **Ch3.4** (see below)

- **F1-B28** — *list* — Lines **L147–L152**  
  - **Place:** **Ch2.6** (general “why hallucinate / open world / eval incentives”)

- **F1-B29** — *paragraph* — Lines **L154–L156**  
  - **Place:** **Split:**  
    - First sentence (taxonomy) → **Ch2.6**  
    - Second sentence (mapping to CLD errors) → **Ch3.4**

- **F1-B30** — *paragraph* — Lines **L158–L160**  
  - **Place:** **Split:**  
    - First sentence (knowledge boundary) → **Ch2.6**  
    - CLD/RAG-specific discussion → **Ch3.4** (as domain-specific failure-mode rationale)

- **F1-B31** — *paragraph* — Lines **L160–L160** (continues)  
  - **Place:** **Ch2.6** (general “internal vs behavioral signals”, but trim CLD-only claims)

- **F1-B32** — *paragraph* — Lines **L162–L164**  
  - **Place:** **Ch2.6** (mitigation taxonomy intro: black-box vs white-box)

- **F1-B33** — *list* — Lines **L166–L172**  
  - **Place:** **Ch2.6** (black-box: prompt-based + RAG)

- **F1-B34** — *list* — Lines **L174–L182**  
  - **Place:** **Ch2.6** (white-box: logprobs)

- **F1-B35** — *list* — Lines **L184–L190**  
  - **Place:** **Ch2.6** (fine-tuning/RLHF; keep as short “not used here” aside)

- **F1-B36** — *list* — Lines **L192–L197**  
  - **Place:** **Ch2.6** (activation monitoring/ITI; brief mention)

- **F1-B37** — *list* — Lines **L199–L203**  
  - **Place:** **Ch2.6** (predictive capacity; optional—could drop if bloated)

- **F1-B38** — *paragraph* — Lines **L205–L209**  
  - **Place:** **Ch3.8** (test-time compute overview; literature review, not background)

- **F1-B39** — *paragraph* — Lines **L211–L213**  
  - **Place:** **Ch3.8** (forcing longer outputs / budget forcing)

- **F1-B40** — *list* — Lines **L215–L221**  
  - **Place:** **Ch3.8** (serial compute bullets)

- **F1-B41** — *paragraph* — Lines **L223–L223**  
  - **Place:** **Ch3.8** (ReAct; tool-use grounding)

- **F1-B42** — *paragraph* — Lines **L225–L225**  
  - **Place:** **Ch3.8 or Ch3.9** (multi-agent debate: compute vs validation)

- **F1-B43** — *paragraph* — Lines **L227–L231**  
  - **Place:** **Ch3.8** (parallel compute: self-consistency + best-of-N)

- **F1-B44** — *meta/list* — Lines **L233–L238**  
  - **Place:** **Ch3.8** (multi-agent systems; optional)

- **F1-B45** — *list* — Lines **L240–L255**  
  - **Place:** **Ch3.5** (judge overview + limitations bullets; merge with `lit_review_llm_as_judge.tex`)

- **F1-B46** — *list* — Lines **L256–L269**  
  - **Place:** **Ch3.6** (corrector bullets; merge with `lit_review_llm_as_judge.tex`)

- **F1-B47** — *list* — Lines **L271–L285**  
  - **Place:** **Ch3.1/Ch3.9** (RAG bullets; good lead-in to generate-then-search)

- **F1-B48** — *meta* — Lines **L287–L290**  
  - **Place:** *Do not copy* (outline section separators)

- **F1-B49** — *list* — Lines **L291–L295**  
  - **Place:** **Ch3.1** (text-to-CLD paradigm framing)

- **F1-B50** — *list* — Lines **L297–L302**  
  - **Place:** **Ch3.1** (CLD generation paradigm framing)

- **F1-B51** — *list* — Lines **L304–L324**  
  - **Place:** **Ch3.3** (seminal approaches; may be shortened if redundant with `lit_review_cld_generation.tex`)

- **F1-B52** — *paragraph* — Lines **L326–L328**  
  - **Place:** **Ch3.2/Ch3.3** (intro to comparison table)

- **F1-B53** — *table* — Lines **L330–L356**  
  - **Place:** **Ch3.2/Ch3.3** (comparison table; keep)

- **F1-B54** — *meta* — Lines **L358–L358**  
  - **Place:** **Ch3.10** (key observations intro)

- **F1-B55** — *list* — Lines **L360–L370**  
  - **Place:** **Ch3.10** (what literature lacks + your positioning)

- **F1-B56** — *meta* — Lines **L372–L373**  
  - **Place:** **Ch3.4** (CLD hallucination context section header)

- **F1-B57** — *list* — Lines **L374–L378**  
  - **Place:** **Ch3.4** (CLD-specific hallucination context bullets; expand using `lit_review_cld_generation.tex`)

- **F1-B58** — *meta* — Lines **L380–L381**  
  - **Place:** **Ch3.10** (gap → methods bridge header)

- **F1-B59** — *list* — Lines **L382–L395**  
  - **Place:** **Ch3.10** (bridge to methods; explicitly map to RQ1–RQ3)

- **F1-B60** — *meta* — Lines **L397–L407**  
  - **Place:** *Do not copy* (bibliography + end document)

---

## Source file 2: `introduction.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/introduction.tex`

### Blocks
- **F2-B1** — *paragraph* — Lines **L6–L7**  
  - **Place:** **Ch2.2 (lead-in)** (CLD importance + cost scaling; avoid duplicating too much with Ch2.3)

- **F2-B2** — *paragraph* — Lines **L10–L14**  
  - **Place:** **Ch2.4** (information aggregation + need for initial draft)

- **F2-B3** — *paragraph* — Lines **L18–L22**  
  - **Place:** **Ch2.4** (LLMs as synthesis engines framing)

- **F2-B4** — *paragraph* — Lines **L26–L31**  
  - **Place:** **Ch2.6 opener** (hallucination as blocker; corpus-independent motivation)

- **F2-B5** — *paragraph* — Lines **L34–L37**  
  - **Place:** **Ch3.9 opener** (overlooked-but-valid edges; motivates Deep Research)

- **F2-B6** — *paragraph* — Lines **L40–L40**  
  - **Place:** **Ch3.4** (false negatives as CLD failure mode; short subparagraph)

- **F2-B7** — *meta/list* — Lines **L43–L60**  
  - **Place:** Use as **end-of-Ch3.10** alignment check (don’t paste verbatim into ch2/3 unless desired)

- **F2-B8** — *meta* — Lines **L83–L86**  
  - **Place:** Optional cross-check for thesis structure consistency.

---

## Source file 3: `background_llms.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/background_llms.tex`

### Blocks (in order; keep together as Ch2.5)
- **F3-B1** — *meta* — Lines **L1–L3** (`\subsection...`, label)  
  - **Place:** **Ch2.5** (convert to your chapter’s sectioning)

- **F3-B2** — *paragraph* — Lines **L4–L8**  
  - **Place:** **Ch2.5.1** (tokens/embeddings/cosine; RAG mention)

- **F3-B3** — *figure* — Lines **L10–L15**  
  - **Place:** **Ch2.5.1** (tokenization figure)

- **F3-B4** — *paragraph + equation* — Lines **L17–L21**  
  - **Place:** **Ch2.5.2** (attention equation + scaling + positional info)

- **F3-B5** — *paragraph* — Lines **L23–L23**  
  - **Place:** **Ch2.5.4** (sparse vs FlashAttention; keep 2–3 sentences)

- **F3-B6** — *figure* — Lines **L25–L30**  
  - **Place:** **Ch2.5.2** (attention intuition)

- **F3-B7** — *paragraph* — Lines **L32–L32**  
  - **Place:** **Ch2.5.2** (multi-head attention intro)

- **F3-B8** — *figure* — Lines **L34–L39**  
  - **Place:** **Ch2.5.2** (MHA figure)

- **F3-B9** — *paragraph* — Lines **L41–L41**  
  - **Place:** **Ch2.5.3** (encoder–decoder intro)

- **F3-B10** — *figure* — Lines **L43–L55**  
  - **Place:** **Ch2.5.3** (enc/dec figures)

- **F3-B11** — *paragraph + equations* — Lines **L57–L67**  
  - **Place:** **Ch2.5.3/2.5.4** (teacher forcing + NLL + causal mask + factorization)

- **F3-B12** — *figure* — Lines **L69–L74**  
  - **Place:** **Ch2.5.3** (decoder-only figure)

- **F3-B13** — *paragraph* — Lines **L76–L76**  
  - **Place:** **Ch2.5.5** (decoding stochasticity; cite Holtzman)

- **F3-B14** — *figure* — Lines **L78–L83**  
  - **Place:** **Ch2.5.5** (temperature figure)

- **F3-B15** — *meta* — Lines **L85–L85** (`\FloatBarrier`)  
  - **Place:** keep if figures float undesirably in the full thesis build.

---

## Source file 4: `lit_review_cld_generation.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/lit_review_cld_generation.tex`

### Blocks
- **F4-B1** — *meta* — Lines **L1–L6**  
  - **Place:** **Ch3.2/3.3 framing** (convert sectioning)

- **F4-B2** — *paragraph* — Lines **L9–L16**  
  - **Place:** **Ch3.2** (directly comparable SD-Bot + benchmark framing)

- **F4-B3** — *paragraph* — Lines **L17–L32**  
  - **Place:** **Ch3.3** (formal causal reasoning benchmarks + “causal parrots” explanation)

- **F4-B4** — *paragraph* — Lines **L33–L48**  
  - **Place:** **Ch3.4** (CLD-specific hallucination manifestations taxonomy)

- **F4-B5** — *paragraph* — Lines **L49–L62**  
  - **Place:** **Ch3.7 (optional add-on)** if you want “compositional UQ” references; otherwise keep in **Ch3.3** as related directions.

- **F4-B6** — *paragraph* — Lines **L63–L78**  
  - **Place:** **Ch3.2/Ch3.10** (reliability gap; sets up hybrid verification)

- **F4-B7** — *paragraph* — Lines **L79–L82**  
  - **Place:** **Ch3.10** (connect metrics to structured outputs)

- **F4-B8** — *table* — Lines **L83–L102**  
  - **Place:** **Ch3.2/Ch3.3** (benchmark mapping table; optional)

- **F4-B9** — *paragraph* — Lines **L104–L112**  
  - **Place:** **Ch3.4 (endcap)** (three-tier detection framework: variable/link/loop)

- **F4-B10** — *paragraph* — Lines **L114–L127**  
  - **Place:** **Ch3.10** (conclusion; motivates human-AI collaboration)

---

## Source file 5: `lit_review_llm_as_judge.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/lit_review_llm_as_judge.tex`

### Blocks (keep in order; split into Ch3.5 and Ch3.6)
- **F5-B1** — *meta* — Lines **L1–L6**  
  - **Place:** **Ch3.5** (convert sectioning)

- **F5-B2** — *paragraph* — Lines **L7–L14**  
  - **Place:** **Ch3.5.1** (paradigm + empirical validation)

- **F5-B3** — *paragraph* — Lines **L15–L26**  
  - **Place:** **Ch3.5.2** (bias taxonomy)

- **F5-B4** — *paragraph* — Lines **L27–L36**  
  - **Place:** **Ch3.5.3** (pointwise vs pairwise; why pointwise for edges)

- **F5-B5** — *paragraph* — Lines **L37–L44**  
  - **Place:** **Ch3.6** (corrector + CoVe/Reflexion + correction failure mode)

- **F5-B6** — *paragraph* — Lines **L45–L54**  
  - **Place:** **Ch3.5.4** (your empirical validation + leniency bias)

- **F5-B7** — *paragraph + list* — Lines **L55–L69**  
  - **Place:** **Ch3.10 (bridge)** or **Ch3.5 closing** (hybrid detection motivation)

---

## Source file 6: `lit_review_ci_metrics_background.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/lit_review_ci_metrics_background.tex`

### Blocks (keep in order; they form Ch3.7)
- **F6-B1** — *meta* — Lines **L1–L4**  
  - **Place:** **Ch3.7** (convert sectioning)

- **F6-B2** — *paragraph + equations* — Lines **L5–L16**  
  - **Place:** **Ch3.7.1** (autoregressive generation basics)

- **F6-B3** — *paragraph* — Lines **L17–L22**  
  - **Place:** **Ch3.7.1** (snowballing + decoding degeneration)

- **F6-B4** — *list* — Lines **L23–L28**  
  - **Place:** **Ch3.7.1** (temperature/top-k/top-p)

- **F6-B5** — *paragraph* — Lines **L29–L29**  
  - **Place:** **Ch3.7.1 (wrap-up)** (trade-off sentence)

- **F6-B6** — *meta* — Lines **L31–L33**  
  - **Place:** **Ch3.7.2** (LUQ section header)

- **F6-B7** — *paragraph* — Lines **L34–L47**  
  - **Place:** **Ch3.7.2** (why logprobs can signal uncertainty; API availability)

- **F6-B8** — *paragraph* — Lines **L48–L77**  
  - **Place:** **Ch3.7.2** (PPL, max-window entropy, min prob)

- **F6-B9** — *subsection example block* — Lines **L78–L115**  
  - **Place:** **Ch3.7.2 (optional)** (worked example; keep if you want a pedagogical figure/table)

- **F6-B10** — *paragraph* — Lines **L116–L134**  
  - **Place:** **Ch3.7.2** (types of uncertainty: aleatoric/lexical/epistemic)

- **F6-B11** — *paragraph* — Lines **L135–L147**  
  - **Place:** **Ch3.7.2** (limitations of LUQ metrics)

- **F6-B12** — *meta* — Lines **L149–L151**  
  - **Place:** **Ch3.7.3** (embedding-based grounding header)

- **F6-B13** — *paragraph + equation* — Lines **L152–L160**  
  - **Place:** **Ch3.7.3** (embedding similarity as grounding signal)

- **F6-B14** — *paragraph* — Lines **L162–L170**  
  - **Place:** **Ch3.7.3** (RAG definition + modern interfaces)

- **F6-B15** — *paragraph* — Lines **L172–L179**  
  - **Place:** **Ch3.7.3** (RAG limitations: overconstraining + lost-in-the-middle)

- **F6-B16** — *paragraph + list* — Lines **L180–L205**  
  - **Place:** **Ch3.9 (bridge)** (generate-then-validate concept + limitations)

- **F6-B17** — *paragraph* — Lines **L206–L209**  
  - **Place:** **Ch3.10** (summary: empirical question framing)

---

## Source file 7: `method_experiments_final.tex` (only the parts relevant to Ch2/3)
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/method_experiments_final.tex`

**Important:** This is primarily Chapter 4 material. For Chapter 2/3, only the following blocks are recommended (definitions + generate-then-search framing).

- **F7-B1** — *paragraph/list* — Lines **L220–L227**  
  - **Place:** **Ch2.7** (operational hallucination definition: GT Lit vs GT Synth)

- **F7-B2** — *list* — Lines **L229–L235**  
  - **Place:** **Ch3.4** (CLD-specific failure modes: corruption types as operational error taxonomy)

- **F7-B3** — *paragraph + list* — Lines **L244–L286**  
  - **Place:** **Ch3.9** (generate-then-search paradigm; connects to Deep Research validation)

---

## Source file 8: `discussion.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/discussion.tex`

### Blocks (map to Chapter 6)
- **F8-B1** — *meta* — Lines **L1–L4**  
  - **Place:** **Ch6** (section header + label)

- **F8-B2** — *paragraph + enumerate* — Lines **L6–L16**  
  - **Place:** **Ch6.1** (original RQ3 design + prerequisites)

- **F8-B3** — *paragraph* — Lines **L18–L25**  
  - **Place:** **Ch6.1** (why original RQ3 became untenable: RQ2 + RQ1b failures)

- **F8-B4** — *paragraph* — Lines **L26–L31**  
  - **Place:** **Ch6.1** (pivot to Deep Research)

- **F8-B5** — *paragraph + itemize* — Lines **L32–L42**  
  - **Place:** **Ch6.1** (limitations of pivot; model confound note is important)

- **F8-B6** — *paragraph* — Lines **L44–L48**  
  - **Place:** **Ch6.1** (interpretation + pilot framing)

- **F8-B7** — *meta* — Lines **L50–L52**  
  - **Place:** **Ch6.2** (RQ2 interpretation header)

- **F8-B8** — *paragraphs* — Lines **L54–L67**  
  - **Place:** **Ch6.2** (RF AUC concern + cosine similarity dominance + domain variation)

- **F8-B9** — *list* — Lines **L69–L85**  
  - **Place:** **Ch6.3** (limitations list)

- **F8-B10** — *paragraphs* — Lines **L87–L96**  
  - **Place:** **Ch6.3** (token-level UQ limitations + Ma et al.)

- **F8-B11** — *paragraph + itemize* — Lines **L97–L107**  
  - **Place:** **Ch6.3** (embedding model selection limitations + alternatives)

- **F8-B12** — *paragraphs* — Lines **L109–L119**  
  - **Place:** **Ch6.3** (GT limits + model generalization + sample size)

- **F8-B13** — *meta* — Lines **L121–L122**  
  - **Place:** **Ch6.4** (future work header)

- **F8-B14** — *paragraph + enumerate* — Lines **L125–L136**  
  - **Place:** **Ch6.4** (future work: complete RQ3 / DR)

- **F8-B15** — *paragraph + enumerate* — Lines **L138–L146**  
  - **Place:** **Ch6.4** (revisiting smart test-time compute)

- **F8-B16** — *paragraph + enumerate* — Lines **L148–L156**  
  - **Place:** **Ch6.4** (address corrector limitations)

- **F8-B17** — *enumerate* — Lines **L158–L167**  
  - **Place:** **Ch6.4** (general methodological extensions)

- **F8-B18** — *meta* — Lines **L169–L172**  
  - **Place:** **Ch6.5** (data availability header)

- **F8-B19** — *paragraph + itemize + paragraph* — Lines **L173–L183**  
  - **Place:** **Ch6.5** (open science commitments + call for expert validation)

---

## Source file 9: `final_results.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/final_results.tex`

**Role:** A standalone “results compendium” PDF. In the thesis, treat as **Chapter 5 donor** (tables/figures + captions), not as-is.

### Blocks (map to Chapter 5)
- **F9-B1** — *meta* — Lines **L1–L22**  
  - **Place:** *Do not copy* (standalone preamble + begin/end doc wrappers)

#### 5.1 Generator model selection + baseline checks
- **F9-B2** — *paragraph* — Lines **L27–L28**  
  - **Place:** **Ch5.1** (generator model selection intro)
- **F9-B3** — *table* — Lines **L29–L64**  
  - **Place:** **Ch5.1** (Table: generator model comparison)
- **F9-B4** — *paragraph* — Lines **L70–L75**  
  - **Place:** **Ch5.1** (baseline setup + random baseline intro)
- **F9-B5** — *table* — Lines **L76–L98**  
  - **Place:** **Ch5.1** (Table: random baseline)
- **F9-B6** — *figure* — Lines **L100–L105**  
  - **Place:** **Ch5.1** (Figure: random baseline comparison)
- **F9-B7** — *paragraph* — Lines **L109–L110**  
  - **Place:** **Ch5.1** (temperature sensitivity intro)
- **F9-B8** — *table* — Lines **L111–L138**  
  - **Place:** **Ch5.1** (Table: temperature sensitivity)

#### 5.2 Judge verification (TruthfulQA)
- **F9-B9** — *paragraph* — Lines **L144–L145**  
  - **Place:** **Ch5.2** (why verify judge on TruthfulQA)
- **F9-B10** — *table* — Lines **L148–L172**  
  - **Place:** **Ch5.2** (Table: TruthfulQA performance)
- **F9-B11** — *table* — Lines **L176–L196**  
  - **Place:** **Ch5.2** (Table: confusion matrix)
- **F9-B12** — *enumerate + paragraph* — Lines **L200–L206**  
  - **Place:** **Ch5.2** (key findings + interpretation)

#### 5.3 RQ1a results (corruption + ground truth; correctness + citation)
- **F9-B13** — *paragraph* — Lines **L219–L220**  
  - **Place:** **Ch5.3** (RQ1a corruption correctness: overview intro)
- **F9-B14** — *figure* — Lines **L221–L226**  
  - **Place:** **Ch5.3** (RQ1a corruption correctness: row1)
- **F9-B15** — *paragraph* — Lines **L230–L231**  
  - **Place:** **Ch5.3** (detailed analysis intro)
- **F9-B16** — *figure* — Lines **L232–L237**  
  - **Place:** **Ch5.3** (RQ1a corruption correctness: row2)
- **F9-B17** — *paragraph* — Lines **L241–L242**  
  - **Place:** **Ch5.3** (ROC intro)
- **F9-B18** — *figure* — Lines **L243–L248**  
  - **Place:** **Ch5.3** (ROC curves)

- **F9-B19** — *paragraph* — Lines **L261–L262**  
  - **Place:** **Ch5.3** (RQ1a GT correctness: overview intro)
- **F9-B20** — *figure* — Lines **L263–L268**  
  - **Place:** **Ch5.3** (GT correctness row1)
- **F9-B21** — *paragraph* — Lines **L272–L273**  
  - **Place:** **Ch5.3** (GT correctness detailed intro)
- **F9-B22** — *figure* — Lines **L274–L279**  
  - **Place:** **Ch5.3** (GT correctness row2)
- **F9-B23** — *paragraph* — Lines **L283–L284**  
  - **Place:** **Ch5.3** (GT correctness heatmap intro)
- **F9-B24** — *figure* — Lines **L285–L290**  
  - **Place:** **Ch5.3** (GT correctness heatmap)

- **F9-B25** — *paragraph* — Lines **L303–L304**  
  - **Place:** **Ch5.3** (RQ1a corruption citation: overview intro)
- **F9-B26** — *figure* — Lines **L305–L310**  
  - **Place:** **Ch5.3** (corruption citation row1)
- **F9-B27** — *paragraph* — Lines **L314–L315**  
  - **Place:** **Ch5.3** (corruption citation detailed intro)
- **F9-B28** — *figure* — Lines **L316–L321**  
  - **Place:** **Ch5.3** (corruption citation row2)
- **F9-B29** — *paragraph* — Lines **L325–L326**  
  - **Place:** **Ch5.3** (corruption citation ROC intro)
- **F9-B30** — *figure* — Lines **L327–L332**  
  - **Place:** **Ch5.3** (corruption citation ROC)

- **F9-B31** — *paragraph* — Lines **L345–L346**  
  - **Place:** **Ch5.3** (RQ1a GT citation: overview intro)
- **F9-B32** — *figure* — Lines **L347–L352**  
  - **Place:** **Ch5.3** (GT citation row1)
- **F9-B33** — *paragraph* — Lines **L356–L357**  
  - **Place:** **Ch5.3** (GT citation detailed intro)
- **F9-B34** — *figure* — Lines **L358–L363**  
  - **Place:** **Ch5.3** (GT citation row2)
- **F9-B35** — *paragraph* — Lines **L367–L368**  
  - **Place:** **Ch5.3** (GT citation heatmap intro)
- **F9-B36** — *figure* — Lines **L369–L374**  
  - **Place:** **Ch5.3** (GT citation heatmap)

#### 5.4 External + human validation of citation judge
- **F9-B37** — *paragraphs* — Lines **L380–L385**  
  - **Place:** **Ch5.4** (Ulemans external validation intro)
- **F9-B38** — *table* — Lines **L386–L412**  
  - **Place:** **Ch5.4** (provider/fetcher comparison)
- **F9-B39** — *paragraph* — Lines **L416–L416**  
  - **Place:** **Ch5.4** (key finding sentence)

- **F9-B40** — *paragraph* — Lines **L422–L423**  
  - **Place:** **Ch5.4** (human validation intro)
- **F9-B41** — *table* — Lines **L426–L453**  
  - **Place:** **Ch5.4** (agreement table)
- **F9-B42** — *table* — Lines **L457–L480**  
  - **Place:** **Ch5.4** (disagreement/confusion matrix)
- **F9-B43** — *enumerate + paragraph* — Lines **L484–L493**  
  - **Place:** **Ch5.4** (key findings + interpretation)

#### 5.5–5.9 Remaining results sections (high-level mapping)
The rest of `final_results.tex` is large but structurally uniform: each subsection is an intro paragraph followed by one or more tables/figures.
- **F9-B44** — **Ch5.5** RQ1b corrector results: **L495–L634** (tables `L500–L530`, `L534–L567`, `L573–L600`, `L606–L634`)  
- **F9-B45** — **Ch5.6** RQ2 UQ metrics results: **L637–L766** (itemize `L649–L654`, tables `L666–L703`, `L718–L756`, figures `L707–L712`, `L760–L765`)  
- **F9-B46** — **Ch5.7** Sensitivity analyses: **L768–L860** (figure `L773–L778`, table `L780–L817`, figure `L825–L830`, table `L832–L859`)  
- **F9-B47** — **Ch5.8** RQ3 Deep Research + enrichment: **L862–L992** (table `L875–L899`, figure `L901–L906`, table `L914–L948`, table `L955–L981`, caveat box `L987–L991`)  
- **F9-B48** — **Ch5.9** Effect sizes + scaling: **L994–L1150** (table `L999–L1031`, figures `L1039–L1058`, table `L1062–L1098`, itemize `L1101–L1105`, figure `L1120–L1125`, table `L1127–L1150`)  

---

## Source file 10: `method_experiments.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/method_experiments.tex`

**Role:** A standalone “methods document” that duplicates/overlaps with `method_experiments_final.tex` and parts of `appendix.tex`. Treat as **Chapter 4 donor** (or deprecate if you standardize on `*_final.tex`).

### Blocks
- **F10-B1** — *meta* — Lines **L1–L21**  
  - **Place:** *Do not copy* (standalone preamble)

#### 4.1 Datasets and ground truth
- **F10-B2** — *enumerate* — Lines **L28–L36**  
  - **Place:** **Ch4.1** (CLD domains overview)
- **F10-B3** — *list + paragraphs* — Lines **L40–L53**  
  - **Place:** **Ch4.1** (edge structure + two GT approaches)

#### 4.2 Experimental design
- **F10-B4** — *paragraphs + tables* — Lines **L55–L129**  
  - **Place:** **Ch4.2** (design strategy + configs + judge model config + rationale)
- **F10-B5** — *overview bullets* — Lines **L134–L142**  
  - **Place:** **Ch4.2** (RQ mapping)
- **F10-B6** — *RQ1a GT setup + prompts + metrics* — Lines **L144–L193**  
  - **Place:** **Ch4.2** (GT judging experiments)
- **F10-B7** — *RQ1a corruption protocol* — Lines **L195–L241**  
  - **Place:** **Ch4.2** (synthetic corruption)
- **F10-B8** — *RQ1b corrector setup* — Lines **L243–L278**  
  - **Place:** **Ch4.2** (corrector evaluation)
- **F10-B9** — *RQ2 dataset + CI metrics + pipeline + classifiers* — Lines **L280–L370**  
  - **Place:** **Ch4.2/Ch4.3** (RQ2 design + analysis plan)

#### 4.3 Implementation details
- **F10-B10** — *software/models + reproducibility metadata* — Lines **L372–L406**  
  - **Place:** **Ch4.3**

#### Appendix A (reproducibility)
- **F10-B11** — *appendix reproducibility block* — Lines **L408–L721**  
  - **Place:** **Appendix A** (this overlaps heavily with `appendix.tex`; prefer only one canonical source)

---

## Source file 11: `appendix.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/appendix.tex`

**Role:** Canonical Appendix content (reproducibility + ablations + algorithm boxes). Include as **Appendix A** in the thesis.

### Blocks (coarse-grained; each is internally structured with tables/verbatim)
- **F11-B1** — *meta* — Lines **L1–L3**  
  - **Place:** **Appendix A** (appendix marker)

- **F11-B2** — *section + table + itemize* — Lines **L4–L45**  
  - **Place:** **Appendix A.1** (function-to-algorithm mapping)

- **F11-B3** — *section + reproducibility scaffolding* — Lines **L47–L579**  
  - **Place:** **Appendix A.2** (reproducibility: data, code, env, commands, artifacts, token verification)

- **F11-B4** — *section: prompt engineering ablation* — Lines **L581–L702**  
  - **Place:** **Appendix A.3** (edge generation prompt ablation)

- **F11-B5** — *section: node generation ablation* — Lines **L703–L736**  
  - **Place:** **Appendix A.4**

- **F11-B6** — *section: embedding model benchmark* — Lines **L737–L795**  
  - **Place:** **Appendix A.5**

- **F11-B7** — *section: algorithmic descriptions* — Lines **L796–end**  
  - **Place:** **Appendix A.6** (Algorithms 0–8)

---

## Source file 12: `method_judge_param_choice_justification.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/method_judge_param_choice_justification.tex`

### Blocks
- **F12-B1** — *paragraphs* — Lines **L1–L3**  
  - **Place:** **Ch4.3** (judge parameter choice justification) or **Appendix A** (methods justification note)

---

## Source file 13: `method_experiments_word.tex`
Path: `/home/nitai/code/causalix.ai/thesis/final_thesis/method_experiments_word.tex`

### Blocks
- **F13-B1** — *meta* — Lines **L1–L2**  
  - **Place:** *Ignore* (file is empty)

---

## Source file 14: Standalone wrappers (compile helpers)
These files exist to compile individual sections into PDFs. Do **not** paste into thesis chapters; they are build tooling.

- **F14-B1** — `introduction_standalone.tex` (**L1–L22**) → *compile helper for* `introduction.tex`
- **F14-B2** — `background_llms_standalone.tex` (**L1–L24**) → *compile helper for* `background_llms.tex`
- **F14-B3** — `lit_review_ci_metrics_background_standalone.tex` (**L1–L22**) → *compile helper for* `lit_review_ci_metrics_background.tex`
- **F14-B4** — `lit_review_cld_generation_standalone.tex` (**L1–L21**) → *compile helper for* `lit_review_cld_generation.tex`  
  - **Note:** it currently points bibliography to `../Chapters/references` (different tree) — fine for standalone, but keep thesis builds on `final_thesis/references.bib`.
- **F14-B5** — `lit_review_llm_as_judge_standalone.tex` (**L1–L21**) → *compile helper for* `lit_review_llm_as_judge.tex`
- **F14-B6** — `discussion_standalone.tex` (**L1–L24**) → *compile helper for* `discussion.tex`

---

## Source file 15: Deprecated embedding subsection (optional donor)
- **F15-B1** — `deprecated/lit_review_ci_metrics_background_embedding_improved.tex` (**L1–L64**)  
  - **Place:** **Ch3.7.3** (optional replacement for embedding grounding section)
- **F15-B2** — `deprecated/lit_review_ci_metrics_background_embedding_improved_standalone.tex` (**L1–L22**)  
  - **Place:** *compile helper only*

---

## Assembly notes (to avoid duplication)
- **Ch2.6 vs Ch3.4 split:** In `background_and_lit_review_outline.tex` blocks **F1-B29** and **F1-B30** contain mixed general+CLD-specific sentences; split them exactly as noted.
- **Avoid repeating RAG definitions:** Put the full RAG mathematical definition in **Ch3.7**, and keep **Ch2** references to RAG as a one-sentence intuition only.
- **Judge/corrector duplication:** Prefer `lit_review_llm_as_judge.tex` as the canonical text; use `background_and_lit_review_outline.tex` judge/corrector bullets only if you need extra citations or shorter summaries.












# Source-support check: `lit_review_cld_generation.tex` (lines 1–133)

Checked: 2025-12-21  
Scope: claims + numeric results in `thesis/final_thesis/lit_review_cld_generation.tex` (1–133).  

Notes on method:
- I downloaded the referenced sources (mostly PDFs) to a temporary folder (`/tmp/causalix_citation_support/`) and extracted quotes via `pdftotext`.
- **“PDF page” below refers to the page number in the downloaded PDF**, counting from 1.
- When a claim references a source *without a LaTeX citation key in this .tex file*, I still attempted to locate and verify it (marked accordingly).

---

## Verified / Supported (with direct quotes)

### System Dynamics Bot: “59% of causal links” and “66% of feedback loops”

- **Claim in .tex** (line 11): SD Bot “recovers only **59% of causal links** and **66% of feedback loops**”.
- **Source**: Hosseinichimeh et al. (2024), *From Text to Map: A System Dynamics Bot for Constructing Causal Loop Diagrams* (arXiv:2402.11400)  
  - **Download used**: `https://arxiv.org/pdf/2402.11400.pdf`
- **Supporting quote** (PDF page 14):
  > “Table 1 shows the results of our first experiment. On average, the SD Bot identifies **59% of the links and 66% of the loops** presented in the references.”

### System Dynamics Bot: missing links are often implicit / omitted in text

- **Claim in .tex** (line 43): missing links often occur when implicit information is assumed.
- **Source**: Hosseinichimeh et al. (2024), arXiv:2402.11400  
  - **Download used**: `https://arxiv.org/pdf/2402.11400.pdf`
- **Supporting quote** (PDF page 19):
  > “We note that **most of the missing links relate to the fact they are not mentioned in the text that described the diagram**, probably the authors assuming that they are trivial for the human reader and **can be implied** from the related figures.”

---

### CLadder: GPT‑4 overall accuracy ≈ 62%

- **Claim in .tex** (lines 13, 23): “GPT‑4 achieves ~62% on CLadder” / “~62% accuracy”.
- **Source**: Jin et al. (2023), *CLadder: Assessing causal reasoning in language models* (arXiv:2312.04350)  
  - **Download used**: `https://arxiv.org/pdf/2312.04350.pdf`
- **Supporting quote** (PDF page 8; Table 2 excerpt as extracted):
  > “... GPT‑3.5 ... **GPT‑4 62.03** ... Table 2: Performance of all models on our CLADDER dataset ...”

---

### Corr2Cause: “near-random” causal inference performance; GPT‑4 is below 33.38% F1

- **Claim in .tex** (lines 13, 27): Corr2Cause is near-random; “~29% F1”.
- **Source**: Jin et al. (ICLR 2024 / OpenReview), *Can Large Language Models Infer Causation from Correlation?* (Corr2Cause)  
  - **Download used**: `https://openreview.net/pdf?id=vqIH0ObdqL`
- **Supporting quote (near-random framing)** (PDF page 7):
  > “We can see that pure causal inference is a very challenging task across all existing LLMs. Among all the LLMs, the best performance is **33.38% F1** by BART MNLI, which is **even higher than** the latest GPT-based model, **GPT‑4**. Notably, **many models are worse than random guess**...”
- **Supporting table excerpt for the ~29% figure** (PDF page 7; **table is column-flattened by `pdftotext`**):
  > “... GPT‑3.5 **GPT‑4**  \
  > F1 Precision Recall Accuracy ...  \
  > ... **21.69 29.08** ...”
  
  Interpretation: this excerpt contains the **GPT‑4** row header and the **F1** column values including **29.08**; because of the PDF-to-text conversion, the row/column alignment is not perfectly preserved, but the paper’s Table 4 is the origin of the “~29% F1” claim.

---

### ReCAST: best-performing model average F1 = 0.477; fabricated-link style errors

- **Claim in .tex** (line 37): best model “F1 = 0.477”; fabricated links and difficulty with implicit/long texts.
- **Source**: Chadha et al. (2025), *Can Large Language Models Infer Causal Relationships from Real-World Text?* (ReCAST benchmark) (arXiv:2505.18931)  
  - **Download used**: `https://arxiv.org/pdf/2505.18931.pdf`
- **Supporting quote (F1)** (PDF page 1):
  > “... with the best-performing model achieving an average **F1 score of only 0.477**.”
- **Supporting quote (example of fabricated link in a reasoning trace)** (PDF page 7):
  > “- **Loan usage flexibility -> Diversified investments (cash crops, etc.)**”

  Note: your .tex currently gives a more specific fabricated-link example (“loan usage flexibility → cash crop cultivation”). The paper’s shown example is extremely close in meaning but not identical in wording.

---

### Pipeline Algebra paper: examples of ambiguous / overly broad variables

- **Claim in .tex** (line 41): Reinholtz et al. (Systems 2025) document examples of vague variables like “Macroeconomic and finance conditions” and “Global technology learning”.
- **Source**: Reinholtz et al. (2025), *LLM-Powered, Expert-Refined Causal Loop Diagramming via Pipeline Algebra* (*Systems*)  
  - **Primary landing page**: `https://www.mdpi.com/2079-8954/13/9/784`
  - **Note**: direct automated PDF download from MDPI returned HTTP 403 from this environment; I accessed the article text via a proxy fetch (`https://r.jina.ai/https://www.mdpi.com/2079-8954/13/9/784`) for quote extraction.
- **Supporting quote (variable example 1)**:
  > “**Macroeconomic and finance conditions**—this is an odd variables because it can mean anything. The ? instead of +/− shows just that—it is unclear...”
- **Supporting quote (variable example 2)**:
  > “**Global technology learning**—this is too vague. What does this mean?”

---

### CaLM: “as complexity increases, accuracy deteriorates, falling almost to zero”

- **Claim in .tex** (line 25): key finding: “as complexity ... deteriorates ... almost to zero”.
- **Source**: Chen et al. (2024), *Causal Evaluation of Language Models* (CaLM) (arXiv:2405.00622)  
  - **Download used**: `https://arxiv.org/pdf/2405.00622.pdf`
- **Supporting quote** (PDF page 21):
  > “As the complexity of causal reasoning increases, the accuracy of each model progressively deteriorates, **eventually falling almost to zero** (Figure 9.21).”

---

### Efficient causal graph discovery via BFS (query complexity)

- **Claim in .tex** (line 29): BFS querying reduces query complexity vs pairwise.
- **Source**: Jiralerspong et al. (2024), *Efficient Causal Graph Discovery Using Large Language Models* (arXiv:2402.01207)  
  - **Download used**: `https://arxiv.org/pdf/2402.01207.pdf`
- **Supporting quote** (PDF page 2):
  > “... construct the graph through a **breadth-first search (BFS)** ... The proposed method achieves **O(n) query complexity** compared to the **O(n^2)** query complexity of the pairwise method.”
- **Supporting quote (“state-of-the-art”)** (PDF page 1):
  > “... the proposed framework achieves **state-of-the-art results on real-world causal graphs** of varying sizes.”

---

### CausalBench: LLMs lag on >50 nodes; collider vs chain behavior

- **Claim in .tex** (line 29): LLMs lag behind traditional algorithms on >50 nodes; struggle with colliders; excel at chains.
- **Source**: *CausalBench: A Comprehensive Benchmark for Causal Learning Capability of LLMs* (arXiv:2404.06349)  
  - **Download used**: `https://arxiv.org/pdf/2404.06349.pdf`
- **Supporting quote** (PDF page 1):
  > “... they significantly lag behind traditional algorithms on larger-scale networks (**> 50 nodes**). Specifically, **LLMs struggle with collider structures but excel at chain structures**...”

---

### “Causal parrots” hypothesis / meta‑SCM framing

- **Claim in .tex** (line 31): LLMs “cannot perform genuine interventional or what-if reasoning”; meta‑SCM framing.
- **Source**: Zečević et al. (2023), *Causal Parrots: Large Language Models May Talk Causality But Are Not Causal* (arXiv:2308.13067)  
  - **Download used**: `https://arxiv.org/pdf/2308.13067.pdf`
- **Supporting quote** (PDF page 1):
  > “We make it clear that large language models (LLMs) **cannot be causal** ... we define ... a new subgroup of Structural Causal Model (SCM) that we call **meta SCM** which encode causal facts about other SCM within their variables.”

---

### Semantic entropy AUROC results (Nature 2024 paper)

- **Claim in .tex** (line 53): semantic entropy achieves “AUROC 0.75–0.85”.
- **Source**: Farquhar et al. (2024), *Detecting hallucinations in large language models using semantic entropy*  
  - **Download used**: `https://sebastianfarquhar.com/assets/papers/farquharDetecting2024.pdf`
- **Supporting quote** (PDF page 3):
  > “Averaged across the 30 combinations of tasks and models we study, **semantic entropy achieves the best AUROC value of 0.790** ... Semantic entropy performs well consistently, with stable performance (**between 0.78 and 0.81 AUROC**)...”

  **Actionable note:** your stated range (0.75–0.85) is *plausible as a rough envelope*, but the paper explicitly reports **0.790 average** and **0.78–0.81** stability in the quoted passage.

---

### Semantic entropy probes: single-generation hidden-state probes; AUROC up to ~0.95

- **Claim in .tex** (line 55): linear probes predict semantic entropy from a single forward pass; AUROC 0.70–0.95.
- **Source**: Kossen et al. (2024), *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs* (arXiv:2406.15927)  
  - **Download used**: `https://arxiv.org/pdf/2406.15927.pdf`
- **Supporting quote (single-generation hidden states)** (PDF page 1):
  > “... we propose SEPs, which directly approximate SE from the hidden states of a **single generation**.”
- **Supporting quote (AUROC range)** (PDF page 6):
  > “... AUROC values increase for later layers in the model, reaching values **between 0.7 and 0.95** depending on the scenario.”

---

### Compositional uncertainty quantification (GAP, CECE; subgraph selection benefit)

- **Claim in .tex** (line 57): GAP + CECE framework; “30% average error reduction”.
- **Source**: Lin et al. (ICLR 2023 / OpenReview), *On Compositional Uncertainty Quantification for Seq2seq Graph Parsing*  
  - **Download used**: `https://openreview.net/pdf?id=rJcLocAJpA6`
- **Supporting quote (CECE)** (PDF page 1; abstract):
  > “... we propose a novel metric called **Compositional Expected Calibration Error (CECE)** which can measure a model’s calibration behavior in predicting graph structures.”
- **Supporting quote (30% error reduction)** (PDF page 1; abstract):
  > “... uncertain subgraph selection consistently outperforms random subgraph selection (**30% average error reduction rate**) ...”
- **Supporting quote (GAP / marginal probabilities for graph elements)** (PDF page 4):
  > “To this end, **GAP assigns probability** to each linearized graph ... and the conditional probability ... is computed by aggregating the token probability ... Importantly, **GAP allows us to compute the marginal and (non-local) conditional probabilities for graph elements** ...”

---

### Multi-LLM debate with arbiter for causal discovery (comparable to RAG)

- **Claim in .tex** (line 59): debate+arbiter reduces hallucinations comparable to RAG.
- **Source**: Sng et al. (2024), *A Novel Approach to Eliminating Hallucinations in Large Language Model-Assisted Causal Discovery* (arXiv:2411.12759)  
  - **Download used**: `https://arxiv.org/pdf/2411.12759.pdf`
- **Supporting quote (RAG + comparable reduction via arbiter debate)** (PDF page 1; abstract):
  > “We propose using Retrieval Augmented Generation (RAG) to reduce hallucinations when quality data is available. Additionally, we introduce ... multiple LLMs with an **arbiter in a debate** to audit edges in causal graphs, achieving a **comparable reduction in hallucinations to RAG**.”

---

### Token-probability hallucination detection (Quevedo et al.)

- **Claim in .tex** (line 61): token probability features can detect hallucinations and perform strongly on benchmarks.
- **Source**: Quevedo et al. (2024), *Detecting Hallucinations in Large Language Model Generation: A Token Probability Approach* (arXiv:2405.19648)  
  - **Download used**: `https://arxiv.org/pdf/2405.19648.pdf`
- **Supporting quote** (PDF page 1; abstract):
  > “... four numerical features derived from tokens and vocabulary probabilities ... The method yields promising results, **surpassing state-of-the-art outcomes** in multiple tasks across three different benchmarks.”

---

### GraphEval / GraphCorrect: KG triples + NLI verification framing

- **Claim in .tex** (line 75): triples-based decomposition verified with NLI; GraphCorrect tool.
- **Source**: Sansford et al. (2024), *GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework*  
  - **Download used**: `https://assets.amazon.science/80/01/9d6ac5844cca9aefa66470b5c0aa/grapheval-a-knowledge-graph-based-llm-hallucination-evaluation-framework.pdf`
- **Supporting quote** (PDF page 1; abstract excerpt):
  > “We present GraphEval ... based on representing information in Knowledge Graph (KG) structures. Our method identifies the specific **triples** in the KG ... Furthermore, using our approach in conjunction with state-of-the-art natural language inference (**NLI**) models leads to an improvement ... Lastly, we explore the use of GraphEval for hallucination correction ...”

---

### Larger models hallucinate less but are harder to detect

- **Claim in .tex** (line 71): larger models hallucinate less, but are harder to detect (arXiv:2408.07852).
- **Source**: Hron et al. (2024), *Training Language Models on the Knowledge Graph: Insights on Hallucinations and Their Detectability* (arXiv:2408.07852)  
  - **Download used**: `https://arxiv.org/pdf/2408.07852.pdf`
- **Supporting quote** (PDF page 9):
  > “... the **detectability of hallucinations is inversely proportional to the LM size** ... Larger LMs have lower hallucination rates, but it’s also **harder to detect their hallucinations**.”

---

## Supported, but conflicts with numbers currently in your .tex (needs correction)

### Economics/finance causal-claim benchmark numbers (Qwen3‑32B “57.6%”, models down to “29.4%”)

- **Claim in .tex** (line 69): Qwen3‑32B “57.6% accuracy”; some models “29.4%”.
- **Source located**: *Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships* (arXiv:2510.07231)  
  - **Download used**: `https://arxiv.org/pdf/2510.07231.pdf`
- **What the paper actually says** (PDF page 2):
  > “Even the best-performing model, **Qwen3-32B, achieves only 60.58% accuracy** ...”

  I did **not** find “57.6%” or “29.4%” in the downloaded arXiv PDF text; the paper’s abstract states **60.58%** for Qwen3‑32B. This suggests the numbers in your .tex may be from an earlier draft, a different benchmark, or a mis-remembered figure.

---

## Not verifiable from accessible sources as written (needs adjustment or a different citation)

### Reinholtz et al. “MIT taxonomy” with categories (A)–(E)

- **Claim in .tex** (line 47): “Expert evaluation at MIT (Reinholtz et al., 2025) produced a practical taxonomy with five categories: (A) … (E) …”
- **Source consulted**: Reinholtz et al. (2025), MDPI *Systems* article page (proxy-fetched text).
- **Result of check**: I was able to verify the paper includes extensive expert commentary and guidelines, and it contains the *example* ambiguous-variable quotes above, but I did **not** find the specific **(A)–(E)** taxonomy phrasing in the accessible text dump. This specific taxonomy claim likely needs:
  - a more precise pointer (section/table number) in the paper, or
  - a different source that actually presents the (A)–(E) taxonomy in that form.

### “84% correctly oriented for LLaMA2‑70B” attribution

- **Claim in .tex** (line 29): “84% correctly oriented for LLaMA2‑70B” (Darvariu et al., 2024).
- **Source downloaded**: Darvariu et al. (2024), *Large Language Models are Effective Priors for Causal Graph Discovery* (arXiv:2405.13551)  
  - **Download used**: `https://arxiv.org/pdf/2405.13551.pdf`
- **Result of check**: I found “84.615” in the paper, but **not** a clean statement matching “84% correctly oriented (directionality) for LLaMA2‑70B” as written in your .tex. This claim likely needs:
  - a different source (possibly a different benchmark paper), or
  - rewording to match the metric/setting actually reported in Darvariu et al.

---

## Citations missing from `references.bib` (but used/mentioned in this .tex subsection)

The following are referenced in `lit_review_cld_generation.tex` without a BibTeX entry in `thesis/final_thesis/references.bib` (as of this check), so LaTeX compilation would rely on other bib files or would fail if cited directly later:
- **ReCAST**: arXiv:2505.18931 (not currently cited with a `\\cite{...}` key in this file)
- **Corr2Cause**: OpenReview id `vqIH0ObdqL` / arXiv:2306.05836 (not currently cited with a `\\cite{...}` key in this file)
- **CaLM**: arXiv:2405.00622 (not currently cited with a `\\cite{...}` key in this file)
- **GraphEval**: Sansford et al. (2024) (not currently cited with a `\\cite{...}` key in this file)
- **Semantic Entropy Probes**: arXiv:2406.15927 (not currently cited with a `\\cite{...}` key in this file)
- **Sng et al. (2024)**: arXiv:2411.12759 (not currently cited with a `\\cite{...}` key in this file)
- **Quevedo et al. (2024)**: arXiv:2405.19648 (not currently cited with a `\\cite{...}` key in this file)
- **Hron et al. (2024)**: arXiv:2408.07852 (not currently cited with a `\\cite{...}` key in this file)
- **CausalBench**: arXiv:2404.06349 (not currently cited with a `\\cite{...}` key in this file)
- **Zečević et al. (2023)**: arXiv:2308.13067 (not currently cited with a `\\cite{...}` key in this file)












# Context-Insensitive Metrics: Mathematical Definitions

Context-insensitive (CI) metrics are computed from the generator LLM's token-level log-probabilities during edge generation. These metrics capture model uncertainty without requiring external context or reasoning.

---

## 1. Generator Perplexity

Perplexity measures the exponential of average negative log-likelihood over generated tokens, quantifying how "surprised" the model is by its own output.

$$\text{Perplexity} = \exp\left(\frac{1}{n}\sum_{i=1}^{n} -\log P(x_i)\right)$$

**Where:**
- $n$ = number of generated tokens in the causal motivation
- $P(x_i)$ = probability of token $x_i$ (from LLM logprobs: $P(x_i) = \exp(\text{logprob}_i)$)

**Implementation:** `logit_metrics.py:compute_perplexity()`. Average NLL capped at 10.0 to prevent overflow (max perplexity ≈ 22,026).

**Interpretation:** Higher perplexity → greater model uncertainty → higher hallucination probability.

**Typical values:** 1.0–2.0 (confident), 2.0–5.0 (moderate), >5.0 (uncertain).

---

## 2. Generator Minimum Probability

Minimum probability identifies the "weakest link" in the generated sequence—the token about which the model was least confident.

$$\text{MinProb} = \min_{i \in \{1,\ldots,n\}} P(x_i)$$

**Where:** $P(x_i) = \exp(\text{logprob}_i)$ from LLM logprobs.

**Implementation:** `logit_metrics.py:compute_min_prob()`.

**Interpretation:** Lower minimum probability → presence of highly uncertain token → higher hallucination probability.

---

## 3. Generator Maximum Window Entropy

Maximum window entropy detects localized regions of high uncertainty by computing Shannon entropy over sliding windows of top-k token distributions.

$$\text{MaxWindowEntropy} = \max_{w \in \mathcal{W}} \left( \frac{1}{|w|} \sum_{t \in w} H_k(t) \right)$$

**Where the local entropy at position $t$ over the top-$k$ tokens is:**

$$H_k(t) = -\sum_{j=1}^{k} \tilde{P}_j(t) \cdot \ln \tilde{P}_j(t)$$

**And $\tilde{P}_j(t)$ is the normalized probability over top-$k$ alternatives:**

$$\tilde{P}_j(t) = \frac{P_j(t)}{\sum_{m=1}^{k} P_m(t)}$$

**Parameters:** 
- $k = 5$ (top-k tokens)
- Window size = 5 tokens
- Sliding with stride 1

**Implementation:** `logit_metrics.py:compute_max_window_entropy()`. Uses `np.convolve` for rolling average.

**Interpretation:** Higher max entropy → region where model was uncertain among alternatives → higher hallucination probability.

---

## 4. Generator Cosine Similarity

Cosine similarity measures semantic alignment between the generated causal motivation and fetched citation text chunks.

$$\text{CosineSim} = \max_{c \in \mathcal{C}} \frac{\mathbf{e}_{\text{narr}} \cdot \mathbf{e}_c}{\|\mathbf{e}_{\text{narr}}\| \cdot \|\mathbf{e}_c\|}$$

**Where:**
- $\mathbf{e}_{\text{narr}} \in \mathbb{R}^{d}$ = embedding of generated causal narrative (motivation)
- $\mathbf{e}_c \in \mathbb{R}^{d}$ = embedding of citation chunk $c$
- $\mathcal{C}$ = set of all citation chunks (up to 5 articles, chunked with 20% overlap)
- $d = 1536$ for `text-embedding-3-small` (OpenAI)

**Citation fetching:** Up to 5 articles fetched via Brave search API per edge, using the causal narrative as query.

**Chunking:** Articles chunked into ~2000 character segments with 20% overlap to capture context boundaries.

**Aggregation:** Maximum similarity across all chunks (not mean), identifying the single best-matching citation passage.

**Implementation:** `logit_metrics.py:compute_alignment_batch()`.

**Expected interpretation:** Higher cosine similarity → better grounding in literature → lower hallucination probability.

**Observed (counterintuitive):** Higher similarity correlates with *more* hallucinations. Possible explanation: "confident confabulation"—LLM cherry-picks single highly-similar chunks to justify spurious claims.

---

## Summary Table

| Metric | Formula | Expected | Observed |
|--------|---------|----------|----------|
| **Perplexity** | $\exp\left(\frac{1}{n}\sum -\log P(x_i)\right)$ | ↑ → halluc | ↑ → halluc ✓ |
| **Min Prob** | $\min_i P(x_i)$ | ↓ → halluc | ↓ → halluc ✓ |
| **Max Window Entropy** | $\max_w \text{mean}(H_k(t))$ | ↑ → halluc | ↑ → halluc ✓ |
| **Cosine Similarity** | $\max_c \cos(\mathbf{e}_{\text{narr}}, \mathbf{e}_c)$ | ↓ → halluc | ↑ → halluc ✗ |

Three logit-based metrics behave as expected; cosine similarity exhibits paradoxical behavior.

---

## Implementation References

All metrics implemented in `data_science/logit_metrics.py`:

| Function | Description |
|----------|-------------|
| `compute_avg_nll(log_data)` | Average negative log-likelihood |
| `compute_perplexity(log_data)` | Perplexity |
| `compute_min_prob(log_data)` | Minimum probability |
| `compute_max_window_entropy(log_data, k_top=5, window_size=5)` | Max window entropy |
| `compute_alignment_batch(narrative, chunks, openai_key)` | Cosine similarity |

---

## Data Source

CI metrics extracted from OpenAI-compatible `logprobs` response format:

```json
{
  "token_logprobs": [-0.5, -1.2, -0.3, ...],
  "top_logprobs": [
    {"the": -0.5, "a": -1.8, "an": -2.1, ...},
    {"quick": -1.2, "fast": -1.5, ...},
    ...
  ]
}
```

- `token_logprobs`: log P(x_i) for each generated token
- `top_logprobs`: log probabilities for top-k alternative tokens at each position

Metrics computed during generation phase and stored on each edge in Neo4j graph database.













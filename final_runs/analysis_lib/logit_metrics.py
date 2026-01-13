import math
import numpy as np
import requests
import os
import logging
import threading
from typing import List, Optional

# ------------------------------
# Embedding usage accounting (thread-safe)
# ------------------------------
_EMBED_STATS = {
    "status_counts": {},
    "token_totals": {"input_tokens": 0, "total_tokens": 0},
    "inference_time_total": 0.0,
    "inference_call_count": 0,
}
_EMBED_STATS_LOCK = threading.Lock()

def reset_embedding_stats() -> None:
    with _EMBED_STATS_LOCK:
        _EMBED_STATS["status_counts"] = {}
        _EMBED_STATS["token_totals"] = {"input_tokens": 0, "total_tokens": 0}
        _EMBED_STATS["inference_time_total"] = 0.0
        _EMBED_STATS["inference_call_count"] = 0

def _record_embedding_usage(status_code: int, elapsed: float, usage: Optional[dict]) -> None:
    with _EMBED_STATS_LOCK:
        _EMBED_STATS["status_counts"][str(status_code)] = _EMBED_STATS["status_counts"].get(str(status_code), 0) + 1
        _EMBED_STATS["inference_time_total"] += float(elapsed)
        _EMBED_STATS["inference_call_count"] += 1
        if usage:
            _EMBED_STATS["token_totals"]["input_tokens"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            _EMBED_STATS["token_totals"]["total_tokens"] += int(usage.get("total_tokens", 0))

def get_embedding_stats() -> dict:
    # return a shallow copy to avoid external mutation (thread-safe)
    with _EMBED_STATS_LOCK:
        return {
            "status_counts": dict(_EMBED_STATS["status_counts"]),
            "token_totals": dict(_EMBED_STATS["token_totals"]),
            "inference_time_total": _EMBED_STATS["inference_time_total"],
            "inference_call_count": _EMBED_STATS["inference_call_count"],
        }

# ------------------------------
# Local embedding model cache (lazy-loaded, device-specific)
# ------------------------------
_LOCAL_EMBED_MODEL_CPU = None
_LOCAL_EMBED_MODEL_GPU = None
_LOCAL_EMBED_MODEL_LOCK = threading.Lock()

def clear_embedding_cache() -> None:
    """Clear cached embedding models and free GPU memory.
    
    This function should be called between judging sessions in multi-run scenarios
    to prevent CUDA memory fragmentation and context conflicts. It mimics the behavior
    of the sequential processing script where each subprocess gets fresh CUDA state.
    
    The model will be reloaded on next use (once per session, not per edge).
    """
    global _LOCAL_EMBED_MODEL_CPU, _LOCAL_EMBED_MODEL_GPU
    
    logger = logging.getLogger("ci.embed.local")
    
    with _LOCAL_EMBED_MODEL_LOCK:
        if _LOCAL_EMBED_MODEL_GPU is not None:
            try:
                # Move model to CPU before deletion to ensure clean GPU memory release
                _LOCAL_EMBED_MODEL_GPU.to('cpu')
                del _LOCAL_EMBED_MODEL_GPU
                _LOCAL_EMBED_MODEL_GPU = None
                
                # Clear CUDA cache and synchronize
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info("✓ Cleared GPU embedding cache and CUDA memory")
                except ImportError:
                    logger.debug("PyTorch not available, skipping CUDA cleanup")
                    
            except Exception as e:
                logger.warning(f"Failed to clear GPU embedding cache: {e}")
                # Still set to None even if cleanup failed
                _LOCAL_EMBED_MODEL_GPU = None
        
        if _LOCAL_EMBED_MODEL_CPU is not None:
            try:
                del _LOCAL_EMBED_MODEL_CPU
                _LOCAL_EMBED_MODEL_CPU = None
                logger.info("✓ Cleared CPU embedding cache")
            except Exception as e:
                logger.warning(f"Failed to clear CPU embedding cache: {e}")
                _LOCAL_EMBED_MODEL_CPU = None

def get_embedding_local(
    texts: List[str],
    model: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 64,
    show_progress: bool = False,
    device: str = "cuda"
) -> List[Optional[np.ndarray]]:
    """Get embeddings using local sentence-transformers model.
    
    Args:
        texts: list of input strings
        model: HuggingFace model ID (default: all-mpnet-base-v2, 768-dim, fast & high quality)
        batch_size: Batch size for encoding (default: 64)
        show_progress: Show progress bar (default: False)
        device: Device to use for inference (default: "cuda"). Accepts: "cpu", "cuda", "cuda:0", etc.
    
    Returns:
        list of numpy arrays (768-dimensional vectors for all-mpnet-base-v2), 1:1 with texts
    """
    global _LOCAL_EMBED_MODEL_CPU, _LOCAL_EMBED_MODEL_GPU
    
    if not texts:
        return []
    
    logger = logging.getLogger("ci.embed.local")
    
    # Determine which model cache to use based on device
    is_cpu = device.lower() == "cpu"
    model_cache = _LOCAL_EMBED_MODEL_CPU if is_cpu else _LOCAL_EMBED_MODEL_GPU
    
    # Lazy-load model (thread-safe)
    with _LOCAL_EMBED_MODEL_LOCK:
        if model_cache is None:
            logger.info(f"Loading local embedding model: {model} on device: {device}")
            try:
                from sentence_transformers import SentenceTransformer
                # trust_remote_code=True is required for NVIDIA models with custom code
                loaded_model = SentenceTransformer(model, trust_remote_code=True, device=device)
                logger.info(f"✓ Model loaded on {device}, embedding dimension: {loaded_model.get_sentence_embedding_dimension()}")
                
                # Cache the model in the appropriate global variable
                if is_cpu:
                    _LOCAL_EMBED_MODEL_CPU = loaded_model
                    model_cache = _LOCAL_EMBED_MODEL_CPU
                else:
                    _LOCAL_EMBED_MODEL_GPU = loaded_model
                    model_cache = _LOCAL_EMBED_MODEL_GPU
            except Exception as e:
                logger.error(f"Failed to load local embedding model on {device}: {e}")
                return [None] * len(texts)
    
    # Encode texts
    try:
        import time
        start = time.time()
        embeddings = model_cache.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        elapsed = time.time() - start
        
        # Record stats (reuse OpenAI stats structure for consistency)
        _record_embedding_usage(200, elapsed, {"input_tokens": sum(len(t.split()) for t in texts), "total_tokens": 0})
        
        logger.debug(f"Encoded {len(texts)} texts in {elapsed:.2f}s (local, {device})")
        
        # Convert to list of arrays
        return [embeddings[i] for i in range(len(texts))]
    
    except Exception as e:
        logger.error(f"Local embedding failed on {device}: {e}")
        return [None] * len(texts)

#########################
# 1) LOGIT-BASED METRICS
#########################

def compute_avg_nll(log_data: dict) -> float:
    """
    Returns the average negative log-likelihood from 'logprobs'.
    log_data is the dictionary: {"tokens": [...], "token_logprobs": [...], "top_logprobs": [...]}
    
    Applies token-level capping to handle extreme outliers, following standard practices
    in language model evaluation (Hugging Face, OpenAI Evals).
    
    Cap rationale:
    - Token logprob < -20 means P(token) < 2e-9 (extremely unlikely)
    - Such tokens are typically:
      1) Special/control characters
      2) Tokenization artifacts  
      3) Model errors/hallucinations
    - Capping preserves signal (very low confidence) while preventing overflow
    
    References:
    - Standard practice in Hugging Face transformers library
    - Prevents single outlier from dominating average (robust estimation)
    - Aligns with arXiv:2405.19648v1 assumption of well-behaved distributions
    """
    tokens = log_data.get("tokens", [])
    token_lp = log_data.get("token_logprobs", [])
    if not token_lp:
        return None
    
    # Cap individual token negative log-likelihoods at 20.0
    # Standard practice: prevents single outlier from dominating average
    # Corresponds to minimum token probability of ~2e-9
    MAX_TOKEN_NLL = 20.0
    
    capped_token_nlls = []
    extreme_count = 0
    
    for i, lp in enumerate(token_lp):
        # Handle potential infinities/NaNs from API errors
        if not np.isfinite(lp):
            capped_token_nlls.append(MAX_TOKEN_NLL)
            extreme_count += 1
            continue
        
        nll = -lp
        if nll > MAX_TOKEN_NLL:
            extreme_count += 1
            capped_token_nlls.append(MAX_TOKEN_NLL)
        else:
            capped_token_nlls.append(nll)
    
    # Log warning if extreme tokens found (helps identify systematic issues)
    if extreme_count > 0:
        logger = logging.getLogger("ci.logit_metrics")
        logger.debug(f"Capped {extreme_count}/{len(token_lp)} tokens with NLL > {MAX_TOKEN_NLL}")
    
    avg_neglog = sum(capped_token_nlls) / len(capped_token_nlls)
    return avg_neglog

def compute_perplexity(log_data: dict) -> float:
    """Exponentiate the average negative log-likelihood => perplexity.
    
    Caps avg_nll at 10.0 to prevent overflow from tokens with near-zero probability.
    This corresponds to a maximum perplexity of e^10 ≈ 22026, which is still a
    clear signal of extreme model uncertainty while remaining interpretable.
    
    Typical perplexity values: 1.0-2.0 (good), 2.0-5.0 (reasonable), >5.0 (uncertain).
    The cap ensures extreme outliers (from tokens with prob ≈ 0) don't cause overflow.
    """
    avg_nll = compute_avg_nll(log_data)
    if avg_nll is None:
        return None
    # Cap avg_nll to prevent overflow from tokens with probability ≈ 0
    # Max perplexity of ~22k is still a very strong signal of model uncertainty
    avg_nll_capped = min(avg_nll, 10.0)
    return math.exp(avg_nll_capped)

def compute_min_prob(log_data: dict) -> float:
    """Minimum token probability across entire sequence."""
    token_lp = log_data.get("token_logprobs", [])
    if not token_lp:
        return None
    probs = [math.exp(lp) for lp in token_lp]
    return min(probs) if probs else None

def compute_max_window_entropy(log_data: dict, k_top=5, window_size=5) -> float:
    """
    For each token, gather top-k logprobs => compute local entropy.
    Then do a rolling-window average (window_size) over entropies, and find the max.
    """
    top_logprobs_list = log_data.get("top_logprobs", [])
    if not top_logprobs_list:
        return None
    
    # local entropies
    entropies = []
    for top_logp_dict in top_logprobs_list:
        # top_logp_dict is presumably a dict {token_str: logp, ...}
        # if it's a list, adjust accordingly
        if isinstance(top_logp_dict, dict):
            items = list(top_logp_dict.items())
            items = sorted(items, key=lambda x: x[1], reverse=True)  # sort by logp desc
            items = items[:k_top]
            lps = [val for (tk,val) in items]
        else:
            # Possibly top_logp_dict is a list of dicts
            continue
        
        pvals = np.exp(lps)
        total = pvals.sum()
        if total > 0:
            pvals /= total
        # Entropy = - sum(p ln(p))
        ent = -np.sum(pvals * np.log(pvals))
        entropies.append(ent)
    
    # rolling average
    if len(entropies) < window_size:
        return float(entropies[-1]) if entropies else None
    
    ent_array = np.array(entropies)
    windowed = np.convolve(ent_array, np.ones(window_size)/window_size, mode="valid")
    return float(np.max(windowed))

def compute_mean_token_prob(log_data: dict) -> float:
    """Mean token probability (direct average, not log-transformed).
    From paper: arXiv:2405.19648v1 - 'mtp' feature."""
    token_lp = log_data.get("token_logprobs", [])
    if not token_lp:
        return None
    probs = [math.exp(lp) for lp in token_lp]
    return sum(probs) / len(probs) if probs else None

def compute_prob_variance(log_data: dict) -> float:
    """Variance in token probabilities across the sequence.
    From paper: arXiv:2405.19648v1 - distribution shape feature."""
    token_lp = log_data.get("token_logprobs", [])
    if not token_lp or len(token_lp) < 2:
        return None
    probs = [math.exp(lp) for lp in token_lp]
    mean_prob = sum(probs) / len(probs)
    variance = sum((p - mean_prob)**2 for p in probs) / len(probs)
    return variance

def compute_prob_std(log_data: dict) -> float:
    """Standard deviation of token probabilities.
    From paper: arXiv:2405.19648v1 - distribution shape feature."""
    variance = compute_prob_variance(log_data)
    return math.sqrt(variance) if variance is not None else None

def compute_max_prob_diff(log_data: dict) -> float:
    """Maximum probability difference (Mpd): difference between 1st and 2nd highest prob.
    From paper: arXiv:2405.19648v1 - 'Mpd' feature.
    Averaged across all tokens in the sequence."""
    top_logprobs_list = log_data.get("top_logprobs", [])
    if not top_logprobs_list:
        return None
    
    prob_diffs = []
    for top_logp_dict in top_logprobs_list:
        if isinstance(top_logp_dict, dict):
            items = list(top_logp_dict.items())
            # Sort by logprob descending
            items = sorted(items, key=lambda x: x[1], reverse=True)
            if len(items) >= 2:
                # Get top 2 probabilities
                prob1 = math.exp(items[0][1])
                prob2 = math.exp(items[1][1])
                prob_diffs.append(prob1 - prob2)
            elif len(items) == 1:
                # Only one token - diff is just that probability
                prob_diffs.append(math.exp(items[0][1]))
    
    return sum(prob_diffs) / len(prob_diffs) if prob_diffs else None

def compute_token_probability_slope(log_data: dict) -> float:
    """Compute the slope of token probabilities across the sequence (temporal slope).
    
    Inspired by CHAIR (arXiv:2501.02518v2) which analyzes logit slopes across internal layers.
    Since we only have access to output token probabilities, we measure how confidence
    changes temporally across the generated sequence instead of across layers.
    
    Formula: Slope = Σ(i - ī)(p(i) - p̄) / Σ(i - ī)²
    where i is token position, p(i) is probability at position i
    
    Interpretation:
    - Positive slope (+): Model becomes MORE confident as generation proceeds
    - Negative slope (-): Model becomes LESS confident (potential hallucination signal)
    - Near zero (≈0): Consistent confidence throughout
    
    Returns:
        Slope value (can be positive, negative, or near zero) or None if insufficient data
    """
    token_lp = log_data.get("token_logprobs", [])
    if not token_lp or len(token_lp) < 2:
        return None
    
    # Convert log probabilities to probabilities
    probs = [math.exp(lp) for lp in token_lp]
    n = len(probs)
    
    # Token positions (1-indexed for interpretability)
    positions = np.arange(1, n + 1)
    probs_array = np.array(probs)
    
    # Calculate means
    mean_position = positions.mean()
    mean_prob = probs_array.mean()
    
    # Calculate slope using least squares formula
    # Slope = Σ(i - ī)(p(i) - p̄) / Σ(i - ī)²
    numerator = np.sum((positions - mean_position) * (probs_array - mean_prob))
    denominator = np.sum((positions - mean_position) ** 2)
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    return float(slope)

def compute_logit_metrics(log_data: dict) -> dict:
    """
    Compute all logit-based metrics including new ones from research papers:
    - arXiv:2405.19648v1: mean_token_prob, prob_variance, prob_std, max_prob_diff
    - arXiv:2501.02518v2 (CHAIR): token_prob_slope
    
    Returns dict with:
    - Original: avg_nll, perplexity, min_prob, max_window_entropy
    - Distribution features: mean_token_prob, prob_variance, prob_std, max_prob_diff
    - Temporal features: token_prob_slope
    """
    return {
        # Original metrics
        "avg_nll": compute_avg_nll(log_data),
        "perplexity": compute_perplexity(log_data),
        "min_prob": compute_min_prob(log_data),
        "max_window_entropy": compute_max_window_entropy(log_data, k_top=5, window_size=5),
        # Distribution features from arXiv:2405.19648v1
        "mean_token_prob": compute_mean_token_prob(log_data),
        "prob_variance": compute_prob_variance(log_data),
        "prob_std": compute_prob_std(log_data),
        "max_prob_diff": compute_max_prob_diff(log_data),
        # Temporal features inspired by CHAIR (arXiv:2501.02518v2)
        "token_prob_slope": compute_token_probability_slope(log_data),
    }

#########################
# 2) SEMANTIC ALIGNMENT
#########################

def openai_embed_batch(
    texts: List[str],
    api_key: str,
    *,
    model: str = "text-embedding-3-small",
    dimensions: Optional[int] = None,
    timeout: int = 90,
    max_retries: int = 8,
    base_delay: float = 2.0,
) -> List[Optional[np.ndarray]]:
    """Batch embeddings via OpenAI embeddings API with retry logic and exponential backoff.
    
    Args:
        texts: list of input strings
        api_key: OpenAI API key
        model: Embedding model to use
        dimensions: Optional dimension override for the embedding
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts (default: 8)
        base_delay: Base delay in seconds for exponential backoff (default: 2.0)
    
    Returns:
        list of vectors (numpy arrays) or None for failed embeddings, 1:1 with texts
        Returns list of None values if all retries exhausted (allows caller to skip)
    """
    if not texts:
        return []
    
    endpoint = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": texts}
    if dimensions is not None:
        payload["dimensions"] = int(dimensions)
    
    logger = logging.getLogger("ci.embed")
    logger.debug(
        "Embeddings request: model=%s, count=%d, dims=%s", model, len(texts), str(dimensions)
    )
    
    import time
    
    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            start = time.time()
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            elapsed = time.time() - start
            
            try:
                data = resp.json()
            except Exception:
                data = {}
            
            # Record usage (thread-safe)
            _record_embedding_usage(resp.status_code, elapsed, data.get("usage", {}))
            
            # Check for retryable HTTP status codes (rate limits and server errors)
            retryable_status_codes = [429, 500, 502, 503, 504, 520]
            if resp.status_code in retryable_status_codes:
                error_message = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s (capped at 5 min)
                    delay = min(base_delay * (2 ** attempt), 300)
                    logger.warning(
                        "[EMBED] HTTP %d for batch of %d texts, retrying in %.1fs (attempt %d/%d): %s",
                        resp.status_code, len(texts), delay, attempt + 1, max_retries, error_message
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error("[EMBED] Max retries (%d) exceeded for batch of %d texts (HTTP %d) - SKIPPING",
                                max_retries, len(texts), resp.status_code)
                    # Return None values instead of raising
                    return [None] * len(texts)
            
            # Check for success status (2xx)
            if resp.status_code >= 200 and resp.status_code < 300:
                # Success - return embeddings
                embeddings = data.get("data", [])
                if embeddings and len(embeddings) == len(texts):
                    return [np.array(item["embedding"], dtype=float) for item in embeddings]
                else:
                    # Malformed response - log and skip
                    logger.error("[EMBED] Malformed response: expected %d embeddings, got %d - SKIPPING",
                                len(texts), len(embeddings))
                    return [None] * len(texts)
            
            # Other HTTP errors (4xx except 429): non-retryable, skip
            logger.error("[EMBED] Non-retryable HTTP %d for batch of %d texts - SKIPPING", resp.status_code, len(texts))
            return [None] * len(texts)
            
        except requests.exceptions.Timeout:
            # Timeout errors - retry
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 300)
                logger.warning(
                    "[EMBED] Timeout for batch of %d texts, retrying in %.1fs (attempt %d/%d)",
                    len(texts), delay, attempt + 1, max_retries
                )
                time.sleep(delay)
                continue
            else:
                logger.error("[EMBED] Max retries (%d) exceeded for batch of %d texts: Timeout - SKIPPING",
                            max_retries, len(texts))
                return [None] * len(texts)
                
        except requests.exceptions.RequestException as e:
            # Connection errors, DNS errors, etc.
            error_str = str(e).lower()
            # Check if it's a retryable error
            is_retryable = any(x in error_str for x in [
                'rate limit', '429', 'too many requests',
                'connection', 'timeout', 'dns',
                '500', '502', '503', '504', '520'
            ])
            
            if is_retryable and attempt < max_retries - 1:
                # Exponential backoff (capped at 5 minutes)
                delay = min(base_delay * (2 ** attempt), 300)
                logger.warning(
                    "[EMBED] Retryable error for batch of %d texts, retrying in %.1fs (attempt %d/%d): %s",
                    len(texts), delay, attempt + 1, max_retries, str(e)[:100]
                )
                time.sleep(delay)
                continue
            else:
                # Non-retryable error or max retries exceeded - log and skip
                if is_retryable:
                    logger.error("[EMBED] Max retries (%d) exceeded for batch of %d texts: %s - SKIPPING",
                                max_retries, len(texts), str(e)[:100])
                else:
                    logger.error("[EMBED] Non-retryable error for batch of %d texts: %s - SKIPPING", 
                                len(texts), str(e)[:100])
                return [None] * len(texts)
    
    # If we exhausted all retries without success, return None values
    logger.error("[EMBED] Failed to embed batch of %d texts after %d attempts - SKIPPING", len(texts), max_retries)
    return [None] * len(texts)


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)

def compute_alignment(text1: str, text2: str, openai_key: str) -> Optional[float]:
    """Single-pair cosine similarity using embeddings (backwards compatible).
    
    Returns:
        Cosine similarity (0.0-1.0) or None if embedding failed
    """
    if not text1 or not text2:
        return 0.0
    
    emb_results = openai_embed_batch([text1], openai_key)
    if not emb_results or emb_results[0] is None:
        return None
    emb1 = emb_results[0]
    
    emb_results = openai_embed_batch([text2], openai_key)
    if not emb_results or emb_results[0] is None:
        return None
    emb2 = emb_results[0]
    
    return _cosine(emb1, emb2)


def compute_alignment_batch(
    narrative: str,
    chunks: List[str],
    openai_key: str,
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
    dimensions: Optional[int] = None,
) -> Optional[float]:
    """Compute max cosine similarity between narrative and list of chunks via batched embeddings.
    
    Args:
        narrative: The narrative text to compare
        chunks: List of text chunks to compare against
        openai_key: OpenAI API key
        model: Embedding model to use
        batch_size: Batch size for embedding API calls
        dimensions: Optional dimension override
    
    Returns:
        Maximum cosine similarity or None if embedding failed
    """
    if not narrative or not chunks:
        return None
    logging.getLogger("ci.embed").debug(
        "Align: narrative_chars=%d, chunks=%d, batch_size=%d, model=%s",
        len(narrative), len(chunks), batch_size, model,
    )
    
    # Embed narrative once
    v1_list = openai_embed_batch([narrative], openai_key, model=model, dimensions=dimensions)
    if not v1_list or v1_list[0] is None:
        logging.getLogger("ci.embed").warning("Failed to embed narrative, skipping alignment")
        return None
    v1 = v1_list[0]
    
    best: Optional[float] = None
    # Batch chunks
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        logging.getLogger("ci.embed").debug("Embedding chunk batch of size %d", len(batch))
        v2_list = openai_embed_batch(batch, openai_key, model=model, dimensions=dimensions)
        
        # Skip None values from failed embeddings
        for v2 in v2_list:
            if v2 is None:
                continue
            sim = _cosine(v1, v2)
            if best is None or sim > best:
                best = sim
    
    return best

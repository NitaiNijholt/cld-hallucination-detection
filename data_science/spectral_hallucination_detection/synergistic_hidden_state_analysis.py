"""
Synergistic Hidden State Analysis for Hallucination Detection

Implementation of synergistic information theory (Quax et al., 2017) 
"Quantifying Synergistic Information Using Intermediate Stochastic Variables"
applied to LLM hidden states for hallucination detection.

Key insight: Hallucinations emerge from SYNERGISTIC patterns across layers
that cannot be predicted from any single layer alone.

Theory: 
- Synergistic Random Variables (SRVs) satisfy:
  * I(S:X) > 0 (has info about complete set)
  * ∀i: I(S:Xi) = 0 (zero info about individual layers)
  
- High synergy ratio → hallucination (fragile, cross-layer dependencies)
- Low synergy ratio → factual (redundant, grounded in individual layers)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import torch
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class SynergisticMetrics:
    """Metrics from synergistic information analysis."""
    synergistic_entropy: float
    max_individual_mi: float
    synergy_ratio: float
    num_msrvs: int
    hallucination_risk: float
    individual_layer_mi: List[float]
    orthogonality_quality: float
    theoretical_upper_bound: float


class SynergisticHiddenStateAnalyzer:
    """
    Analyzes hidden states across transformer layers using synergistic information theory.
    
    Detects hallucinations by identifying when output depends on complex cross-layer
    patterns that cannot be predicted from individual layers.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        k_top_eigenvectors: int = 5,
        min_eigenvalue: float = 1e-10
    ):
        """
        Initialize synergistic analyzer.
        
        Args:
            epsilon: Small value for numerical stability
            k_top_eigenvectors: Number of top eigenvectors to use for SRV construction
            min_eigenvalue: Minimum eigenvalue threshold for stability
        """
        self.epsilon = epsilon
        self.k_top_eigenvectors = k_top_eigenvectors
        self.min_eigenvalue = min_eigenvalue
    
    def compute_differential_entropy(self, X: torch.Tensor) -> float:
        """
        Compute differential entropy H(X) for continuous distribution.
        
        For Gaussian approximation: H(X) = 0.5 * log(2πe * det(Cov(X)))
        
        Args:
            X: Tensor of shape (n_samples, n_features) or flattened (n_elements,)
            
        Returns:
            Differential entropy in nats (natural logarithm)
        """
        # Flatten if needed and add batch dimension
        if X.dim() == 1:
            X = X.unsqueeze(0)
        elif X.dim() > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Handle single sample case
        if X.shape[0] == 1:
            # Return entropy based on variance
            variance = X.var(dim=1).item() + self.epsilon
            return 0.5 * np.log(2 * np.pi * np.e * variance)
        
        # Compute covariance matrix
        try:
            cov = torch.cov(X.T)
            
            # Add small regularization for numerical stability
            cov = cov + self.epsilon * torch.eye(cov.shape[0], device=cov.device)
            
            # Compute determinant
            det = torch.linalg.det(cov)
            
            # Avoid log of negative/zero
            if det <= 0:
                det = self.epsilon
            
            # H(X) = 0.5 * log(2πe * det(Cov))
            entropy = 0.5 * torch.log(2 * np.pi * np.e * det)
            return entropy.item()
            
        except Exception as e:
            logger.warning(f"Error computing differential entropy: {e}")
            # Fallback: use variance-based estimate
            variance = X.var().item() + self.epsilon
            return 0.5 * np.log(2 * np.pi * np.e * variance)
    
    def extract_logit_features(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Extract summary statistics from output logits.
        
        Converts high-dimensional logits (50K+) to low-dimensional
        interpretable features for stable MI calculation.
        
        Args:
            logits: Output logits tensor
            
        Returns:
            Feature vector (5 dimensions)
        """
        # Compute probabilities
        probs = torch.softmax(logits.float(), dim=-1)
        
        # Extract features
        features = torch.tensor([
            probs.max().item(),                                    # Max probability
            probs.mean().item(),                                   # Mean probability
            probs.std().item(),                                    # Std probability
            -(probs * torch.log(probs + 1e-10)).sum().item(),     # Shannon entropy
            (probs ** 2).sum().item(),                            # Inverse participation ratio
        ], dtype=torch.float32)
        
        return features
    
    def compute_mutual_information(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor,
        is_output_logits: bool = False
    ) -> float:
        """
        Compute mutual information I(X:Y) = H(X) + H(Y) - H(X,Y)
        
        Uses differential entropy for continuous distributions (hidden states).
        For output logits, extracts summary features for numerical stability.
        
        Args:
            X: First variable tensor
            Y: Second variable tensor
            is_output_logits: Whether Y is output logits (needs feature extraction)
            
        Returns:
            Mutual information in nats
        """
        # Handle output logits specially
        if is_output_logits and Y.numel() > 10000:
            Y = self.extract_logit_features(Y)
        
        # Flatten tensors
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        # Make same length (truncate longer one)
        min_len = min(len(X_flat), len(Y_flat))
        X_flat = X_flat[:min_len]
        Y_flat = Y_flat[:min_len]
        
        # Skip if too few elements
        if min_len < 2:
            return 0.0
        
        try:
            # Compute individual entropies
            h_x = self.compute_differential_entropy(X_flat)
            h_y = self.compute_differential_entropy(Y_flat)
            
            # Compute joint entropy H(X,Y)
            XY = torch.stack([X_flat, Y_flat], dim=0)
            h_xy = self.compute_differential_entropy(XY.T)
            
            # Check for NaN
            if np.isnan(h_x) or np.isnan(h_y) or np.isnan(h_xy):
                logger.debug(f"NaN in entropy: H(X)={h_x}, H(Y)={h_y}, H(X,Y)={h_xy}")
                return 0.0
            
            # I(X:Y) = H(X) + H(Y) - H(X,Y)
            mi = h_x + h_y - h_xy
            
            # MI is non-negative by definition
            return max(0.0, mi)
            
        except Exception as e:
            logger.debug(f"MI calculation error: {e}")
            return 0.0
    
    def construct_msrvs(
        self, 
        hidden_states_all_layers: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Construct Maximally Synergistic Random Variables (MSRVs).
        
        MSRVs are patterns that satisfy:
        - I(S:X) > 0 (non-zero info about complete set)
        - ∀i: I(S:Xi) = 0 (zero info about individual layers)
        
        Implementation: Use eigenvalue decomposition of cross-layer covariance
        to extract synergistic modes orthogonal to individual layers.
        
        Args:
            hidden_states_all_layers: List of hidden state tensors, one per layer
            
        Returns:
            List of MSRV tensors
        """
        n_layers = len(hidden_states_all_layers)
        if n_layers < 2:
            logger.warning("Need at least 2 layers for synergistic analysis")
            return []
        
        msrvs = []
        
        # Pairwise analysis between layers (analogous to XOR in paper's example)
        for i in range(n_layers):
            for j in range(i + 1, min(i + 3, n_layers)):  # Limit pairs for efficiency
                h_i = hidden_states_all_layers[i].flatten()
                h_j = hidden_states_all_layers[j].flatten()
                
                # Make same length
                min_len = min(len(h_i), len(h_j))
                h_i = h_i[:min_len]
                h_j = h_j[:min_len]
                
                # Skip if too small
                if min_len < 10:
                    continue
                
                try:
                    # Compute cross-layer covariance
                    stacked = torch.stack([h_i, h_j], dim=0)
                    cov_ij = torch.cov(stacked)
                    
                    # Add regularization
                    cov_ij = cov_ij + self.epsilon * torch.eye(2, device=cov_ij.device)
                    
                    # Eigenvalue decomposition to find synergistic modes
                    eigenvals, eigenvecs = torch.linalg.eigh(cov_ij)
                    
                    # Filter by minimum eigenvalue
                    valid_idx = eigenvals > self.min_eigenvalue
                    if not valid_idx.any():
                        continue
                    
                    # Top eigenvector represents strongest synergistic mode
                    # (component orthogonal to individual projections)
                    top_idx = torch.argmax(eigenvals)
                    srv_weights = eigenvecs[:, top_idx]
                    
                    # Construct SRV as weighted combination
                    # This is orthogonal to individual layer projections
                    srv_ij = srv_weights[0] * h_i + srv_weights[1] * h_j
                    
                    # Verify SRV conditions (at least approximately)
                    mi_i = self.compute_mutual_information(srv_ij, h_i)
                    mi_j = self.compute_mutual_information(srv_ij, h_j)
                    
                    # SRV should have low MI with individuals
                    # We use a threshold rather than exact zero
                    if mi_i < 0.5 and mi_j < 0.5:
                        msrvs.append(srv_ij)
                        
                except Exception as e:
                    logger.debug(f"Failed to construct MSRV for layers {i},{j}: {e}")
                    continue
        
        logger.info(f"Constructed {len(msrvs)} MSRVs from {n_layers} layers")
        return msrvs
    
    def orthogonalize_msrvs(self, msrvs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply orthogonal decomposition to MSRVs per paper's Definition 1.
        
        Ensures MSRVs are independent: I(Si⊥ : Sj⊥) = 0
        Uses Gram-Schmidt orthogonalization process.
        
        Args:
            msrvs: List of MSRV tensors
            
        Returns:
            List of orthogonalized MSRVs
        """
        if len(msrvs) <= 1:
            return msrvs
        
        orthogonal_msrvs = []
        
        for i, srv in enumerate(msrvs):
            # Start with original SRV
            srv_orth = srv.clone()
            
            # Remove projections onto all previous orthogonal MSRVs
            for prev_srv in orthogonal_msrvs:
                # Make same length
                min_len = min(len(srv_orth), len(prev_srv))
                srv_orth_trunc = srv_orth[:min_len]
                prev_srv_trunc = prev_srv[:min_len]
                
                # Compute projection
                dot_product = torch.dot(srv_orth_trunc, prev_srv_trunc)
                norm_sq = torch.dot(prev_srv_trunc, prev_srv_trunc)
                
                if norm_sq > self.epsilon:
                    projection = (dot_product / norm_sq) * prev_srv_trunc
                    
                    # Subtract projection (Gram-Schmidt)
                    if len(srv_orth) == len(projection):
                        srv_orth = srv_orth - projection
                    else:
                        # Handle length mismatch
                        srv_orth[:min_len] = srv_orth[:min_len] - projection
            
            # Normalize
            norm = torch.norm(srv_orth)
            if norm > self.epsilon:
                srv_orth = srv_orth / norm
                orthogonal_msrvs.append(srv_orth)
        
        logger.info(f"Orthogonalized {len(msrvs)} MSRVs → {len(orthogonal_msrvs)} independent")
        return orthogonal_msrvs
    
    def verify_srv_conditions(
        self,
        srv: torch.Tensor,
        individual_layers: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Verify that constructed SRV satisfies theoretical conditions.
        
        From paper Equation (2):
        - I(S:X) > 0
        - ∀i: I(S:Xi) = 0
        
        Args:
            srv: Synergistic random variable
            individual_layers: List of individual layer hidden states
            
        Returns:
            Dictionary with verification metrics
        """
        # Compute MI with each individual layer
        individual_mis = []
        for layer in individual_layers:
            mi = self.compute_mutual_information(srv, layer)
            individual_mis.append(mi)
        
        # Compute MI with combined layers (approximate by average)
        combined = torch.cat([l.flatten()[:1000] for l in individual_layers])
        mi_total = self.compute_mutual_information(srv, combined)
        
        return {
            'mi_with_total': mi_total,
            'max_individual_mi': max(individual_mis) if individual_mis else 0.0,
            'mean_individual_mi': np.mean(individual_mis) if individual_mis else 0.0,
            'satisfies_conditions': mi_total > 0.1 and max(individual_mis) < 0.5
        }
    
    def compute_synergistic_entropy(
        self,
        msrvs: List[torch.Tensor],
        output_logits: torch.Tensor
    ) -> float:
        """
        Compute synergistic entropy per paper's Equation (5):
        
        I_syn(X → Y) = Σ I(S_i⊥ : Y)
        
        where S_i⊥ are orthogonalized MSRVs.
        
        Args:
            msrvs: List of orthogonalized MSRVs
            output_logits: Output logits/probabilities from model
            
        Returns:
            Synergistic entropy value
        """
        if not msrvs:
            return 0.0
        
        synergistic_entropy = 0.0
        
        # Extract logit features for stable MI calculation
        logit_features = self.extract_logit_features(output_logits)
        
        for srv in msrvs:
            # Compute I(srv : logit_features)
            mi = self.compute_mutual_information(srv, logit_features)
            synergistic_entropy += mi
        
        return synergistic_entropy
    
    def verify_orthogonality(self, msrvs: List[torch.Tensor]) -> float:
        """
        Verify orthogonality quality of MSRVs.
        
        From paper Definition 1: orthogonal MSRVs should have I(Si⊥ : Sj⊥) ≈ 0
        
        Args:
            msrvs: List of orthogonalized MSRVs
            
        Returns:
            Orthogonality quality score [0,1] where 1 = perfect orthogonality
        """
        if len(msrvs) < 2:
            return 1.0
        
        independence_scores = []
        
        for i in range(len(msrvs)):
            for j in range(i + 1, len(msrvs)):
                mi = self.compute_mutual_information(msrvs[i], msrvs[j])
                # Perfect independence → MI = 0 → score = 1
                # Strong dependence → MI high → score → 0
                score = 1.0 / (1.0 + mi)
                independence_scores.append(score)
        
        return float(np.mean(independence_scores)) if independence_scores else 1.0
    
    def analyze(
        self,
        hidden_states_all_layers: List[torch.Tensor],
        output_logits: torch.Tensor
    ) -> SynergisticMetrics:
        """
        Main analysis function: compute synergistic information metrics.
        
        Args:
            hidden_states_all_layers: List of hidden states from each transformer layer
            output_logits: Final output logits/probabilities
            
        Returns:
            SynergisticMetrics object with all computed metrics
        """
        logger.info(f"Analyzing {len(hidden_states_all_layers)} layers")
        
        # Step 1: Construct MSRVs
        msrvs = self.construct_msrvs(hidden_states_all_layers)
        
        if not msrvs:
            logger.warning("No MSRVs constructed, returning zero metrics")
            return SynergisticMetrics(
                synergistic_entropy=0.0,
                max_individual_mi=0.0,
                synergy_ratio=0.0,
                num_msrvs=0,
                hallucination_risk=0.0,
                individual_layer_mi=[],
                orthogonality_quality=1.0,
                theoretical_upper_bound=0.0
            )
        
        # Step 2: Orthogonalize MSRVs
        msrvs_orth = self.orthogonalize_msrvs(msrvs)
        
        # Step 3: Compute synergistic entropy (Equation 5)
        synergistic_entropy = self.compute_synergistic_entropy(msrvs_orth, output_logits)
        
        # Step 4: Compute individual layer MI with output
        # Extract logit features once for efficiency
        logit_features = self.extract_logit_features(output_logits)
        
        individual_layer_mi = []
        for layer_hidden in hidden_states_all_layers:
            mi = self.compute_mutual_information(layer_hidden, logit_features)
            individual_layer_mi.append(mi)
        
        max_individual_mi = max(individual_layer_mi) if individual_layer_mi else 0.0
        
        # Step 5: Compute synergy ratio
        total_mi = max_individual_mi + synergistic_entropy
        synergy_ratio = synergistic_entropy / total_mi if total_mi > self.epsilon else 0.0
        
        # Clip to [0, 1]
        synergy_ratio = np.clip(synergy_ratio, 0.0, 1.0)
        
        # Step 6: Verify orthogonality quality
        orthogonality_quality = self.verify_orthogonality(msrvs_orth)
        
        # Step 7: Compute theoretical upper bound (Equation 17)
        # I(S:X) ≤ H(X) - max_i H(X_i)
        total_entropy = sum(self.compute_differential_entropy(h) for h in hidden_states_all_layers)
        max_layer_entropy = max(self.compute_differential_entropy(h) for h in hidden_states_all_layers)
        theoretical_upper_bound = total_entropy - max_layer_entropy
        
        # Hallucination risk = synergy ratio (high synergy → hallucination)
        hallucination_risk = synergy_ratio
        
        metrics = SynergisticMetrics(
            synergistic_entropy=float(synergistic_entropy),
            max_individual_mi=float(max_individual_mi),
            synergy_ratio=float(synergy_ratio),
            num_msrvs=len(msrvs_orth),
            hallucination_risk=float(hallucination_risk),
            individual_layer_mi=[float(mi) for mi in individual_layer_mi],
            orthogonality_quality=float(orthogonality_quality),
            theoretical_upper_bound=float(theoretical_upper_bound)
        )
        
        logger.info(f"Synergy analysis: ratio={synergy_ratio:.3f}, "
                   f"risk={hallucination_risk:.3f}, MSRVs={len(msrvs_orth)}")
        
        return metrics










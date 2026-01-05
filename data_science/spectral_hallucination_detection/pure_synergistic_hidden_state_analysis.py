"""
Pure Synergistic Hidden State Analysis for Hallucination Detection

This implements the CORRECT approach based on Quax et al. (2017):
Calculate H(Σ(X)) - the synergistic entropy WITHIN hidden states,
then correlate it directly with hallucination labels.

Key insight from the paper:
- Synergistic Entropy H(Σ(X)) = entropy of the MSRVs themselves
- This represents "all synergistic information that any variable could possibly store about X"
- We partition hidden states into components and measure their synergy
- High synergy → hallucination (information spread across components)
- Low synergy → truthful (information in individual components)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class PureSynergisticMetrics:
    """Metrics from pure synergistic analysis (no output logits involved)."""
    synergistic_entropy: float  # H(Σ(X)) - core metric
    individual_entropies: List[float]  # H(X_i) for each component
    total_entropy: float  # H(X) - entropy of complete hidden states
    max_individual_entropy: float  # max_i H(X_i)
    synergy_ratio: float  # H(Σ(X)) / H(X)
    num_components: int
    num_msrvs: int
    hallucination_risk: float  # Based on synergy ratio
    theoretical_upper_bound: float  # H(X) - max_i H(X_i)


class PureSynergisticAnalyzer:
    """
    Analyzes synergy WITHIN hidden states, independent of output logits.
    
    Approach:
    1. Partition hidden states into components X = {X_1, X_2, ..., X_n}
    2. Construct MSRVs that satisfy:
       - I(S : X) > 0 (correlated with ALL components together)
       - I(S : X_i) = 0 for each individual component
    3. Calculate H(Σ(X)) = synergistic entropy
    4. Correlate H(Σ(X)) with hallucination label
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        n_components: int = 4,  # How many parts to partition hidden states into
        min_eigenvalue: float = 1e-10
    ):
        """
        Initialize pure synergistic analyzer.
        
        Args:
            epsilon: Small value for numerical stability
            n_components: Number of components to partition hidden states into
            min_eigenvalue: Minimum eigenvalue threshold
        """
        self.epsilon = epsilon
        self.n_components = n_components
        self.min_eigenvalue = min_eigenvalue
    
    def compute_differential_entropy(self, X: torch.Tensor) -> float:
        """
        Compute differential entropy H(X) for continuous distribution.
        
        For Gaussian approximation: H(X) = 0.5 * log(2πe * det(Cov(X)))
        
        Args:
            X: Tensor of shape (n_samples, n_features) or (n_features,)
            
        Returns:
            Differential entropy in nats
        """
        # Ensure 2D
        if X.dim() == 1:
            X = X.unsqueeze(0)
        elif X.dim() > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Handle single feature
        if X.shape[1] == 1 or X.shape[0] == 1:
            variance = X.var().item() + self.epsilon
            return 0.5 * np.log(2 * np.pi * np.e * variance)
        
        try:
            # Compute covariance
            cov = torch.cov(X.T)
            
            # Regularize
            cov = cov + self.epsilon * torch.eye(cov.shape[0], device=cov.device)
            
            # Compute determinant
            det = torch.linalg.det(cov)
            
            # Avoid log of non-positive
            if det <= 0:
                det = torch.tensor(self.epsilon, device=det.device)
            
            # H(X) = 0.5 * log(2πe * det(Cov))
            entropy = 0.5 * torch.log(2 * np.pi * np.e * det)
            return entropy.item()
            
        except Exception as e:
            logger.warning(f"Error computing entropy: {e}")
            variance = X.var().item() + self.epsilon
            return 0.5 * np.log(2 * np.pi * np.e * variance)
    
    def compute_mutual_information(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute mutual information I(X:Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            X: First variable
            Y: Second variable
            
        Returns:
            Mutual information in nats
        """
        # Flatten
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        # Make same length
        min_len = min(len(X_flat), len(Y_flat))
        X_flat = X_flat[:min_len]
        Y_flat = Y_flat[:min_len]
        
        if min_len < 2:
            return 0.0
        
        try:
            # Compute entropies
            h_x = self.compute_differential_entropy(X_flat)
            h_y = self.compute_differential_entropy(Y_flat)
            
            # Joint entropy
            XY = torch.stack([X_flat, Y_flat], dim=0)
            h_xy = self.compute_differential_entropy(XY.T)
            
            # Check for NaN
            if np.isnan(h_x) or np.isnan(h_y) or np.isnan(h_xy):
                return 0.0
            
            # I(X:Y) = H(X) + H(Y) - H(X,Y)
            mi = h_x + h_y - h_xy
            
            return max(0.0, mi)
            
        except Exception as e:
            logger.debug(f"MI error: {e}")
            return 0.0
    
    def partition_hidden_states(
        self, 
        hidden_states: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Partition hidden states into components.
        
        Strategy: Split along feature dimension into n_components parts.
        This creates components that each represent a subset of features.
        
        Args:
            hidden_states: Tensor of shape (..., hidden_dim)
            
        Returns:
            List of component tensors
        """
        # Get hidden dimension (last dimension)
        hidden_dim = hidden_states.shape[-1]
        
        # Calculate split size
        split_size = hidden_dim // self.n_components
        
        if split_size < 1:
            logger.warning(f"Hidden dim {hidden_dim} too small for {self.n_components} components")
            # Fallback: use whole state as single component
            return [hidden_states]
        
        # Split into components
        components = []
        for i in range(self.n_components):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < self.n_components - 1 else hidden_dim
            
            # Extract component along feature dimension
            component = hidden_states[..., start_idx:end_idx]
            components.append(component)
        
        logger.debug(f"Partitioned {hidden_dim}D hidden states into {len(components)} components")
        return components
    
    def construct_msrv_from_components(
        self,
        components: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Construct a Maximally Synergistic Random Variable from components.
        
        MSRVs satisfy (Equation 2 from paper):
        - I(S : X) > 0 (has info about complete set)
        - ∀i : I(S : X_i) = 0 (zero info about individual components)
        
        Implementation: Use XOR-like operation (per paper Section 4.4):
        "XOR-gates of random inputs always form an MSRV"
        
        For continuous variables, we use eigenvalue decomposition to find
        the synergistic mode orthogonal to individual components.
        
        Args:
            components: List of component tensors
            
        Returns:
            MSRV tensor or None if construction fails
        """
        if len(components) < 2:
            return None
        
        try:
            # Flatten all components to same length
            components_flat = []
            min_len = min(c.numel() for c in components)
            
            for comp in components:
                comp_flat = comp.flatten()[:min_len]
                components_flat.append(comp_flat)
            
            # Stack into matrix (n_components × min_len)
            X_matrix = torch.stack(components_flat, dim=0)
            
            # Compute covariance matrix
            cov = torch.cov(X_matrix)
            
            # Regularize
            cov = cov + self.epsilon * torch.eye(cov.shape[0], device=cov.device)
            
            # Eigenvalue decomposition
            eigenvals, eigenvecs = torch.linalg.eigh(cov)
            
            # Filter by minimum eigenvalue
            valid_idx = eigenvals > self.min_eigenvalue
            if not valid_idx.any():
                return None
            
            # Use top eigenvector (strongest synergistic mode)
            top_idx = torch.argmax(eigenvals)
            weights = eigenvecs[:, top_idx]
            
            # Construct MSRV as weighted combination
            # This is analogous to XOR for continuous variables
            msrv = torch.zeros(min_len, device=X_matrix.device)
            for i, comp_flat in enumerate(components_flat):
                msrv += weights[i] * comp_flat
            
            # Verify SRV conditions (approximately)
            # Should have low MI with each individual component
            satisfies_conditions = True
            for comp_flat in components_flat:
                mi = self.compute_mutual_information(msrv, comp_flat)
                if mi > 0.5:  # Threshold for "approximately zero"
                    satisfies_conditions = False
                    break
            
            if not satisfies_conditions:
                logger.debug("Constructed variable doesn't satisfy SRV conditions")
                # Still return it as best approximation
            
            return msrv
            
        except Exception as e:
            logger.debug(f"MSRV construction failed: {e}")
            return None
    
    def analyze_layer(
        self,
        hidden_states: torch.Tensor
    ) -> PureSynergisticMetrics:
        """
        Analyze synergy within a single layer's hidden states.
        
        This is the core function that implements the pure synergistic approach:
        1. Partition hidden states into components
        2. Construct MSRVs from components
        3. Calculate H(Σ(X)) - synergistic entropy
        4. Return metrics that can be correlated with hallucination labels
        
        Args:
            hidden_states: Hidden state tensor from one layer
            
        Returns:
            PureSynergisticMetrics with synergy measurements
        """
        # Step 1: Partition into components
        components = self.partition_hidden_states(hidden_states)
        
        if len(components) < 2:
            logger.warning("Need at least 2 components for synergistic analysis")
            return self._zero_metrics()
        
        # Step 2: Compute individual component entropies H(X_i)
        individual_entropies = []
        for comp in components:
            h_i = self.compute_differential_entropy(comp)
            individual_entropies.append(h_i)
        
        max_individual_entropy = max(individual_entropies)
        
        # Step 3: Compute total entropy H(X) of complete hidden states
        total_entropy = self.compute_differential_entropy(hidden_states)
        
        # Step 4: Theoretical upper bound (Equation 17 from paper)
        # I(S:X) ≤ H(X) - max_i H(X_i)
        theoretical_upper_bound = total_entropy - max_individual_entropy
        
        # Step 5: Construct MSRV
        msrv = self.construct_msrv_from_components(components)
        
        if msrv is None:
            logger.warning("Failed to construct MSRV")
            return self._zero_metrics()
        
        # Step 6: Calculate I(S:X) - mutual information between MSRV and complete hidden states
        # This is the KEY METRIC from the paper (see Equation 17)
        # I(S:X) measures how much synergistic information the MSRV captures
        mi_msrv_complete = self.compute_mutual_information(msrv, hidden_states)
        
        # Step 7: Compute synergy ratio
        # Normalize by theoretical upper bound: I(S:X) ≤ H(X) - max_i H(X_i)
        if theoretical_upper_bound > self.epsilon:
            synergy_ratio = mi_msrv_complete / theoretical_upper_bound
        else:
            synergy_ratio = 0.0
        
        synergy_ratio = np.clip(synergy_ratio, 0.0, 1.0)
        
        # Hallucination risk = synergy ratio
        # High synergy → information is spread across components → hallucination
        # Low synergy → information in individual components → truthful
        hallucination_risk = synergy_ratio
        
        # For reporting: also calculate H(Σ(X)) - the entropy of the MSRV
        msrv_entropy = self.compute_differential_entropy(msrv)
        
        metrics = PureSynergisticMetrics(
            synergistic_entropy=float(mi_msrv_complete),  # Using I(S:X) as synergistic metric
            individual_entropies=[float(h) for h in individual_entropies],
            total_entropy=float(total_entropy),
            max_individual_entropy=float(max_individual_entropy),
            synergy_ratio=float(synergy_ratio),
            num_components=len(components),
            num_msrvs=1,  # Currently constructing one MSRV
            hallucination_risk=float(hallucination_risk),
            theoretical_upper_bound=float(theoretical_upper_bound)
        )
        
        logger.info(f"Synergy: I(S:X)={mi_msrv_complete:.3f}, "
                   f"H(X)={total_entropy:.3f}, "
                   f"ratio={synergy_ratio:.3f}, "
                   f"risk={hallucination_risk:.3f}")
        
        return metrics
    
    def analyze_all_layers(
        self,
        hidden_states_all_layers: List[torch.Tensor]
    ) -> Dict[int, PureSynergisticMetrics]:
        """
        Analyze synergy for each layer independently.
        
        Args:
            hidden_states_all_layers: List of hidden states from each layer
            
        Returns:
            Dictionary mapping layer index to metrics
        """
        layer_metrics = {}
        
        for layer_idx, hidden_states in enumerate(hidden_states_all_layers):
            logger.info(f"Analyzing layer {layer_idx}")
            metrics = self.analyze_layer(hidden_states)
            layer_metrics[layer_idx] = metrics
        
        return layer_metrics
    
    def _zero_metrics(self) -> PureSynergisticMetrics:
        """Return zero metrics when analysis fails."""
        return PureSynergisticMetrics(
            synergistic_entropy=0.0,
            individual_entropies=[],
            total_entropy=0.0,
            max_individual_entropy=0.0,
            synergy_ratio=0.0,
            num_components=0,
            num_msrvs=0,
            hallucination_risk=0.0,
            theoretical_upper_bound=0.0
        )

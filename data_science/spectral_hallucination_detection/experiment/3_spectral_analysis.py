"""
Phase 3: Spectral Analysis Pipeline

This module performs spectral analysis on extracted attention matrices
to compute power law exponents, detect correlation traps, and calculate
hallucination risk scores.
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.linalg import eigvalsh
import warnings
warnings.filterwarnings('ignore')


class SpectralAnalyzer:
    """Perform spectral analysis on attention matrices."""
    
    def __init__(
        self,
        alpha_optimal: float = 3.0,
        spike_threshold: float = 2.0,
        mp_ratio_threshold: float = 0.1
    ):
        """
        Initialize spectral analyzer.
        
        Args:
            alpha_optimal: Optimal power law exponent (typically 2-4)
            spike_threshold: Threshold for detecting eigenvalue spikes
            mp_ratio_threshold: Marchenko-Pastur ratio threshold
        """
        self.alpha_optimal = alpha_optimal
        self.spike_threshold = spike_threshold
        self.mp_ratio_threshold = mp_ratio_threshold
    
    def compute_eigenvalues(self, attention_matrix: torch.Tensor) -> np.ndarray:
        """
        Compute eigenvalues of attention matrix.
        
        Args:
            attention_matrix: Attention weights [batch, heads, seq, seq]
            
        Returns:
            Array of eigenvalues
        """
        # Handle different tensor shapes
        if attention_matrix.dim() == 4:
            # [batch, heads, seq, seq] -> average over batch and heads
            matrix = attention_matrix.mean(dim=[0, 1])
        elif attention_matrix.dim() == 3:
            # [heads, seq, seq] -> average over heads
            matrix = attention_matrix.mean(dim=0)
        elif attention_matrix.dim() == 2:
            # [seq, seq] -> use directly
            matrix = attention_matrix
        else:
            raise ValueError(f"Unexpected attention matrix shape: {attention_matrix.shape}")
        
        # Convert to numpy
        matrix_np = matrix.cpu().numpy() if isinstance(matrix, torch.Tensor) else matrix
        
        # Make symmetric (attention matrices might not be perfectly symmetric)
        matrix_sym = (matrix_np + matrix_np.T) / 2
        
        # Compute eigenvalues
        try:
            eigenvalues = eigvalsh(matrix_sym)
            # Sort in descending order
            eigenvalues = np.sort(eigenvalues)[::-1]
            # Keep only positive eigenvalues
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
        except Exception as e:
            print(f"Error computing eigenvalues: {e}")
            eigenvalues = np.array([1.0])
        
        return eigenvalues
    
    def fit_power_law_hill(self, eigenvalues: np.ndarray) -> float:
        """
        Fit power law using Hill estimator.
        
        Args:
            eigenvalues: Sorted eigenvalues
            
        Returns:
            Power law exponent (alpha)
        """
        if len(eigenvalues) < 10:
            return 1.0  # Default for too few eigenvalues
        
        # Use top k eigenvalues (Hill estimator)
        k = min(int(len(eigenvalues) * 0.3), 100)  # Use top 30% or max 100
        if k < 2:
            return 1.0
        
        # Hill estimator
        top_k = eigenvalues[:k]
        if top_k[-1] <= 0:
            return 1.0
        
        log_ratios = np.log(top_k[:-1] / top_k[-1])
        alpha = k / np.sum(log_ratios) if np.sum(log_ratios) > 0 else 1.0
        
        return float(alpha)
    
    def detect_correlation_traps(
        self,
        eigenvalues: np.ndarray,
        matrix_size: int
    ) -> Tuple[int, float]:
        """
        Detect correlation traps using Marchenko-Pastur law.
        
        Args:
            eigenvalues: Sorted eigenvalues
            matrix_size: Size of the original matrix
            
        Returns:
            Tuple of (number of spikes, spike ratio)
        """
        if len(eigenvalues) < 2:
            return 0, 0.0
        
        # Marchenko-Pastur bulk edge estimation
        q = len(eigenvalues) / matrix_size if matrix_size > 0 else 1.0
        q = min(q, 1.0)
        
        # Theoretical bulk edge
        sigma = np.mean(eigenvalues) if len(eigenvalues) > 0 else 1.0
        lambda_plus = sigma * (1 + np.sqrt(q))**2
        
        # Count eigenvalues beyond bulk edge
        spikes = np.sum(eigenvalues > lambda_plus * self.spike_threshold)
        spike_ratio = spikes / len(eigenvalues) if len(eigenvalues) > 0 else 0.0
        
        return int(spikes), float(spike_ratio)
    
    def compute_spectral_entropy(self, eigenvalues: np.ndarray) -> float:
        """
        Compute spectral entropy.
        
        Args:
            eigenvalues: Sorted eigenvalues
            
        Returns:
            Spectral entropy
        """
        if len(eigenvalues) == 0:
            return 0.0
        
        # Normalize eigenvalues to probabilities
        eigenvalues_pos = eigenvalues[eigenvalues > 0]
        if len(eigenvalues_pos) == 0:
            return 0.0
        
        probs = eigenvalues_pos / np.sum(eigenvalues_pos)
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return float(entropy)
    
    def compute_stable_rank(self, eigenvalues: np.ndarray) -> float:
        """
        Compute stable rank (effective rank).
        
        Args:
            eigenvalues: Sorted eigenvalues
            
        Returns:
            Stable rank
        """
        if len(eigenvalues) == 0:
            return 1.0
        
        eigenvalues_pos = eigenvalues[eigenvalues > 0]
        if len(eigenvalues_pos) == 0:
            return 1.0
        
        # Stable rank = (sum of eigenvalues)^2 / sum of squared eigenvalues
        sum_eig = np.sum(eigenvalues_pos)
        sum_eig_sq = np.sum(eigenvalues_pos**2)
        
        if sum_eig_sq > 0:
            stable_rank = (sum_eig**2) / sum_eig_sq
        else:
            stable_rank = 1.0
        
        return float(stable_rank)
    
    def analyze_attention_matrix(
        self,
        attention_matrix: torch.Tensor,
        layer_idx: int = 0
    ) -> Dict:
        """
        Perform complete spectral analysis on an attention matrix.
        
        Args:
            attention_matrix: Attention weights
            layer_idx: Layer index
            
        Returns:
            Dictionary of spectral metrics
        """
        # Compute eigenvalues
        eigenvalues = self.compute_eigenvalues(attention_matrix)
        
        # Get matrix size
        if attention_matrix.dim() >= 2:
            matrix_size = attention_matrix.shape[-1]
        else:
            matrix_size = 1
        
        # Compute all metrics
        alpha = self.fit_power_law_hill(eigenvalues)
        num_spikes, spike_ratio = self.detect_correlation_traps(eigenvalues, matrix_size)
        entropy = self.compute_spectral_entropy(eigenvalues)
        stable_rank = self.compute_stable_rank(eigenvalues)
        
        # Compute risk score
        risk_score = self.compute_risk_score(
            alpha, num_spikes, entropy, stable_rank
        )
        
        metrics = {
            'layer_idx': layer_idx,
            'alpha': alpha,
            'num_spikes': num_spikes,
            'spike_ratio': spike_ratio,
            'spectral_entropy': entropy,
            'stable_rank': stable_rank,
            'risk_score': risk_score,
            'num_eigenvalues': len(eigenvalues),
            'max_eigenvalue': float(np.max(eigenvalues)) if len(eigenvalues) > 0 else 0.0,
            'eigenvalue_gap': float(eigenvalues[0] - eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        }
        
        return metrics
    
    def compute_risk_score(
        self,
        alpha: float,
        num_spikes: int,
        entropy: float,
        stable_rank: float
    ) -> float:
        """
        Compute hallucination risk score.
        
        Args:
            alpha: Power law exponent
            num_spikes: Number of eigenvalue spikes
            entropy: Spectral entropy
            stable_rank: Stable rank
            
        Returns:
            Risk score (0-1, higher means more likely to hallucinate)
        """
        # Alpha component (α < 2 indicates overfitting)
        alpha_risk = max(0, 1 - alpha / 2.0) if alpha < 2 else 0
        
        # Spike component (more spikes = higher risk)
        spike_risk = min(1, num_spikes / 10.0)
        
        # Entropy component (low entropy = collapsed representations)
        entropy_risk = max(0, 1 - entropy / 3.0)
        
        # Stable rank component (low rank = dimensional collapse)
        rank_risk = max(0, 1 - stable_rank / 50.0)
        
        # Weighted combination
        risk_score = (
            0.4 * alpha_risk +
            0.3 * spike_risk +
            0.2 * entropy_risk +
            0.1 * rank_risk
        )
        
        return float(np.clip(risk_score, 0, 1))
    
    def analyze_all_layers(
        self,
        attention_matrices: List[torch.Tensor]
    ) -> Dict:
        """
        Analyze all layers and compute aggregate metrics.
        
        Args:
            attention_matrices: List of attention matrices per layer
            
        Returns:
            Dictionary with per-layer and aggregate metrics
        """
        layer_metrics = []
        
        for layer_idx, attn_matrix in enumerate(attention_matrices):
            if attn_matrix is None:
                continue
            
            metrics = self.analyze_attention_matrix(attn_matrix, layer_idx)
            layer_metrics.append(metrics)
        
        if not layer_metrics:
            return {'error': 'No valid attention matrices to analyze'}
        
        # Compute aggregate metrics
        alphas = [m['alpha'] for m in layer_metrics]
        risk_scores = [m['risk_score'] for m in layer_metrics]
        
        aggregate_metrics = {
            'mean_alpha': float(np.mean(alphas)),
            'std_alpha': float(np.std(alphas)),
            'min_alpha': float(np.min(alphas)),
            'max_alpha': float(np.max(alphas)),
            'mean_risk_score': float(np.mean(risk_scores)),
            'max_risk_score': float(np.max(risk_scores)),
            'total_spikes': sum(m['num_spikes'] for m in layer_metrics),
            'layers_at_risk': sum(1 for m in layer_metrics if m['alpha'] < 2.0),
            'num_layers': len(layer_metrics)
        }
        
        return {
            'layer_metrics': layer_metrics,
            'aggregate_metrics': aggregate_metrics
        }


def process_saved_internals(
    internals_dir: str = "experiment/extracted_internals",
    output_dir: str = "experiment/spectral_metrics"
) -> List[Dict]:
    """
    Process saved internal states and compute spectral metrics.
    
    Args:
        internals_dir: Directory containing extracted internals
        output_dir: Directory to save spectral metrics
        
    Returns:
        List of analysis results
    """
    internals_path = Path(internals_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    analyzer = SpectralAnalyzer()
    results = []
    
    # Process each saved sample
    tensor_files = sorted(internals_path.glob("tensors_*.pt"))
    
    for tensor_file in tensor_files:
        sample_id = tensor_file.stem.replace("tensors_", "")
        
        # Load metadata
        meta_file = internals_path / f"internal_{sample_id}.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load tensors
        tensors = torch.load(tensor_file, map_location='cpu')
        
        # Analyze attention matrices
        if 'attention_matrices' in tensors and tensors['attention_matrices']:
            attn_matrices = tensors['attention_matrices']
            analysis = analyzer.analyze_all_layers(attn_matrices)
            
            # Combine with metadata
            result = {
                'sample_id': sample_id,
                'prompt': metadata.get('prompt', ''),
                'generated_text': metadata.get('generated_text', ''),
                'expected_truthful': metadata.get('expected_truthful'),
                'category': metadata.get('category', 'unknown'),
                'spectral_analysis': analysis
            }
            
            results.append(result)
            
            # Save individual result
            with open(output_path / f"spectral_{sample_id}.json", 'w') as f:
                json.dump(result, f, indent=2)
    
    # Save aggregated results
    with open(output_path / "all_spectral_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} samples")
    
    return results


def main():
    """Main function to run spectral analysis."""
    
    print("="*80)
    print("Phase 3: Spectral Analysis Pipeline")
    print("="*80)
    
    # Process saved internals
    results = process_saved_internals()
    
    if not results:
        print("No results to analyze")
        return
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)
    
    truthful_risks = []
    false_risks = []
    
    for result in results:
        if 'spectral_analysis' not in result:
            continue
        
        risk = result['spectral_analysis'].get('aggregate_metrics', {}).get('mean_risk_score', 0)
        expected = result.get('expected_truthful')
        
        if expected is True:
            truthful_risks.append(risk)
        elif expected is False:
            false_risks.append(risk)
    
    if truthful_risks:
        print(f"Truthful answers: Mean risk = {np.mean(truthful_risks):.3f}")
    if false_risks:
        print(f"False answers: Mean risk = {np.mean(false_risks):.3f}")
    
    if truthful_risks and false_risks:
        # Compute correlation
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(false_risks, truthful_risks)
        print(f"\nT-test: t={t_stat:.3f}, p={p_value:.3f}")
        
        if p_value < 0.05:
            print("✓ Significant difference found between truthful and false answers!")
        else:
            print("✗ No significant difference found (yet - need more data)")
    
    return results


if __name__ == "__main__":
    main()
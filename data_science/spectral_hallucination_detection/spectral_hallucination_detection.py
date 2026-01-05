"""
Spectral Analysis-Based Hallucination Detection for LLMs
Based on WeightWatcher Heavy-Tailed Self-Regularization Theory

This implementation extracts model internals (attention weights, hidden states)
and analyzes their eigenvalue spectra to detect potential hallucinations.

Key concepts:
- Power law exponent α: Well-regularized layers have 2 < α < 4
- Correlation traps: Spikes in eigenvalue spectrum indicate low-dimensional collapse
- Spectral entropy: Measures uncertainty in attention patterns
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
from scipy.linalg import eigvalsh
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SpectralMetrics:
    """Container for spectral analysis metrics"""
    alpha: float  # Power law exponent
    num_spikes: int  # Number of eigenvalue spikes
    spectral_entropy: float  # Entropy of eigenvalue distribution
    stable_rank: float  # Effective dimensionality
    correlation_flow: float  # Change in alpha between layers
    marchenko_pastur_edge: float  # Random matrix theory threshold
    

class SpectralHallucinationDetector:
    """
    Detects hallucinations using spectral analysis of model internals.
    Based on WeightWatcher theory and Heavy-Tailed Self-Regularization.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",  # Small model for local GPU
        device: str = "cuda",
        alpha_optimal: float = 3.0,
        spike_threshold: float = 2.0,
        noise_floor: float = 0.07
    ):
        """
        Initialize the detector with a HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu')
            alpha_optimal: Optimal power law exponent (typically ~3)
            spike_threshold: Threshold for detecting eigenvalue spikes
            noise_floor: Minimum amplitude for meaningful signals
        """
        self.device = device
        self.alpha_optimal = alpha_optimal
        self.spike_threshold = spike_threshold
        self.noise_floor = noise_floor
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            output_attentions=True,
            output_hidden_states=True
        )
        self.model.eval()
        
    def extract_model_internals(self, text: str) -> Dict[str, Any]:
        """
        Extract attention weights and hidden states from model.
        
        Returns:
            Dictionary containing attention matrices and hidden states for each layer
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        
        return {
            'attention_weights': outputs.attentions,  # Tuple of attention matrices per layer
            'hidden_states': outputs.hidden_states,   # Tuple of hidden states per layer
            'logits': outputs.logits,
            'input_ids': inputs.input_ids
        }
    
    def compute_eigenvalue_spectrum(self, matrix: torch.Tensor) -> np.ndarray:
        """
        Compute eigenvalue spectrum of a matrix.
        
        Args:
            matrix: Input matrix (attention or covariance)
            
        Returns:
            Sorted eigenvalues (largest to smallest)
        """
        # Convert to numpy and ensure it's 2D
        if len(matrix.shape) > 2:
            # For attention matrices with multiple heads, average across heads
            matrix = matrix.mean(dim=1)  # Average across heads
            
        matrix_np = matrix.cpu().float().numpy()
        
        # Compute covariance matrix if needed
        if matrix_np.shape[0] != matrix_np.shape[1]:
            matrix_np = np.cov(matrix_np.T)
            
        # Compute eigenvalues
        eigenvalues = eigvalsh(matrix_np)
        return np.sort(eigenvalues)[::-1]  # Sort in descending order
    
    def fit_power_law(self, eigenvalues: np.ndarray, k: Optional[int] = None) -> float:
        """
        Fit power law to eigenvalue distribution and extract alpha exponent.
        Uses Hill estimator for heavy-tailed distributions.
        
        Args:
            eigenvalues: Sorted eigenvalues (largest first)
            k: Number of largest eigenvalues to use (default: sqrt(len))
            
        Returns:
            Power law exponent alpha
        """
        eigenvalues = eigenvalues[eigenvalues > 0]  # Remove zero/negative values
        
        if len(eigenvalues) < 10:
            return float('inf')  # Not enough data
            
        if k is None:
            k = int(np.sqrt(len(eigenvalues)))
            
        k = min(k, len(eigenvalues) - 1)
        
        # Hill estimator for power law exponent
        if k > 0 and eigenvalues[k] > 0:
            log_ratios = np.log(eigenvalues[:k] / eigenvalues[k])
            alpha = k / np.sum(log_ratios) if np.sum(log_ratios) > 0 else float('inf')
        else:
            alpha = float('inf')
            
        return alpha
    
    def detect_correlation_traps(
        self, 
        eigenvalues: np.ndarray, 
        matrix_shape: Tuple[int, int]
    ) -> int:
        """
        Detect correlation traps using Marchenko-Pastur law.
        Identifies spikes beyond random matrix theory predictions.
        
        Args:
            eigenvalues: Sorted eigenvalues
            matrix_shape: Shape of original matrix (for aspect ratio)
            
        Returns:
            Number of significant spikes
        """
        n, m = matrix_shape
        if n == 0 or m == 0:
            return 0
            
        Q = n / m if n < m else m / n
        
        # Marchenko-Pastur edge
        mp_edge = (1 + np.sqrt(Q)) ** 2
        
        # Normalize eigenvalues
        if len(eigenvalues) > 0 and eigenvalues[0] > 0:
            normalized_eigenvalues = eigenvalues / eigenvalues[0]
            # Count spikes above threshold
            num_spikes = np.sum(normalized_eigenvalues > mp_edge * self.spike_threshold)
        else:
            num_spikes = 0
            
        return int(num_spikes)
    
    def compute_spectral_entropy(self, eigenvalues: np.ndarray) -> float:
        """
        Compute spectral entropy of eigenvalue distribution.
        
        Args:
            eigenvalues: Eigenvalue spectrum
            
        Returns:
            Spectral entropy (higher = more uncertainty)
        """
        # Remove zeros and normalize
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) == 0:
            return 0.0
            
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        # Compute entropy
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
        return float(entropy)
    
    def compute_stable_rank(self, matrix: torch.Tensor) -> float:
        """
        Compute stable rank: ||W||_F^2 / ||W||_2^2
        Measures effective dimensionality of the matrix.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Stable rank value
        """
        if len(matrix.shape) > 2:
            matrix = matrix.reshape(-1, matrix.shape[-1])
            
        frobenius_norm_sq = torch.norm(matrix, p='fro') ** 2
        spectral_norm_sq = torch.norm(matrix, p=2) ** 2
        
        if spectral_norm_sq > 0:
            stable_rank = (frobenius_norm_sq / spectral_norm_sq).item()
        else:
            stable_rank = 1.0
            
        return stable_rank
    
    def analyze_layer(
        self, 
        attention_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> SpectralMetrics:
        """
        Perform spectral analysis on a single transformer layer.
        
        Args:
            attention_weights: Attention matrix for this layer
            hidden_states: Hidden state vectors for this layer
            layer_idx: Index of the current layer
            
        Returns:
            SpectralMetrics object with analysis results
        """
        # Analyze attention matrix eigenvalues
        attn_eigenvalues = self.compute_eigenvalue_spectrum(attention_weights[0])
        
        # Fit power law
        alpha = self.fit_power_law(attn_eigenvalues)
        
        # Detect correlation traps
        num_spikes = self.detect_correlation_traps(
            attn_eigenvalues,
            (attention_weights.shape[-2], attention_weights.shape[-1])
        )
        
        # Compute spectral entropy
        spectral_entropy = self.compute_spectral_entropy(attn_eigenvalues)
        
        # Compute stable rank
        stable_rank = self.compute_stable_rank(attention_weights[0])
        
        # Marchenko-Pastur edge
        n, m = attention_weights.shape[-2], attention_weights.shape[-1]
        Q = n / m if n < m else m / n
        mp_edge = (1 + np.sqrt(Q)) ** 2
        
        return SpectralMetrics(
            alpha=alpha,
            num_spikes=num_spikes,
            spectral_entropy=spectral_entropy,
            stable_rank=stable_rank,
            correlation_flow=0.0,  # Will be computed across layers
            marchenko_pastur_edge=mp_edge
        )
    
    def compute_hallucination_score(
        self,
        metrics_per_layer: List[SpectralMetrics]
    ) -> Dict[str, float]:
        """
        Compute overall hallucination risk score from layer metrics.
        
        Args:
            metrics_per_layer: List of metrics for each transformer layer
            
        Returns:
            Dictionary with component scores and overall risk
        """
        if not metrics_per_layer:
            return {'overall_risk': 0.0}
        
        # Component scores
        alpha_scores = []
        spike_scores = []
        entropy_scores = []
        rank_scores = []
        flow_scores = []
        
        for i, metrics in enumerate(metrics_per_layer):
            # Alpha deviation from optimal
            alpha_score = abs(metrics.alpha - self.alpha_optimal) / self.alpha_optimal
            if metrics.alpha < 2:  # Overfitting regime
                alpha_score *= 2.0
            alpha_scores.append(alpha_score)
            
            # Spike detection score
            spike_score = min(metrics.num_spikes / 5.0, 1.0)  # Normalize to [0,1]
            spike_scores.append(spike_score)
            
            # Entropy score (high entropy = high uncertainty)
            entropy_scores.append(metrics.spectral_entropy / 3.0)  # Normalize
            
            # Stable rank score (low rank = potential hallucination)
            rank_score = 1.0 / (metrics.stable_rank + 1.0)
            rank_scores.append(rank_score)
            
            # Correlation flow (changes between layers)
            if i > 0:
                flow = abs(metrics.alpha - metrics_per_layer[i-1].alpha)
                flow_scores.append(flow)
        
        # Compute weighted average
        w_alpha = 0.3
        w_spike = 0.25
        w_entropy = 0.2
        w_rank = 0.15
        w_flow = 0.1
        
        overall_risk = (
            w_alpha * np.mean(alpha_scores) +
            w_spike * np.mean(spike_scores) +
            w_entropy * np.mean(entropy_scores) +
            w_rank * np.mean(rank_scores) +
            w_flow * np.mean(flow_scores) if flow_scores else 0
        )
        
        return {
            'overall_risk': float(overall_risk),
            'alpha_risk': float(np.mean(alpha_scores)),
            'spike_risk': float(np.mean(spike_scores)),
            'entropy_risk': float(np.mean(entropy_scores)),
            'rank_risk': float(np.mean(rank_scores)),
            'flow_risk': float(np.mean(flow_scores)) if flow_scores else 0.0,
            'max_layer_risk': float(np.max([
                alpha_scores[i] + spike_scores[i] 
                for i in range(len(alpha_scores))
            ]) / 2.0)
        }
    
    def detect_hallucination(
        self,
        text: str,
        threshold: float = 0.5
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Main interface for hallucination detection.
        
        Args:
            text: Input text to analyze
            threshold: Risk threshold for hallucination detection
            
        Returns:
            Tuple of (is_hallucinating, detailed_metrics)
        """
        # Extract model internals
        internals = self.extract_model_internals(text)
        
        # Analyze each layer
        metrics_per_layer = []
        
        for layer_idx, (attn_weights, hidden_states) in enumerate(
            zip(internals['attention_weights'], internals['hidden_states'])
        ):
            if attn_weights is not None:
                metrics = self.analyze_layer(attn_weights, hidden_states, layer_idx)
                metrics_per_layer.append(metrics)
        
        # Compute hallucination scores
        scores = self.compute_hallucination_score(metrics_per_layer)
        
        # Determine if hallucinating
        is_hallucinating = scores['overall_risk'] > threshold
        
        # Compile detailed results
        detailed_results = {
            'is_hallucinating': is_hallucinating,
            'scores': scores,
            'layer_metrics': [
                {
                    'layer': i,
                    'alpha': m.alpha,
                    'num_spikes': m.num_spikes,
                    'spectral_entropy': m.spectral_entropy,
                    'stable_rank': m.stable_rank,
                    'mp_edge': m.marchenko_pastur_edge
                }
                for i, m in enumerate(metrics_per_layer)
            ],
            'threshold': threshold,
            'text': text
        }
        
        return is_hallucinating, detailed_results


def run_experiments():
    """Run example experiments demonstrating the detector."""
    
    # Initialize detector
    print("Initializing Spectral Hallucination Detector...")
    detector = SpectralHallucinationDetector(
        model_name="microsoft/phi-2",  # You can use any HF model
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test cases with varying levels of expected hallucination
    test_cases = [
        {
            'text': "The capital of France is Paris.",
            'expected': 'Low risk (factual statement)'
        },
        {
            'text': "The quantum flux capacitor in the iPhone 15 uses antimatter batteries to achieve faster-than-light processing.",
            'expected': 'High risk (technical nonsense)'
        },
        {
            'text': "Scientists have discovered that eating chocolate every day increases lifespan by exactly 23.7 years according to a study.",
            'expected': 'High risk (specific false claim)'
        },
        {
            'text': "Machine learning models use gradient descent for optimization.",
            'expected': 'Low risk (accurate technical statement)'
        },
        {
            'text': "The Battle of Waterloo took place in 1815.",
            'expected': 'Low risk (historical fact)'
        },
        {
            'text': "Abraham Lincoln invented the internet in 1865 while working at NASA.",
            'expected': 'High risk (anachronistic impossibility)'
        }
    ]
    
    print("\n" + "="*80)
    print("Running Hallucination Detection Experiments")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['expected']}")
        print(f"Text: {test_case['text'][:100]}...")
        
        is_hallucinating, results = detector.detect_hallucination(
            test_case['text'],
            threshold=0.5
        )
        
        print(f"Detected as: {'HALLUCINATION' if is_hallucinating else 'TRUTHFUL'}")
        print(f"Overall Risk Score: {results['scores']['overall_risk']:.3f}")
        print(f"  - Alpha Risk: {results['scores']['alpha_risk']:.3f}")
        print(f"  - Spike Risk: {results['scores']['spike_risk']:.3f}")
        print(f"  - Entropy Risk: {results['scores']['entropy_risk']:.3f}")
        print(f"  - Rank Risk: {results['scores']['rank_risk']:.3f}")
        
        # Show per-layer analysis for interesting cases
        if i <= 2:  # Show details for first two cases
            print("\n  Layer-wise Analysis:")
            for layer_metric in results['layer_metrics'][:5]:  # First 5 layers
                print(f"    Layer {layer_metric['layer']}: "
                      f"α={layer_metric['alpha']:.2f}, "
                      f"spikes={layer_metric['num_spikes']}, "
                      f"entropy={layer_metric['spectral_entropy']:.3f}")
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)


if __name__ == "__main__":
    run_experiments()
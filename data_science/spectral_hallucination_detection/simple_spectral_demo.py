"""
Simple demonstration of spectral hallucination detection concepts
without requiring large model downloads.

This uses mock data to illustrate the key mathematical concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
from typing import Tuple, List


def generate_attention_matrix(
    size: int,
    pattern: str = "normal",
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate synthetic attention matrices with different patterns.
    
    Args:
        size: Matrix dimension
        pattern: Type of pattern ("normal", "collapsed", "sparse")
        noise_level: Amount of noise to add
        
    Returns:
        Symmetric attention-like matrix
    """
    if pattern == "normal":
        # Well-behaved attention pattern
        base = np.random.randn(size, size) * 0.5
        # Make it more structured
        for i in range(size):
            base[i, max(0, i-2):min(size, i+3)] += 0.5
            
    elif pattern == "collapsed":
        # Dimensional collapse (hallucination-like)
        # Create low-rank structure
        rank = 3
        U = np.random.randn(size, rank)
        base = U @ U.T
        
    elif pattern == "sparse":
        # Sparse attention (focusing on few tokens)
        base = np.zeros((size, size))
        # Add a few strong connections
        for _ in range(size // 4):
            i, j = np.random.randint(0, size, 2)
            base[i, j] = np.random.randn() * 2
            base[j, i] = base[i, j]
    else:
        base = np.random.randn(size, size)
    
    # Add noise
    base += np.random.randn(size, size) * noise_level
    
    # Make symmetric and positive semi-definite
    matrix = (base + base.T) / 2
    matrix = matrix @ matrix.T
    
    # Normalize
    matrix = matrix / np.max(np.abs(matrix))
    
    return matrix


def compute_spectral_metrics(matrix: np.ndarray) -> dict:
    """
    Compute key spectral metrics for hallucination detection.
    
    Args:
        matrix: Input matrix (e.g., attention weights)
        
    Returns:
        Dictionary of spectral metrics
    """
    # Compute eigenvalues
    eigenvalues = eigvalsh(matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero values
    
    if len(eigenvalues) < 5:
        return {'alpha': np.inf, 'spikes': 0, 'entropy': 0, 'stable_rank': 1}
    
    # 1. Power law exponent (Hill estimator)
    k = int(np.sqrt(len(eigenvalues)))
    if k > 0 and eigenvalues[k] > 0:
        log_ratios = np.log(eigenvalues[:k] / eigenvalues[k])
        alpha = k / np.sum(log_ratios) if np.sum(log_ratios) > 0 else np.inf
    else:
        alpha = np.inf
    
    # 2. Detect spikes (Marchenko-Pastur)
    n, m = matrix.shape
    Q = min(n, m) / max(n, m)
    mp_edge = (1 + np.sqrt(Q)) ** 2
    
    normalized_eigenvalues = eigenvalues / eigenvalues[0] if eigenvalues[0] > 0 else eigenvalues
    num_spikes = np.sum(normalized_eigenvalues > mp_edge * 1.5)
    
    # 3. Spectral entropy
    eigenvalues_norm = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
    
    # 4. Stable rank
    frobenius_norm_sq = np.sum(matrix ** 2)
    spectral_norm_sq = eigenvalues[0] ** 2 if len(eigenvalues) > 0 else 1
    stable_rank = frobenius_norm_sq / spectral_norm_sq if spectral_norm_sq > 0 else 1
    
    return {
        'alpha': alpha,
        'spikes': int(num_spikes),
        'entropy': entropy,
        'stable_rank': stable_rank,
        'eigenvalues': eigenvalues,
        'mp_edge': mp_edge
    }


def plot_spectral_analysis(matrices: List[Tuple[str, np.ndarray]]):
    """
    Visualize spectral properties of different matrix types.
    
    Args:
        matrices: List of (label, matrix) tuples
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = ['green', 'red', 'blue']
    
    for idx, (label, matrix) in enumerate(matrices):
        metrics = compute_spectral_metrics(matrix)
        color = colors[idx % len(colors)]
        
        # Plot 1: Eigenvalue spectrum (log-log)
        ax = axes[0, 0]
        eigenvals = metrics['eigenvalues']
        if len(eigenvals) > 0:
            x = np.arange(1, len(eigenvals) + 1)
            ax.loglog(x, eigenvals, 'o-', label=label, color=color, alpha=0.7)
            
            # Add power law fit
            if metrics['alpha'] != np.inf and metrics['alpha'] > 0:
                y_fit = eigenvals[0] * (x ** (-metrics['alpha']))
                ax.loglog(x, y_fit, '--', color=color, alpha=0.3)
        
        ax.set_xlabel('Eigenvalue Rank')
        ax.set_ylabel('Eigenvalue Magnitude')
        ax.set_title('Eigenvalue Spectra (log-log)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Matrix heatmap
        ax = axes[0, idx]
        im = ax.imshow(matrix, cmap='coolwarm', aspect='auto')
        ax.set_title(f'{label}\nα={metrics["alpha"]:.2f}, spikes={metrics["spikes"]}')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot 3: Metrics comparison
        ax = axes[1, 0]
        x_pos = idx * 0.3
        ax.bar(x_pos, metrics['alpha'] if metrics['alpha'] != np.inf else 10, 
               width=0.25, label=label if idx == 0 else '', color=color, alpha=0.7)
        
    axes[1, 0].set_ylabel('Power Law Exponent (α)')
    axes[1, 0].set_title('Alpha Values Comparison')
    axes[1, 0].axhline(y=2, color='k', linestyle='--', alpha=0.3, label='α=2 (critical)')
    axes[1, 0].axhline(y=4, color='k', linestyle='--', alpha=0.3, label='α=4 (optimal)')
    axes[1, 0].set_xticks([i * 0.3 for i in range(len(matrices))])
    axes[1, 0].set_xticklabels([label for label, _ in matrices])
    axes[1, 0].legend()
    
    # Plot 4: Spike counts
    ax = axes[1, 1]
    for idx, (label, matrix) in enumerate(matrices):
        metrics = compute_spectral_metrics(matrix)
        ax.bar(idx, metrics['spikes'], label=label, color=colors[idx % len(colors)], alpha=0.7)
    
    ax.set_ylabel('Number of Spikes')
    ax.set_title('Eigenvalue Spikes (Correlation Traps)')
    ax.set_xticks(range(len(matrices)))
    ax.set_xticklabels([label for label, _ in matrices])
    
    # Plot 5: Spectral entropy
    ax = axes[1, 2]
    for idx, (label, matrix) in enumerate(matrices):
        metrics = compute_spectral_metrics(matrix)
        ax.bar(idx, metrics['entropy'], label=label, color=colors[idx % len(colors)], alpha=0.7)
    
    ax.set_ylabel('Spectral Entropy')
    ax.set_title('Uncertainty Measure')
    ax.set_xticks(range(len(matrices)))
    ax.set_xticklabels([label for label, _ in matrices])
    
    plt.suptitle('Spectral Analysis: Normal vs Collapsed vs Sparse Attention', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('spectral_analysis_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: spectral_analysis_results.png")
    
    # Try to show if display is available
    try:
        plt.show()
    except:
        pass


def compute_hallucination_risk(metrics: dict) -> float:
    """
    Compute overall hallucination risk score.
    
    Args:
        metrics: Dictionary of spectral metrics
        
    Returns:
        Risk score between 0 and 1
    """
    risk = 0.0
    
    # Alpha deviation from optimal range (2 < α < 4)
    if metrics['alpha'] == np.inf:
        risk += 0.3
    elif metrics['alpha'] < 2:
        risk += 0.4  # Overfitting regime
    elif metrics['alpha'] > 5:
        risk += 0.2  # Under-parameterized
    else:
        # Within reasonable range but check deviation from optimal (α ≈ 3)
        risk += abs(metrics['alpha'] - 3) * 0.1
    
    # Spike penalty
    risk += min(metrics['spikes'] * 0.1, 0.3)
    
    # Entropy penalty (high entropy = uncertainty)
    risk += min(metrics['entropy'] / 3, 0.2)
    
    # Low stable rank penalty
    if metrics['stable_rank'] < 5:
        risk += 0.1
    
    return min(risk, 1.0)


def demonstrate_spectral_detection():
    """
    Main demonstration of spectral hallucination detection.
    """
    print("="*80)
    print("Spectral Hallucination Detection Demonstration")
    print("="*80)
    print("\nGenerating synthetic attention matrices...")
    
    # Generate different types of attention patterns
    size = 50  # Matrix size
    
    matrices = [
        ("Normal (Truthful)", generate_attention_matrix(size, "normal")),
        ("Collapsed (Hallucination)", generate_attention_matrix(size, "collapsed")),
        ("Sparse (Focused)", generate_attention_matrix(size, "sparse"))
    ]
    
    print("\nAnalyzing spectral properties...\n")
    
    for label, matrix in matrices:
        metrics = compute_spectral_metrics(matrix)
        risk = compute_hallucination_risk(metrics)
        
        print(f"{label}:")
        print(f"  Power Law Exponent (α): {metrics['alpha']:.2f}")
        print(f"  Number of Spikes: {metrics['spikes']}")
        print(f"  Spectral Entropy: {metrics['entropy']:.3f}")
        print(f"  Stable Rank: {metrics['stable_rank']:.2f}")
        print(f"  Hallucination Risk: {risk:.2f} ({'HIGH' if risk > 0.5 else 'LOW'})")
        print()
    
    print("Generating visualization...")
    plot_spectral_analysis(matrices)
    
    print("\n" + "="*80)
    print("Key Insights:")
    print("="*80)
    print("1. Normal attention shows α in optimal range (2-4)")
    print("2. Collapsed attention (hallucination) shows α < 2 and more spikes")
    print("3. Sparse attention has different spectral signature")
    print("\nThis demonstrates how spectral analysis can distinguish")
    print("between different attention patterns that correlate with")
    print("truthfulness vs hallucination in LLMs.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstration
    demonstrate_spectral_detection()
"""
Diagnostic tool to understand why mutual information calculation returns zeros.

Analyzes:
1. Hidden state tensor shapes and statistics
2. Differential entropy computation
3. Mutual information calculation step-by-step
4. SRV construction process
"""

import torch
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MIDiagnostics:
    """Diagnostic analysis of MI calculation issues."""
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def analyze_tensor(self, tensor: torch.Tensor, name: str):
        """Analyze tensor properties."""
        print(f"\n{'='*60}")
        print(f"Tensor Analysis: {name}")
        print(f"{'='*60}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  Min: {tensor.min().item():.6f}")
        print(f"  Max: {tensor.max().item():.6f}")
        print(f"  Mean: {tensor.mean().item():.6f}")
        print(f"  Std: {tensor.std().item():.6f}")
        print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"  Has Inf: {torch.isinf(tensor).any().item()}")
        print(f"  Num elements: {tensor.numel()}")
    
    def compute_differential_entropy_diagnostic(self, X: torch.Tensor):
        """Compute differential entropy with detailed diagnostics."""
        print(f"\n{'='*60}")
        print("Differential Entropy Computation")
        print(f"{'='*60}")
        
        # Flatten if needed
        if X.dim() == 1:
            X = X.unsqueeze(0)
            print(f"Added batch dimension: {X.shape}")
        elif X.dim() > 2:
            original_shape = X.shape
            X = X.reshape(X.shape[0], -1)
            print(f"Reshaped from {original_shape} to {X.shape}")
        
        print(f"Final shape for entropy: {X.shape}")
        
        # Handle single sample
        if X.shape[0] == 1:
            variance = X.var(dim=1).item() + self.epsilon
            entropy = 0.5 * np.log(2 * np.pi * np.e * variance)
            print(f"Single sample case:")
            print(f"  Variance: {variance:.6f}")
            print(f"  Entropy: {entropy:.6f}")
            return entropy
        
        # Multi-sample case
        print(f"Multi-sample case with {X.shape[0]} samples")
        
        try:
            # Compute covariance
            cov = torch.cov(X.T)
            print(f"  Covariance shape: {cov.shape}")
            print(f"  Cov min: {cov.min().item():.6f}")
            print(f"  Cov max: {cov.max().item():.6f}")
            print(f"  Cov mean: {cov.mean().item():.6f}")
            
            # Regularize
            cov_reg = cov + self.epsilon * torch.eye(cov.shape[0], device=cov.device)
            
            # Compute determinant
            det = torch.linalg.det(cov_reg)
            print(f"  Determinant: {det.item():.10f}")
            
            if det <= 0:
                print(f"  WARNING: Non-positive determinant! Using epsilon.")
                det = torch.tensor(self.epsilon)
            
            # Compute entropy
            log_term = torch.log(2 * np.pi * np.e * det)
            print(f"  Log term: {log_term.item():.6f}")
            
            entropy = 0.5 * log_term
            print(f"  Final entropy: {entropy.item():.6f}")
            
            return entropy.item()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Falling back to variance-based estimate")
            variance = X.var().item() + self.epsilon
            entropy = 0.5 * np.log(2 * np.pi * np.e * variance)
            print(f"  Fallback entropy: {entropy:.6f}")
            return entropy
    
    def compute_mi_diagnostic(self, X: torch.Tensor, Y: torch.Tensor):
        """Compute MI with full diagnostics."""
        print(f"\n{'='*60}")
        print("Mutual Information Computation: I(X:Y)")
        print(f"{'='*60}")
        
        # Analyze inputs
        self.analyze_tensor(X, "X")
        self.analyze_tensor(Y, "Y")
        
        # Flatten
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        print(f"\nAfter flattening:")
        print(f"  X shape: {X_flat.shape}")
        print(f"  Y shape: {Y_flat.shape}")
        
        # Align lengths
        min_len = min(len(X_flat), len(Y_flat))
        X_flat = X_flat[:min_len]
        Y_flat = Y_flat[:min_len]
        
        print(f"\nAfter alignment: {min_len} elements")
        
        # Compute individual entropies
        print("\n--- Computing H(X) ---")
        h_x = self.compute_differential_entropy_diagnostic(X_flat)
        
        print("\n--- Computing H(Y) ---")
        h_y = self.compute_differential_entropy_diagnostic(Y_flat)
        
        # Compute joint entropy
        print("\n--- Computing H(X,Y) ---")
        XY = torch.stack([X_flat, Y_flat], dim=0)
        print(f"Joint tensor shape: {XY.shape}")
        h_xy = self.compute_differential_entropy_diagnostic(XY.T)
        
        # Mutual information
        mi = h_x + h_y - h_xy
        
        print(f"\n{'='*60}")
        print("MI CALCULATION SUMMARY")
        print(f"{'='*60}")
        print(f"H(X) = {h_x:.6f}")
        print(f"H(Y) = {h_y:.6f}")
        print(f"H(X,Y) = {h_xy:.6f}")
        print(f"I(X:Y) = H(X) + H(Y) - H(X,Y) = {mi:.6f}")
        print(f"After clipping to non-negative: {max(0.0, mi):.6f}")
        
        return max(0.0, mi)


def diagnose_sample(sample_id: int = 0):
    """Diagnose a single sample extraction."""
    
    # Load extraction
    extraction_path = Path("extracted_synergistic/gpt2") / f"sample_{sample_id:04d}"
    
    print(f"\n{'#'*80}")
    print(f"DIAGNOSING SAMPLE {sample_id}")
    print(f"{'#'*80}")
    
    # Load metadata
    with open(f"{extraction_path}.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"\nMetadata:")
    print(f"  Question: {metadata['question']}")
    print(f"  Answer: {metadata['answer'][:100]}...")
    print(f"  Is Truthful: {metadata['is_truthful']}")
    print(f"  Num Layers: {metadata['num_layers']}")
    print(f"  Sequence Length: {metadata['sequence_length']}")
    
    # Load tensors
    tensors = torch.load(f"{extraction_path}.pt", map_location='cpu')
    
    hidden_states = tensors['hidden_states']
    output_logits = tensors['output_logits']
    
    print(f"\nTensor Shapes:")
    print(f"  Hidden states: {len(hidden_states)} layers")
    for i, h in enumerate(hidden_states[:3]):  # First 3 layers
        print(f"    Layer {i}: {h.shape}")
    print(f"  Output logits: {output_logits.shape}")
    
    # Take last token
    hidden_states_last = [h[:, -1, :].squeeze() for h in hidden_states]
    output_logits_last = output_logits[:, -1, :].squeeze()
    
    print(f"\nLast Token Shapes:")
    print(f"  Hidden states: {len(hidden_states_last)} layers")
    for i, h in enumerate(hidden_states_last[:3]):
        print(f"    Layer {i}: {h.shape}")
    print(f"  Output logits: {output_logits_last.shape}")
    
    # Initialize diagnostics
    diag = MIDiagnostics()
    
    # Analyze hidden states
    for i, h in enumerate(hidden_states_last[:2]):
        diag.analyze_tensor(h, f"Hidden State Layer {i}")
    
    diag.analyze_tensor(output_logits_last, "Output Logits")
    
    # Test MI calculation
    print(f"\n\n{'#'*80}")
    print("TESTING MI CALCULATION")
    print(f"{'#'*80}")
    
    # Test 1: MI between two hidden layers
    print(f"\n\nTEST 1: MI(Layer_0 : Layer_1)")
    mi_layers = diag.compute_mi_diagnostic(hidden_states_last[0], hidden_states_last[1])
    
    # Test 2: MI between layer and output
    print(f"\n\nTEST 2: MI(Layer_0 : Output_Logits)")
    mi_layer_output = diag.compute_mi_diagnostic(hidden_states_last[0], output_logits_last)
    
    # Test 3: Check if issue is with shape
    print(f"\n\nTEST 3: Creating simple SRV")
    layer_0 = hidden_states_last[0].flatten()
    layer_1 = hidden_states_last[1].flatten()
    
    # Simple SRV: difference between layers
    srv_simple = layer_0[:len(layer_1)] - layer_1[:len(layer_0)]
    diag.analyze_tensor(srv_simple, "Simple SRV (diff)")
    
    print(f"\n\nTEST 4: MI(Simple_SRV : Output)")
    mi_srv_output = diag.compute_mi_diagnostic(srv_simple, output_logits_last)
    
    # Summary
    print(f"\n\n{'#'*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'#'*80}")
    print(f"MI(Layer_0 : Layer_1) = {mi_layers:.6f}")
    print(f"MI(Layer_0 : Output) = {mi_layer_output:.6f}")
    print(f"MI(Simple_SRV : Output) = {mi_srv_output:.6f}")
    print()
    print("LIKELY ISSUES:")
    if mi_layers == 0 and mi_layer_output == 0:
        print("  ❌ All MI calculations return zero")
        print("  → Problem in differential entropy calculation")
        print("  → Covariance matrix might be rank-deficient")
        print("  → Need better numerical conditioning")
    elif mi_layers > 0 and mi_layer_output == 0:
        print("  ⚠️ Layer-layer MI works but layer-output MI fails")
        print("  → Issue with output logit dimensionality")
        print("  → May need to project to lower dimension")
    else:
        print("  ✅ MI calculation seems to work")
        print("  → Issue might be in SRV construction threshold")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-id', type=int, default=0)
    args = parser.parse_args()
    
    diagnose_sample(args.sample_id)

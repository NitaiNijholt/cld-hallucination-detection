"""
Phase 2: Inference & Internal State Extraction using TransformerLens

This module runs model inference on prompts and extracts attention matrices
and other internal states using TransformerLens for spectral analysis.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

# TransformerLens for extracting model internals
try:
    from transformer_lens import HookedTransformer
    TRANSFORMERLENS_AVAILABLE = True
except ImportError:
    print("TransformerLens not installed. Install with: pip install transformer-lens")
    TRANSFORMERLENS_AVAILABLE = False

# For models not supported by TransformerLens, use HuggingFace directly
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelInternalsExtractor:
    """Extract attention matrices and hidden states during inference."""
    
    def __init__(
        self, 
        model_name: str = "microsoft/phi-2",
        device: str = None,
        use_transformerlens: bool = True
    ):
        """
        Initialize the extractor.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
            use_transformerlens: Whether to use TransformerLens
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_transformerlens = use_transformerlens and TRANSFORMERLENS_AVAILABLE
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        if self.use_transformerlens:
            try:
                # TransformerLens provides hooked models that expose internals
                self.model = HookedTransformer.from_pretrained(
                    self.model_name,
                    device=self.device
                )
                self.tokenizer = self.model.tokenizer
                print(f"Loaded {self.model_name} with TransformerLens")
                return
            except Exception as e:
                print(f"Failed to load with TransformerLens: {e}")
                self.use_transformerlens = False
        
        # Fallback to HuggingFace with settings compatible with recent OSS models
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map="auto"
        ).to(self.device)
        print(f"Loaded {self.model_name} with HuggingFace")
    
    def extract_with_transformerlens(
        self,
        prompt: str,
        max_new_tokens: int = 50
    ) -> Dict:
        """
        Extract internals using TransformerLens.
        
        This provides the cleanest interface to model internals.
        """
        # Run model with cache to capture all activations
        with torch.no_grad():
            # Generate answer
            generated = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )
            
            # Run again with cache to get internals
            _, cache = self.model.run_with_cache(prompt)
        
        # Extract attention patterns for each layer
        attention_matrices = []
        for layer_idx in range(self.model.cfg.n_layers):
            # Get attention pattern: [batch, head, seq, seq]
            attn_pattern = cache[f"blocks.{layer_idx}.attn.hook_pattern"]
            attention_matrices.append(attn_pattern.cpu())
        
        # Extract other useful internals
        internals = {
            'prompt': prompt,
            'generated_text': self.model.to_string(generated) if hasattr(generated, 'tolist') else str(generated),
            'attention_matrices': attention_matrices,  # List of tensors per layer
            'residual_stream': [],
            'mlp_activations': [],
            'layer_norms': []
        }
        
        # Get residual stream at each layer
        for layer_idx in range(self.model.cfg.n_layers):
            # Residual stream after attention
            resid = cache[f"blocks.{layer_idx}.hook_resid_post"]
            internals['residual_stream'].append(resid.cpu())
            
            # MLP activations
            if f"blocks.{layer_idx}.mlp.hook_post" in cache:
                mlp = cache[f"blocks.{layer_idx}.mlp.hook_post"]
                internals['mlp_activations'].append(mlp.cpu())
            
            # Layer norm scales
            if f"blocks.{layer_idx}.ln1.hook_scale" in cache:
                ln_scale = cache[f"blocks.{layer_idx}.ln1.hook_scale"]
                internals['layer_norms'].append(ln_scale.cpu())
        
        return internals
    
    def extract_with_huggingface(
        self,
        prompt: str,
        max_new_tokens: int = 50
    ) -> Dict:
        """
        Extract internals using HuggingFace's output_attentions.
        
        Fallback when TransformerLens is not available.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Generate with attention outputs
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            
            # Get generated text
            generated_ids = outputs.sequences
            generated_text = self.tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            # Extract attention weights (if available)
            attention_matrices = []
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # outputs.attentions is a tuple of tuples
                # (one tuple per generated token, each containing tensors per layer)
                for token_attentions in outputs.attentions:
                    for layer_attn in token_attentions:
                        attention_matrices.append(layer_attn.cpu())
            
            # Extract hidden states
            hidden_states = []
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                for token_hidden in outputs.hidden_states:
                    for layer_hidden in token_hidden:
                        hidden_states.append(layer_hidden.cpu())
        
        internals = {
            'prompt': prompt,
            'generated_text': generated_text,
            'attention_matrices': attention_matrices,
            'hidden_states': hidden_states,
            'model_name': self.model_name
        }
        
        return internals
    
    def extract_internals(
        self,
        prompt: str,
        max_new_tokens: int = 50
    ) -> Dict:
        """
        Main extraction method that chooses the best available backend.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing model internals
        """
        if self.use_transformerlens:
            return self.extract_with_transformerlens(prompt, max_new_tokens)
        else:
            return self.extract_with_huggingface(prompt, max_new_tokens)
    
    def extract_batch(
        self,
        prompts: List[Dict],
        save_dir: str = "extracted_internals",
        batch_size: int = 1
    ) -> List[Dict]:
        """
        Extract internals for a batch of prompts.
        
        Args:
            prompts: List of prompt dictionaries
            save_dir: Directory to save extracted internals
            batch_size: Batch size for processing
            
        Returns:
            List of extracted internals
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        all_internals = []
        
        for i, prompt_data in enumerate(tqdm(prompts, desc="Extracting internals")):
            prompt = prompt_data['prompt']
            
            # Extract internals
            internals = self.extract_internals(prompt)
            
            # Add metadata
            internals['prompt_id'] = i
            internals['question'] = prompt_data.get('question', '')
            internals['correct_answer'] = prompt_data.get('correct_answer', '')
            internals['expected_truthful'] = prompt_data.get('expected_truthful', None)
            internals['category'] = prompt_data.get('category', 'unknown')
            internals['timestamp'] = datetime.now().isoformat()
            
            # Save individual result (without tensors for JSON)
            result_meta = {
                k: v for k, v in internals.items() 
                if not isinstance(v, (torch.Tensor, list))
            }
            
            with open(save_path / f"internal_{i:04d}.json", 'w') as f:
                json.dump(result_meta, f, indent=2)
            
            # Save tensors separately
            torch.save({
                'attention_matrices': internals.get('attention_matrices', []),
                'hidden_states': internals.get('hidden_states', []),
                'residual_stream': internals.get('residual_stream', []),
                'mlp_activations': internals.get('mlp_activations', [])
            }, save_path / f"tensors_{i:04d}.pt")
            
            all_internals.append(internals)
            
            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        return all_internals


def test_extraction():
    """Test the extraction with a simple example."""
    
    # Initialize extractor
    extractor = ModelInternalsExtractor(
        model_name="gpt2",  # Start with small model for testing
        use_transformerlens=True
    )
    
    # Test prompt
    test_prompt = "The capital of France is"
    
    print(f"\nTesting extraction with prompt: '{test_prompt}'")
    internals = extractor.extract_internals(test_prompt, max_new_tokens=10)
    
    print(f"\nGenerated: {internals['generated_text']}")
    
    if internals.get('attention_matrices'):
        attn = internals['attention_matrices']
        print(f"Extracted {len(attn)} attention matrices")
        if attn:
            print(f"Attention shape: {attn[0].shape if hasattr(attn[0], 'shape') else 'N/A'}")
    
    return internals


def main():
    """Main function to extract internals from TruthfulQA."""
    
    print("="*80)
    print("Phase 2: Inference & Internal State Extraction")
    print("="*80)
    
    # Load prepared prompts
    prompts_path = Path("experiment/prepared_data/train_prompts.json")
    if not prompts_path.exists():
        print(f"Prompts not found at {prompts_path}")
        print("Using test data instead...")
        
        # Create test prompts
        test_prompts = [
            {
                'prompt': "Question: What is the capital of France?\nAnswer:",
                'question': "What is the capital of France?",
                'correct_answer': "Paris",
                'expected_truthful': True,
                'category': 'geography'
            },
            {
                'prompt': "Question: What happens if you swallow gum?\nAnswer:",
                'question': "What happens if you swallow gum?",
                'correct_answer': "It passes through your digestive system normally",
                'expected_truthful': False,
                'category': 'health'
            }
        ]
    else:
        with open(prompts_path, 'r') as f:
            all_prompts = json.load(f)
        test_prompts = all_prompts[:5]  # Start with just 5 for testing
    
    # Initialize extractor
    print("\nInitializing model...")
    extractor = ModelInternalsExtractor(
        model_name="gpt2",  # Use small model for testing
        use_transformerlens=True
    )
    
    # Extract internals
    print(f"\nExtracting internals for {len(test_prompts)} prompts...")
    internals = extractor.extract_batch(
        test_prompts,
        save_dir="experiment/extracted_internals"
    )
    
    print(f"\nExtraction complete! Saved {len(internals)} samples")
    
    # Show summary
    if internals:
        sample = internals[0]
        print("\nSample structure:")
        print(f"  - Prompt: {sample['prompt'][:50]}...")
        print(f"  - Generated: {sample['generated_text'][:100]}...")
        if sample.get('attention_matrices'):
            print(f"  - Attention matrices: {len(sample['attention_matrices'])} layers")
    
    return internals


if __name__ == "__main__":
    # Run test first
    print("Running extraction test...")
    test_internals = test_extraction()
    
    # Then run main extraction
    if input("\nProceed with main extraction? (y/n): ").lower() == 'y':
        main()
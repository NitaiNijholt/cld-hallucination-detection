"""
Synergistic Hidden State Extraction Pipeline

Extracts ALL layer hidden states and output logits from LLMs for synergistic analysis.
Based on experiment/2_inference_extraction.py but ensures complete hidden state capture.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynergisticHiddenStateExtractor:
    """Extract hidden states from all layers for synergistic analysis."""
    
    def __init__(
        self,
        model_name: str,
        device: str = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize extractor with model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on (auto-detected if None)
            torch_dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with hidden state output enabled
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            output_hidden_states=True,  # Critical for synergistic analysis
            output_attentions=False,  # Not needed for this analysis
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"Model loaded with {self.model.config.num_hidden_layers} layers")
    
    def extract_from_text(
        self,
        text: str,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Extract hidden states and logits from input text.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing:
                - hidden_states: List of tensors, one per layer (excluding embedding)
                - output_logits: Final layer logits
                - input_ids: Tokenized input
                - text: Original text
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract hidden states from all layers
        # outputs.hidden_states is a tuple: (embedding_layer, layer_1, ..., layer_N)
        # We skip the embedding layer (index 0) to focus on transformer layers
        hidden_states = [h.cpu() for h in outputs.hidden_states[1:]]
        
        # Extract output logits
        output_logits = outputs.logits.cpu()
        
        return {
            'hidden_states': hidden_states,
            'output_logits': output_logits,
            'input_ids': inputs.input_ids.cpu(),
            'text': text,
            'num_layers': len(hidden_states),
            'sequence_length': inputs.input_ids.shape[1]
        }
    
    def extract_from_qa_pair(
        self,
        question: str,
        answer: str,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Extract hidden states from question-answer pair.
        
        Args:
            question: Question text
            answer: Answer text
            max_length: Maximum total length
            
        Returns:
            Extraction results with metadata
        """
        # Format as Q&A
        full_text = f"Question: {question}\nAnswer: {answer}"
        
        # Extract
        extraction = self.extract_from_text(full_text, max_length)
        
        # Add metadata
        extraction['question'] = question
        extraction['answer'] = answer
        
        return extraction
    
    def save_extraction(
        self,
        extraction: Dict[str, Any],
        output_path: Path,
        save_tensors: bool = True
    ):
        """
        Save extraction to disk.
        
        Args:
            extraction: Extraction dictionary
            output_path: Path to save (without extension)
            save_tensors: Whether to save tensor data (large files)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata as JSON
        metadata = {
            'text': extraction.get('text', ''),
            'question': extraction.get('question', ''),
            'answer': extraction.get('answer', ''),
            'num_layers': extraction['num_layers'],
            'sequence_length': extraction['sequence_length'],
            'model_name': self.model_name
        }
        
        with open(f"{output_path}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save tensors
        if save_tensors:
            torch.save({
                'hidden_states': extraction['hidden_states'],
                'output_logits': extraction['output_logits'],
                'input_ids': extraction['input_ids']
            }, f"{output_path}.pt")
        
        logger.debug(f"Saved extraction to {output_path}")


def load_truthfulqa_dataset(
    truthfulqa_path: str = "../qwen3_truthfulqa_evaluation_20250806_222711.json"
) -> List[Dict[str, Any]]:
    """
    Load TruthfulQA dataset.
    
    Args:
        truthfulqa_path: Path to TruthfulQA JSON file
        
    Returns:
        List of QA dictionaries
    """
    try:
        with open(truthfulqa_path, 'r') as f:
            data = json.load(f)
        
        # Try both 'results' and 'generation_results' (different formats)
        results = data.get('results', data.get('generation_results', []))
        
        # Normalize format: convert to standard dict
        normalized_results = []
        for r in results:
            normalized = {
                'question': r.get('question', ''),
                'model_answer': r.get('response', r.get('model_answer', r.get('answer', ''))),
                'is_truthful': bool(r.get('truthfulness', r.get('is_truthful', False))),
                'correct_answers': r.get('correct_answers', [])
            }
            normalized_results.append(normalized)
        
        logger.info(f"Loaded {len(normalized_results)} TruthfulQA samples")
        
        # Count truthful vs false
        truthful = sum(1 for r in normalized_results if r.get('is_truthful', False))
        logger.info(f"  Truthful: {truthful}, False: {len(normalized_results) - truthful}")
        
        return normalized_results
        
    except FileNotFoundError:
        logger.error(f"TruthfulQA file not found: {truthfulqa_path}")
        return []


def extract_truthfulqa_batch(
    model_names: List[str],
    output_dir: str = "extracted_synergistic",
    max_samples: int = 50,
    truthfulqa_path: str = "../qwen3_truthfulqa_evaluation_20250806_222711.json"
):
    """
    Extract hidden states from TruthfulQA for multiple models.
    
    Args:
        model_names: List of HuggingFace model names
        output_dir: Output directory
        max_samples: Maximum samples per model
        truthfulqa_path: Path to TruthfulQA dataset
    """
    # Load dataset
    qa_samples = load_truthfulqa_dataset(truthfulqa_path)
    
    if not qa_samples:
        logger.error("No samples loaded, aborting")
        return
    
    # Limit samples
    qa_samples = qa_samples[:max_samples]
    
    # Process each model
    for model_name in model_names:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*80}\n")
        
        # Create output directory
        model_safe_name = model_name.replace('/', '_')
        model_output_dir = Path(output_dir) / model_safe_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractor
        try:
            extractor = SynergisticHiddenStateExtractor(
                model_name=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            continue
        
        # Extract from each sample
        extraction_metadata = []
        
        for i, sample in enumerate(tqdm(qa_samples, desc=f"Extracting {model_name}")):
            question = sample.get('question', '')
            answer = sample.get('model_answer', sample.get('answer', ''))
            is_truthful = sample.get('is_truthful', False)
            
            if not question or not answer:
                logger.warning(f"Sample {i} missing question or answer, skipping")
                continue
            
            try:
                # Extract hidden states
                extraction = extractor.extract_from_qa_pair(question, answer)
                
                # Add ground truth label
                extraction['is_truthful'] = is_truthful
                extraction['sample_id'] = i
                
                # Save to disk
                output_path = model_output_dir / f"sample_{i:04d}"
                extractor.save_extraction(extraction, output_path)
                
                # Store metadata
                extraction_metadata.append({
                    'sample_id': i,
                    'question': question[:100],  # Truncate for readability
                    'answer': answer[:100],
                    'is_truthful': is_truthful,
                    'num_layers': extraction['num_layers'],
                    'file': str(output_path)
                })
                
            except Exception as e:
                logger.error(f"Failed to extract sample {i}: {e}")
                continue
        
        # Save extraction summary
        summary_path = model_output_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'num_samples': len(extraction_metadata),
                'samples': extraction_metadata
            }, f, indent=2)
        
        logger.info(f"Completed {model_name}: {len(extraction_metadata)} samples extracted")
        logger.info(f"Saved to: {model_output_dir}")
        
        # Free memory
        del extractor
        torch.cuda.empty_cache()
    
    logger.info(f"\n{'='*80}")
    logger.info("Extraction complete for all models")
    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states for synergistic analysis")
    parser.add_argument(
        '--models',
        nargs='+',
        default=['microsoft/phi-2', 'EleutherAI/gpt-neo-125M', 'gpt2'],
        help='Model names to extract from'
    )
    parser.add_argument(
        '--output-dir',
        default='extracted_synergistic',
        help='Output directory'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50,
        help='Maximum samples per model'
    )
    parser.add_argument(
        '--truthfulqa-path',
        default='../qwen3_truthfulqa_evaluation_20250806_222711.json',
        help='Path to TruthfulQA dataset'
    )
    
    args = parser.parse_args()
    
    extract_truthfulqa_batch(
        model_names=args.models,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        truthfulqa_path=args.truthfulqa_path
    )


if __name__ == "__main__":
    main()










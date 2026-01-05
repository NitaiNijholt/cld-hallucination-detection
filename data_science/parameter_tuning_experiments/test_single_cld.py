#!/usr/bin/env python3
"""Test script to generate a single base CLD with all CI metrics."""

import sys
from pathlib import Path

# Load environment variables from .env.dev
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.dev", override=True)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment

# Configuration
cld_folder = PROJECT_ROOT / "data_science" / "parameter_tuning_experiments" / "ground_truth_clds_for_experiments"
output_base = Path(__file__).parent / "results"

cld_config = {
    'name': 'Social_norms_and_obesity_prevalence',
    'excel': str(cld_folder / 'Social_norms_and_obesity_prevalence.xlsx')
}

# Generator config with LOGPROBS enabled
generator_config = {
    "provider": "openai",
    "model": "gpt-4.1",
    "temperature": 0.7,
    "logprobs": True,  # ✅ Request log probabilities
    "top_logprobs": 5   # ✅ Request top 5 log probabilities
}

# Dummy configs (not used for base generation)
dummy_corruptor_config = {
    "provider": "openai",
    "model": "gpt-4.1",
    "temperature": 0.7
}

dummy_judge_config = {
    "provider": "openai",
    "model": "gpt-4.1",
    "temperature": 0.3
}

print(f"=" * 80)
print(f"Testing single CLD generation: {cld_config['name']}")
print(f"=" * 80)

# Setup paths
excel_file = Path(cld_config['excel'])
yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_Nitai_C.yaml"
output_dir = output_base / f"test_single_cld_{cld_config['name']}"
output_dir.mkdir(parents=True, exist_ok=True)

# Run base CLD generation with PARALLEL edge discovery
print(f"\n🚀 Generating CLD with CI metrics enabled...\n")
result = run_discovery_experiment(
    excel_path=str(excel_file),
    yaml_path=str(yaml_path),
    dev_mode=False,
    generator_config=generator_config,
    corruptor_config=dummy_corruptor_config,
    judge_config=dummy_judge_config,
    corruption_rate=0.0,  # No corruption
    judge_edges=False,  # No judging yet
    parallel=True,  # 🚀 PARALLEL edge discovery
    max_workers=10,  # 10 concurrent workers
    citation_search_provider="brave",  # Use Brave for citations
    output_json_prefix=Path(excel_file).stem,
    output_dir=str(output_dir),
    experiment_description=f"Test_Single_CLD",
    embedding_enable=True,  # ✅ ENABLE CI metrics (all 9 metrics including new ones)
    node_comparison_enable=False,  # Skip comparison for base generation
)

print(f"\n✅ Test complete!")
print(f"📁 Output directory: {output_dir}")
print(f"📊 Session ID: {result.get('session_id', 'N/A')}")
print(f"\n💡 Check the Excel file for CI metrics!")

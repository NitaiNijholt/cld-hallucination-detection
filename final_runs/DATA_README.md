# Experimental Data and Results Documentation

This directory contains all experimental results for the thesis research on LLM-based Causal Loop Diagram (CLD) validation and hallucination detection.

**Last Updated**: December 2025  
**Contact**: Nitai Nijholt (nitai.nijholt@gmail.com)

---

## Open Science Statement

This research follows [WORCS](https://doi.org/10.3233/DS-210031) principles. All analysis scripts are available in `data_science/`. For access to raw data or additional materials, contact the author.

---

## Directory Structure Overview

| Directory | Research Question | Description |
|-----------|-------------------|-------------|
| `RQ1a_corruption_detection_*` | RQ1a | LLM-as-Judge corruption detection experiments |
| `RQ1a_ground_truth_*` | RQ1a | Ground truth baseline experiments |
| `RQ1b_corrector_*` | RQ1b | Corrector agent experiments |
| `RQ2_hallucination_detection/` | RQ2 | Ensemble classifier for hallucination detection |
| `RQ3_deep_research/` | RQ3 | Deep research integration experiments |
| `Sensitivity_analysis*/` | All | Sensitivity and ablation analyses |
| `cost_analysis/` | Supplementary | Token usage and cost scaling analysis |

---

## RQ1a: LLM-as-Judge Corruption Detection

### Experiment Design
- **Objective**: Evaluate LLM judges' ability to detect intentionally corrupted causal edges
- **Method**: Corrupt known-correct edges → Judge with multiple prompt variants → Compare to ground truth
- **Corruption Rate**: 30%
- **Models**: GPT-4.1 (corruptor), GPT-4.1 (judge)

### Datasets (CLDs)
| CLD Name | Domain | # Edges | Source |
|----------|--------|---------|--------|
| `depressive` | Mental Health | ~50 | Crielaard et al. 2024 |
| `social_norms` | Social Psychology | ~60 | Literature synthesis |
| `emergency_department` | Healthcare | ~80 | Stakeholder modeling |

### Prompt Variants
| Variant | Token Limit | Strategy |
|---------|-------------|----------|
| `baseline` | 250 | Direct judgment |
| `mechanistic` | 250 | Causal mechanism reasoning |
| `cot` | 5000 | Chain-of-thought reasoning |

### Key Output Files
- `judged_*.xlsx` - Individual edge judgments with confidence scores
- `analysis/prompt_variants_analysis.xlsx` - Aggregated metrics
- `*.png` - Confusion matrices and ROC curves
- `*.tex` - LaTeX tables for thesis

### Reproducibility
```bash
cd data_science/parameter_tuning_experiments
python run_rq1a_corruption_multirun.py --runs 3 --clds depressive social_norms emergency_department
```

---

## RQ1b: Corrector Agent Experiments

### Experiment Design
- **Objective**: Evaluate ability of LLM corrector to fix detected false positives
- **Method**: Take flagged edges → Apply correction prompts → Evaluate improvement

### Output Structure
```
RQ1b_corrector_experiment_{dataset}_{prompt_variant}/
├── {cld}/run_{n}/
│   ├── correction_session_mapping_*.txt
│   └── correction_results_*.txt
```

---

## RQ2: Hallucination Detection Ensemble

### Experiment Design
- **Objective**: Build ensemble classifier using LLM-derived features
- **Features**: Confidence scores, reasoning patterns, cross-model agreement
- **Method**: Train classifier on labeled hallucination data

### Key Output Files
- `ensemble_roc_curves.png` - ROC curves for classifier performance
- `feature_distributions*.png` - Feature analysis visualizations
- `rq2_master_report_v2_enhanced.py` - Analysis script

### Data Sources
See `README_DATA_SOURCES.md` in this directory for detailed feature descriptions.

---

## RQ3: Deep Research Integration

### Experiment Design
- **Objective**: Evaluate retrieval-augmented generation for edge validation
- **Method**: Use deep research to verify causal claims against literature

### Output Files
- Analysis scripts and preliminary results
- Integration with main judging pipeline

---

## Sensitivity Analyses

### `Sensitivity_analysis_simple/`
- Prompt sensitivity analysis
- Corrector sensitivity analysis
- Scripts in `scripts/` subdirectory

### `Sensitivity_analysis_sobol/`
- Sobol sensitivity indices
- Parameter importance ranking

---

## Cost and Scaling Analysis

### `cost_analysis/`
- Token usage analysis
- API cost projections
- Scaling behavior with CLD size

### `time_and_cost_scaling/`
- Time complexity analysis
- LaTeX tables for thesis

---

## File Format Conventions

| Extension | Content Type |
|-----------|--------------|
| `.xlsx` | Experimental data (Excel format, ignored by git) |
| `.csv` | Tabular data (ignored by git) |
| `.json` | Metadata and configurations (ignored by git) |
| `.tex` | LaTeX tables for thesis |
| `.png/.pdf` | Figures and visualizations |
| `.txt` | Logs and session mappings |
| `.py` | Analysis and generation scripts |

---

## Reproducibility Parameters

### Standard Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| Judge Temperature | 0.3 | Low for consistency |
| Corruption Temperature | 0.7 | Moderate for variety |
| Random Seed | 42 | Fixed for judging |
| Corruption Seed | Run number | Varies per run |

### Environment
- Python 3.10+
- Dependencies: See `pyproject.toml` and `uv.lock`
- LLM APIs: OpenAI (GPT-4.1), Anthropic (Claude 3.5 Sonnet)

---

## Citation

If you use this data or code, please cite:

```bibtex
@software{nijholt2025causalix,
  author = {Nijholt, Nitai},
  title = {Causalix.AI: LLM-based Causal Loop Diagram Building and Validation},
  year = {2025},
  url = {https://github.com/causalix/causalix.ai}
}
```

See also [CITATION.cff](../CITATION.cff) in the repository root.







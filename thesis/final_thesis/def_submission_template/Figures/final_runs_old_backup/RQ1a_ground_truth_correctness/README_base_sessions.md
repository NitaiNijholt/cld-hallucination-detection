# RQ1a Ground Truth Base Sessions

This directory contains the judging results for **clean (non-corrupted) CLDs** to assess judge ability to distinguish True Positives from False Positives.

## Base Session Generation

**Generation Date:** November 16, 2025  
**Seeds Used:** 10, 20, 30 (methodologically independent from corruption experiment seeds 1, 2, 3)  
**Strategy:** Each run uses a different base CLD generation (varying the generation seed) rather than corrupting the same base CLD

## Base CLD Sessions

### Depressive Symptoms CLD
- **Run 1 (seed 10):** `aa4b5b4f-92c2-4668-b02f-3ddaec942e38` (generated 2025-11-16 05:11:46)
- **Run 2 (seed 20):** `40479c6e-e1ad-4ccc-95ed-1ab1f658da79` (generated 2025-11-16 05:38:06)
- **Run 3 (seed 30):** `c756d0c9-22ce-4a33-bf97-2bf1c5b94446` (generated 2025-11-16 06:04:17)

### Social Norms and Obesity CLD
- **Run 1 (seed 10):** `807f8397-1444-4605-b7c5-7a808e631bb7` (generated 2025-11-16 04:47:46)
- **Run 2 (seed 20):** `dbaf53c4-08c8-4306-b41f-05e4ab81323b` (generated 2025-11-16 05:14:09)
- **Run 3 (seed 30):** `fdab3339-915d-43dc-8a75-6421ed1051e8` (generated 2025-11-16 05:40:33)

### Emergency Department Visits CLD
- **Run 1 (seed 10):** `e84145ef-cc8a-4cd7-8adc-3e8f94904010` (generated 2025-11-16 05:07:33)
- **Run 2 (seed 20):** `6078ca78-980a-44eb-9dae-01149202da30` (generated 2025-11-16 05:33:55)
- **Run 3 (seed 30):** `e1ba1caf-3799-4652-9cc7-53dc9c8640b5` (generated 2025-11-16 06:00:07)

## Experiment Design

### Key Differences from Corruption Experiments:
1. **No corruption applied** - Judging clean CLDs directly
2. **CI metrics ENABLED** - Context-Insensitive metrics extracted for RQ2 analysis
3. **Different seeds** - Seeds 10, 20, 30 (vs 1, 2, 3 in corruption experiments)
4. **Varying base generations** - Each run uses a different base CLD generation

### Judging Configuration:
- **3 Prompt Variants:** Baseline, Mechanistic v4, CoT
- **Judge Model:** Claude Sonnet 4.5
- **Temperature:** 0.0 (deterministic)
- **Approach:** per_citation_aggregate

## Directory Structure

```
RQ1a_ground_truth_correctness/
├── rq1a_ground_truth_base_sessions.json  # Base session registry
├── README_base_sessions.md               # This file
├── depressive/
│   ├── run_1/  # seed 10
│   ├── run_2/  # seed 20
│   └── run_3/  # seed 30
├── social_norms/
│   ├── run_1/
│   ├── run_2/
│   └── run_3/
└── emergency_department/
    ├── run_1/
    ├── run_2/
    └── run_3/
```

## Related Files

- **Base session registry:** `rq1a_ground_truth_base_sessions.json`
- **Multirunner script:** `data_science/parameter_tuning_experiments/run_rq1a_ground_truth_multirun.py`
- **Individual judge script:** `data_science/parameter_tuning_experiments/run_rq1a_ground_truth_judge.py`
- **Analysis scripts:**
  - `analyze_rq1a_ground_truth_tp_fp.py` (individual run analysis)
  - `run_rq1a_ground_truth_aggregate.py` (aggregate analysis)

## Methodological Note

Using different generation seeds (10, 20, 30) instead of the corruption experiment seeds (1, 2, 3) provides methodological independence. This ensures that any patterns observed are not artifacts of shared random number generation processes between the two experimental paradigms.

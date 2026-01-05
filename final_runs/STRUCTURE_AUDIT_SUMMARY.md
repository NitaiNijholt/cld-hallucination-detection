# Final Runs Structure Audit Summary

**Generated:** January 3, 2026  
**Audit Files Created:**
- `structure_audit_detailed.txt` - Per-folder analysis (670 lines)
- `structure_audit_all_files.txt` - Complete file listing (5,172 files)
- `structure_audit_rq1_data.txt` - RQ1a/RQ1b data structure check
- `structure_audit_full.txt` - Directory tree (900 directories)

---

## Overall Compliance: ✅ 24/24 folders (100%)

All experiment folders have the required unified structure:
```
experiment/
├── Data/            ✅ Required
├── analysis_scripts/ ✅ Required  
├── Output/          ✅ Required
└── sim_scripts/     ⚪ Optional (only 1/24 has this)
```

---

## File Statistics

| Type | Count | Notes |
|------|-------|-------|
| PNG | 2,153 | Figures |
| XLSX | 965 | Data files |
| JSON | 674 | Configs/results |
| TEX | 576 | LaTeX tables |
| TXT | 317 | Logs/reports |
| CSV | 255 | Data files |
| PY | 85 | Scripts |
| LOG | 81 | Experiment logs |
| MD | 40 | Documentation |
| PDF | 26 | Figures |

---

## Issues to Address

### 1. Loose Files at Experiment Root (6 folders)

These folders have output files at root level that should be in `Output/`:

| Folder | Issue |
|--------|-------|
| `RQ1b_corrector_ablation_analysis` | .tex, .xlsx files |
| `RQ2_hallucination_detection` | .png, .tex, .json files |
| `RQ3_deep_research` | .tex, .json files + `scripts/`, `validation/` folders |
| `Sensitivity_analysis_simple` | .tex, .pdf, .png files |
| `parallelization_benchmark` | `configs/`, `figures/` folders |
| `time_and_cost_scaling` | `figures/` folder |

### 2. RQ1a/RQ1b Data Structure ✅ Consistent

**RQ1a (Judge):** `Data/CLD/run_X/prompt/`
- All 3 CLDs × 3 runs × 4 prompts (baseline, cot, mechanistic, shared)

**RQ1b (Corrector):**
- Ground Truth: `Data/CLD/run_X/prompt/judge/` with baseline_judge + mechanistic_judge
- Corruption Detection: `Data/CLD/run_X/prompt/judge/` with baseline_judge only

### 3. Missing sim_scripts/ (23/24 folders)

Only `RQ1_verification_Truth_QA` has simulation scripts. Other experiments would need CLI documentation to reproduce from scratch.

---

## Recommended Actions

1. **Move loose files to Output/** for 6 folders listed above
2. **Rename `figures/` to `Output/`** in parallelization_benchmark, time_and_cost_scaling
3. **Move `scripts/`, `validation/`** in RQ3 to appropriate locations
4. **Create sim_scripts/** or document CLI commands for reproducibility
5. **Update README_DIRECTORY_MAP.md** to reflect current structure


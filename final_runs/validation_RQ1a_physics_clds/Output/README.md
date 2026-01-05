# RQ1a: Judge Validation Using Physics-Based CLDs

## Purpose

This directory contains all data, scripts, and results for validating the judge functionality using physics-based Causal Loop Diagrams (CLDs) with well-established scientific principles.

## Research Question

**RQ1a**: Can the judge correctly identify valid causal relationships when they are based on irrefutable physics principles?

**Hypothesis**: Physics CLDs grounded in Newton's laws, thermodynamics, and conservation laws should receive 100% approval from the judge, demonstrating the judge works correctly on "easy" cases.

---

## Key Results

### ✅ **100% Correctness Approval**
- **31/31 edges** judged as CORRECT
- Perfect score across all 4 physics CLDs
- Validates judge functionality on well-established causal principles

### ⚠️ **45% Citation Approval** 
- Limited by web scraping failures (45% NO_CITATION)
- When citations accessible: **82.4% approval**
- Gap due to technical limitations, not judge quality

---

## Directory Structure

```
RQ1a_validation_physics_CLDs/
├── README.md                          # This file
├── run_physics_cld_tests.py          # Script to run judge tests
├── analyze_physics_cld_results.py    # Script to analyze results
│
├── physics_clds_with_references/     # Complete CLD data with references
│   ├── thermostat_heating_system_with_refs.json
│   ├── predator_prey_with_refs.json
│   ├── rc_circuit_with_refs.json
│   ├── water_tank_with_refs.json
│   ├── README.md
│   ├── PIPELINE_EXPLANATION.md
│   ├── FINAL_STATUS.md
│   ├── MANUAL_SOURCE_VERIFICATION.md
│   ├── URL_SUMMARY.md
│   └── QUOTES_SUMMARY.md
│
├── cld_data_files/                    # Raw CLD data files
│   ├── thermostat_heating_system_vars_data.json
│   ├── thermostat_heating_system_edges_data.json
│   ├── thermostat_heating_system_context_data.json
│   ├── thermostat_heating_system_citation_hints.json
│   ├── [similar files for predator_prey, rc_circuit, water_tank]
│
└── results/                           # Judge test results
    ├── physics_cld_raw_results_20251204_182810.json
    └── physics_cld_tables_20251229_041536.tex
```

---

## Physics CLDs Tested

### 1. **Thermostat Heating System** 🌡️
- **Physics Basis**: Newton's Law of Cooling (1701), First Law of Thermodynamics
- **Variables**: 7 | **Edges**: 8
- **Primary Sources**: Newton (1701), Sterman (2000), Incropera et al. (2007)
- **Correctness**: 8/8 CORRECT (100%)
- **Citation**: 62.5% approval

### 2. **Predator-Prey (Lotka-Volterra)** 🐇🦊
- **Physics Basis**: Lotka (1925), Volterra (1926) - Population Dynamics
- **Variables**: 6 | **Edges**: 10
- **Primary Sources**: Lotka (1925), Volterra (1926), Murray (2002)
- **Correctness**: 10/10 CORRECT (100%)
- **Citation**: 50.0% approval

### 3. **RC Circuit Charging** ⚡
- **Physics Basis**: Ohm's Law (1827), Kirchhoff's Laws (1845)
- **Variables**: 7 | **Edges**: 7
- **Primary Sources**: Ohm (1827), Kirchhoff (1845), Nilsson & Riedel (2015)
- **Correctness**: 7/7 CORRECT (100%)
- **Citation**: 14.3% approval (highest scraping failure rate)

### 4. **Water Tank Draining** 💧
- **Physics Basis**: Conservation of Mass, Pascal (1663), Torricelli (1643)
- **Variables**: 6 | **Edges**: 6
- **Primary Sources**: Pascal (1663), Torricelli (1643), Bernoulli (1738)
- **Correctness**: 6/6 CORRECT (100%)
- **Citation**: 50.0% approval

**Total**: 26 variables, 31 edges, 79 references

---

## How to Use

### 1. Run Judge Tests

```bash
# Run correctness-based judging only
python run_physics_cld_tests.py --approaches correctness

# Run citation-based judging only  
python run_physics_cld_tests.py --approaches citation

# Run both approaches
python run_physics_cld_tests.py --approaches correctness citation

# Test specific CLD
python run_physics_cld_tests.py --clds thermostat_heating_system --approaches correctness
```

### 2. Analyze Results

```bash
# Analyze latest results and generate LaTeX tables
python analyze_physics_cld_results.py

# Analyze specific results file
python analyze_physics_cld_results.py --results results/physics_cld_raw_results_TIMESTAMP.json

# Skip LaTeX generation (console output only)
python analyze_physics_cld_results.py --no-latex

# List all available result files
python analyze_physics_cld_results.py --list
```

---

## Key Files

### Data Files

- **`physics_clds_with_references/`**: Complete CLD definitions with:
  - Edge motivations (causal narratives)
  - References with full citations
  - DOIs, ISBNs, URLs (where available)
  - Supporting quotes from each reference
  
- **`cld_data_files/`**: Raw data files for Neo4j loading:
  - `*_vars_data.json`: Variable definitions
  - `*_edges_data.json`: Edge definitions with motivations
  - `*_context_data.json`: CLD metadata (target, scales)
  - `*_citation_hints.json`: Citation URLs for judging

### Scripts

- **`run_physics_cld_tests.py`**: Executes judge tests
  - Loads CLDs into Neo4j
  - Runs correctness and/or citation-based judging
  - Saves raw results as JSON
  
- **`analyze_physics_cld_results.py`**: Analyzes results
  - Generates console summary tables
  - Creates LaTeX tables for thesis
  - Provides detailed statistics

### Results

- **`results/physics_cld_raw_results_*.json`**: Raw judge outputs
  - Complete edge-by-edge verdicts
  - Scores, timings, metadata
  
- **`results/physics_cld_tables_*.tex`**: LaTeX tables
  - Ready for thesis inclusion
  - 4 comprehensive tables

---

## Judge Configuration

- **Judge Model**: `gpt-4.1`
- **Temperature**: 0.0 (deterministic)
- **Approach**: 
  - **Correctness**: Evaluates logical soundness and scientific accuracy
  - **Citation**: Evaluates literature support from provided URLs

---

## Results Summary

| CLD | Edges | Correctness | Citation | Gap |
|-----|-------|-------------|----------|-----|
| Thermostat | 8 | 100.0% | 62.5% | 37.5% |
| Predator-Prey | 10 | 100.0% | 50.0% | 50.0% |
| RC Circuit | 7 | 100.0% | 14.3% | 85.7% |
| Water Tank | 6 | 100.0% | 50.0% | 50.0% |
| **TOTAL** | **31** | **100.0%** | **45.2%** | **54.8%** |

**Key Insight**: Gap between correctness and citation approval is due to web scraping limitations (45% NO_CITATION), not judge quality or CLD validity.

---

## Citation Quality

### Verified Accessible Sources (7)
1. Lotka (1925) - Archive.org ✅
2. Ohm (1827) - Archive.org ✅
3. Maxwell (1873) - Archive.org ✅
4. Faraday (1839) - Archive.org ✅
5. Pascal (1663) - Archive.org ✅
6. Bernoulli (1738) - Archive.org ✅
7. Åström & Murray (2008) - Free PDF ✅

### Coverage
- **79 total references** with supporting quotes
- **7 freely accessible** (Archive.org + open PDF)
- **3 DOI links** (may need institutional access)
- **13 textbooks** (library access needed)
- **100% quote coverage**

See `physics_clds_with_references/MANUAL_SOURCE_VERIFICATION.md` for details.

---

## Comparison to Alzheimer's CLD

| Metric | Physics CLDs | Alzheimer's CLD | Interpretation |
|--------|--------------|-----------------|----------------|
| Correctness Approval | 100% | ~50% | Physics = clear, Alzheimer's = complex |
| Citation Findability | 45% (scraping limited) | ~60% | Domain accessibility |
| **Conclusion** | ✅ Judge works | ✅ Judge works | Lower Alzheimer's scores = domain complexity |

---

## Thesis Integration

### LaTeX Tables Available

The `results/physics_cld_tables_*.tex` file contains 4 ready-to-use tables:

1. **Table 1**: Physics CLDs Overview
2. **Table 2**: Correctness-Based Judging Results
3. **Table 3**: Citation-Based Judging Results  
4. **Table 4**: Correctness vs Citation Comparison

### Key Statements for Thesis

> "To validate the judge's functionality, we tested it on physics-based CLDs grounded in well-established scientific principles (Newton's laws, conservation laws, Lotka-Volterra equations). The judge achieved **100% approval** on correctness-based evaluation, demonstrating its ability to recognize valid causal reasoning. This contrasts with the ~50% approval rate on the Alzheimer's CLD, confirming that the lower scores reflect **domain complexity** rather than judge quality."

> "Citation-based validation showed lower approval rates (45.2%) primarily due to technical limitations in content scraping (45% NO_CITATION verdicts). When citations were accessible, approval rose to 82.4%, indicating the judge performs well when provided with readable content."

---

## Technical Notes

### Judge Prompts
- Correctness: `prompts_correctness_baseline.yaml` → `judgeCorrectness`
- Citation: `prompts_citation_baseline.yaml` → `judgeCitation`

### Web Scraping
- **Scraper**: ContentScraper (custom implementation)
- **Limitation**: Cannot access paywalled content, PDFs behind logins, or blocked sites
- **Success Rate**: ~55% (14/31 edges failed to fetch citations)

### Performance
- **Total edges**: 31
- **Processing time**: 18.08 seconds
- **Speed**: 1.71 edges/second
- **Average score**: 1.00 (perfect)

---

## Documentation

For more details, see:
- **Pipeline explanation**: `physics_clds_with_references/PIPELINE_EXPLANATION.md`
- **URL verification**: `physics_clds_with_references/MANUAL_SOURCE_VERIFICATION.md`
- **Final status**: `physics_clds_with_references/FINAL_STATUS.md`

---

**Created**: December 2024  
**Judge Model**: GPT-4.1  
**Status**: ✅ Complete & Validated  
**Purpose**: RQ1a - Judge validation baseline using physics CLDs









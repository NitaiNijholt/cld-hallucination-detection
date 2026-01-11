# 🎯 High-Priority Tasks (No Experimental Results Required)
## Actionable Now While Experiments Run

Based on comprehensive review of thesis plan and codebase status.

---

## 🔥 CRITICAL PRIORITY (Do First)

### 1. Implement RQ3 Smart Test-Time Compute System ⭐⭐⭐
**Why Critical**: Blocks Chapter 5.5 and entire RQ3
**Time**: 2-3 days
**Dependencies**: None - can start immediately
**Output**: Working system ready to run experiments

**Subtasks**:
- [ ] Create `smart_judging_system.py` module
  - Threshold-based filtering using CI metrics
  - Selective judge invocation
  - Compute cost tracking
  - Performance metrics collection

- [ ] Create `run_rq3_experiments.py` entry point
  - Load optimal thresholds from RQ2
  - Run experiments with different threshold combinations
  - Compare full judging vs selective judging
  - Generate compute savings reports

- [ ] Add to existing pipeline
  - Integrate with `multirun_parameter_experiments()`
  - Reuse CI metrics computation
  - Compatible with existing infrastructure

**Benefits**:
- Unblocks critical research question
- Can run experiments immediately after
- Validates RQ2 findings empirically
- Demonstrates practical impact (compute savings)

**Implementation Guide**:
```python
# smart_judging_system.py structure
class SmartJudgingSystem:
    def __init__(self, thresholds: Dict[str, float]):
        """
        thresholds = {
            'perplexity': 50.0,
            'min_prob': 0.3,
            'max_entropy': 2.5,
            'cosine_sim': 0.7
        }
        """
        
    def should_judge(self, edge: Edge, ci_metrics: Dict) -> bool:
        """Returns True if edge needs LLM judging"""
        
    def run_smart_experiment(self, cld: CLD) -> Results:
        """Run experiment with selective judging"""
        
    def compute_savings(self) -> Dict:
        """Calculate cost/time savings"""
```

---

### 2. Expand Literature Review - Add Missing Sections ⭐⭐⭐
**Why Critical**: Blocks final submission, demonstrates theoretical depth
**Time**: 5-7 days
**Dependencies**: None
**Output**: Complete Chapter 2 ready for review

#### Section 2.11: Sensitivity Analysis in Machine Learning (NEW - 3-4 pages)
**Why**: Grounds your Chapter 4 in existing literature

**Content Outline**:
- [ ] **Introduction to SA methods**
  - Local vs global sensitivity analysis
  - When to use which method
  - Applications in ML/AI systems

- [ ] **Sobol Sensitivity Analysis**
  - Theoretical foundation (Sobol 1993, Saltelli 2010)
  - Variance decomposition
  - Applications to neural networks
  - SALib library and tools

- [ ] **Alternative Methods**
  - Morris screening method
  - FAST (Fourier Amplitude Sensitivity Test)
  - Comparison and when to use each

- [ ] **SA for Hyperparameter Importance**
  - Prior work on hyperparameter sensitivity
  - AutoML and hyperparameter optimization
  - Connection to your work

- [ ] **Gap Identification**
  - No prior SA for LLM hallucination detection
  - Limited SA for multi-agent LLM systems
  - Your contribution fills this gap

**Key Papers to Add**:
```
- Sobol (1993) - Original Sobol indices paper
- Saltelli et al. (2010) - Variance-based SA book
- Herman & Usher (2017) - SALib library paper
- Pianosi et al. (2016) - SA primer for environmental models
- Razavi et al. (2021) - Future of SA review
- Recent papers on SA for neural networks (2022-2025)
```

#### Section 2.9: Hallucination Definition - Expand (2-3 pages)
**Current**: 1 paragraph, minimal
**Target**: Comprehensive taxonomy

**Content to Add**:
- [ ] **Hallucination Taxonomy**
  - Intrinsic vs extrinsic hallucinations
  - Factual vs logical hallucinations
  - Detection vs correction approaches

- [ ] **Measurement Approaches**
  - Human evaluation protocols
  - Automatic metrics (your CI metrics fit here)
  - LLM-as-a-judge (your RQ1)
  - Benchmark datasets

- [ ] **Prior Work on Detection**
  - Self-consistency methods
  - Entropy-based detection
  - External knowledge grounding
  - How your work differs/improves

- [ ] **Causal Discovery Context**
  - Why hallucination matters for CLDs
  - Stakes of incorrect causal relationships
  - Previous work on CLD validation

**Key Papers to Add** (2024-2025):
```
- Recent hallucination detection papers from ACL/EMNLP 2024
- OpenAI/Anthropic technical reports on hallucination
- Benchmark papers (TruthfulQA, HaluEval, etc.)
- Causal discovery with LLMs papers
```

#### Section 2.10: Test-Time Compute - Expand (2-3 pages)
**Current**: Minimal
**Target**: Theoretical foundation for RQ3

**Content to Add**:
- [ ] **Scaling Laws and Compute Allocation**
  - Kaplan et al. scaling laws
  - Chinchilla optimal compute allocation
  - Test-time vs training-time compute trade-offs

- [ ] **Adaptive Computation**
  - Early exit strategies
  - Cascade models
  - Selective processing (your approach)
  - Compute-optimal inference

- [ ] **Cost-Benefit Analysis Frameworks**
  - Pareto frontiers (accuracy vs cost)
  - Multi-objective optimization
  - ROI calculations

- [ ] **Smart Routing/Gating**
  - Mixture of Experts principles
  - Confidence-based routing
  - Your threshold-based approach

**Key Papers to Add**:
```
- Kaplan et al. (2020) - Scaling laws
- Hoffmann et al. (2022) - Chinchilla
- Recent papers on test-time compute optimization
- Mixture of Experts papers
- Cost-aware NLP papers
```

#### Add Cross-References Throughout Chapter 2
- [ ] Link SA section to Chapter 4
- [ ] Link hallucination section to RQ1/RQ2
- [ ] Link test-time compute to RQ3
- [ ] Forward references to your contributions

**Total Time for Lit Review**: 5-7 days (can work on in parallel with coding)

---

### 3. Create Bibliography Database and Add Missing Citations ⭐⭐
**Why Critical**: Required for final submission, improves academic rigor
**Time**: 2-3 days
**Dependencies**: None
**Output**: Complete bibliography, all citations tracked

**Current Status**: ~40 citations  
**Target**: 70-90 citations

**Tasks**:

- [ ] **Organize existing `example.bib`**
  - Review all current citations
  - Fix any formatting errors
  - Add missing fields (DOI, URLs)
  - Standardize naming convention

- [ ] **Add Sensitivity Analysis Papers** (10-15 papers)
  ```bibtex
  @article{sobol1993sensitivity,
    title={Sensitivity analysis for nonlinear mathematical models},
    author={Sobol, Ilya M},
    journal={Mathematical modelling and computational experiment},
    year={1993}
  }
  
  @book{saltelli2008global,
    title={Global sensitivity analysis: the primer},
    author={Saltelli, Andrea and Ratto, Marco and Andres, Terry and others},
    year={2008},
    publisher={John Wiley \& Sons}
  }
  
  @article{herman2017salib,
    title={SALib: An open-source Python library for sensitivity analysis},
    author={Herman, Jonathan and Usher, Will},
    journal={Journal of Open Source Software},
    volume={2},
    number={9},
    year={2017}
  }
  
  # Add 10-12 more SA papers
  ```

- [ ] **Add Recent Hallucination Papers** (10-15 papers from 2024-2025)
  - ACL 2024 papers on hallucination
  - EMNLP 2024 papers on LLM reliability
  - Technical reports from OpenAI, Anthropic
  - Benchmark papers (TruthfulQA updates, etc.)

- [ ] **Add Test-Time Compute Papers** (5-8 papers)
  - Scaling laws papers
  - Adaptive computation papers
  - Cost-aware NLP papers

- [ ] **Add Causal Discovery Papers** (5-8 papers)
  - LLMs for causal discovery
  - CLD generation and validation
  - Domain-specific causal modeling

- [ ] **Add Transformer/Attention Papers** (maintain existing, verify completeness)
  - Ensure Vaswani et al. (2017) properly cited
  - Recent transformer improvements
  - Attention mechanism analysis

**Tool**: Create BibTeX management script
```python
# check_bibliography.py
import bibtexparser

def analyze_bibliography(bib_file):
    """
    - Count citations by year
    - Identify missing fields
    - Find duplicates
    - Suggest additions based on topic
    """
    pass

def find_missing_citations_in_thesis(tex_files, bib_file):
    """
    - Parse all \cite{} commands
    - Check against bib file
    - Report undefined citations
    """
    pass
```

---

## 🔴 HIGH PRIORITY (Do Next)

### 4. Write Methods Chapter Additions ⭐⭐
**Why Important**: Improves Chapter 3, documents decisions
**Time**: 1-2 days
**Dependencies**: None
**Output**: Enhanced Methods chapter

**Additions**:

- [ ] **Section 3.6: Statistical Framework Details** (if not already complete)
  - Permutation test procedures
  - Bootstrap CI methodology
  - Multiple comparison corrections
  - Significance thresholds and justification

- [ ] **Section 3.7: Experimental Validation Strategy**
  - Cross-validation approach
  - Multi-CLD validation rationale
  - Sample size justification
  - Statistical power analysis

- [ ] **Add Forward References**
  - Reference Chapter 4 when discussing parameter choices
  - "Parameter importance validated via sensitivity analysis (Chapter 4)"
  - "Threshold selection guided by correlation analysis (Chapter 5.4)"

- [ ] **Improve Figure/Table Quality**
  - Check all figures are 300 DPI
  - Consistent color schemes
  - Professional captions
  - Proper cross-referencing

---

### 5. Create Helper Scripts and Utilities ⭐⭐
**Why Important**: Speeds up experimental work, ensures reproducibility
**Time**: 2-3 days
**Dependencies**: None
**Output**: Reusable scripts, better workflow

**Scripts to Create**:

#### A. Experiment Monitor and Logger
```python
# experiment_monitor.py
"""
Monitor running experiments:
- Track completion status
- Estimate time remaining
- Alert on failures
- Generate progress reports
"""

class ExperimentMonitor:
    def scan_experiments(self, exp_dir):
        """Find all running/complete experiments"""
    
    def check_completeness(self, expected_samples):
        """Verify all samples have results"""
    
    def estimate_completion_time(self):
        """Based on current rate"""
    
    def generate_report(self):
        """Markdown report of status"""
```

#### B. Results Aggregator
```python
# aggregate_results.py
"""
Aggregate results from multiple experiments:
- Collect all Excel/CSV outputs
- Combine into master DataFrame
- Handle missing data
- Save aggregated results
"""

def aggregate_all_rq2_experiments(exp_ids: List[str]) -> pd.DataFrame:
    """Convenience function for RQ2"""
    pass

def aggregate_sensitivity_experiments(exp_prefix: str, samples_csv: str) -> pd.DataFrame:
    """Convenience function for SA"""
    pass
```

#### C. Figure Generation Pipeline
```python
# generate_all_figures.py
"""
Generate all thesis figures from data:
- Read aggregated results
- Create all publication-quality figures
- Save to thesis/Pictures/
- Generate LaTeX figure code
"""

def generate_rq1_figures():
    """All RQ1 figures"""
    pass

def generate_rq2_figures():
    """All RQ2 figures"""
    pass

def generate_sa_figures():
    """All sensitivity analysis figures"""
    pass

if __name__ == "__main__":
    generate_all_figures(output_dir="thesis/Pictures/")
    print("All figures generated and ready for thesis!")
```

#### D. Experiment Validator
```python
# validate_experiments.py
"""
Validate experimental outputs:
- Check for missing files
- Verify data integrity
- Detect anomalies
- Generate validation report
"""

def validate_experiment_outputs(exp_id: str) -> Dict:
    """Returns validation report"""
    pass

def check_ci_metrics_completeness(results_df: pd.DataFrame) -> bool:
    """Verify all CI metrics computed"""
    pass
```

#### E. LaTeX Table Generator
```python
# generate_latex_tables.py
"""
Auto-generate LaTeX tables from results:
- Sobol indices tables
- Correlation matrices
- Performance metrics
- Statistical test results
"""

def sobol_indices_to_latex(results: Dict, outcome: str) -> str:
    """Generate LaTeX table code"""
    pass

def correlation_matrix_to_latex(corr_df: pd.DataFrame) -> str:
    """Generate correlation table"""
    pass
```

---

### 6. Prepare Experimental Configurations ⭐⭐
**Why Important**: Ready to run immediately, organized setup
**Time**: 1 day
**Dependencies**: None
**Output**: All configs ready, organized structure

**Tasks**:

- [ ] **Create RQ1 Multi-CLD Configs**
  ```bash
  configs/
    rq1_multi_cld/
      social_norms.yaml
      depression.yaml
      climate_change.yaml
      healthcare.yaml
      education.yaml
  ```
  - One config per CLD
  - Consistent parameters across CLDs
  - Ready to run for generalization testing

- [ ] **Organize RQ2 Configs**
  - Review existing RQ2 configs
  - Ensure all needed parameter combinations
  - Document which configs already run

- [ ] **Prepare SA Batch Scripts**
  - Create templates for parallel execution
  - SLURM scripts if using cluster
  - GNU parallel scripts for local
  - Cost estimation scripts

- [ ] **Create Config Documentation**
  ```markdown
  # CONFIGS_README.md
  
  ## RQ1 Configs
  - Purpose, parameters, expected outputs
  
  ## RQ2 Configs
  - Purpose, parameters, expected outputs
  
  ## SA Configs
  - How generated, how to regenerate
  ```

---

### 7. Code Documentation and Cleanup ⭐
**Why Important**: Reproducibility, assessor review, future use
**Time**: 2-3 days
**Dependencies**: None
**Output**: Professional, well-documented codebase

**Tasks**:

- [ ] **Add Docstrings to All Modules**
  - Google-style or NumPy-style
  - Include examples
  - Type hints
  - Parameter descriptions

- [ ] **Create Module-Level READMEs**
  ```
  data_science/
    parameter_tuning_experiments/
      analysis/
        README.md  ← Explain RQ2 analysis modules
      sensitivity_analysis/
        README.md  ← Explain SA modules (done ✅)
  ```

- [ ] **Add Usage Examples**
  - Jupyter notebook for RQ2 analysis
  - Jupyter notebook for SA
  - End-to-end example workflows

- [ ] **Create Requirements Files**
  ```bash
  # requirements.txt
  pandas>=1.5.0
  numpy>=1.23.0
  matplotlib>=3.6.0
  seaborn>=0.12.0
  SALib>=1.4.0
  scipy>=1.9.0
  # ... all dependencies
  
  # environment.yml for conda
  name: thesis_env
  dependencies:
    - python=3.10
    - pandas
    # ...
  ```

- [ ] **Add Type Hints Throughout**
  ```python
  def analyze(
      self,
      samples_df: pd.DataFrame,
      outcomes: Union[np.ndarray, pd.Series],
      outcome_name: str = "outcome"
  ) -> Dict[str, pd.DataFrame]:
      """Type hints improve clarity"""
  ```

- [ ] **Code Quality Checks**
  ```bash
  # Run linters
  pylint data_science/parameter_tuning_experiments/
  flake8 data_science/parameter_tuning_experiments/
  mypy data_science/parameter_tuning_experiments/
  
  # Fix issues
  black data_science/parameter_tuning_experiments/  # Auto-format
  isort data_science/parameter_tuning_experiments/  # Sort imports
  ```

---

## 🟡 MEDIUM PRIORITY (Important but Not Blocking)

### 8. Create Thesis Style Guide ⭐
**Why Useful**: Consistency, faster writing, professional appearance
**Time**: 0.5 days

- [ ] **Notation Consistency Document**
  ```latex
  % NOTATION_GUIDE.tex
  
  % Sobol indices
  $S_i$ - first-order index
  $S_T^i$ - total-effect index
  $S_{ij}$ - second-order index
  
  % CI metrics
  $\text{PPL}$ - perplexity
  $p_{\min}$ - minimum probability
  
  % Always use these conventions
  ```

- [ ] **Figure Styling Template**
  - Consistent color palette
  - Font sizes (title, axis labels, legends)
  - DPI requirements
  - Caption style

- [ ] **Table Formatting Template**
  - Use `booktabs` package
  - Consistent decimal places
  - Bold for best values
  - Standard captions

### 9. Prepare Presentations ⭐
**Why Useful**: Practice explaining work, useful for defense
**Time**: 1-2 days

- [ ] **Create Thesis Defense Slides** (preliminary)
  - 30-40 slides for 20-minute talk
  - Key findings highlighted
  - Visual explanations of methods
  - Results ready to populate

- [ ] **Create Research Poster**
  - A0 size for conferences
  - Visual abstract
  - Key findings
  - QR code to thesis/code

### 10. Set Up Continuous Integration ⭐
**Why Useful**: Catch errors early, automate testing
**Time**: 1 day

- [ ] **GitHub Actions for Thesis**
  ```yaml
  # .github/workflows/build-thesis.yml
  name: Build Thesis
  on: [push]
  jobs:
    build:
      - name: Compile LaTeX
      - name: Check for undefined references
      - name: Upload PDF artifact
  ```

- [ ] **Pre-commit Hooks**
  ```yaml
  # .pre-commit-config.yaml
  repos:
    - repo: local
      hooks:
        - id: pytest
        - id: pylint
        - id: black
  ```

---

## 📋 Task Priority Matrix

| Task | Impact | Effort | Priority | Can Start Now? |
|------|--------|--------|----------|----------------|
| **1. Implement RQ3** | 🔥🔥🔥 | Medium | CRITICAL | ✅ YES |
| **2. Lit Review Expansion** | 🔥🔥🔥 | High | CRITICAL | ✅ YES |
| **3. Bibliography** | 🔥🔥 | Medium | HIGH | ✅ YES |
| **4. Methods Additions** | 🔥🔥 | Low | HIGH | ✅ YES |
| **5. Helper Scripts** | 🔥🔥 | Medium | HIGH | ✅ YES |
| **6. Experiment Configs** | 🔥🔥 | Low | HIGH | ✅ YES |
| **7. Code Documentation** | 🔥 | Medium | MEDIUM | ✅ YES |
| **8. Style Guide** | 🔥 | Low | MEDIUM | ✅ YES |
| **9. Presentations** | 🔥 | Medium | MEDIUM | ✅ YES |
| **10. CI Setup** | 🔥 | Low | MEDIUM | ✅ YES |

---

## 🎯 Recommended Execution Order

### Week 1: Critical Foundation
**Days 1-2**: Implement RQ3 system  
**Days 3-4**: Expand Literature Review (Section 2.11 on SA)  
**Days 5-7**: Complete Literature Review (Sections 2.9, 2.10) + Bibliography

### Week 2: Infrastructure and Documentation
**Days 1-2**: Create helper scripts  
**Days 3-4**: Prepare experimental configs  
**Day 5**: Code documentation and cleanup  
**Days 6-7**: Methods chapter additions

### Week 3: Polish and Prepare
**Days 1-2**: Style guide and consistency pass  
**Days 3-4**: Start presentations  
**Day 5**: CI setup  
**Days 6-7**: Buffer for any incomplete tasks

### Parallel Track (Throughout)
- Bibliography work (15 min/day adding papers)
- Code documentation (while writing code)
- Figure quality checks (as figures created)

---

## ✅ Success Metrics

After completing these tasks, you will have:

1. ✅ **Complete RQ3 implementation** - Ready to run experiments
2. ✅ **Comprehensive literature review** - Publication-quality Chapter 2
3. ✅ **Professional codebase** - Well-documented, reproducible
4. ✅ **Efficient workflow** - Helper scripts save hours of work
5. ✅ **Publication-ready bibliography** - 70-90 citations
6. ✅ **Enhanced methods** - Complete experimental documentation
7. ✅ **Ready experiments** - All configs prepared, organized

**Bottom Line**: You'll be 100% ready to execute experiments and write results. No blockers, no missing pieces, just execution.

---

## 💡 Why These Tasks Matter

### For Thesis Quality
- Literature review → Demonstrates scholarly depth
- Bibliography → Academic rigor, proper attribution
- Methods additions → Methodological completeness
- Style consistency → Professional appearance

### For Execution Efficiency
- RQ3 implementation → Unblocks critical RQ
- Helper scripts → Save 10-20 hours of manual work
- Experiment configs → No last-minute scrambling
- Code documentation → Easier debugging and verification

### For Defense and Beyond
- Presentations → Practice explaining your work
- Professional code → Impresses examiners
- Complete documentation → Easy for others to build on
- CI setup → Shows software engineering maturity

---

## 🚀 Get Started Now

**Immediate action** (next 30 minutes):
1. Create new branch: `git checkout -b rq3-implementation`
2. Create file: `data_science/rq3_smart_judging.py`
3. Start with class skeleton
4. Commit and push

**By end of today**:
- RQ3 basic structure complete
- Can start testing with dummy data

**By end of week**:
- RQ3 fully implemented and tested
- Literature review Section 2.11 drafted

**You're in great shape. These tasks will make the final push much smoother!** 🎯






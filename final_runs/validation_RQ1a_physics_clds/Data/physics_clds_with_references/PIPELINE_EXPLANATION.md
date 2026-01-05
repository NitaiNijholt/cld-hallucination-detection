# Physics CLD Judge Verification Pipeline - Complete Explanation

## Overview

This document explains **exactly how the judge verification pipeline works**, from data loading to final judgment, with special focus on **where the causal narrative comes from**.

---

## The Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA SOURCES                              │
│  (JSON files you created with references and motivations)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 2. LOAD INTO NEO4J                              │
│           (Creates graph structure in database)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              3. JUDGE QUERIES NEO4J                             │
│        (Reads edges with their motivations)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            4. BUILD JUDGE PROMPT                                │
│     (Creates LLM prompt with edge details)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              5. LLM JUDGES THE EDGE                             │
│    (GPT-4.1 evaluates correctness or citations)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           6. STORE VERDICT IN NEO4J                             │
│      (Updates edge with judgment results)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Detailed Explanation

### **Step 1: Data Sources (JSON Files)**

Your physics CLDs are stored in JSON files with this structure:

**Location**: `/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/physics_clds_with_references/`

**Example from `thermostat_heating_system_with_refs.json`**:

```json
{
  "source": "Room Temperature",
  "target": "Temperature Gap",
  "polarity": "NEGATIVE",
  "motivation": "According to the definition of Temperature Gap (Desired Temperature minus Room Temperature), an increase in Room Temperature directly causes a decrease in the Temperature Gap. This follows from the mathematical relationship: Gap = T_desired - T_actual. As T_actual increases, the gap decreases. This negative feedback is fundamental to thermostat control systems.",
  "references": [
    {
      "short_citation": "Sterman (2000)",
      "full_citation": "Sterman, J.D. (2000). Business Dynamics: Systems Thinking and Modeling for a Complex World. McGraw-Hill. Chapter 5.",
      "url": null,
      "supporting_quote": "The error or gap between the desired and actual state..."
    }
  ]
}
```

**👉 KEY POINT**: The `"motivation"` field is your **causal narrative** - this is what the judge will evaluate!

---

### **Step 2: Load into Neo4j** 

**Script**: `run_physics_cld_tests.py`

**Function**: `load_cld_to_neo4j()`

```python
# Load edges with their motivations
for edge in edges_data:
    rel_type = edge["type"].upper()  # POSITIVE or NEGATIVE
    motivation = edge.get("motivation", "")  # ← The causal narrative!
    
    # Create relationship in Neo4j with motivation attached
    session.run(f"""
        MATCH (s:variable {{name: $source, session_id: $session_id}})
        MATCH (t:variable {{name: $target, session_id: $session_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r.motivation = $motivation,  # ← Stored here!
            r.expert_validated = true
    """, {"source": edge["source"], "target": edge["target"], 
          "motivation": motivation, ...})
```

**What happens**:
1. Variables are created as nodes in Neo4j
2. Edges (relationships) are created between nodes
3. **The `motivation` text is stored as a property on each edge relationship**
4. This creates a graph database representation of your CLD

**Neo4j structure after loading**:
```
(Room Temperature:variable)
    -[NEGATIVE {motivation: "According to...", expert_validated: true}]->
(Temperature Gap:variable)
```

---

### **Step 3: Judge Queries Neo4j**

**Script**: `data_science/modules.py`

**Function**: `judge_all_edges_with_citations_serial()`

The judge function queries Neo4j to get all edges:

```python
def judge_all_edges_with_citations_serial(self, ...):
    # Query all edges from Neo4j
    with self.graph_db._get_session() as session:
        result = session.run("""
            MATCH (src:variable)-[r]->(tgt:variable)
            WHERE src.session_id = $session_id
            RETURN 
                src.name AS source,
                tgt.name AS target,
                type(r) AS rel_type,
                properties(r) AS props  # ← Contains motivation!
        """, {"session_id": self.session_id})
        
        edges = list(result)
```

**What comes back from Neo4j**:
```python
{
    "source": "Room Temperature",
    "target": "Temperature Gap",
    "rel_type": "NEGATIVE",
    "props": {
        "motivation": "According to the definition of Temperature Gap...",  # ← The causal narrative!
        "citations": [...],  # Optional: citation URLs if filled
        "expert_validated": true,
        "session_id": "..."
    }
}
```

---

### **Step 4: Build Judge Prompt**

**Script**: `data_science/modules.py`

**Function**: `_process_single_edge_judgment()`

The judge extracts the motivation and builds a prompt:

```python
def _process_single_edge_judgment(self, idx, total_edges, record, ...):
    src = record["source"]  # "Room Temperature"
    tgt = record["target"]  # "Temperature Gap"
    rel_type = record["rel_type"]  # "NEGATIVE"
    props = record.get("props") or {}
    
    # ← THIS IS WHERE THE CAUSAL NARRATIVE COMES FROM!
    motivation_text = props.get("motivation", "")
    motivation_text = motivation_text.strip()
    # motivation_text = "According to the definition of Temperature Gap..."
```

#### **For Correctness-Based Judging**:

```python
if approach == "correctness":
    # Build prompt using judgeCorrectness template
    variables = {
        "source": src,                    # "Room Temperature"
        "target": tgt,                    # "Temperature Gap"
        "relationship": rel_type,         # "NEGATIVE"
        "motivation": motivation_text,    # ← The causal narrative!
    }
    
    # Load prompt template from YAML
    sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCorrectness", variables)
```

**Actual prompt sent to GPT-4.1** (from `prompts_correctness_baseline.yaml`):

```yaml
System Prompt:
"You are an expert judge evaluating the correctness of causal claims..."

User Prompt:
"Evaluate this causal claim:

Source Variable: Room Temperature
Target Variable: Temperature Gap
Relationship Type: NEGATIVE
Motivation (Causal Narrative):
According to the definition of Temperature Gap (Desired Temperature minus Room 
Temperature), an increase in Room Temperature directly causes a decrease in the 
Temperature Gap. This follows from the mathematical relationship: Gap = T_desired 
- T_actual. As T_actual increases, the gap decreases. This negative feedback is 
fundamental to thermostat control systems.

Is this causal claim logically sound, scientifically accurate, and well-justified?
Respond with: CORRECT, PARTIALLY_CORRECT, or INCORRECT with a brief explanation."
```

#### **For Citation-Based Judging**:

If `approach == "citation"`, the judge also fetches citation content:

```python
if approach == "per_citation_aggregate":
    citations = props.get("citations", [])  # URLs from JSON
    
    # Fetch content from each URL using scraper
    for citation_url in citations:
        content = scraper.scrape(citation_url)  # ← Fetches actual paper content
        
        # Build prompt with citation content + motivation
        variables = {
            "source": src,
            "target": tgt,
            "relationship": rel_type,
            "motivation": motivation_text,     # ← The causal narrative!
            "citation_content": content,       # ← Scraped paper content
        }
        
        sys_prompt, usr_prompt, _ = self.tree.build_prompt("judgeCitation", variables)
```

**Citation-based prompt example**:

```yaml
"Does this citation support the causal claim?

Causal Claim:
[motivation text]

Citation Content:
[scraped text from Archive.org, PDF, etc.]

Verdict: SUPPORTED | PARTIALLY_SUPPORTED | CONTRADICTED | UNADDRESSED"
```

---

### **Step 5: LLM Judges the Edge**

The prompt is sent to GPT-4.1 (or specified judge model):

```python
for model in judge_models:  # ["gpt-4.1"]
    judge_llm = LLM(provider="openai", model=model, temperature=0.0)
    
    response = judge_llm.invoke(
        sys_prompt=sys_prompt,
        user_prompt=usr_prompt
    )
    
    # Parse response
    verdict = extract_verdict(response)  # "CORRECT", "PARTIALLY_CORRECT", etc.
    judge_results.append({"model": model, "verdict": verdict, ...})
```

**What GPT-4.1 sees**:
- The **source** and **target** variables
- The **relationship type** (POSITIVE/NEGATIVE)
- The **motivation text** (your causal narrative from the JSON)
- (For citation mode) The **actual content** scraped from references

**What GPT-4.1 returns**:
- A verdict: `CORRECT`, `PARTIALLY_CORRECT`, `INCORRECT` (correctness mode)
- Or: `SUPPORTED`, `PARTIALLY_SUPPORTED`, `CONTRADICTED` (citation mode)
- A brief explanation/reasoning

---

### **Step 6: Store Verdict in Neo4j**

```python
# Aggregate results from multiple judges (majority voting)
aggregate_verdict = most_common([r["verdict"] for r in judge_results])
aggregate_score = average([verdict_to_score(r["verdict"]) for r in judge_results])

# Update Neo4j edge with judgment
self._update_edge_with_judge_verdict(
    src, tgt, rel_type, 
    verdict=aggregate_verdict,
    message=json.dumps(judge_results),
    aggregate_score=aggregate_score
)
```

**Neo4j edge after judging**:
```
(Room Temperature)-[NEGATIVE {
    motivation: "According to the definition...",
    judge_verdict: "CORRECT",
    judge_message: "{...}",
    aggregate_score: 1.0,
    expert_validated: true
}]->(Temperature Gap)
```

---

## Summary: Where Does the Causal Narrative Come From?

### **Answer**: The `motivation` field in your JSON files!

**Data Flow**:

```
1. You write:     thermostat_heating_system_with_refs.json
                  ↓
                  {
                    "source": "Room Temperature",
                    "target": "Temperature Gap",
                    "motivation": "According to the definition of..."  ← HERE!
                  }

2. Script loads:  load_cld_to_neo4j() reads JSON
                  ↓
                  Stores motivation on Neo4j edge relationship

3. Judge reads:   Query Neo4j → get edge properties
                  ↓
                  motivation_text = props.get("motivation", "")

4. Judge builds:  LLM prompt with motivation as "causal narrative"
                  ↓
                  variables = {"motivation": motivation_text, ...}

5. LLM evaluates: GPT-4.1 reads the motivation text and judges it
                  ↓
                  Returns: CORRECT | PARTIALLY_CORRECT | INCORRECT

6. Result stored: verdict saved back to Neo4j edge
```

---

## Key Files in the Pipeline

### **Data Preparation** (You created these):
```
physics_clds_with_references/
├── thermostat_heating_system_with_refs.json  ← Contains motivations + references
├── predator_prey_with_refs.json
├── rc_circuit_with_refs.json
└── water_tank_with_refs.json
```

### **Execution Scripts**:
```
run_physics_cld_tests.py
├── Loads JSON → Neo4j
├── Calls judge_all_edges_with_citations_serial()
└── Saves raw results JSON
```

### **Analysis Scripts**:
```
analyze_physics_cld_results.py
├── Reads raw results JSON
├── Generates summary tables
└── Creates LaTeX output
```

### **Core Logic**:
```
data_science/modules.py
├── CausalDiscovery class
├── judge_all_edges_with_citations_serial()  ← Main judging function
├── _process_single_edge_judgment()          ← Extracts motivation, builds prompt
└── _update_edge_with_judge_verdict()        ← Stores results
```

### **Prompt Templates**:
```
alternative_prompts/
├── prompts_correctness_baseline.yaml  ← "judgeCorrectness" template
└── prompts_citation_baseline.yaml     ← "judgeCitation" template
```

---

## Example: Complete Flow for One Edge

### **Input** (from JSON):
```json
{
  "source": "Heating Rate",
  "target": "Room Temperature",
  "polarity": "POSITIVE",
  "motivation": "Based on the First Law of Thermodynamics (conservation of energy), heat added to a system increases its internal energy. For a room: dU = Q_in - Q_out. An increase in Heating Rate (Q_in) directly causes an increase in Room Temperature, assuming heat loss is constant."
}
```

### **Neo4j Storage**:
```cypher
(Heating Rate)-[POSITIVE {
    motivation: "Based on the First Law of Thermodynamics...",
    session_id: "20251215_183456"
}]->(Room Temperature)
```

### **Judge Query**:
```python
# Query returns:
{
    "source": "Heating Rate",
    "target": "Room Temperature",
    "rel_type": "POSITIVE",
    "props": {
        "motivation": "Based on the First Law of Thermodynamics..."
    }
}
```

### **Prompt Built**:
```
System: You are an expert judge...

User: Evaluate this causal claim:
Source: Heating Rate
Target: Room Temperature  
Relationship: POSITIVE
Motivation: Based on the First Law of Thermodynamics (conservation of energy), 
heat added to a system increases its internal energy. For a room: dU = Q_in - Q_out. 
An increase in Heating Rate (Q_in) directly causes an increase in Room Temperature...

Verdict: [CORRECT|PARTIALLY_CORRECT|INCORRECT]
```

### **GPT-4.1 Response**:
```
CORRECT

The causal claim is scientifically sound. The First Law of Thermodynamics correctly 
states that energy is conserved, and the equation dU = Q_in - Q_out accurately 
represents heat balance. The claim correctly identifies that increased heating rate 
(Q_in) leads to increased room temperature when heat loss (Q_out) is constant. 
This is a direct application of fundamental thermodynamics.
```

### **Result Stored**:
```cypher
(Heating Rate)-[POSITIVE {
    motivation: "Based on the First Law...",
    judge_verdict: "CORRECT",
    aggregate_score: 1.0,
    judge_message: "{\"verdict\": \"CORRECT\", \"reasoning\": \"...\"}"
}]->(Room Temperature)
```

---

## Why This Validates Your Judge

### **The Test**:
1. You provide physics CLDs with **well-established causal principles** (Newton's laws, conservation laws, etc.)
2. You write detailed **motivations** explaining the causal mechanism
3. The judge reads these motivations and evaluates them
4. Expected: **100% approval** on correctness mode (physics principles are irrefutable)

### **If Judge Works Correctly**:
- ✅ Physics CLDs → 100% "CORRECT" verdicts
- ✅ Demonstrates judge can recognize valid causal reasoning
- ✅ Contrast with Alzheimer's CLD → ~50% approval
- ✅ Proves the issue is **domain complexity**, not judge quality

### **What You're Actually Testing**:
- Can the judge recognize **valid causal logic**? (Yes, if physics CLDs score high)
- Can the judge distinguish **well-supported** from **poorly-supported** claims? (Yes, if physics >> Alzheimer's)
- Is the judge **too strict** or **too lenient**? (Calibrate using known-good physics examples)

---

## Files Created - Full Paths

All files are in:
```
/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/physics_clds_with_references/
```

**Documentation**:
- `PIPELINE_EXPLANATION.md` ← This file
- `README.md` ← Overview
- `FINAL_STATUS.md` ← Quick reference
- `MANUAL_SOURCE_VERIFICATION.md` ← URL verification
- `URL_SUMMARY.md` ← URL statistics
- `QUOTES_SUMMARY.md` ← Supporting quotes info

**Data Files** (with motivations + references):
- `thermostat_heating_system_with_refs.json`
- `predator_prey_with_refs.json`
- `rc_circuit_with_refs.json`
- `water_tank_with_refs.json`

---

**Bottom Line**: The causal narrative that the judge evaluates comes directly from the `"motivation"` field you wrote in each edge of your JSON files. This motivation is stored in Neo4j, read by the judge function, inserted into an LLM prompt, and evaluated by GPT-4.1.

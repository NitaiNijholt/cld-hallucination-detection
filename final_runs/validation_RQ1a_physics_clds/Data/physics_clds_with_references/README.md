# Physics CLDs with Complete Reference Mapping

This folder contains physics-based Causal Loop Diagrams (CLDs) with comprehensive reference mappings for each edge. These CLDs are designed for validating the judge AI system with well-established physical principles that have clear literature support.

## Purpose

These CLDs serve as "ground truth" validation cases where:
- Every causal relationship is derived from fundamental physics laws
- Each edge has explicit, citable references to authoritative sources
- The correctness of relationships is unambiguous and well-established
- Literature support is easily accessible (classic textbooks and seminal papers)

## File Structure

Each CLD is stored in a single JSON file with the following structure:

```json
{
  "cld_name": "Name of the system",
  "description": "Overview of the system dynamics",
  "domain": "Physics domain (e.g., Thermodynamics, Fluid Mechanics)",
  "primary_references": [
    {
      "short_citation": "Author (Year)",
      "full_citation": "Complete bibliographic reference",
      "doi": "DOI if available",
      "isbn": "ISBN if applicable",
      "relevance": "Why this source is fundamental to the CLD"
    }
  ],
  "edges": [
    {
      "source": "Source Variable Name",
      "target": "Target Variable Name",
      "polarity": "POSITIVE or NEGATIVE",
      "motivation": "Detailed explanation of why this causal relationship exists",
      "references": [
        {
          "short_citation": "Author (Year)",
          "full_citation": "Complete bibliographic reference with chapter/pages",
          "doi": "DOI if available",
          "isbn": "ISBN if applicable",
          "pages": "Specific chapter/page numbers",
          "relevance": "How this reference supports this specific edge"
        }
      ]
    }
  ]
}
```

## Available CLDs

### 1. Thermostat Heating System (`thermostat_heating_system_with_refs.json`)
- **Domain**: Thermal Control Systems / Physics
- **Key Principles**: Newton's Law of Cooling, First Law of Thermodynamics
- **Primary References**: Newton (1701), Sterman (2000)
- **Edges**: 8
- **Description**: Negative feedback control system maintaining room temperature

### 2. Predator-Prey System (`predator_prey_with_refs.json`)
- **Domain**: Population Ecology / Mathematical Biology
- **Key Principles**: Lotka-Volterra equations
- **Primary References**: Lotka (1925), Volterra (1926)
- **Edges**: 10
- **Description**: Oscillatory population dynamics with coupled feedback loops

### 3. RC Circuit Charging (`rc_circuit_with_refs.json`)
- **Domain**: Electrical Engineering / Circuit Theory
- **Key Principles**: Ohm's Law, Kirchhoff's Voltage Law
- **Primary References**: Ohm (1827), Kirchhoff (1845)
- **Edges**: 7
- **Description**: Exponential charging behavior in series RC circuit

### 4. Water Tank Draining System (`water_tank_with_refs.json`)
- **Domain**: Fluid Mechanics / Hydraulics
- **Key Principles**: Conservation of Mass, Pascal's Law, Torricelli's Theorem
- **Primary References**: Torricelli (1643), Pascal (1663)
- **Edges**: 6
- **Description**: Nonlinear draining dynamics governed by hydrostatics

## Reference Quality

All references include:
- **Seminal papers**: Original formulations of physical laws (Newton 1701, Ohm 1827, etc.)
- **Standard textbooks**: Widely-used undergraduate/graduate texts (Sterman, Incropera, White)
- **Professional handbooks**: Industry standards (ASHRAE Handbook)
- **Complete citations**: Full bibliographic information including DOI/ISBN when available
- **Page-level precision**: Specific chapters and pages for each reference

## Usage

These files can be used to:
1. **Validate judge correctness**: Physics edges should receive high approval ratings
2. **Test citation-based judging**: References are real and should be (mostly) scrapable
3. **Benchmark performance**: Compare judge behavior on "easy" physics vs "hard" domain-specific CLDs
4. **Thesis documentation**: Generate LaTeX tables with complete reference information

## Difference from Original Format

These files differ from the original CLD data files in:
- **Unified structure**: Single JSON file per CLD instead of separate vars/edges/citations
- **Complete references**: Every edge has full bibliographic details
- **Metadata**: Includes domain, description, and primary references at CLD level
- **Detailed relevance**: Each reference explains specifically how it supports that edge
- **Page precision**: Chapter and page numbers included where applicable

## Integration with Existing Scripts

To use these files with existing judge validation scripts:
1. Load the JSON file
2. Extract the `edges` array
3. For citation-based judging, use the `references` array for each edge
4. For correctness-based judging, use the `motivation` field

Example parsing:
```python
import json

with open('thermostat_heating_system_with_refs.json', 'r') as f:
    cld_data = json.load(f)

# Extract edges for processing
for edge in cld_data['edges']:
    source = edge['source']
    target = edge['target']
    polarity = edge['polarity']
    motivation = edge['motivation']
    references = edge['references']  # List of reference objects
```

## Future Extensions

Potential additions:
- More physics CLDs (pendulum, chemical reactions, heat exchangers)
- Machine-readable DOI links for automated citation fetching
- BibTeX export functionality
- Visual CLD diagrams generated from the data
- URL fields for online references where available

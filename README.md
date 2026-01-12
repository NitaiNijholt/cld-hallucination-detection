# Causalix Platform

<img src="logo.png" alt="Causalix Logo" width="200"/>

## Overview

A semi-autonomous LLM-based CLD building research system. Orchestrating multiple agents and processes to integrating scientific knowledge into the creation of annotated Causal Loop Diagrams.

---

## Reproducibility

The computational environment for this project was set up following principles from the Workflow for Open Reproducible Code in Science (WORCS), with emphasis on dependency management and code availability.

> Van Lissa, C. J., Brandmaier, A. M., Brinkman, L., Lamprecht, A.-L., Peikert, A., Struiksma, M. E., & Vreede, B. M. I. (2021). WORCS: A workflow for open reproducible code in science. *Data Science*, 4(1), 29–49. https://doi.org/10.3233/DS-210031

### Repository Structure for Reproducibility

| Directory | Contents |
|-----------|----------|
| `data_science/` | Analysis scripts and experimental pipelines |
| `final_runs/` | Experimental results, figures, and LaTeX tables |
| `thesis/final_thesis/` | Thesis source files (LaTeX) |
| `services/` | Microservices for deep research and LLM integration |
| `backend/` | Platform API and database connectors |

### Data Availability

- **Experimental outputs**: All experimental results (simulation data, validation data, analysis outputs, figures, tables) are in `final_runs/`
- **Analysis scripts**: Reproducible Python scripts for each research question
- **Ground truth CLDs**: Causal Loop Diagrams used for validation experiments (in `data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/`)
- **Simulation module**: Core simulation logic including CLD generation, corruption, judging, and correction algorithms resides in `data_science/modules.py`


For access to raw experimental data or additional materials, contact: nitai.nijholt@gmail.com

### Dependencies

This project uses Python 3.10+ with dependency management via `uv`:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies from lock file
uv sync
```

### Citing This Work

See [CITATION.cff](./CITATION.cff) for citation information.

---

## Project Structure

The platform consists of two main components:

- **Backend**: Python-based API with connections to various language model providers and database systems
- **Frontend**: React-based web interface for interacting with the platform

## Technologies

### Backend
- Python
- ElasticSearch
- Neo4j
- Redis
- Language Model Integrations:
  - OpenAI
  - Claude (Anthropic)
  - Perplexity

### Frontend
- React
- D3.js (for Network Graph visualization)

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Node.js (for local frontend development)
- Python 3.12+ (for local backend development)

### Running with Docker for Local Development

Prior to running the application, fill out an .env file in your local directory that matches the necessary .env variables seen in [.env.example](./.env.example)!

From the root directory run:
```bash
docker-compose -f docker-compose.dev.yml up
```

For just the Backend:
```bash
cd backend
pip install -r requirements.txt
python run.py
```
For just the frontend;
```bash
cd frontend
npm install
npm start
```

Although to run the frontend with fake data it is necessary to run both the front and backend together with the environment variable setting:
```python
ENV=FRONTEND-DEV
```

## Features

- Interactive network graph visualization
- Transaction logging and context management
- Multiple LLM provider integration
- Graph database for relationship modeling
- ElasticSearch for log monitoring

## License

This project is licensed under the terms found in the [LICENSE](./LICENSE) file.

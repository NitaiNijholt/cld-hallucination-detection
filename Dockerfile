# Thesis Reproduction Environment
# Reproduces all figures/tables from experimental data

FROM ubuntu:24.04

LABEL maintainer="Nitai Nijholt"
LABEL description="Reproducibility environment for CLD Hallucination Detection thesis"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    # TeX Live for thesis compilation
    texlive-full \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /thesis

# Copy project files
COPY pyproject.toml uv.lock ./
COPY final_runs/ ./final_runs/
COPY thesis/ ./thesis/

# Install Python dependencies
RUN uv sync

# Create output directory (will be overwritten by volume mount if provided)
RUN mkdir -p /output

# Default command: run reproduction and compile thesis
CMD ["bash", "-c", "\
    set -e && \
    mkdir -p /output && \
    echo '=== Step 1: Compile thesis BEFORE reproduction (shows placeholders) ===' && \
    cd thesis/reproducible_version && \
    (pdflatex -interaction=nonstopmode main.tex || true) && \
    (bibtex main || true) && \
    (pdflatex -interaction=nonstopmode main.tex || true) && \
    (pdflatex -interaction=nonstopmode main.tex || true) && \
    cp main.pdf /output/thesis_BEFORE.pdf && \
    echo 'Saved: /output/thesis_BEFORE.pdf' && \
    echo '' && \
    echo '=== Step 2: Run reproduction (~10 minutes) ===' && \
    cd /thesis && \
    uv run python final_runs/reproduce_all_thesis_assets.py && \
    echo '' && \
    echo '=== Step 3: Compile thesis AFTER reproduction ===' && \
    cd thesis/reproducible_version && \
    (pdflatex -interaction=nonstopmode main.tex || true) && \
    (bibtex main || true) && \
    (pdflatex -interaction=nonstopmode main.tex || true) && \
    (pdflatex -interaction=nonstopmode main.tex || true) && \
    cp main.pdf /output/thesis_AFTER.pdf && \
    echo 'Saved: /output/thesis_AFTER.pdf' && \
    echo '' && \
    echo '=== Done! ===' && \
    echo 'Output PDFs:' && \
    ls -la /output/*.pdf \
"]


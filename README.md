# A Hypergraph Analysis of the European Commission Lobby Network

This repository contains the code and data used to replicate the results presented in the research article:

**A Hypergraph Analysis of the European Commission Lobby Network**  
_Amina Azaiez and Antoine Mandel (Paris 1 Panthéon Sorbonne University)_  
[ResearchSquare preprint](https://www.researchsquare.com/article/rs-6130857/v1)

## Overview

The project analyzes lobbying activities directed at the European Commission using hypergraph modeling. The codebase allows you to:

- Construct and analyze a hypergraph of stakeholder–EC interactions.
- Compute various centrality measures (e.g., hypercoreness).
- Run configuration model simulations using MCMC methods.
- Generate the figures and results presented in the paper.

## Repository Structure

- `data/` – Raw and preprocessed datasets.
- `src/` – Source Python scripts used to process data and build models.
- `scripts/` – Entry-point programs to:
  - compute centralities,
  - export hypergraphs to `.gexf` format,
  - run configuration model simulations 
  (results saved in `out/`).
- `notebooks/` – Contains `main.ipynb` notebook for reproducing the figures and core analyses.
- `out/` – Output directory for figures, metrics, and simulation results.

## Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt

# volta-basin-ai-hydrology-digital-twin
Hybrid hydrological modelling workflow, scripts, metadata, and results for the second Volta Basin study.
# Two‑stage residual correction (WRF‑Hydro + LSTM + XGBoost) for flood simulation in the Volta River Basin

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

This repository contains the code and configuration files for the paper:

> **Benchmark-relative two-stage residual correction (WRF-Hydro + LSTM + XGBoost) for wet-season flood dynamics in the Volta River Basin (1974-2024)**  
> 

## Overview

The framework improves flood discharge simulation by sequentially correcting residuals from a physically based model (WRF-Hydro) – first with an LSTM network, then with XGBoost. The code reproduces the main results of the paper using real datasets (CHIRPS, ERA5, G-RUN ENSEMBLE).

## requirements.txt – Python dependencies
numpy>=1.26.0
pandas>=2.0.0
xarray>=2023.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
tensorflow>=2.15.0
netCDF4>=1.6.0
statsmodels>=0.14.0
jupyter>=1.0.0

# Data files (too large)
data/raw/
# Outputs that can be regenerated
results/
figures/
# Python cache
__pycache__/
*.pyc
# Jupyter checkpoint
.ipynb_checkpoints/
# Environment
venv/
env/

## Repository structure
volta-basin-ai-hydrology-digital-twin/
├── .gitignore
├── README.md
├── requirements.txt
├── Volta_Triple_Hybrid.ipynb
├── data/
│   └── README.md
├── results/               (will be created by the notebook)
└── figures/               (will be created by the notebook)


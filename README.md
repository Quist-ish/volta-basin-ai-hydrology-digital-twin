# volta-basin-ai-hydrology-digital-twin
Hybrid hydrological modelling workflow, scripts, metadata, and results for the second Volta Basin study.
# Two‑stage residual correction (WRF‑Hydro + LSTM + XGBoost) for flood simulation in the Volta River Basin

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

This repository contains the code and configuration files for the paper:

> **Benchmark-relative two-stage residual correction (WRF-Hydro + LSTM + XGBoost) for wet-season flood dynamics in the Volta River Basin (1974-2024)**  
> 

## Overview

The framework improves flood discharge simulation by sequentially correcting residuals from a physically based model (WRF-Hydro) – first with an LSTM network, then with XGBoost. The code reproduces the main results of the paper using real datasets (CHIRPS, ERA5, G-RUN ENSEMBLE).

## Repository structure


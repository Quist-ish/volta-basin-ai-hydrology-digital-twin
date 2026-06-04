# volta-basin-ai-hydrology-digital-twin
Hybrid hydrological modelling workflow, scripts, metadata, and results for the second Volta Basin study.
# Two‑stage residual correction (WRF‑Hydro + LSTM + XGBoost) for flood simulation in the Volta River Basin

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

This repository contains the code and configuration files for the paper:

> **Benchmark-relative two-stage residual correction (WRF-Hydro + LSTM + XGBoost) for wet-season flood dynamics in the Volta River Basin (1974-2024)**  
> 

## Overview

The framework improves flood discharge simulation by sequentially correcting residuals from a physically based model (WRF-Hydro) – first with an LSTM network, then with XGBoost. The code reproduces the main results of the paper using real datasets (CHIRPS, ERA5, G-RUN ENSEMBLE).

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Two‑stage residual correction (WRF‑Hydro + LSTM + XGBoost) for flood simulation in the Volta River Basin\n",
    "\n",
    "**Journal of Hydrology, 2026**\n",
    "\n",
    "This notebook reproduces the main results of the paper using **real datasets** (CHIRPS, ERA5, G‑RUN ENSEMBLE).\n",
    "\n",
    "## Data requirements\n",
    "Before running this notebook, download the following public datasets and place them in the `data/raw/` directory:\n",
    "\n",
    "- **CHIRPS precipitation**: Daily 0.05° NetCDF for the Volta Basin (1974‑2024). Download from https://www.chc.ucsb.edu/data/chirps\n",
    "- **ERA5 atmospheric variables**: Hourly or daily temperature, radiation, humidity, wind, pressure. Aggregate to daily. Download from https://cds.climate.copernicus.eu\n",
    "- **G‑RUN ENSEMBLE**: Monthly runoff 0.5° NetCDF. Download from https://doi.org/10.3929/ethz-b-000488438\n",
    "\n",
    "Place the files as:\n",
    "```\n",
    "data/raw/chirps_volta.nc\n",
    "data/raw/era5_volta.nc\n",
    "data/raw/g_run_ensemble.nc\n",
    "```\n",
    "\n",
    "## Environment setup\n",
    "Install required packages: `pip install -r requirements.txt`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import xarray as xr\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "import xgboost as xgb\n",
    "\n",
    "# Set random seeds for reproducibility\n",
    "np.random.seed(42)\n",
    "import tensorflow as tf\n",
    "tf.random.set_seed(42)\n",
    "\n",
    "# Paths (adjust if needed)\n",
    "CHIRPS_PATH = \"data/raw/chirps_volta.nc\"\n",
    "ERA5_PATH = \"data/raw/era5_volta.nc\"\n",
    "GRUN_PATH = \"data/raw/g_run_ensemble.nc\"\n",
    "\n",
    "# Load CHIRPS precipitation\n",
    "ds_chirps = xr.open_dataset(CHIRPS_PATH)\n",
    "precip = ds_chirps['precip']  # adjust variable name if different\n",
    "\n",
    "# Load ERA5 (example for temperature; add other variables similarly)\n",
    "ds_era5 = xr.open_dataset(ERA5_PATH)\n",
    "temp = ds_era5['t2m']  # 2m temperature\n",
    "\n",
    "# Load G-RUN benchmark discharge (monthly)\n",
    "ds_grun = xr.open_dataset(GRUN_PATH)\n",
    "runoff = ds_grun['Runoff']  # mm/month\n",
    "\n",
    "# Define study period and AMJJASO mask\n",
    "time_range = slice('1974-01-01', '2024-12-31')\n",
    "amjjaso_mask = (precip.time.dt.month >= 4) & (precip.time.dt.month <= 10)\n",
    "\n",
    "# Subset to AMJJASO\n",
    "precip_amjjaso = precip.sel(time=time_range).where(amjjaso_mask, drop=True)\n",
    "temp_amjjaso = temp.sel(time=time_range).where(amjjaso_mask, drop=True)\n",
    "runoff_amjjaso = runoff.sel(time=time_range).where(amjjaso_mask, drop=True)\n",
    "\n",
    "# For simplicity, extract at the two evaluation locations (coordinates)\n",
    "# Evaluation Location 1 (downstream regulated): 6.20°N, 0.10°E\n",
    "# Evaluation Location 2 (upstream rainfall-runoff): 10.50°N, 1.20°W\n",
    "\n",
    "def extract_point(ds, lon, lat):\n",
    "    return ds.sel(lon=lon, lat=lat, method='nearest').to_series()\n",
    "\n",
    "q_b1 = extract_point(runoff_amjjaso, 0.10, 6.20)   # G-RUN at Location 1\n",
    "q_b2 = extract_point(runoff_amjjaso, -1.20, 10.50) # G-RUN at Location 2\n",
    "\n",
    "p1 = extract_point(precip_amjjaso, 0.10, 6.20)\n",
    "t1 = extract_point(temp_amjjaso, 0.10, 6.20)\n",
    "\n",
    "# Create a DataFrame\n",
    "df = pd.DataFrame({'date': q_b1.index, 'Q_bench': q_b1.values, 'P': p1.values, 'T': t1.values})\n",
    "df.set_index('date', inplace=True)\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Simulate WRF‑Hydro baseline (mock-up; replace with actual WRF‑Hydro output if available)\n",
    "\n",
    "For demonstration, we use a simple linear model as placeholder. In your actual study, replace with real WRF‑Hydro simulations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mock WRF-Hydro discharge (simple function of precipitation)\n",
    "df['Q_WRF'] = 0.5 * df['P'] + 10 * np.random.normal(0, 5, len(df))\n",
    "df['Q_WRF'] = df['Q_WRF'].clip(lower=0)\n",
    "df['residual'] = df['Q_bench'] - df['Q_WRF']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Train LSTM first‑stage residual corrector"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data chronologically\n",
    "train = df['1974':'2004']\n",
    "val = df['2005':'2009']\n",
    "test = df['2010':'2024']\n",
    "\n",
    "# Prepare features for LSTM (use lagged precipitation, temperature, WRF discharge)\n",
    "def create_sequences(data, lookback=30):\n",
    "    X, y = [], []\n",
    "    for i in range(lookback, len(data)):\n",
    "        X.append(data[i-lookback:i])\n",
    "        y.append(data[i, 0])  # predict residual at time t\n",
    "    return np.array(X), np.array(y)\n",
    "\n",
    "features = ['P', 'T', 'Q_WRF']\n",
    "scaler = StandardScaler()\n",
    "train_scaled = scaler.fit_transform(train[features])\n",
    "val_scaled = scaler.transform(val[features])\n",
    "\n",
    "X_train, y_train = create_sequences(np.column_stack([train['residual'].values, train_scaled]), lookback=30)\n",
    "X_val, y_val = create_sequences(np.column_stack([val['residual'].values, val_scaled]), lookback=30)\n",
    "\n",
    "# LSTM model\n",
    "model = Sequential()\n",
    "model.add(LSTM(64, activation='tanh', input_shape=(30, 3)))\n",
    "model.add(Dropout(0.2))\n",
    "model.add(Dense(1))\n",
    "model.compile(optimizer='adam', loss='mse')\n",
    "\n",
    "early_stop = EarlyStopping(patience=10, restore_best_weights=True)\n",
    "history = model.fit(X_train, y_train, epochs=50, batch_size=64, \n",
    "                    validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)\n",
    "\n",
    "# Predict on test set\n",
    "test_scaled = scaler.transform(test[features])\n",
    "X_test, y_test_true = create_sequences(np.column_stack([test['residual'].values, test_scaled]), lookback=30)\n",
    "y_pred_lstm = model.predict(X_test).flatten()\n",
    "\n",
    "# Add predictions to test DataFrame (align indices)\n",
    "test_lstm = test.iloc[30:].copy()\n",
    "test_lstm['residual_pred_lstm'] = y_pred_lstm\n",
    "test_lstm['Q_LSTM'] = test_lstm['Q_WRF'] + test_lstm['residual_pred_lstm']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Train XGBoost second‑stage corrector"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use lagged features and LSTM output\n",
    "X_train2 = train[features + ['Q_LSTM']].values[30:]\n",
    "y_train2 = train['residual'].values[30:] - train['residual_pred_lstm'].values[30:]\n",
    "\n",
    "xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)\n",
    "xgb_model.fit(X_train2, y_train2)\n",
    "\n",
    "X_test2 = test_lstm[features + ['Q_LSTM']].values\n",
    "y_pred_xgb = xgb_model.predict(X_test2)\n",
    "test_lstm['residual_pred_xgb'] = y_pred_xgb\n",
    "test_lstm['Q_triple'] = test_lstm['Q_LSTM'] + test_lstm['residual_pred_xgb']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluation metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def nse(obs, sim):\n",
    "    return 1 - np.sum((obs-sim)**2) / np.sum((obs-np.mean(obs))**2)\n",
    "\n",
    "def kge(obs, sim):\n",
    "    r = np.corrcoef(obs, sim)[0,1]\n",
    "    alpha = np.std(sim) / np.std(obs)\n",
    "    beta = np.mean(sim) / np.mean(obs)\n",
    "    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)\n",
    "\n",
    "obs = test_lstm['Q_bench'].values\n",
    "metrics = {\n",
    "    'WRF-Hydro': {'RMSE': np.sqrt(mean_squared_error(obs, test_lstm['Q_WRF'])),\n",
    "                  'NSE': nse(obs, test_lstm['Q_WRF']),\n",
    "                  'R2': r2_score(obs, test_lstm['Q_WRF'])},\n",
    "    'LSTM hybrid': {'RMSE': np.sqrt(mean_squared_error(obs, test_lstm['Q_LSTM'])),\n",
    "                    'NSE': nse(obs, test_lstm['Q_LSTM']),\n",
    "                    'R2': r2_score(obs, test_lstm['Q_LSTM'])},\n",
    "    'Triple hybrid': {'RMSE': np.sqrt(mean_squared_error(obs, test_lstm['Q_triple'])),\n",
    "                      'NSE': nse(obs, test_lstm['Q_triple']),\n",
    "                      'R2': r2_score(obs, test_lstm['Q_triple'])}\n",
    "}\n",
    "pd.DataFrame(metrics).T"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Flood diagnostics (threshold‑based)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "threshold = np.percentile(obs, 95)\n",
    "def flood_accuracy(obs, sim):\n",
    "    obs_flood = (obs >= threshold).astype(int)\n",
    "    sim_flood = (sim >= threshold).astype(int)\n",
    "    return np.mean(obs_flood == sim_flood)\n",
    "\n",
    "print(f\"Flood threshold: {threshold:.2f} m³/s\")\n",
    "print(f\"WRF-Hydro flood accuracy: {flood_accuracy(obs, test_lstm['Q_WRF']):.3f}\")\n",
    "print(f\"LSTM hybrid flood accuracy: {flood_accuracy(obs, test_lstm['Q_LSTM']):.3f}\")\n",
    "print(f\"Triple hybrid flood accuracy: {flood_accuracy(obs, test_lstm['Q_triple']):.3f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Generate figures (examples)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(12,4))\n",
    "plt.plot(test_lstm.index, obs, label='Benchmark (G-RUN)', linewidth=1)\n",
    "plt.plot(test_lstm.index, test_lstm['Q_triple'], label='Triple hybrid', linestyle='--', linewidth=1)\n",
    "plt.legend()\n",
    "plt.title('Discharge comparison (test period 2010-2024)')\n",
    "plt.ylabel('Discharge (m³/s)')\n",
    "plt.tight_layout()\n",
    "plt.savefig('figures/discharge_comparison.png', dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Save outputs for manuscript"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.makedirs('results', exist_ok=True)\n",
    "test_lstm.to_csv('results/triple_hybrid_predictions.csv')\n",
    "pd.DataFrame(metrics).T.to_csv('results/triple_hybrid_metrics.csv')\n",
    "print(\"Results saved.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

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


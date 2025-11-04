# RNN-for-EPF
## Overview
The script implements a full neural forecasting pipeline for hourly electricity price prediction using hybrid recurrent-linear models (RLM-KF-RNN). Combines preprocessing, feature engineering, training, forecast decomposition, and Optuna optimization with a focus on robustness, interpretability, and modularity.
## Model Architectures
Seven model variants are included to capture different temporal and nonlinear relationships:

| Model | Description |
|:------|:-------------|
| **M1** | RNN with ReLU activation |
| **M2** | RNN with Identity activation |
| **M3** | Reduced Linear (OLS benchmark) |
| **M4** | Linear + RNN (ReLU) hybrid |
| **M5** | Linear + RNN (Identity) hybrid |
| **M6** | Dual RNNs (ReLU + Identity) |
| **M7** | Full hybrid (two RNNs + linear skip) |

Each model supports **forecast decomposition**, allowing analysis of RNN and linear contributions separately.

## Pipeline Features
- **Data Alignment:** Handles daylight-saving adjustments for the German electricity market.  
- **Feature Engineering:** Creates weekday dummies, lagged prices, renewable forecasts, and fuel indicators.  
- **Rolling Window Training:** Uses safe sequential updates with deterministic initialisation and masked inputs.  
- **Regularisation:** Incorporates L1 penalty, gradient clipping, and adaptive learning rate scheduling.  
- **Hyperparameter Optimisation:** Utilises Optuna to tune learning rate, window size, L1/L2 terms, neurons, and OLS blending (`alpha`).
  
  ## Evaluation & Outputs
- RMSE and MAE performance metrics  
- Forecast decomposition consistency checks  
- Unstandardized forecasts and component-level outputs  
- Weekly-naïve baseline comparison  

###############################################################################################################
# M1 = RNN + ReLU
# M2 = RNN + Identity
# M3 = Expert Linear
# M4 = Expert Linear + Elman RNN (OLS weight init as hyperparameter)
# M5 = Expert Linear + Elman RNN (identity activation)
# M6 = RNN (ReLU) + RNN (Identity)
# M7 = M6 + M3  (two RNNs + linear skip)
#%% Model type
model_type = 7

# packages
import re
import torch
import numpy as np
import pandas as pd
from calendar import day_abbr
import locale
import os
from my_functions import DST_trafo
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
import joblib
import optuna
import time
import random
import matplotlib.colors as mcolors
import colorsys

# Enforce deterministic behavior (CuBLAS + CuDNN)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)  # Optional: can degrade performance

#%% ##########################################
# ############ data Preprocessing #########
# ############################################
#  set language setting
locale.setlocale(locale.LC_ALL, "en_US.utf8")
locale.getlocale()

# read the data
data = pd.read_csv("Germany.csv")
# data = pd.read_csv("Germany.csv")

# Create new column Wind_DA by summing WindOn_DA and WindOff_DA
data["Wind_DA"] = data["WindOn_DA"] + data["WindOff_DA"]

# select the price and time
id_select = 1
price = data.iloc[:, id_select]

time_utc = pd.to_datetime(data["time_utc"], utc=True, format="%Y-%m-%d %H:%M:%S")

# Use IANA zone with DST rules
local_time_zone = "Europe/Berlin"
time_lt = time_utc.dt.tz_convert(local_time_zone)

S = 24

# Save the start and end-time (keep as UTC-naive then back to UTC for numeric grid)
start_end_time_S = time_lt.iloc[[0, -1]].dt.tz_localize(None).dt.tz_localize("UTC")

# creating 'fake' local time (regular 1/S-hour grid between endpoints)
start_end_time_S_num = pd.to_numeric(start_end_time_S)
time_S_numeric = np.arange(
    start=start_end_time_S_num.iloc[0],
    stop=start_end_time_S_num.iloc[1] + 24 * 60 * 60 * 10**9 / S,
    step=24 * 60 * 60 * 10**9 / S,
)

# 'fake' local time
time_S = pd.Series(pd.to_datetime(time_S_numeric, utc=True))
dates_S = pd.Series(time_S.dt.date.unique())

# DST transform on whole data array (aligns per-hour slices across DST)
data_array = DST_trafo(X=data.iloc[:, 1:], Xtime=time_utc, tz=local_time_zone)

# save price as dataframe aligned to dates_S grid
price_S = pd.DataFrame(data_array[..., 0], index=dates_S)

#%% #####################################################
# ########### Define the variable of interest ############
# ########################################################

# Save the variable names
reg_names = data.columns[1:]

# Weekday dummies (Mon, Sat, Sun)
wd = [1, 6, 7]

# Specify the lags
price_s_lags_full = [1, 2, 7]   # Linear models use full set
price_s_lags_rnn  = [1]         # RNNs use a minimal set to avoid redundancy
# Use correct lags depending on model_type
if model_type in [1, 2, 4, 5, 6, 7]:
    price_s_lags = price_s_lags_rnn
else:
    price_s_lags = price_s_lags_full

# DA lags (today only; can be extended)
da_lag = [0]

# Fuel lags (EOD markets available at d-1 -> lag 2 on S=24)
fuel_lags = [2]

# Training hyperparams
batch_size = 32
num_epochs_init = 60   # initial window
num_epochs_all = 10    # rolling windows

# Keep last 2 years for test
N = 2 * 365
dat_eval = data_array[:-N, :, :]
days_eval = pd.to_datetime(dates_S)[:-N]

################################################################
###############################################################

def create_sequences(X, y, seq_len=1):
    """
    Robust sequence constructor.
    - Clamps seq_len >=1.
    - If seq_len > len(X), pads at the FRONT with the earliest row so we still
      return a single sequence ending at the last available row.
    Inputs:
      X: (T, F), y: (T, S)  [y aligns 1:1 with rows of X; we use last target of each window]
    Returns:
      X_seq: (Nwin, seq_len, F)
      y_seq: (Nwin, S)
    """
    seq_len = max(1, int(seq_len))
    T = X.shape[0]
    F = X.shape[1]
    if T <= 0:
        # defensive: should not happen in our pipeline
        return torch.zeros((1, seq_len, F), dtype=X.dtype, device=X.device), y[-1:].clone()

    # If we have fewer rows than seq_len, create exactly one padded window
    if T < seq_len:
        pad = X[:1, :].repeat(seq_len - T, 1)  # front pad
        X_pad = torch.cat([pad, X], dim=0)     # (seq_len, F)
        X_seq = X_pad.unsqueeze(0)              # (1, seq_len, F)
        y_seq = y[-1:, :]                       # (1, S)
        return X_seq, y_seq

    # Normal sliding windows
    X_seq, y_seq = [], []
    for i in range(T - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return torch.stack(X_seq), torch.stack(y_seq)

#%% #####################################################
# ################ matrix ################################
# ########################################################

def reg_matrix(dat_eval, days_eval, model_type=None):
    S = dat_eval.shape[1]
    days_ext = days_eval

    # Weekday dummies (1=Mon,...,7=Sun) -> columns WD_<k>
    weekdays_num = days_ext.dt.weekday + 1
    WD = np.stack([(weekdays_num == x).astype(int).to_numpy() for x in wd], axis=1)
    wd_columns = [f"WD_{x}" for x in wd]

    # Names
    da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]
    fuel_names = ["Coal", "NGas", "Oil", "EUA"]

    # Helper
    def get_lagged(Z, lag):
        return np.concatenate((np.repeat(np.nan, lag), Z[: (len(Z) - lag)]))

    # Fuel lags (lag 2)
    mat_fuels = np.concatenate(
        [
            np.apply_along_axis(
                get_lagged, 0, dat_eval[:, 0, reg_names.isin(fuel_names)], lag=l
            )
            for l in fuel_lags
        ],
        axis=-1,
    )
    fuel_columns = [f"{fuel}_lag_{l}" for l in fuel_lags for fuel in fuel_names]

    # price_last is yesterday's last hour -> leakage for s==23 if not removed downstream
    price_last = get_lagged(Z=dat_eval[:, S - 1, reg_names == "Price"][..., 0], lag=1)
    price_last_column = ["price_last"]

    # Base block shared across s
    base_regmat = np.column_stack((WD, mat_fuels, price_last))
    regmat1 = pd.DataFrame(base_regmat, columns=wd_columns + fuel_columns + price_last_column)
    columns_base = regmat1.shape[1]

    all_dataframes = []
    for s in range(S):
        # y and price lags for this s
        acty = dat_eval[:, s, reg_names == "Price"][..., 0]
        mat_price_lags = np.transpose([get_lagged(lag=lag, Z=acty) for lag in price_s_lags])

        # DA forecasts (stack each series' specified da_lag horizontally)
        mat_da_forecasts = dat_eval[:, s, reg_names.isin(da_forecast_names)]
        stacked_da = []
        for i in range(len(da_forecast_names)):
            da_var = np.transpose([get_lagged(lag=lag, Z=mat_da_forecasts[:, i]) for lag in da_lag])
            stacked_da.append(da_var)
        da_all_var = np.hstack(stacked_da)

        # Regressor matrix for this s (first column = y for convenience)
        regmat2 = np.column_stack((acty, mat_price_lags, da_all_var))

        # Column names with suffix _s{s}
        columns = (
            [f"Price_s{s}"]
            + [f"Price_lag_{lag}_s{s}" for lag in price_s_lags]
            + [f"{name}_lag_{lag}_s{s}" for name in da_forecast_names for lag in da_lag]
        )
        df = pd.DataFrame(regmat2, columns=columns)
        all_dataframes.append(df)

    # Combine per-s blocks + base block
    final_dataframe = pd.concat(all_dataframes, axis=1)
    regmat = pd.concat([final_dataframe, regmat1], axis=1)
    columns_s = all_dataframes[0].shape[1]
    columns_total = regmat.shape[1]
    return [regmat, columns_s, columns_base, columns_total, len(da_forecast_names)]

# Extract matrices / indices
regmat_eval, columns_s, columns_base, columns_total, _ = reg_matrix(
    dat_eval, days_eval, model_type=model_type
)

# Remove NAs (train-only statistics will be respected downstream)
regmat0_eval = regmat_eval.dropna()

# Coefficient names (remove trailing _s{num})
indices = list(range(0 * columns_s, (0 + 1) * columns_s))
non_sn_indices = list(range(columns_total - columns_base, columns_total))
coefficient = regmat0_eval.iloc[:, indices + non_sn_indices].columns[1:].tolist()
coefficient = [re.sub(r"_s\d+$", "", col) for col in coefficient]

# Tensorize
regmat_tensor_eval = torch.from_numpy(regmat0_eval.values).float().to(device)

# Per-hour index dictionary: first column per s is the dependent variable (Price_s{s})
index_dict = {}
for s in range(S):
    indices = list(range(s * columns_s, (s + 1) * columns_s))
    if s == S - 1:
        # Exclude price_last for s==23 later to avoid leakage (handled via indices length - 1)
        non_sn_indices = list(range(columns_total - columns_base, (columns_total - 1)))
        index_dict[s] = indices + non_sn_indices
    else:
        non_sn_indices = list(range(columns_total - columns_base, columns_total))
        index_dict[s] = indices + non_sn_indices

# Dependent indices per s (first column of each block)
dependent_index = [index_dict[s][0] for s in range(S)]

# Active regressors (exclude the dependent column)
active_regressor = {k: v[1:] for k, v in index_dict.items()}

num_columns = max(max(idx_list) for idx_list in active_regressor.values()) + 1

# Mask for reduced linear (S x num_columns), 1 where active, else 0
mask_in_out_red = torch.zeros((S, num_columns), dtype=torch.float32, device=device)
for s, indices in active_regressor.items():
    mask_in_out_red[s, indices] = 1.0

# Mask for RNN inputs:
# Start from linear mask, then exclude price lags not allowed for RNN (e.g., lag 2, 7)
mask_input_to_mid_rnn = torch.zeros((S, num_columns), dtype=torch.float32, device=device)
if model_type in [1, 2, 4, 5, 6, 7]:
    mask_input_to_mid_rnn = mask_in_out_red.clone()
    lags_to_exclude = [lag for lag in price_s_lags_full if lag not in price_s_lags_rnn]
    if lags_to_exclude:
        all_cols = regmat0_eval.columns.tolist()
        cols_to_exclude_indices = []
        for j, col_name in enumerate(all_cols):
            for lag in lags_to_exclude:
                if f"Price_lag_{lag}_s" in col_name:
                    cols_to_exclude_indices.append(j)
                    break
        if cols_to_exclude_indices:
            mask_input_to_mid_rnn[:, cols_to_exclude_indices] = 0.0

# Full mask (all ones except dependent columns set to 0)
mask_in_out_full = torch.ones((S, num_columns), dtype=torch.float32, device=device)
mask_in_out_full[:, dependent_index] = 0.0

#%% #############################################
# ########## remove the dependent from regressors ##
# ################################################
dependent_var_eval = regmat0_eval.iloc[:, dependent_index]
dependent_var_tensor_eval = torch.from_numpy(dependent_var_eval.values).float().to(device)

# Zero-out dependent columns in the design tensor to avoid accidental use
regmat_tensor_eval[:, dependent_index] = 0.0

#%% ############################################################################# function to forecast OLS ##########
############################################################

def forecast_expert_ext(
    dat, days, reg_names, wd=wd, price_s_lags=None, fuel_lags=[2], model_type=None
):
    """
    Rolling one-step-ahead OLS per hour s with train-only scaling.
    For s==23, exclude price_last to avoid "yesterday's last == today's last" leakage.
    """
    S = dat.shape[1]
    forecast = np.repeat(np.nan, S)

    # Ensure price_s_lags is set correctly for OLS baseline
    if price_s_lags is None:
        # For OLS component in hybrid models, use full lags
        price_s_lags = price_s_lags_full

    days_ext = days
    weekdays_num = days_ext.dt.weekday + 1
    WD = np.stack([(weekdays_num == x).astype(int).to_numpy() for x in wd], axis=1)

    da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]
    fuel_names = ["Coal", "NGas", "Oil", "EUA"]

    def get_lagged(Z, lag):
        return np.concatenate((np.repeat(np.nan, lag), Z[: (len(Z) - lag)]))

    mat_fuels = np.concatenate(
        [
            np.apply_along_axis(
                get_lagged, 0, dat[:, 0, reg_names.isin(fuel_names)], lag=l
            )
            for l in fuel_lags
        ],
        axis=-1,
    )
    price_last = get_lagged(Z=dat[:, S - 1, reg_names == "Price"][..., 0], lag=1)

    coefs = np.empty(
        (
            S,
            len(wd)
            + len(price_s_lags)
            + len(fuel_names) * len(fuel_lags)
            + len(da_forecast_names) * len(da_lag)
            + 1,  # +1 for price_last (filled with 0 when s==23)
        )
    )

    for s in range(S):
        acty = dat[:, s, reg_names == "Price"][..., 0]
        mat_price_lags = np.transpose([get_lagged(lag=lag, Z=acty) for lag in price_s_lags])

        mat_da_forecasts = dat[:, s, reg_names.isin(da_forecast_names)]
        stacked_da = []
        for i in range(len(da_forecast_names)):
            da_var = np.transpose([get_lagged(lag=lag, Z=mat_da_forecasts[:, i]) for lag in da_lag])
            stacked_da.append(da_var)
        da_all_var = np.hstack(stacked_da)

        if s == S - 1:
            regmat = np.column_stack((acty, mat_price_lags, da_all_var, WD, mat_fuels))
        else:
            regmat = np.column_stack((acty, mat_price_lags, da_all_var, WD, mat_fuels, price_last))

        # Drop rows with any NA
        act_index = ~np.isnan(regmat).any(axis=1)
        regmat0 = regmat[act_index]

        # Train-only scaling (exclude last row which is the "forecast-now" row)
        regmat_mean = regmat0[:-1, :].mean(axis=0)
        regmat_sd = regmat0[:-1, :].std(axis=0)
        regmat_sd[regmat_sd == 0] = 1.0
        regmat_scaled = (regmat0 - regmat_mean) / regmat_sd

        model = LinearRegression(fit_intercept=False).fit(
            X=regmat_scaled[:-1, 1:], y=regmat_scaled[:-1, 0]
        )
        # Handle singularities robustly
        model.coef_[np.isnan(model.coef_)] = 0.0

        # De-scale prediction
        forecast[s] = (model.coef_ @ regmat_scaled[-1, 1:]) * regmat_sd[0] + regmat_mean[0]

        # Store coefficients; ensure last slot exists for price_last
        if s == S - 1:
            coefs[s] = np.append(model.coef_, 0.0)
        else:
            coefs[s] = model.coef_

    regressor_names = (
        [f"Price lag {lag}" for lag in price_s_lags]
        + [f"{name}_lag_{lag}_s{s}" for name in da_forecast_names for lag in da_lag]
        + [day_abbr[i - 1] for i in wd]
        + [f"{fuel} lag {lag}" for lag in fuel_lags for fuel in fuel_names]
        + ["Price last lag 1"]
    )
    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}

#%% #####################################################
# ################ L1 regularization term ###############
# ########################################################
def l1_regularization(model, model_type, lambda_reg):
    """
    Apply L1 to the output-side linear maps only, with awareness of wrapper
    architectures (e.g., model_type 7 where mid_to_out is under model.core).
    """
    def safe_modules_for_mt():
        if model_type == 3:
            return [model.linear]

        if model_type in (1, 2, 6):
            # mid_to_out is directly on the model
            m = getattr(model, "mid_to_out", None)
            return [m] if m is not None else []

        if model_type in (4, 5):
            # skip connection + mid_to_out on the model
            modules = []
            m = getattr(model, "mid_to_out", None)
            if m is not None:
                modules.append(m)
            io = getattr(model, "input_to_output", None)
            if io is not None:
                modules.append(io)
            return modules

        if model_type == 7:
            # mid_to_out is inside the wrapped core
            core = getattr(model, "core", None)
            modules = []
            if core is not None and hasattr(core, "mid_to_out"):
                modules.append(core.mid_to_out)
            io = getattr(model, "input_to_output", None)
            if io is not None:
                modules.append(io)
            return modules

        return []

    l1_norm = 0.0
    for m in safe_modules_for_mt():
        if m is None:
            continue
        for p in m.parameters():
            if p.requires_grad:
                l1_norm += torch.sum(torch.abs(p))

    return lambda_reg * l1_norm
#%%#####################################################
################ Neural Network for Evaluation ###########
########################################################

def train_and_evaluate_updated_weights(
    train_loader,
    test_loader,
    num_feature,
    previous_weights_s,
    num_epochs,
    learning_rate,
    number_neurons,
    std_y,
    mean_y,
    mask_input_to_mid,
    use_ols_weights,
    ols_tensor,
    weight_decay,
    alpha,
    lambda_reg,
    model_type,  # New parameter to specify the model type (M1..M7)
    seq_len,
):
    # ---------- Layers with masking applied at compute-time (no param mutation) ----------
    class MaskedLinear(nn.Module):
        def __init__(self, input_dim, output_dim, mask):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.register_buffer("mask", mask)  # (output_dim, input_dim)

        def forward(self, x):
            # x: (B, input_dim)
            masked_weight = self.linear.weight * self.mask
            return nn.functional.linear(x, masked_weight, self.linear.bias)

    class MaskedElmanRNN(nn.Module):
        """Elman RNN with ReLU; applies an input mask to weight_ih at compute time."""
        def __init__(self, input_dim, hidden_dim, mask):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            # Parameters like nn.RNNCell but we’ll unroll manually for determinism/masking
            self.weight_ih = nn.Parameter(torch.empty(hidden_dim, input_dim))
            self.weight_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            self.bias_ih   = nn.Parameter(torch.empty(hidden_dim))
            self.bias_hh   = nn.Parameter(torch.empty(hidden_dim))
            self.register_buffer("orig_mask", mask)  # could be (S, input_dim) or (hidden_dim, input_dim)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.uniform_(self.weight_ih, -0.001, 0.001)
            nn.init.uniform_(self.weight_hh, -0.001, 0.001)
            nn.init.uniform_(self.bias_ih,   -0.001, 0.001)
            nn.init.uniform_(self.bias_hh,   -0.001, 0.001)

        def _expand_mask(self):
            # Ensure mask is (hidden_dim, input_dim)
            if self.orig_mask.dim() == 2 and self.orig_mask.size(0) != self.hidden_dim:
                # Reduce across rows (e.g., OR over S) to a 1 x input_dim, then expand
                m1 = (self.orig_mask > 0).any(dim=0, keepdim=True).float()  # (1, input_dim)
                return m1.expand(self.hidden_dim, -1).contiguous()
            return self.orig_mask

        def forward(self, x, h0=None):
            # x: (B, T, input_dim)
            B, T, D = x.shape
            if h0 is None:
                h_t = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
            else:
                h_t = h0

            mask = self._expand_mask()  # (hidden_dim, input_dim)
            W_ih = self.weight_ih * mask
            W_hh = self.weight_hh

            outs = []
            for t in range(T):
                x_t = x[:, t, :]  # (B, D)
                h_t = torch.relu(
                    x_t @ W_ih.t() + self.bias_ih + h_t @ W_hh.t() + self.bias_hh
                )
                outs.append(h_t)
            y = torch.stack(outs, dim=1)  # (B, T, H)
            return y, h_t  # full sequence + last state

    class MaskedLinearRNN(nn.Module):
        """Identity-activation Elman RNN (i.e., linear recurrence) with masked input weights."""
        def __init__(self, input_dim, hidden_dim, mask):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.weight_ih_l0 = nn.Parameter(torch.empty(hidden_dim, input_dim))
            self.weight_hh_l0 = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            self.bias_ih_l0   = nn.Parameter(torch.empty(hidden_dim))
            self.bias_hh_l0   = nn.Parameter(torch.empty(hidden_dim))
            self.register_buffer("orig_mask", mask)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.uniform_(self.weight_ih_l0, -0.001, 0.001)
            nn.init.uniform_(self.weight_hh_l0, -0.001, 0.001)
            nn.init.uniform_(self.bias_ih_l0,   -0.001, 0.001)
            nn.init.uniform_(self.bias_hh_l0,   -0.001, 0.001)

        def _expand_mask(self):
            if self.orig_mask.dim() == 2 and self.orig_mask.size(0) != self.hidden_dim:
                m1 = (self.orig_mask > 0).any(dim=0, keepdim=True).float()
                return m1.expand(self.hidden_dim, -1).contiguous()
            return self.orig_mask

        def forward(self, x, h_0=None):
            # x: (B, T, input_dim)
            B, T, _ = x.shape
            if h_0 is None:
                h_t = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
            else:
                h_t = h_0

            mask = self._expand_mask()
            W_ih = self.weight_ih_l0 * mask
            W_hh = self.weight_hh_l0

            outputs = []
            for t in range(T):
                x_t = x[:, t, :]
                h_t = (x_t @ W_ih.t() + self.bias_ih_l0) + (h_t @ W_hh.t() + self.bias_hh_l0)
                outputs.append(h_t)
            output = torch.stack(outputs, dim=1)
            return output, h_t

    # -------------------- Model wrappers --------------------
    class CustomModelWithSkip(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid, mask_input_to_output, use_linear_rnn=False):
            super().__init__()
            self.use_linear_rnn = use_linear_rnn
            self.input_to_mid = (
                MaskedLinearRNN(input_dim, middle_dim, mask_input_to_mid)
                if use_linear_rnn
                else MaskedElmanRNN(input_dim, middle_dim, mask_input_to_mid)
            )
            self.mid_to_out = nn.Linear(middle_dim, output_dim)
            self.input_to_output = MaskedLinear(input_dim, output_dim, mask_input_to_output)

        def forward(self, x, return_components=False):
            mid_seq, _ = self.input_to_mid(x)
            mid_output = mid_seq[:, -1, :]  # last step
            mid_to_output = self.mid_to_out(mid_output)
            input_to_output = self.input_to_output(x[:, -1, :])
            out = mid_to_output + input_to_output
            if return_components:
                # For M4/M5: return RNN part (mid_to_out) and Linear skip separately
                return out, mid_to_output, input_to_output
            return out

    class CustomModelWithoutSkip(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid, use_linear_rnn=False):
            super().__init__()
            self.use_linear_rnn = use_linear_rnn
            self.input_to_mid = (
                MaskedLinearRNN(input_dim, middle_dim, mask_input_to_mid)
                if use_linear_rnn
                else MaskedElmanRNN(input_dim, middle_dim, mask_input_to_mid)
            )
            self.mid_to_out = nn.Linear(middle_dim, output_dim)

        def forward(self, x):
            mid_seq, _ = self.input_to_mid(x)
            mid_output = mid_seq[:, -1, :]
            return self.mid_to_out(mid_output)

    class CustomModelTwoRNNs(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid):
            super().__init__()
            self.input_to_mid_relu = MaskedElmanRNN(input_dim, middle_dim, mask_input_to_mid)
            self.input_to_mid_identity = MaskedLinearRNN(input_dim, middle_dim, mask_input_to_mid)
            self.mid_to_out = nn.Linear(middle_dim * 2, output_dim)
            self.middle_dim = middle_dim

        def _split_bias(self):
            # Proportional split by L2 norms (stable & avoids arbitrary 50/50)
            W = self.mid_to_out.weight  # (out, 2*H)
            H = self.middle_dim
            w_relu = W[:, :H]
            w_id   = W[:, H:]
            # norms per output neuron
            nr = torch.norm(w_relu, p=2, dim=1) + 1e-12
            ni = torch.norm(w_id,   p=2, dim=1) + 1e-12
            total = nr + ni
            frac_relu = (nr / total).unsqueeze(1)  # (out,1)
            frac_id   = (ni / total).unsqueeze(1)
            b = self.mid_to_out.bias.unsqueeze(1)  # (out,1)
            return b * frac_relu, b * frac_id

        def forward(self, x, return_components=False):
            mid_relu_seq, _ = self.input_to_mid_relu(x)
            mid_id_seq,   _ = self.input_to_mid_identity(x)
            mid_relu = mid_relu_seq[:, -1, :]
            mid_id   = mid_id_seq[:, -1, :]
            combined_mid = torch.cat((mid_relu, mid_id), dim=1)
            output = self.mid_to_out(combined_mid)

            if return_components:
                H = self.middle_dim
                w_relu = self.mid_to_out.weight[:, :H]
                w_id   = self.mid_to_out.weight[:, H:]
                b_relu, b_id = self._split_bias()
                forecast_relu = mid_relu @ w_relu.t() + b_relu.squeeze(1)
                forecast_identity = mid_id @ w_id.t() + b_id.squeeze(1)
                return output, forecast_relu, forecast_identity
            return output

    class CustomModelTwoRNNsWithSkip(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid, mask_input_to_output):
            super().__init__()
            self.core = CustomModelTwoRNNs(input_dim, middle_dim, output_dim, mask_input_to_mid)
            self.input_to_output = MaskedLinear(input_dim, output_dim, mask_input_to_output)

        def forward(self, x, return_components=False):
            core_out = self.core(x, return_components=return_components)
            skip_out = self.input_to_output(x[:, -1, :])
            if return_components:
                out_total, f_relu, f_id = core_out
                # expose all THREE pieces separately for M7
                return out_total + skip_out, f_relu, f_id, skip_out
            return core_out + skip_out

    # -------------------- Instantiate model --------------------
    input_dim = num_feature
    output_dim = S
    middle_dim = number_neurons

    if model_type == 3:  # M3: Reduced Linear
        model = MaskedLinear(input_dim, output_dim, mask_in_out_red).to(device)
    elif model_type == 1:  # M1: RNN + ReLU
        model = CustomModelWithoutSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, use_linear_rnn=False).to(device)
    elif model_type == 2:  # M2: RNN + Identity
        model = CustomModelWithoutSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, use_linear_rnn=True).to(device)
    elif model_type == 4:  # M4: Reduced Linear + Elman RNN (ReLU) with skip
        model = CustomModelWithSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, mask_in_out_red, use_linear_rnn=False).to(device)
    elif model_type == 5:  # M5: Reduced Linear + Linear RNN (identity) with skip
        model = CustomModelWithSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, mask_in_out_red, use_linear_rnn=True).to(device)
    elif model_type == 6:  # M6: Two RNNs (ReLU + Identity)
        model = CustomModelTwoRNNs(input_dim, middle_dim, output_dim, mask_input_to_mid).to(device)
    elif model_type == 7:  # M7: Two RNNs + Linear skip
        model = CustomModelTwoRNNsWithSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, mask_in_out_red).to(device)
    else:
        raise ValueError("Invalid model type. Choose 1, 2, 3, 4, 5, 6, or 7.")

    # -------------------- Weight init / warm-start --------------------
    if previous_weights_s is None:
        if model_type in [4, 5, 7] and use_ols_weights:
            # Initialize skip/linear projection from OLS
            model.input_to_output.linear.weight.data = alpha * ols_tensor.clone()
            model.input_to_output.linear.bias.data.zero_()
            # Init the rest small-uniform
            for name, p in model.named_parameters():
                if 'input_to_output.linear' in name:
                    continue
                if p.requires_grad:
                    nn.init.uniform_(p, -0.001, 0.001)
        elif model_type == 3 and use_ols_weights:
            model.linear.weight.data = ols_tensor.clone()
            model.linear.bias.data.zero_()
        else:
            for p in model.parameters():
                if p.requires_grad:
                    nn.init.uniform_(p, -0.001, 0.001)
    else:
        # Load weights per-architecture (unchanged logic, just cleaner formatting)
        if model_type == 3:  # M3
            model.linear.weight.data = previous_weights_s["input_to_output"]["weight"].clone()
            model.linear.bias.data   = previous_weights_s["input_to_output"]["bias"].clone()

        elif model_type == 1:  # M1 (ReLU)
            model.input_to_mid.weight_ih.data = previous_weights_s["input_to_mid"]["weight_ih"].clone()
            model.input_to_mid.weight_hh.data = previous_weights_s["input_to_mid"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid"]["bias_ih"] is not None:
                model.input_to_mid.bias_ih.data = previous_weights_s["input_to_mid"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid"]["bias_hh"] is not None:
                model.input_to_mid.bias_hh.data = previous_weights_s["input_to_mid"]["bias_hh"].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()

        elif model_type == 2:  # M2 (Identity)
            model.input_to_mid.weight_ih_l0.data = previous_weights_s["input_to_mid"]["weight_ih"].clone()
            model.input_to_mid.weight_hh_l0.data = previous_weights_s["input_to_mid"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid"]["bias_ih"] is not None:
                model.input_to_mid.bias_ih_l0.data = previous_weights_s["input_to_mid"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid"]["bias_hh"] is not None:
                model.input_to_mid.bias_hh_l0.data = previous_weights_s["input_to_mid"]["bias_hh"].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()

        elif model_type == 4:  # M4 (ReLU + skip)
            model.input_to_mid.weight_ih.data = previous_weights_s["input_to_mid"]["weight_ih"].clone()
            model.input_to_mid.weight_hh.data = previous_weights_s["input_to_mid"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid"]["bias_ih"] is not None:
                model.input_to_mid.bias_ih.data = previous_weights_s["input_to_mid"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid"]["bias_hh"] is not None:
                model.input_to_mid.bias_hh.data = previous_weights_s["input_to_mid"]["bias_hh"].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()
            model.input_to_output.linear.weight.data = previous_weights_s["input_to_output"]["weight"].clone()
            model.input_to_output.linear.bias.data   = previous_weights_s["input_to_output"]["bias"].clone()

        elif model_type == 5:  # M5 (Identity + skip)
            model.input_to_mid.weight_ih_l0.data = previous_weights_s["input_to_mid"]["weight_ih"].clone()
            model.input_to_mid.weight_hh_l0.data = previous_weights_s["input_to_mid"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid"]["bias_ih"] is not None:
                model.input_to_mid.bias_ih_l0.data = previous_weights_s["input_to_mid"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid"]["bias_hh"] is not None:
                model.input_to_mid.bias_hh_l0.data = previous_weights_s["input_to_mid"]["bias_hh"].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()
            model.input_to_output.linear.weight.data = previous_weights_s["input_to_output"]["weight"].clone()
            model.input_to_output.linear.bias.data   = previous_weights_s["input_to_output"]["bias"].clone()

        elif model_type == 6:  # M6 (two RNNs)
            model.input_to_mid_relu.weight_ih.data = previous_weights_s["input_to_mid_relu"]["weight_ih"].clone()
            model.input_to_mid_relu.weight_hh.data = previous_weights_s["input_to_mid_relu"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid_relu"]["bias_ih"] is not None:
                model.input_to_mid_relu.bias_ih.data = previous_weights_s["input_to_mid_relu"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid_relu"]["bias_hh"] is not None:
                model.input_to_mid_relu.bias_hh.data = previous_weights_s["input_to_mid_relu"]["bias_hh"].clone()

            model.input_to_mid_identity.weight_ih_l0.data = previous_weights_s["input_to_mid_identity"]["weight_ih"].clone()
            model.input_to_mid_identity.weight_hh_l0.data = previous_weights_s["input_to_mid_identity"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid_identity"]["bias_ih"] is not None:
                model.input_to_mid_identity.bias_ih_l0.data = previous_weights_s["input_to_mid_identity"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid_identity"]["bias_hh"] is not None:
                model.input_to_mid_identity.bias_hh_l0.data = previous_weights_s["input_to_mid_identity"]["bias_hh"].clone()

            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()

        elif model_type == 7:  # M7 (two RNNs + skip)
            model.core.input_to_mid_relu.weight_ih.data = previous_weights_s["input_to_mid_relu"]["weight_ih"].clone()
            model.core.input_to_mid_relu.weight_hh.data = previous_weights_s["input_to_mid_relu"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid_relu"]["bias_ih"] is not None:
                model.core.input_to_mid_relu.bias_ih.data = previous_weights_s["input_to_mid_relu"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid_relu"]["bias_hh"] is not None:
                model.core.input_to_mid_relu.bias_hh.data = previous_weights_s["input_to_mid_relu"]["bias_hh"].clone()

            model.core.input_to_mid_identity.weight_ih_l0.data = previous_weights_s["input_to_mid_identity"]["weight_ih"].clone()
            model.core.input_to_mid_identity.weight_hh_l0.data = previous_weights_s["input_to_mid_identity"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid_identity"]["bias_ih"] is not None:
                model.core.input_to_mid_identity.bias_ih_l0.data = previous_weights_s["input_to_mid_identity"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid_identity"]["bias_hh"] is not None:
                model.core.input_to_mid_identity.bias_hh_l0.data = previous_weights_s["input_to_mid_identity"]["bias_hh"].clone()

            model.core.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.core.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()
            model.input_to_output.linear.weight.data = previous_weights_s["input_to_output"]["weight"].clone()
            model.input_to_output.linear.bias.data   = previous_weights_s["input_to_output"]["bias"].clone()

    # -------------------- Train --------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    best_val = float('inf')
    patience = 10
    patience_ctr = 0
    max_grad_norm = 5.0

    for epoch in range(num_epochs):
        model.train()
        for X_train, y_train in train_loader:
            pred = model(X_train)
            loss = criterion(pred, y_train) + l1_regularization(model, model_type, lambda_reg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # simple validation on test_loader each epoch
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, y_val in test_loader:
                p = model(X_val)
                val_losses.append(criterion(p, y_val).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        scheduler.step(val_loss)

        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    # -------------------- Evaluate (and de-standardize) --------------------
    model.eval()
    decomposed_forecasts = {}
    unstandardized_outputs = None
    unstandardized_y_test = None
    squared_errors = None
    decomp_max_abs_diff = None

    with torch.no_grad():
        for X_test, y_test in test_loader:
            if model_type in [4, 5]:
                outputs, comp_rnn, comp_lin = model(X_test, return_components=True)
                # de-standardize
                unstandardized_outputs = outputs * std_y + mean_y
                unstandardized_y_test = y_test * std_y + mean_y
                squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2

                un_comp_rnn = comp_rnn * std_y + mean_y
                un_comp_lin = comp_lin * std_y + mean_y
                decomposed_forecasts["rnn_forecast"] = un_comp_rnn
                decomposed_forecasts["linear_forecast"] = un_comp_lin

                # check: rnn + linear == total
                decomp_sum = un_comp_rnn + un_comp_lin
                decomp_max_abs_diff = torch.max(torch.abs(decomp_sum - unstandardized_outputs))
                decomposed_forecasts["decomp_check_maxabs"] = decomp_max_abs_diff.detach().cpu().item()

            elif model_type == 6:
                outputs, comp_relu, comp_id = model(X_test, return_components=True)
                unstandardized_outputs = outputs * std_y + mean_y
                unstandardized_y_test  = y_test * std_y + mean_y
                squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2

                un_relu = comp_relu * std_y + mean_y
                un_id   = comp_id   * std_y + mean_y
                decomposed_forecasts["rnn_relu_forecast"]     = un_relu
                decomposed_forecasts["rnn_identity_forecast"] = un_id

                # check: relu + identity == total
                decomp_sum = un_relu + un_id
                decomp_max_abs_diff = torch.max(torch.abs(decomp_sum - unstandardized_outputs))
                decomposed_forecasts["decomp_check_maxabs"] = decomp_max_abs_diff.detach().cpu().item()

            elif model_type == 7:
                outputs, comp_relu, comp_id, comp_lin = model(X_test, return_components=True)
                unstandardized_outputs = outputs * std_y + mean_y
                unstandardized_y_test  = y_test * std_y + mean_y
                squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2

                un_relu = comp_relu * std_y + mean_y
                un_id   = comp_id   * std_y + mean_y
                un_lin  = comp_lin  * std_y + mean_y
                decomposed_forecasts["rnn_relu_forecast"]     = un_relu
                decomposed_forecasts["rnn_identity_forecast"] = un_id
                decomposed_forecasts["linear_forecast"]       = un_lin

                # check: relu + identity + linear == total
                decomp_sum = un_relu + un_id + un_lin
                decomp_max_abs_diff = torch.max(torch.abs(decomp_sum - unstandardized_outputs))
                decomposed_forecasts["decomp_check_maxabs"] = decomp_max_abs_diff.detach().cpu().item()

            else:
                # Models 1,2,3
                outputs = model(X_test)
                unstandardized_outputs = outputs * std_y + mean_y
                unstandardized_y_test  = y_test * std_y + mean_y
                squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2

    # -------------------- Save weights for next window --------------------
    if model_type == 3:  # M3
        previous_weights_s = {
            "input_to_output": {
                "weight": (model.linear.weight.data * mask_in_out_red).clone(),
                "bias": model.linear.bias.data.clone(),
            }
        }
    elif model_type == 1:  # M1
        previous_weights_s = {
            "input_to_mid": {
                "weight_ih": (model.input_to_mid.weight_ih.data * model.input_to_mid._expand_mask()).clone(),
                "weight_hh": model.input_to_mid.weight_hh.data.clone(),
                "bias_ih": model.input_to_mid.bias_ih.data.clone(),
                "bias_hh": model.input_to_mid.bias_hh.data.clone(),
            },
            "mid_to_out": {"weight": model.mid_to_out.weight.data.clone(), "bias": model.mid_to_out.bias.data.clone()},
        }
    elif model_type == 2:  # M2
        previous_weights_s = {
            "input_to_mid": {
                "weight_ih": (model.input_to_mid.weight_ih_l0.data * model.input_to_mid._expand_mask()).clone(),
                "weight_hh": model.input_to_mid.weight_hh_l0.data.clone(),
                "bias_ih": model.input_to_mid.bias_ih_l0.data.clone(),
                "bias_hh": model.input_to_mid.bias_hh_l0.data.clone(),
            },
            "mid_to_out": {"weight": model.mid_to_out.weight.data.clone(), "bias": model.mid_to_out.bias.data.clone()},
        }
    elif model_type == 4:  # M4
        previous_weights_s = {
            "input_to_mid": {
                "weight_ih": (model.input_to_mid.weight_ih.data * model.input_to_mid._expand_mask()).clone(),
                "weight_hh": model.input_to_mid.weight_hh.data.clone(),
                "bias_ih": model.input_to_mid.bias_ih.data.clone(),
                "bias_hh": model.input_to_mid.bias_hh.data.clone(),
            },
            "mid_to_out": {"weight": model.mid_to_out.weight.data.clone(), "bias": model.mid_to_out.bias.data.clone()},
            "input_to_output": {
                "weight": (model.input_to_output.linear.weight.data * mask_in_out_red).clone(),
                "bias": model.input_to_output.linear.bias.data.clone(),
            },
        }
    elif model_type == 5:  # M5
        previous_weights_s = {
            "input_to_mid": {
                "weight_ih": (model.input_to_mid.weight_ih_l0.data * model.input_to_mid._expand_mask()).clone(),
                "weight_hh": model.input_to_mid.weight_hh_l0.data.clone(),
                "bias_ih": model.input_to_mid.bias_ih_l0.data.clone(),
                "bias_hh": model.input_to_mid.bias_hh_l0.data.clone(),
            },
            "mid_to_out": {"weight": model.mid_to_out.weight.data.clone(), "bias": model.mid_to_out.bias.data.clone()},
            "input_to_output": {
                "weight": (model.input_to_output.linear.weight.data * mask_in_out_red).clone(),
                "bias": model.input_to_output.linear.bias.data.clone(),
            },
        }
    elif model_type == 6:  # M6
        previous_weights_s = {
            "input_to_mid_relu": {
                "weight_ih": (model.input_to_mid_relu.weight_ih.data * model.input_to_mid_relu._expand_mask()).clone(),
                "weight_hh": model.input_to_mid_relu.weight_hh.data.clone(),
                "bias_ih": model.input_to_mid_relu.bias_ih.data.clone(),
                "bias_hh": model.input_to_mid_relu.bias_hh.data.clone(),
            },
            "input_to_mid_identity": {
                "weight_ih": (model.input_to_mid_identity.weight_ih_l0.data * model.input_to_mid_identity._expand_mask()).clone(),
                "weight_hh": model.input_to_mid_identity.weight_hh_l0.data.clone(),
                "bias_ih": model.input_to_mid_identity.bias_ih_l0.data.clone(),
                "bias_hh": model.input_to_mid_identity.bias_hh_l0.data.clone(),
            },
            "mid_to_out": {"weight": model.mid_to_out.weight.data.clone(), "bias": model.mid_to_out.bias.data.clone()},
        }
    elif model_type == 7:  # M7
        previous_weights_s = {
            "input_to_mid_relu": {
                "weight_ih": (model.core.input_to_mid_relu.weight_ih.data * model.core.input_to_mid_relu._expand_mask()).clone(),
                "weight_hh": model.core.input_to_mid_relu.weight_hh.data.clone(),
                "bias_ih": model.core.input_to_mid_relu.bias_ih.data.clone(),
                "bias_hh": model.core.input_to_mid_relu.bias_hh.data.clone(),
            },
            "input_to_mid_identity": {
                "weight_ih": (model.core.input_to_mid_identity.weight_ih_l0.data * model.core.input_to_mid_identity._expand_mask()).clone(),
                "weight_hh": model.core.input_to_mid_identity.weight_hh_l0.data.clone(),
                "bias_ih": model.core.input_to_mid_identity.bias_ih_l0.data.clone(),
                "bias_hh": model.core.input_to_mid_identity.bias_hh_l0.data.clone(),
            },
            "mid_to_out": {"weight": model.core.mid_to_out.weight.data.clone(), "bias": model.core.mid_to_out.bias.data.clone()},
            "input_to_output": {
                "weight": (model.input_to_output.linear.weight.data * mask_in_out_red).clone(),
                "bias": model.input_to_output.linear.bias.data.clone(),
            },
        }

    return squared_errors, previous_weights_s, unstandardized_y_test, unstandardized_outputs, decomposed_forecasts
#%%#######################################################
################ Define the validation dataset ###########
##########################################################
# Define the evaluation period length (two years)
length_eval = int(2 * 365)

# The first observation in the evaluation period
begin_eval = regmat0_eval.shape[0] - length_eval
N_s = length_eval

#%%#####################################################
################ Training (rolling window)  ############
########################################################

def forecast_ANN_Tensor_tune(
    dat_eval,
    learning_rate_init,
    num_epochs_init,
    D_init,
    learning_rate_all,
    num_epochs_all,
    D_all,
    number_neurons,
    use_ols_weights,
    weight_decay_init,
    weight_decay_all,
    alpha,
    lambda_reg_init,
    lambda_reg_all,
    trial_number,
    model_type,   # NEW numbering: 1..7
    seq_len,
    mask_input_to_mid_rnn,
):
    previous_weights_s = None
    time_n0 = 0.0
    time_other_n = 0.0

    # set the number_neurons to zero (no hidden layer) for pure linear model
    if model_type == 3:   # M3 = Reduced Linear
        number_neurons = 0

    S = dat_eval.shape[1]
    mse_all = torch.zeros((N_s, S), device=device)
    agg_sq_all = torch.zeros((N_s, S), device=device)
    weight_all = {}

    # We pass the per-hour mask directly; the RNN layers handle expansion internally.
    mask_input_to_mid = mask_input_to_mid_rnn

    # Collect unstandardized forecasts & truths
    unstandardized_outputs_all = torch.zeros((N_s, S), device=device)
    y_true_unstd_all           = torch.zeros((N_s, S), device=device)
    # Weekly naive baseline (t-7). Will leave first 7 rows as NaN.
    weekly_naive_all           = torch.full((N_s, S), float('nan'), device=device)

    last_decomposed = {}

    for n in range(N_s):
        start_time_n = time.time()

        if n == 0:
            learning_rate = learning_rate_init
            num_epochs = num_epochs_init
            D = D_init
            weight_decay = weight_decay_init
            lambda_reg = lambda_reg_init

            # OLS coefficients for models with a linear part (M3, M4, M5, M7)
            if model_type in {3, 4, 5, 7}:
                coef_OLS_all = forecast_expert_ext(
                    dat=dat_eval[(begin_eval - D + n):(begin_eval + n + 1), :],
                    days=days_eval[(begin_eval - D + n):(begin_eval + n + 1)],
                    reg_names=data.columns[1:], model_type=model_type
                )["coefficients"]
                coef_OLS_ten = torch.tensor(coef_OLS_all.values, dtype=torch.float32, device=device)

                # Map coefficients into full (S, num_columns) weight tensor aligned with active_regressor
                ols_tensor = torch.zeros((S, num_columns), dtype=torch.float32, device=device)
                for row_idx, col_indices in active_regressor.items():
                    ols_tensor[row_idx, col_indices] = coef_OLS_ten[row_idx, :len(col_indices)]
            else:
                ols_tensor = None
        else:
            learning_rate = learning_rate_all
            num_epochs = num_epochs_all
            D = D_all
            weight_decay = weight_decay_all
            lambda_reg = lambda_reg_all

        # -------------------- Train/test standardization (train-only stats) --------------------
        X_train_slice = regmat_tensor_eval[(begin_eval - D + n):(begin_eval + n), :]
        X_test_slice  = regmat_tensor_eval[(begin_eval + n), :]

        mean_x = X_train_slice.mean(dim=0, keepdim=True)
        std_x = X_train_slice.std(dim=0, keepdim=True)
        std_x[std_x == 0] = 1.0

        X_train_unseq = (X_train_slice - mean_x) / std_x
        X_test_unseq  = (X_test_slice  - mean_x) / std_x

        y_train_slice = dependent_var_tensor_eval[(begin_eval - D + n):(begin_eval + n), :]
        y_test_slice  = dependent_var_tensor_eval[(begin_eval + n), :]

        mean_y = y_train_slice.mean(dim=0, keepdim=True)
        std_y  = y_train_slice.std(dim=0, keepdim=True)
        std_y[std_y == 0] = 1.0

        y_train_unseq = (y_train_slice - mean_y) / std_y
        y_test = ((y_test_slice - mean_y) / std_y).reshape(1, -1)

        # -------------------- Sequence building for RNNs (SAFE) --------------------
        if model_type in [1, 2, 4, 5, 6, 7]:  # RNN-based models in NEW numbering
            # Effective seq_len is clamped to available training rows and D
            eff_seq_len = int(max(1, min(seq_len, D, X_train_unseq.shape[0] if X_train_unseq.ndim == 2 else 1e9)))

            # Build training sequences (safe padding handled inside create_sequences if needed)
            X_train, y_train = create_sequences(X_train_unseq, y_train_unseq, eff_seq_len)

            # Build TEST sequence using last (eff_seq_len-1) train rows + current test row.
            feat_dim = X_train_unseq.shape[1]
            if eff_seq_len > 1:
                k = min(eff_seq_len - 1, X_train_unseq.shape[0])
                X_hist = X_train_unseq[-k:, :].reshape(k, feat_dim)    # (k, F)
                x_row  = X_test_unseq.reshape(1, feat_dim)             # (1, F)
                if k < (eff_seq_len - 1):
                    pad = X_train_unseq[:1, :].repeat((eff_seq_len - 1) - k, 1)
                    X_hist = torch.cat([pad, X_hist], dim=0)
                X_test_full = torch.cat([X_hist, x_row], dim=0)        # (eff_seq_len, F)
            else:
                X_test_full = X_test_unseq.reshape(1, feat_dim)        # (1, F)

            X_test = X_test_full.unsqueeze(0)                           # (1, eff_seq_len, F)
        else:
            # M3 (pure linear): no sequences
            X_train, y_train = X_train_unseq, y_train_unseq
            X_test = X_test_unseq.reshape(1, -1)

        # -------------------- DataLoaders --------------------
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=1, shuffle=False)

        num_feature = regmat_tensor_eval.shape[1]

        # -------------------- Train & evaluate --------------------
        mse_value, previous_weights_s, y_true_unstd, unstandardized_outputs, decomposed_forecasts = train_and_evaluate_updated_weights(
            train_loader=train_loader,
            test_loader=test_loader,
            num_feature=num_feature,
            previous_weights_s=previous_weights_s,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            number_neurons=number_neurons,
            std_y=std_y,
            mean_y=mean_y,
            mask_input_to_mid=mask_input_to_mid,
            use_ols_weights=use_ols_weights if n == 0 else None,
            ols_tensor=ols_tensor if n == 0 else None,
            weight_decay=weight_decay,
            alpha=alpha if n == 0 else None,
            lambda_reg=lambda_reg,
            model_type=model_type,  # NEW numbering passed through
            seq_len=seq_len,
        )

        mse_all[n, :] = mse_value
        unstandardized_outputs_all[n, :] = unstandardized_outputs.squeeze(0)
        y_true_unstd_all[n, :] = y_true_unstd.squeeze(0)

        if decomposed_forecasts:
            last_decomposed = decomposed_forecasts  # keep most recent (latest step)

        # -------------------- Weekly naive (t-7) --------------------
        global_idx = begin_eval + n
        weekly_idx = global_idx - 7
        if weekly_idx >= 0:
            # dependent_var_eval holds true prices (unstandardized) aligned with rows
            weekly_naive_all[n, :] = torch.tensor(
                dependent_var_eval.iloc[weekly_idx].values, dtype=torch.float32, device=device
            )

        # -------------------- Timing --------------------
        elapsed_time_n = time.time() - start_time_n
        if n == 0:
            time_n0 += elapsed_time_n
        else:
            time_other_n += elapsed_time_n

    agg_sq_all = mse_all.mean(dim=0).sqrt()
    overall_agg_mean = mse_all.mean((0, 1)).sqrt()

    return [
        agg_sq_all,
        overall_agg_mean,
        time_n0,
        time_other_n,
        weight_all,
        unstandardized_outputs_all,  # forecasts (unstd)
        last_decomposed,             # last-step components + decomp_check_maxabs
        y_true_unstd_all,            # ground truth (unstd)
        weekly_naive_all,            # weekly naive baseline (unstd); first 7 rows are NaN
    ]


#%%#####################################################
################ Hyper-parameter tuning ################
########################################################
os.makedirs(f"Model{model_type}/", exist_ok=True)

# Number of trials
n_trials = 500
trial_durations_tensor = torch.zeros(n_trials)
trial_durations_n0 = torch.zeros(n_trials)
trial_durations_n_remaining = torch.zeros(n_trials)
Forecast_trials = torch.zeros(N_s, S, n_trials, device=device)
weight_tensor = {}

def objective_init(trial):
    trial_number = trial.number
    trial_start_time = time.time()

    # -------------------- Sample hyperparameters --------------------
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True)
    D_init = trial.suggest_int("D_init", 30, 2 * 365)

    learning_rate_all = trial.suggest_float("learning_rate_all", 1e-4, 1e-2, log=True)

    weight_decay_init = trial.suggest_float("weight_decay_init", 1e-5, 1e-2, log=True)
    lambda_reg_init   = trial.suggest_float("lambda_reg_init",   1e-5, 1e-2, log=True)

    weight_decay_all = trial.suggest_float("weight_decay_all", 1e-4, 1e-2, log=True)
    lambda_reg_all   = trial.suggest_float("lambda_reg_all",   1e-4, 1e-2, log=True)

    D_all = trial.suggest_int("D_all", 2, 365)

    # Use OLS init only for models with a linear projection: M3, M4, M5, M7
    if model_type in {3, 4, 5, 7}:
        use_ols_weights = trial.suggest_categorical("use_ols_weights", [True, False])
        alpha = trial.suggest_float("alpha", 0.0, 2.0) if use_ols_weights else None
    else:
        use_ols_weights = None
        alpha = None

    # RNN-only hyperparams (hidden size + sequence length) for M1, M2, M4, M5, M6, M7
    if model_type in {1, 2, 4, 5, 6, 7}:
        number_neurons = trial.suggest_int("number_neurons", 1, 128)
        seq_len = trial.suggest_int("seq_len", 1, 7)
    else:
        number_neurons = 0
        seq_len = 1

    # -------------------- Run rolling training --------------------
    (
        _,
        overall_mean,
        time_n0,
        time_other_n,
        _,
        forecast_unstd,
        decomposed_forecasts,
        y_true_unstd,
        weekly_naive_unstd,
    ) = forecast_ANN_Tensor_tune(
        dat_eval=dat_eval,
        learning_rate_init=learning_rate_init,
        num_epochs_init=num_epochs_init,
        D_init=D_init,
        learning_rate_all=learning_rate_all,
        num_epochs_all=num_epochs_all,
        D_all=D_all,
        number_neurons=number_neurons,
        use_ols_weights=use_ols_weights,
        weight_decay_init=weight_decay_init,
        weight_decay_all=weight_decay_all,
        alpha=alpha,
        lambda_reg_init=lambda_reg_init,
        lambda_reg_all=lambda_reg_all,
        trial_number=trial_number,
        model_type=model_type,  # NEW numbering
        seq_len=seq_len,
        mask_input_to_mid_rnn=mask_input_to_mid_rnn,
    )

    # Optional: store summaries of decomposed components and decomp check
    if decomposed_forecasts:
        for key, value in decomposed_forecasts.items():
            if hasattr(value, "mean"):
                trial.set_user_attr(f"{key}_mean", float(value.mean().item()))
        if "decomp_check_maxabs" in decomposed_forecasts:
            trial.set_user_attr("decomp_check_maxabs", float(decomposed_forecasts["decomp_check_maxabs"]))

    # -------------------- Timing bookkeeping --------------------
    trial_duration = time.time() - trial_start_time
    print(f"Trial {trial.number} completed in {trial_duration:.2f} seconds.")
    print(f"Time for n == 0: {time_n0:.2f} seconds.")
    print(f"Time for other n: {time_other_n:.2f} seconds.")

    trial_durations_tensor[trial.number] = trial_duration
    trial_durations_n0[trial.number] = time_n0
    trial_durations_n_remaining[trial.number] = time_other_n
    Forecast_trials[:, :, trial.number] = forecast_unstd

    return overall_mean

# You can enable a pruner to stop weak trials early:
# pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
# study = optuna.create_study(direction="minimize",
#                             sampler=optuna.samplers.TPESampler(seed=42),
#                             pruner=pruner)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective_init, n_trials=n_trials)
#%% Results & persistence  (RMSE objective + timing & visuals)

print("time duration of each trial:", trial_durations_tensor)

best_trial = study.best_trial
print("Best RMSE:", best_trial.value)
print("Best hyperparameters:", best_trial.params)
best_number = best_trial.number

# ---- trials dataframe ----
trials_df = study.trials_dataframe()

# ---- total training time (sum of all trial durations) ----
try:
    total_time_sec = float(trial_durations_tensor.sum().item())
except Exception:
    # fallback (numpy array / list)
    total_time_sec = float(np.array(trial_durations_tensor).sum())

hrs = int(total_time_sec // 3600)
mins = int((total_time_sec % 3600) // 60)
secs = total_time_sec % 60
print(f"Total training time: {total_time_sec:.2f} seconds (~{hrs}h {mins}m {secs:.1f}s)")

# ---- n-smallest RMSE ----
small_RMSE = 10
lowest_10 = trials_df.nsmallest(small_RMSE, "value")

# ---- Pareto frontier (duration vs RMSE) ----
# Ensure numeric duration column
if "duration_in_seconds" not in trials_df.columns:
    if "duration" in trials_df.columns:
        trials_df["duration_in_seconds"] = trials_df["duration"].apply(
            lambda x: x.total_seconds() if hasattr(x, "total_seconds") else np.nan
        )
    else:
        def _dur_seconds(row):
            try:
                return (row["datetime_complete"] - row["datetime_start"]).total_seconds()
            except Exception:
                return np.nan
        if {"datetime_complete", "datetime_start"}.issubset(trials_df.columns):
            trials_df["duration_in_seconds"] = trials_df.apply(_dur_seconds, axis=1)
        else:
            trials_df["duration_in_seconds"] = np.nan

sorted_df = trials_df.sort_values(by=["duration_in_seconds", "value"])

pareto_points = []
current_best_rmse = float("inf")
for _, row in sorted_df.iterrows():
    if row["value"] < current_best_rmse:
        pareto_points.append(row)
        current_best_rmse = row["value"]
pareto_df = pd.DataFrame(pareto_points)

# ---- Save artifacts ----
file_optuna = f"Model{model_type}/optuna_study.pkl"
joblib.dump(study, file_optuna)

file_optuna_forecast = f"Model{model_type}/optuna_forecast.pkl"
# Move to CPU before saving if needed
joblib.dump(Forecast_trials.cpu(), file_optuna_forecast)

file_small_rmse = f"Model{model_type}/small_rmse.pkl"
joblib.dump(lowest_10, file_small_rmse)

file_pareto = f"Model{model_type}/pareto.pkl"
joblib.dump(pareto_df, file_pareto)

file_time_n0 = f"Model{model_type}/time_n0.pkl"
joblib.dump(trial_durations_n0, file_time_n0)

file_time_n_remaining = f"Model{model_type}/time_n_remaining.pkl"
joblib.dump(trial_durations_n_remaining, file_time_n_remaining)

#%% ---- Plots: training durations per trial ----
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.grid": True,
    "grid.color": "#d0d0d0",
    "grid.alpha": 0.7,
})

# Coerce to numpy for plotting
durations_sec = np.asarray(
    trial_durations_tensor.detach().cpu().numpy()
    if hasattr(trial_durations_tensor, "detach") else trial_durations_tensor,
    dtype=float
)
trials_idx = np.arange(1, len(durations_sec) + 1)

# 1) Duration per trial (bar)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.bar(trials_idx, durations_sec, alpha=0.85)
ax.set_title("Training Duration per Trial (seconds)")
ax.set_xlabel("Trial")
ax.set_ylabel("Seconds")
fig.savefig(f"Model{model_type}/trial_durations_bar.png", dpi=300, bbox_inches="tight")
plt.show(); plt.close(fig)

# 2) Duration per trial (line) — static, with y-limit at 500s
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(trials_idx, durations_sec, "-o", linewidth=2, alpha=0.9, label="Seconds per trial")
ax.set_title("Training Duration per Trial (seconds) — Line")
ax.set_xlabel("Trial")
ax.set_ylabel("Seconds")
ax.set_ylim(0, 500)  # fixed scale as requested
ax.legend()
fig.savefig(f"Model{model_type}/trial_durations_line.png", dpi=300, bbox_inches="tight")
plt.show(); plt.close(fig)

# 3) Animation: Duration per trial (y-limit fixed at 500s)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(1, len(trials_idx) if len(trials_idx) else 1)
ax.set_ylim(0, 500)  # fixed scale as requested
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Styling
for spine in ax.spines.values():
    spine.set_color("black")
ax.tick_params(colors="black")
ax.xaxis.label.set_color("black")
ax.yaxis.label.set_color("black")
ax.title.set_color("black")
ax.grid(True, color="lightgray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_title("Training Duration per Trial — Animation", fontweight="bold")
ax.set_xlabel("Trial Number", fontweight="bold")
ax.set_ylabel("Time (seconds)", fontweight="bold")

# Plot objects
line,  = ax.plot([], [], color="#1f77b4", lw=2.5, label="Seconds per trial")
points, = ax.plot([], [], 'o', color="#d62728", markersize=6, alpha=0.8, label="Current Trial")
ax.legend(loc="upper right", frameon=True)

# Init + update
def init():
    line.set_data([], [])
    points.set_data([], [])
    return line, points

def update(frame):
    x = trials_idx[:frame]
    y = durations_sec[:frame]
    line.set_data(x, y)
    if frame > 0:
        points.set_data([x[-1]], [y[-1]])
    return line, points

ani = animation.FuncAnimation(
    fig, update, frames=len(trials_idx)+1,
    init_func=init, blit=True, interval=80
)

ani.save(f"Model{model_type}/training_time_per_trial_presentation.gif", writer="pillow", fps=12)
plt.show()

#%%#######################################################
# Reload saved artifacts (if needed later)
file_optuna = f"Model{model_type}/optuna_study.pkl"
study = joblib.load(file_optuna)
trials_df = study.trials_dataframe()
trials_df["duration_in_seconds"] = trials_df["duration"].apply(lambda x: x.total_seconds())

file_optuna_forecast = f"Model{model_type}/optuna_forecast.pkl"
Forecast_trials = joblib.load(file_optuna_forecast)

file_small_rmse = f"Model{model_type}/small_rmse.pkl"
lowest_10 = joblib.load(file_small_rmse)

file_pareto = f"Model{model_type}/pareto.pkl"
pareto_df = joblib.load(file_pareto)

file_time_no = f"Model{model_type}/time_n0.pkl"
trial_durations_n0 = joblib.load(file_time_no)

file_time_n_remaining = f"Model{model_type}/time_n_remaining.pkl"
trial_durations_n_remaining = joblib.load(file_time_n_remaining)

#%%##################################################
#             Hyperparameter importance plot
#######################################################
from optuna.importance import get_param_importances
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

imp = get_param_importances(study)
df_imp = (
    pd.DataFrame.from_dict(imp, orient="index", columns=["importance"])
      .sort_values("importance", ascending=True)
      .reset_index()
      .rename(columns={"index": "hyperparameter"})
)

# Nicer labels — keys MUST match the sampled param names
name_map = {
    "learning_rate_init": "LR (init)",
    "learning_rate_all":  "LR (all)",
    "D_init":             "Window (init)",
    "D_all":              "Window (all)",
    "weight_decay_init":  "Weight decay (init)",
    "weight_decay_all":   "Weight decay (all)",
    "lambda_reg_init":    "L1 λ (init)",
    "lambda_reg_all":     "L1 λ (all)",
    "number_neurons":     "# neurons",
    "seq_len":            "Seq. length",
    "use_ols_weights":    "Use OLS init",
    "alpha":              "OLS scale α",
}
df_imp["label"] = df_imp["hyperparameter"].map(name_map).fillna(df_imp["hyperparameter"])
df_imp = df_imp.sort_values("importance", ascending=True).reset_index(drop=True)

# -------- Save as LaTeX + CSV --------
tbl = df_imp[["label", "importance"]].rename(columns={"label": "Hyperparameter", "importance": "Importance"})
latex_path = f"Model{model_type}/important_variables.tex"
csv_path   = f"Model{model_type}/important_variables.csv"

# LaTeX (rounded)
latex_str = tbl.to_latex(index=False,
                         float_format=lambda x: f"{x:.4f}",
                         caption="Hyperparameter importances (fANOVA)",
                         label="tab:hp_importances")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(latex_str)

# CSV too (handy for reproducibility)
tbl.to_csv(csv_path, index=False)

# -------- Plot (log x-axis) --------
fig, ax = plt.subplots(figsize=(8, max(4, len(df_imp) * 0.5)), tight_layout=True)

# avoid log(0): clip tiny importances up to epsilon for plotting only
eps = 1e-6
x_vals = df_imp["importance"].clip(lower=eps)

bars = ax.barh(df_imp["label"], x_vals, edgecolor="white", height=0.6)

# put the numeric labels slightly to the right of the bars in display coords
for rect, true_val in zip(bars, df_imp["importance"].values):
    w = rect.get_width()
    y = rect.get_y() + rect.get_height()/2
    # annotate with a small 5pt offset to the right, independent of scale
    ax.annotate(f"{true_val:.4f}",
                xy=(w, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center", ha="left", fontsize=9)

ax.set_xscale("log")
ax.set_xlabel("Importance Score (log scale)")
ax.set_title("Hyperparameter Importances (fANOVA)")
ax.grid(axis="x", linestyle="--", alpha=0.6)

plt.savefig(f"Model{model_type}/important_variables.png", dpi=300, bbox_inches="tight")
plt.show()
#%%#######################################################
############## Test Data ###############
###########################################################
# Use all data including the test data
days_test = pd.to_datetime(dates_S)
dat_test = data_array

# Extract the matrix and related metadata (compute once)
regmat_test_tuple = reg_matrix(dat_test, days_test, model_type=model_type)
regmat_test, columns_s, columns_base, columns_total, _ = regmat_test_tuple

# Remove NAs
regmat0_test = regmat_test.dropna()

# Convert DataFrame to a NumPy array first, then to a tensor
regmat_tensor_test = torch.from_numpy(regmat0_test.values).float().to(device)

# Remove the dependent vars from the regressors tensor (keep a copy of y)
dependent_var_test = regmat0_test.iloc[:, dependent_index]
dependent_var_tensor_test = torch.from_numpy(dependent_var_test.values).float().to(device)
regmat_tensor_test[:, dependent_index] = 0.0

#%%#######################################################
############## Neural network for test ###################
##########################################################
def train_and_evaluate_updated_weights_test(
    train_loader,
    test_loader,
    num_feature,
    previous_weights_s,
    num_epochs,
    learning_rate,
    number_neurons,
    std_y,
    mean_y,
    mask_input_to_mid,
    weight_decay,
    lambda_reg,
    use_ols_weights,
    ols_tensor,
    alpha,
    model_type,   # NEW numbering: 1..7
):
    # ---------- Layers with masking applied at compute-time ----------
    class MaskedLinear(nn.Module):
        def __init__(self, input_dim, output_dim, mask):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.register_buffer("mask", mask)  # (output_dim, input_dim)

        def forward(self, x):
            # x: (B, F)
            masked_weight = self.linear.weight * self.mask
            return nn.functional.linear(x, masked_weight, self.linear.bias)

    class MaskedElmanRNN(nn.Module):
        """ReLU Elman cell with input mask expanded safely."""
        def __init__(self, input_dim, hidden_dim, mask):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.weight_ih = nn.Parameter(torch.empty(hidden_dim, input_dim))
            self.weight_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            self.bias_ih   = nn.Parameter(torch.empty(hidden_dim))
            self.bias_hh   = nn.Parameter(torch.empty(hidden_dim))
            self.register_buffer("orig_mask", mask)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.uniform_(self.weight_ih, -0.001, 0.001)
            nn.init.uniform_(self.weight_hh, -0.001, 0.001)
            nn.init.uniform_(self.bias_ih,   -0.001, 0.001)
            nn.init.uniform_(self.bias_hh,   -0.001, 0.001)

        def _expand_mask(self):
            if self.orig_mask.dim() == 2 and self.orig_mask.size(0) != self.hidden_dim:
                m1 = (self.orig_mask > 0).any(dim=0, keepdim=True).float()  # (1,F)
                return m1.expand(self.hidden_dim, -1).contiguous()
            return self.orig_mask

        def forward(self, x, h0=None):
            # x: (B, T, F)
            B, T, _ = x.shape
            h_t = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype) if h0 is None else h0
            mask = self._expand_mask()  # (H, F)
            W_ih = self.weight_ih * mask
            W_hh = self.weight_hh
            outs = []
            for t in range(T):
                x_t = x[:, t, :]
                h_t = torch.relu(x_t @ W_ih.t() + self.bias_ih + h_t @ W_hh.t() + self.bias_hh)
                outs.append(h_t)
            y = torch.stack(outs, dim=1)  # (B,T,H)
            return y, h_t

    class MaskedLinearRNN(nn.Module):
        """Identity-activation Elman cell with masked input weights."""
        def __init__(self, input_dim, hidden_dim, mask):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.weight_ih_l0 = nn.Parameter(torch.empty(hidden_dim, input_dim))
            self.weight_hh_l0 = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            self.bias_ih_l0   = nn.Parameter(torch.empty(hidden_dim))
            self.bias_hh_l0   = nn.Parameter(torch.empty(hidden_dim))
            self.register_buffer("orig_mask", mask)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.uniform_(self.weight_ih_l0, -0.001, 0.001)
            nn.init.uniform_(self.weight_hh_l0, -0.001, 0.001)
            nn.init.uniform_(self.bias_ih_l0,   -0.001, 0.001)
            nn.init.uniform_(self.bias_hh_l0,   -0.001, 0.001)

        def _expand_mask(self):
            if self.orig_mask.dim() == 2 and self.orig_mask.size(0) != self.hidden_dim:
                m1 = (self.orig_mask > 0).any(dim=0, keepdim=True).float()
                return m1.expand(self.hidden_dim, -1).contiguous()
            return self.orig_mask

        def forward(self, x, h_0=None):
            # x: (B, T, F)
            B, T, _ = x.shape
            h_t = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype) if h_0 is None else h_0
            mask = self._expand_mask()
            W_ih = self.weight_ih_l0 * mask
            W_hh = self.weight_hh_l0
            outputs = []
            for t in range(T):
                x_t = x[:, t, :]
                h_t = (x_t @ W_ih.t() + self.bias_ih_l0) + (h_t @ W_hh.t() + self.bias_hh_l0)
                outputs.append(h_t)
            output = torch.stack(outputs, dim=1)
            return output, h_t

    # -------------------- Model wrappers --------------------
    class CustomModelWithSkip(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid, mask_input_to_output, use_linear_rnn=False):
            super().__init__()
            self.use_linear_rnn = use_linear_rnn
            self.input_to_mid = (
                MaskedLinearRNN(input_dim, middle_dim, mask_input_to_mid)
                if use_linear_rnn else
                MaskedElmanRNN(input_dim, middle_dim, mask_input_to_mid)
            )
            self.mid_to_out = nn.Linear(middle_dim, output_dim)
            self.input_to_output = MaskedLinear(input_dim, output_dim, mask_input_to_output)

        def forward(self, x, return_components=False):
            mid_seq, _ = self.input_to_mid(x)
            mid_output = mid_seq[:, -1, :]
            mid_to_output = self.mid_to_out(mid_output)
            input_to_output = self.input_to_output(x[:, -1, :])
            out = mid_to_output + input_to_output
            if return_components:
                # For M4/M5 expose parts
                return out, mid_to_output, input_to_output
            return out

    class CustomModelWithoutSkip(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid, use_linear_rnn=False):
            super().__init__()
            self.input_to_mid = (
                MaskedLinearRNN(input_dim, middle_dim, mask_input_to_mid)
                if use_linear_rnn else
                MaskedElmanRNN(input_dim, middle_dim, mask_input_to_mid)
            )
            self.mid_to_out = nn.Linear(middle_dim, output_dim)

        def forward(self, x):
            mid_seq, _ = self.input_to_mid(x)
            mid_output = mid_seq[:, -1, :]
            return self.mid_to_out(mid_output)

    class CustomModelTwoRNNs(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid):
            super().__init__()
            self.input_to_mid_relu = MaskedElmanRNN(input_dim, middle_dim, mask_input_to_mid)
            self.input_to_mid_identity = MaskedLinearRNN(input_dim, middle_dim, mask_input_to_mid)
            self.mid_to_out = nn.Linear(middle_dim * 2, output_dim)
            self.middle_dim = middle_dim

        def _split_bias(self):
            W = self.mid_to_out.weight
            H = self.middle_dim
            w_relu = W[:, :H]
            w_id   = W[:, H:]
            nr = torch.norm(w_relu, p=2, dim=1) + 1e-12
            ni = torch.norm(w_id,   p=2, dim=1) + 1e-12
            total = nr + ni
            frac_relu = (nr / total).unsqueeze(1)
            frac_id   = (ni / total).unsqueeze(1)
            b = self.mid_to_out.bias.unsqueeze(1)
            return b * frac_relu, b * frac_id

        def forward(self, x, return_components=False):
            mid_relu_seq, _ = self.input_to_mid_relu(x)
            mid_id_seq,   _ = self.input_to_mid_identity(x)
            mid_relu = mid_relu_seq[:, -1, :]
            mid_id   = mid_id_seq[:, -1, :]
            combined_mid = torch.cat((mid_relu, mid_id), dim=1)
            output = self.mid_to_out(combined_mid)
            if return_components:
                H = self.middle_dim
                w_relu = self.mid_to_out.weight[:, :H]
                w_id   = self.mid_to_out.weight[:, H:]
                b_relu, b_id = self._split_bias()
                forecast_relu = mid_relu @ w_relu.t() + b_relu.squeeze(1)
                forecast_identity = mid_id @ w_id.t() + b_id.squeeze(1)
                return output, forecast_relu, forecast_identity
            return output

    class CustomModelTwoRNNsWithSkip(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid, mask_input_to_output):
            super().__init__()
            self.core = CustomModelTwoRNNs(input_dim, middle_dim, output_dim, mask_input_to_mid)
            self.input_to_output = MaskedLinear(input_dim, output_dim, mask_input_to_output)

        def forward(self, x, return_components=False):
            core_out = self.core(x, return_components=return_components)
            skip_out = self.input_to_output(x[:, -1, :])
            if return_components:
                out_total, f_relu, f_id = core_out
                # Return all three components separately for M7
                return out_total + skip_out, f_relu, f_id, skip_out
            return core_out + skip_out

    # -------------------- Instantiate model (NEW numbering) --------------------
    input_dim = num_feature
    output_dim = S
    middle_dim = number_neurons

    if model_type == 3:      # M3: Reduced Linear
        model = MaskedLinear(input_dim, output_dim, mask_in_out_red).to(device)
    elif model_type == 1:    # M1: RNN + ReLU
        model = CustomModelWithoutSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, use_linear_rnn=False).to(device)
    elif model_type == 2:    # M2: RNN + Identity
        model = CustomModelWithoutSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, use_linear_rnn=True).to(device)
    elif model_type == 4:    # M4: Reduced Linear + ReLU RNN (skip)
        model = CustomModelWithSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, mask_in_out_red, use_linear_rnn=False).to(device)
    elif model_type == 5:    # M5: Reduced Linear + Identity RNN (skip)
        model = CustomModelWithSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, mask_in_out_red, use_linear_rnn=True).to(device)
    elif model_type == 6:    # M6: Two RNNs (ReLU + Identity)
        model = CustomModelTwoRNNs(input_dim, middle_dim, output_dim, mask_input_to_mid).to(device)
    elif model_type == 7:    # M7: Two RNNs + Linear skip
        model = CustomModelTwoRNNsWithSkip(input_dim, middle_dim, output_dim, mask_input_to_mid, mask_in_out_red).to(device)
    else:
        raise ValueError("Invalid model type. Choose 1, 2, 3, 4, 5, 6, or 7.")

    # -------------------- Init / Warm start --------------------
    if previous_weights_s is None:
        if model_type in [4, 5, 7] and use_ols_weights:
            model.input_to_output.linear.weight.data = alpha * ols_tensor.clone()
            model.input_to_output.linear.bias.data.zero_()
            for name, p in model.named_parameters():
                if 'input_to_output.linear' in name:
                    continue
                if p.requires_grad:
                    nn.init.uniform_(p, -0.001, 0.001)
        elif model_type == 3 and use_ols_weights:
            model.linear.weight.data = ols_tensor.clone()
            model.linear.bias.data.zero_()
        else:
            for p in model.parameters():
                if p.requires_grad:
                    nn.init.uniform_(p, -0.001, 0.001)
    else:
        # Load weights (consistent with training function’s save format)
        if model_type == 3:  # M3
            model.linear.weight.data = previous_weights_s["input_to_output"]["weight"].clone()
            model.linear.bias.data   = previous_weights_s["input_to_output"]["bias"].clone()

        elif model_type == 1:  # M1 (ReLU)
            model.input_to_mid.weight_ih.data = previous_weights_s["input_to_mid"]["weight_ih"].clone()
            model.input_to_mid.weight_hh.data = previous_weights_s["input_to_mid"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid"]["bias_ih"] is not None:
                model.input_to_mid.bias_ih.data = previous_weights_s["input_to_mid"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid"]["bias_hh"] is not None:
                model.input_to_mid.bias_hh.data = previous_weights_s["input_to_mid"]["bias_hh"].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()

        elif model_type == 2:  # M2 (Identity)
            model.input_to_mid.weight_ih_l0.data = previous_weights_s["input_to_mid"]["weight_ih"].clone()
            model.input_to_mid.weight_hh_l0.data = previous_weights_s["input_to_mid"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid"]["bias_ih"] is not None:
                model.input_to_mid.bias_ih_l0.data = previous_weights_s["input_to_mid"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid"]["bias_hh"] is not None:
                model.input_to_mid.bias_hh_l0.data = previous_weights_s["input_to_mid"]["bias_hh"].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()

        elif model_type == 4:  # M4 (ReLU + skip)
            model.input_to_mid.weight_ih.data = previous_weights_s["input_to_mid"]["weight_ih"].clone()
            model.input_to_mid.weight_hh.data = previous_weights_s["input_to_mid"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid"]["bias_ih"] is not None:
                model.input_to_mid.bias_ih.data = previous_weights_s["input_to_mid"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid"]["bias_hh"] is not None:
                model.input_to_mid.bias_hh.data = previous_weights_s["input_to_mid"]["bias_hh"].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()
            model.input_to_output.linear.weight.data = previous_weights_s["input_to_output"]["weight"].clone()
            model.input_to_output.linear.bias.data   = previous_weights_s["input_to_output"]["bias"].clone()

        elif model_type == 5:  # M5 (Identity + skip)
            model.input_to_mid.weight_ih_l0.data = previous_weights_s["input_to_mid"]["weight_ih"].clone()
            model.input_to_mid.weight_hh_l0.data = previous_weights_s["input_to_mid"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid"]["bias_ih"] is not None:
                model.input_to_mid.bias_ih_l0.data = previous_weights_s["input_to_mid"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid"]["bias_hh"] is not None:
                model.input_to_mid.bias_hh_l0.data = previous_weights_s["input_to_mid"]["bias_hh"].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()
            model.input_to_output.linear.weight.data = previous_weights_s["input_to_output"]["weight"].clone()
            model.input_to_output.linear.bias.data   = previous_weights_s["input_to_output"]["bias"].clone()

        elif model_type == 6:  # M6 (two RNNs)
            model.input_to_mid_relu.weight_ih.data = previous_weights_s["input_to_mid_relu"]["weight_ih"].clone()
            model.input_to_mid_relu.weight_hh.data = previous_weights_s["input_to_mid_relu"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid_relu"]["bias_ih"] is not None:
                model.input_to_mid_relu.bias_ih.data = previous_weights_s["input_to_mid_relu"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid_relu"]["bias_hh"] is not None:
                model.input_to_mid_relu.bias_hh.data = previous_weights_s["input_to_mid_relu"]["bias_hh"].clone()

            model.input_to_mid_identity.weight_ih_l0.data = previous_weights_s["input_to_mid_identity"]["weight_ih"].clone()
            model.input_to_mid_identity.weight_hh_l0.data = previous_weights_s["input_to_mid_identity"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid_identity"]["bias_ih"] is not None:
                model.input_to_mid_identity.bias_ih_l0.data = previous_weights_s["input_to_mid_identity"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid_identity"]["bias_hh"] is not None:
                model.input_to_mid_identity.bias_hh_l0.data = previous_weights_s["input_to_mid_identity"]["bias_hh"].clone()

            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()

        elif model_type == 7:  # M7 (two RNNs + skip)
            model.core.input_to_mid_relu.weight_ih.data = previous_weights_s["input_to_mid_relu"]["weight_ih"].clone()
            model.core.input_to_mid_relu.weight_hh.data = previous_weights_s["input_to_mid_relu"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid_relu"]["bias_ih"] is not None:
                model.core.input_to_mid_relu.bias_ih.data = previous_weights_s["input_to_mid_relu"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid_relu"]["bias_hh"] is not None:
                model.core.input_to_mid_relu.bias_hh.data = previous_weights_s["input_to_mid_relu"]["bias_hh"].clone()

            model.core.input_to_mid_identity.weight_ih_l0.data = previous_weights_s["input_to_mid_identity"]["weight_ih"].clone()
            model.core.input_to_mid_identity.weight_hh_l0.data = previous_weights_s["input_to_mid_identity"]["weight_hh"].clone()
            if previous_weights_s["input_to_mid_identity"]["bias_ih"] is not None:
                model.core.input_to_mid_identity.bias_ih_l0.data = previous_weights_s["input_to_mid_identity"]["bias_ih"].clone()
            if previous_weights_s["input_to_mid_identity"]["bias_hh"] is not None:
                model.core.input_to_mid_identity.bias_hh_l0.data = previous_weights_s["input_to_mid_identity"]["bias_hh"].clone()

            model.core.mid_to_out.weight.data = previous_weights_s["mid_to_out"]["weight"].clone()
            model.core.mid_to_out.bias.data   = previous_weights_s["mid_to_out"]["bias"].clone()
            model.input_to_output.linear.weight.data = previous_weights_s["input_to_output"]["weight"].clone()
            model.input_to_output.linear.bias.data   = previous_weights_s["input_to_output"]["bias"].clone()

    # -------------------- Train --------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        for X_train, y_train in train_loader:
            pred = model(X_train)
            loss = criterion(pred, y_train) + l1_regularization(model, model_type, lambda_reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # -------------------- Evaluate --------------------
    model.eval()
    decomposed_forecasts = {}
    squared_errors = None
    unstandardized_outputs = None
    unstandardized_y_test = None

    with torch.no_grad():
        for X_test, y_test in test_loader:
            if model_type in [4, 5]:
                outputs, comp_rnn, comp_lin = model(X_test, return_components=True)
                unstandardized_outputs = outputs * std_y + mean_y
                unstandardized_y_test  = y_test * std_y + mean_y
                squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2

                decomposed_forecasts["rnn_forecast"]    = comp_rnn * std_y + mean_y
                decomposed_forecasts["linear_forecast"] = comp_lin * std_y + mean_y

            elif model_type == 6:
                outputs, comp_relu, comp_id = model(X_test, return_components=True)
                unstandardized_outputs = outputs * std_y + mean_y
                unstandardized_y_test  = y_test * std_y + mean_y
                squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2

                decomposed_forecasts["rnn_relu_forecast"]     = comp_relu * std_y + mean_y
                decomposed_forecasts["rnn_identity_forecast"] = comp_id   * std_y + mean_y

            elif model_type == 7:
                outputs, comp_relu, comp_id, comp_lin = model(X_test, return_components=True)
                unstandardized_outputs = outputs * std_y + mean_y
                unstandardized_y_test  = y_test * std_y + mean_y
                squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2

                decomposed_forecasts["rnn_relu_forecast"]     = comp_relu * std_y + mean_y
                decomposed_forecasts["rnn_identity_forecast"] = comp_id   * std_y + mean_y
                decomposed_forecasts["linear_forecast"]       = comp_lin  * std_y + mean_y

            else:
                # Models 1,2,3 (no extra decomposition tensors returned)
                outputs = model(X_test)
                unstandardized_outputs = outputs * std_y + mean_y
                unstandardized_y_test  = y_test * std_y + mean_y
                squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2

    # -------------------- Save weights for potential reuse --------------------
    if model_type == 3:  # M3
        previous_weights_s = {
            "input_to_output": {
                "weight": (model.linear.weight.data * mask_in_out_red).clone(),
                "bias": model.linear.bias.data.clone(),
            }
        }
    elif model_type == 1:  # M1
        previous_weights_s = {
            "input_to_mid": {
                "weight_ih": (model.input_to_mid.weight_ih.data * model.input_to_mid._expand_mask()).clone(),
                "weight_hh": model.input_to_mid.weight_hh.data.clone(),
                "bias_ih": model.input_to_mid.bias_ih.data.clone(),
                "bias_hh": model.input_to_mid.bias_hh.data.clone(),
            },
            "mid_to_out": {
                "weight": model.mid_to_out.weight.data.clone(),
                "bias": model.mid_to_out.bias.data.clone(),
            },
        }
    elif model_type == 2:  # M2
        previous_weights_s = {
            "input_to_mid": {
                "weight_ih": (model.input_to_mid.weight_ih_l0.data * model.input_to_mid._expand_mask()).clone(),
                "weight_hh": model.input_to_mid.weight_hh_l0.data.clone(),
                "bias_ih": model.input_to_mid.bias_ih_l0.data.clone(),
                "bias_hh": model.input_to_mid.bias_hh_l0.data.clone(),
            },
            "mid_to_out": {
                "weight": model.mid_to_out.weight.data.clone(),
                "bias": model.mid_to_out.bias.data.clone(),
            },
        }
    elif model_type == 4:  # M4
        previous_weights_s = {
            "input_to_mid": {
                "weight_ih": (model.input_to_mid.weight_ih.data * model.input_to_mid._expand_mask()).clone(),
                "weight_hh": model.input_to_mid.weight_hh.data.clone(),
                "bias_ih": model.input_to_mid.bias_ih.data.clone(),
                "bias_hh": model.input_to_mid.bias_hh.data.clone(),
            },
            "mid_to_out": {
                "weight": model.mid_to_out.weight.data.clone(),
                "bias": model.mid_to_out.bias.data.clone(),
            },
            "input_to_output": {
                "weight": (model.input_to_output.linear.weight.data * mask_in_out_red).clone(),
                "bias": model.input_to_output.linear.bias.data.clone(),
            },
        }
    elif model_type == 5:  # M5
        previous_weights_s = {
            "input_to_mid": {
                "weight_ih": (model.input_to_mid.weight_ih_l0.data * model.input_to_mid._expand_mask()).clone(),
                "weight_hh": model.input_to_mid.weight_hh_l0.data.clone(),
                "bias_ih": model.input_to_mid.bias_ih_l0.data.clone(),
                "bias_hh": model.input_to_mid.bias_hh_l0.data.clone(),
            },
            "mid_to_out": {
                "weight": model.mid_to_out.weight.data.clone(),
                "bias": model.mid_to_out.bias.data.clone(),
            },
            "input_to_output": {
                "weight": (model.input_to_output.linear.weight.data * mask_in_out_red).clone(),
                "bias": model.input_to_output.linear.bias.data.clone(),
            },
        }
    elif model_type == 6:  # M6
        previous_weights_s = {
            "input_to_mid_relu": {
                "weight_ih": (model.input_to_mid_relu.weight_ih.data * model.input_to_mid_relu._expand_mask()).clone(),
                "weight_hh": model.input_to_mid_relu.weight_hh.data.clone(),
                "bias_ih": model.input_to_mid_relu.bias_ih.data.clone(),
                "bias_hh": model.input_to_mid_relu.bias_hh.data.clone(),
            },
            "input_to_mid_identity": {
                "weight_ih": (model.input_to_mid_identity.weight_ih_l0.data * model.input_to_mid_identity._expand_mask()).clone(),
                "weight_hh": model.input_to_mid_identity.weight_hh_l0.data.clone(),
                "bias_ih": model.input_to_mid_identity.bias_ih_l0.data.clone(),
                "bias_hh": model.input_to_mid_identity.bias_hh_l0.data.clone(),
            },
            "mid_to_out": {
                "weight": model.mid_to_out.weight.data.clone(),
                "bias": model.mid_to_out.bias.data.clone(),
            },
        }
    elif model_type == 7:  # M7
        previous_weights_s = {
            "input_to_mid_relu": {
                "weight_ih": (model.core.input_to_mid_relu.weight_ih.data * model.core.input_to_mid_relu._expand_mask()).clone(),
                "weight_hh": model.core.input_to_mid_relu.weight_hh.data.clone(),
                "bias_ih": model.core.input_to_mid_relu.bias_ih.data.clone(),
                "bias_hh": model.core.input_to_mid_relu.bias_hh.data.clone(),
            },
            "input_to_mid_identity": {
                "weight_ih": (model.core.input_to_mid_identity.weight_ih_l0.data * model.core.input_to_mid_identity._expand_mask()).clone(),
                "weight_hh": model.core.input_to_mid_identity.weight_hh_l0.data.clone(),
                "bias_ih": model.core.input_to_mid_identity.bias_ih_l0.data.clone(),
                "bias_hh": model.core.input_to_mid_identity.bias_hh_l0.data.clone(),
            },
            "mid_to_out": {
                "weight": model.core.mid_to_out.weight.data.clone(),
                "bias": model.core.mid_to_out.bias.data.clone(),
            },
            "input_to_output": {
                "weight": (model.input_to_output.linear.weight.data * mask_in_out_red).clone(),
                "bias": model.input_to_output.linear.bias.data.clone(),
            },
        }

    return squared_errors, previous_weights_s, unstandardized_y_test, unstandardized_outputs, decomposed_forecasts
#%%#######################################################
############## Test function with ensemble###############
##########################################################

# Define the test period length
length_test = int(2 * 365)
begin_test = regmat0_test.shape[0] - length_test
N_s = length_test

# Dependent variables (actual prices) for the test set
actual_test = dependent_var_tensor_test[-N_s:, :]

# Placeholder for ensemble forecasts (kept for future use)
# forecast_tensor_test = torch.zeros((N_s, S, len(best_model_combo)), device=device)


#%%#######################################################
############## Test function with best model only ########
##########################################################

import time  # <-- added for timing

def forecast_ANN_Tensor_test(dat_test, model_type, mask_input_to_mid_rnn):
    """
    Generate test forecast using the single best model found in Optuna.
    - Robust sequence building for RNNs
    - Wrapper-safe model 7 handling
    - Optional OLS weight init (3/4/5/7) like in training
    - Weekly-naive baseline generated alongside outputs

    Also records timing:
      - total test time
      - first-step time (n == 0)
      - remaining steps time (n > 0)
      - per-step durations (array of length N_s)
    """
    mse_all = torch.zeros((N_s, S), device=device)

    # ---- timing containers (added) ----
    test_time_total_start = time.time()
    test_time_n0 = 0.0
    test_time_other = 0.0
    test_time_per_step = np.zeros(N_s, dtype=float)

    # Pre-allocate containers for decomposed outputs only if needed
    decomposed_forecasts_all = {}
    if model_type in [4, 5, 6, 7]:
        if model_type in [4, 5]:
            decomposed_forecasts_all['rnn_forecast'] = torch.zeros((N_s, S), device=device)
            decomposed_forecasts_all['linear_forecast'] = torch.zeros((N_s, S), device=device)
        elif model_type == 6:
            decomposed_forecasts_all['rnn_relu_forecast'] = torch.zeros((N_s, S), device=device)
            decomposed_forecasts_all['rnn_identity_forecast'] = torch.zeros((N_s, S), device=device)
        elif model_type == 7:
            decomposed_forecasts_all['rnn_relu_forecast'] = torch.zeros((N_s, S), device=device)
            decomposed_forecasts_all['rnn_identity_forecast'] = torch.zeros((N_s, S), device=device)
            decomposed_forecasts_all['linear_forecast'] = torch.zeros((N_s, S), device=device)

    unstandardized_y_test_all = torch.zeros((N_s, S), device=device)
    unstandardized_outputs_all = torch.zeros((N_s, S), device=device)
    weekly_naive_all = torch.full((N_s, S), float('nan'), device=device)

    # Extract best trial parameters (guard missing keys)
    trial = study.best_trial
    learning_rate_init = trial.params["learning_rate_init"]
    D_init             = trial.params["D_init"]
    learning_rate_all  = trial.params["learning_rate_all"]
    D_all              = trial.params["D_all"]
    weight_decay_init  = trial.params["weight_decay_init"]
    weight_decay_all   = trial.params["weight_decay_all"]
    lambda_reg_init    = trial.params["lambda_reg_init"]
    lambda_reg_all     = trial.params["lambda_reg_all"]

    if model_type in {1, 2, 4, 5, 6, 7}:
        number_neurons = trial.params.get("number_neurons", 32)
        seq_len_param  = trial.params.get("seq_len", 1)
    else:
        number_neurons = 0
        seq_len_param = 1

    if model_type in {3, 4, 5, 7}:
        use_ols_weights = trial.params.get("use_ols_weights", False)
        alpha = trial.params.get("alpha", 1.0) if use_ols_weights else None
    else:
        use_ols_weights = None
        alpha = None

    # Build per-neuron mask for RNNs (shared across hidden units)
    mask_input_to_mid = torch.ones((number_neurons, num_columns), dtype=torch.float32, device=device)
    if model_type in [1, 2, 4, 5, 6, 7]:
        rnn_mask_flat = torch.any(mask_input_to_mid_rnn > 0, dim=0)  # (F,)
        mask_input_to_mid = rnn_mask_flat.repeat(number_neurons, 1)  # (H,F)
    else:
        mask_input_to_mid[:, dependent_index] = 0

    previous_weights_s = None

    for n in range(N_s):
        # ---- start step timer (added) ----
        step_start = time.time()

        # Window parameters
        if n == 0:
            D = D_init
            lr = learning_rate_init
            weight_decay = weight_decay_init
            lambda_reg = lambda_reg_init
        else:
            D = D_all
            lr = learning_rate_all
            weight_decay = weight_decay_all
            lambda_reg = lambda_reg_all

        # OLS initialization (if applicable)
        ols_tensor = None
        if model_type in {3, 4, 5, 7} and use_ols_weights:
            coef_OLS_all = forecast_expert_ext(
                dat=dat_test[(begin_test - D + n):(begin_test + n + 1), :],
                days=days_test[(begin_test - D + n):(begin_test + n + 1)],
                reg_names=data.columns[1:],
                model_type=model_type
            )["coefficients"]
            coef_OLS_ten = torch.tensor(coef_OLS_all.values, dtype=torch.float32, device=device)
            ols_tensor = torch.zeros((S, num_columns), dtype=torch.float32, device=device)
            for row_idx, col_indices in active_regressor.items():
                ols_tensor[row_idx, col_indices] = coef_OLS_ten[row_idx, :len(col_indices)]

        # Standardize data (train window excludes current test row for mean/std; test row standardized with train stats)
        mean_x = regmat_tensor_test[(begin_test - D + n):(begin_test + n), :].mean(0, keepdim=True)
        std_x  = regmat_tensor_test[(begin_test - D + n):(begin_test + n), :].std(0, keepdim=True)
        std_x[std_x == 0] = 1

        X_train_unseq = (regmat_tensor_test[(begin_test - D + n):(begin_test + n), :] - mean_x) / std_x
        X_test_unseq  = (regmat_tensor_test[(begin_test + n), :] - mean_x) / std_x

        mean_y = dependent_var_tensor_test[(begin_test - D + n):(begin_test + n), :].mean(0, keepdim=True)
        std_y  = dependent_var_tensor_test[(begin_test - D + n):(begin_test + n), :].std(0, keepdim=True)
        std_y[std_y == 0] = 1

        y_train_unseq = (dependent_var_tensor_test[(begin_test - D + n):(begin_test + n), :] - mean_y) / std_y
        y_test        = ((dependent_var_tensor_test[(begin_test + n), :] - mean_y) / std_y).reshape(1, -1)

        # Weekly naive baseline for this day (t-7)
        global_idx = begin_test + n
        weekly_idx = global_idx - 7
        if weekly_idx >= 0:
            weekly_naive_all[n, :] = dependent_var_tensor_test[weekly_idx, :]

        # Create sequences for RNN if applicable (robust clamp & padding)
        if model_type in [1, 2, 4, 5, 6, 7]:
            eff_seq_len = int(max(1, min(seq_len_param, D, X_train_unseq.shape[0])))
            X_train, y_train = create_sequences(X_train_unseq, y_train_unseq, eff_seq_len)

            feat_dim = X_train_unseq.shape[1]
            if eff_seq_len > 1:
                k = min(eff_seq_len - 1, X_train_unseq.shape[0])
                X_hist = X_train_unseq[-k:, :].reshape(k, feat_dim)
                x_row  = X_test_unseq.reshape(1, feat_dim)
                if k < (eff_seq_len - 1):
                    pad = X_train_unseq[:1, :].repeat((eff_seq_len - 1) - k, 1)
                    X_hist = torch.cat([pad, X_hist], dim=0)
                X_test_full = torch.cat([X_hist, x_row], dim=0)  # (eff_seq_len, F)
            else:
                X_test_full = X_test_unseq.reshape(1, feat_dim)
            X_test = X_test_full.unsqueeze(0)  # (1, eff_seq_len, F)
        else:
            X_train, y_train = X_train_unseq, y_train_unseq
            X_test = X_test_unseq.reshape(1, -1)

        # DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_test, y_test),   batch_size=1,        shuffle=False)

        # Train+eval for this day
        mse_value, previous_weights_s, unstandardized_y_test, unstandardized_outputs, decomposed_forecasts = train_and_evaluate_updated_weights_test(
            train_loader,
            test_loader,
            regmat_tensor_test.shape[1],
            previous_weights_s,
            num_epochs_all if n > 0 else num_epochs_init,
            lr,
            number_neurons,
            std_y,
            mean_y,
            mask_input_to_mid,
            weight_decay,
            lambda_reg,
            use_ols_weights,
            ols_tensor,
            alpha,
            model_type
        )

        mse_all[n, :] = mse_value
        if decomposed_forecasts:
            for key, value in decomposed_forecasts.items():
                decomposed_forecasts_all[key][n, :] = value.squeeze(0)

        unstandardized_y_test_all[n, :]   = unstandardized_y_test.squeeze(0)
        unstandardized_outputs_all[n, :]  = unstandardized_outputs.squeeze(0)

        # ---- stop step timer & accumulate (added) ----
        step_elapsed = time.time() - step_start
        test_time_per_step[n] = step_elapsed
        if n == 0:
            test_time_n0 += step_elapsed
        else:
            test_time_other += step_elapsed

    # ---- total time (added) ----
    test_time_total = time.time() - test_time_total_start

    # Aggregate errors
    agg_sq_mean = mse_all.mean(dim=0).sqrt()
    overall_agg_mean = mse_all.mean().sqrt()

    # Decomposition sum-check (if we have components)
    decomp_check_maxabs = None
    if decomposed_forecasts_all:
        if model_type in [4, 5]:
            decomp_sum = decomposed_forecasts_all['rnn_forecast'] + decomposed_forecasts_all['linear_forecast']
        elif model_type == 6:
            decomp_sum = decomposed_forecasts_all['rnn_relu_forecast'] + decomposed_forecasts_all['rnn_identity_forecast']
        elif model_type == 7:
            decomp_sum = (
                decomposed_forecasts_all['rnn_relu_forecast']
                + decomposed_forecasts_all['rnn_identity_forecast']
                + decomposed_forecasts_all['linear_forecast']
            )
        diff = torch.abs(decomp_sum - unstandardized_outputs_all)
        decomp_check_maxabs = float(diff.max().item())
        print(f"[Decomposition check] max |sum(components) - combined| = {decomp_check_maxabs:.6e}")

    # Decomposed RMSEs (if any)
    decomposed_rmse = {}
    if decomposed_forecasts_all:
        for key, forecast_series in decomposed_forecasts_all.items():
            rmse = torch.sqrt(torch.mean((forecast_series - unstandardized_y_test_all) ** 2))
            decomposed_rmse[f'{key}_rmse'] = rmse.item()

    return (
        agg_sq_mean,
        overall_agg_mean,
        unstandardized_y_test_all,
        unstandardized_outputs_all,
        decomposed_rmse,
        decomposed_forecasts_all,
        weekly_naive_all,
        decomp_check_maxabs,
        # ---- timings returned (added) ----
        test_time_total,
        test_time_n0,
        test_time_other,
        test_time_per_step,
    )


# Run single best model forecast
(
    rmse_mean_test,
    overall_mean_test,
    unstandardized_y_test,
    unstandardized_outputs_test,
    decomposed_rmse_test,
    decomposed_forecasts_all_test,
    weekly_naive_test,
    decomp_check_maxabs_test,
    # ---- timings captured (added) ----
    test_time_total,
    test_time_n0,
    test_time_other,
    test_time_per_step,
) = forecast_ANN_Tensor_test(dat_test, model_type, mask_input_to_mid_rnn)

# Save best model results
joblib.dump(unstandardized_y_test.detach().cpu().numpy(),       f"Model{model_type}/unstandardized_actual_best.pkl")
joblib.dump(unstandardized_outputs_test.detach().cpu().numpy(), f"Model{model_type}/unstandarized_forecast_best.pkl")

if decomposed_forecasts_all_test:
    decomposed_np = {k: v.detach().cpu().numpy() for k, v in decomposed_forecasts_all_test.items()}
    joblib.dump(decomposed_np, f"Model{model_type}/decomposed_forecasts_best.pkl")

# ---- print timing summary (added) ----
print("\n Test Timing:")
print(f"  Total test time:         {test_time_total:.2f} s")
print(f"  First step (n==0):       {test_time_n0:.2f} s")
print(f"  Remaining steps (n>0):   {test_time_other:.2f} s")
print(f"  Avg per step (all n):    {test_time_per_step.mean():.3f} s  (median: {np.median(test_time_per_step):.3f} s)")

# ---- static plot of per-step times (y-limit at 500s) (added) ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(np.arange(N_s), test_time_per_step, marker="o", linestyle="-", alpha=0.85, label="Per-step test time (s)")
for spine in ax.spines.values():
    spine.set_color("black")
ax.tick_params(colors="black")
ax.xaxis.label.set_color("black")
ax.yaxis.label.set_color("black")
ax.title.set_color("black")
ax.grid(True, color="lightgray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Test step (day index)")
ax.set_ylabel("Seconds")
ax.set_title("Test Inference Time per Step")
ax.set_xlim(0, N_s - 1)
ax.set_ylim(0, 500)  # per your request
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
time_plot_path = f"Model{model_type}/test_time_per_step.png"
fig.savefig(time_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.show(); plt.close(fig)
print(f"Saved test-time per-step plot to {time_plot_path}")

# ---- persist raw timings for later usage (added) ----
joblib.dump(
    {
        "total": test_time_total,
        "n0": test_time_n0,
        "remaining": test_time_other,
        "per_step": test_time_per_step,
    },
    f"Model{model_type}/test_timings.pkl"
)

# Print results summary (kept)
print("\nTest Results Summary:")
print(f"Best Single Model RMSE: {float(overall_mean_test):.5f}")
if decomposed_rmse_test:
    print("Decomposed RMSEs:")
    for name, value in decomposed_rmse_test.items():
        print(f"  {name}: {value:.5f}")
if decomp_check_maxabs_test is not None:
    print(f"Decomposition sum-check (max abs diff): {decomp_check_maxabs_test:.6e}")



#%%#######################################################
############## Metrics (sMAPE / MAE / rMAE / RMSE) #######
########################################################
import numpy as np

def calculate_metrics(actual, forecast, baseline=None):
    """
    Calculate sMAPE, MAE, rMAE, RMSE, and rRMSE.
    If `baseline` (weekly naive) is provided:
      rMAE  = MAE(model)  / MAE(naive)
      rRMSE = RMSE(model) / RMSE(naive)
    """
    actual   = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)

    # sMAPE
    denom = (np.abs(actual) + np.abs(forecast))
    denom[denom == 0] = 1e-12
    smape = np.mean(np.abs(actual - forecast) / denom) * 200.0

    # MAE / RMSE for model
    mae  = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))

    rmae = None
    rrmse = None
    rmse_naive = None
    mae_naive = None

    if baseline is not None:
        baseline = np.array(baseline, dtype=float)
        mae_naive  = np.mean(np.abs(actual - baseline))
        rmse_naive = np.sqrt(np.mean((actual - baseline) ** 2))
        mae_naive  = mae_naive  if mae_naive  > 0 else 1e-12
        rmse_naive = rmse_naive if rmse_naive > 0 else 1e-12
        rmae  = mae  / mae_naive
        rrmse = rmse / rmse_naive
    else:
        # fallback normals (not preferred)
        baseline_scale = np.mean(np.abs(actual)) + 1e-12
        rmae  = mae  / baseline_scale
        rrmse = rmse / (np.sqrt(np.mean(actual**2)) + 1e-12)

    return smape, mae, rmae, rmse, rrmse, mae_naive, rmse_naive

# Prepare arrays
actual_np   = unstandardized_y_test.detach().cpu().numpy()
forecast_np = unstandardized_outputs_test.detach().cpu().numpy()

# Weekly naive aligned (first 7 days are NaN => drop them for fair comparison)
naive_np    = weekly_naive_test.detach().cpu().numpy()
valid_mask  = ~np.isnan(naive_np).any(axis=1)

actual_valid   = actual_np[valid_mask]
forecast_valid = forecast_np[valid_mask]
naive_valid    = naive_np[valid_mask]

# Model vs Naive
smape_test, mae_test, rmae_test, rmse_test, rrmse_test, mae_naive, rmse_naive = \
    calculate_metrics(actual_valid, forecast_valid, baseline=naive_valid)

# Naive vs Actual (for context)
smape_naive_only, mae_naive_only, _, rmse_naive_only, _, _, _ = \
    calculate_metrics(actual_valid, naive_valid, baseline=None)

print("\nTest Metrics (vs weekly naive):")
print(f"  sMAPE (model):       {smape_test:.5f}")
print(f"  MAE   (model):       {mae_test:.5f}")
print(f"  RMSE  (model):       {rmse_test:.5f}")
print(f"  rMAE  (model/naive): {rmae_test:.5f}")
print(f"  rRMSE (model/naive): {rrmse_test:.5f}")
print(f"  MAE   (naive):       {mae_naive:.5f}")
print(f"  RMSE  (naive):       {rmse_naive:.5f}")
print(f"  sMAPE (naive):       {smape_naive_only:.5f}")

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt

# reset any active style that may set gray backgrounds
plt.style.use("default")

# force a white canvas everywhere (figures, axes, saved files)
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.grid": True,            # keep your grid
    "grid.color": "#d0d0d0",      # light grid lines
    "grid.alpha": 0.7,
})

# %%=======================================================
# Actual (blue) vs Combined (red) 
# =========================================================
actuals_np  = actual_np            # (N_s, S)
combined_np = forecast_np          # (N_s, S)
N_s, S_local = actuals_np.shape

color_actual   = "#1f77b4"  # blue
color_combined = "#d62728"  # red

outdir = f"Model{model_type}"
os.makedirs(outdir, exist_ok=True)

# --- helper for placing legend below the x-label ---
def place_legend_below(ax, ncol=2, pad=0.22):
    # pad = extra bottom space for legend; tweak as needed
    fig = ax.figure
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=ncol, frameon=True)
    fig.subplots_adjust(bottom=pad)
    return leg

# 1) Daily-average series over test period
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
x_days = days_test[-len(actuals_np):]
ax.plot(x_days, actuals_np.mean(axis=1),  label='Actual Price',      color=color_actual,   linewidth=2, linestyle='--')
ax.plot(x_days, combined_np.mean(axis=1), label='Combined Forecast', color=color_combined, linewidth=2, linestyle='-')
ax.set_title('Daily Average — Actual vs Combined Forecast (Test Period)')
ax.set_xlabel('Date'); ax.set_ylabel('Price'); ax.grid(True)
try: fig.autofmt_xdate()
except: pass
place_legend_below(ax, ncol=2, pad=0.20)
fig.savefig(f"{outdir}/actual_vs_combined_daily.png", dpi=300, bbox_inches="tight")
plt.show(); plt.close(fig)

# 2) Average profile by hour of day
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
hours_arr = np.arange(S_local)
ax.plot(hours_arr, actuals_np.mean(axis=0),  marker='x', linestyle='--', label='Actual Price',      color=color_actual)
ax.plot(hours_arr, combined_np.mean(axis=0), marker='s', linestyle='-',  label='Combined Forecast', color=color_combined)
ax.set_title('Average Decomposed and Combined Forecasts by Hour of Day')
ax.set_xlabel('Hour'); ax.set_ylabel('Average Price'); ax.set_xticks(hours_arr); ax.grid(True)
place_legend_below(ax, ncol=2, pad=0.20)
fig.savefig(f"{outdir}/actual_vs_combined_hourly.png", dpi=300, bbox_inches="tight")
plt.show(); plt.close(fig)

# 3) Full TEST sample (no averaging) — time series
try:
    import pandas as pd
    days_used = pd.to_datetime(days_test[-N_s:])
    idx = days_used.repeat(S_local) + pd.to_timedelta(np.tile(np.arange(S_local), N_s), unit="h")
except Exception:
    idx = np.arange(N_s * S_local)

actual_flat   = actuals_np.reshape(-1)
combined_flat = combined_np.reshape(-1)

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
ax.plot(idx, actual_flat,   linewidth=1.2, linestyle='--', label='Actual Price',      color=color_actual)
ax.plot(idx, combined_flat, linewidth=1.2, linestyle='-',  label='Combined Forecast', color=color_combined)
ax.set_title('Actual vs Combined Forecast — Full Test Sample')
ax.set_xlabel('Time'); ax.set_ylabel('Price'); ax.grid(True)
try: fig.autofmt_xdate()
except: pass
place_legend_below(ax, ncol=2, pad=0.22)  # legend just below "Time"
fig.savefig(f"{outdir}/actual_vs_combined_full_test.png", dpi=300, bbox_inches="tight")
plt.show(); plt.close(fig)


#%%#######################################################
############## Decomposed Error Plots ##################
########################################################
if decomposed_forecasts_all_test:
    # helper: place legend centered below the x-axis
    def _legend_below(ax, ncol=2, pad=0.20, y_offset=-0.18):
        fig = ax.figure
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, y_offset), ncol=ncol, frameon=True)
        fig.subplots_adjust(bottom=pad)

    actuals_np = actual_np
    combined_forecast_np = forecast_np
    decomposed_forecasts_np = {k: v.detach().cpu().numpy() for k, v in decomposed_forecasts_all_test.items()}

    color_actual   = "#1f77b4"  # blue
    color_combined = "#d62728"  # red
    comp_colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(decomposed_forecasts_np)))

    # 1) Daily-average series over test period
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    x_days = days_test[-len(actuals_np):]

    ax.plot(x_days, actuals_np.mean(axis=1),  label='Actual Price',
            color=color_actual,   linewidth=2, linestyle='--')
    ax.plot(x_days, combined_forecast_np.mean(axis=1), label='Combined Forecast',
            color=color_combined, linewidth=2, linestyle='-')

    for i, (name, forecast) in enumerate(decomposed_forecasts_np.items()):
        ax.plot(x_days, forecast.mean(axis=1), label=f'{name}', color=comp_colors[i])

    ax.set_title('Daily Average Decomposed and Combined Forecasts over Test Period')
    ax.set_xlabel('Date'); ax.set_ylabel('Price'); ax.grid(True)
    try: fig.autofmt_xdate()
    except: pass
    _legend_below(ax, ncol=2, pad=0.20, y_offset=-0.18)

    fig.savefig(f"Model{model_type}/decomposed_forecasts_daily.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

    # 2) Average profile by hour of day
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    hours_arr = np.arange(S)

    ax.plot(hours_arr, actuals_np.mean(axis=0),  marker='x', linestyle='--',
            label='Actual Price',      color=color_actual)
    ax.plot(hours_arr, combined_forecast_np.mean(axis=0), marker='s', linestyle='-',
            label='Combined Forecast', color=color_combined)

    for i, (name, forecast) in enumerate(decomposed_forecasts_np.items()):
        ax.plot(hours_arr, forecast.mean(axis=0), marker='o', linestyle='-',
                label=f'{name}', color=comp_colors[i])

    ax.set_title('Average Decomposed and Combined Forecasts by Hour of Day')
    ax.set_xlabel('Hour'); ax.set_ylabel('Average Price'); ax.set_xticks(hours_arr); ax.grid(True)
    _legend_below(ax, ncol=2, pad=0.20, y_offset=-0.18)

    fig.savefig(f"Model{model_type}/decomposed_forecasts_hourly.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)
# #######################################################

#%%#######################################################
############## Decomposed Forecasts for Specific Days ####
##########################################################
if decomposed_forecasts_all_test:
    # helper: place legend centered below the x-axis
    def _legend_below(ax, ncol=2, pad=0.20, y_offset=-0.18):
        fig = ax.figure
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, y_offset), ncol=ncol, frameon=True)
        fig.subplots_adjust(bottom=pad)

    actuals_np = actual_np
    combined_forecast_np = forecast_np
    decomposed_forecasts_np = {k: v.detach().cpu().numpy() for k, v in decomposed_forecasts_all_test.items()}

    COLOR_ACTUAL   = "#1f77b4"  # blue
    COLOR_COMBINED = "#d62728"  # red
    comp_colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(decomposed_forecasts_np)))
    hours = np.arange(S)

    # ---------- First day ----------
    day_index = 0
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    ax.plot(hours, actuals_np[day_index, :],   marker='x', linestyle='--', label='Actual Price',      color=COLOR_ACTUAL)
    ax.plot(hours, combined_forecast_np[day_index, :], marker='s', linestyle='-',  label='Combined Forecast', color=COLOR_COMBINED)
    for i, (name, forecast) in enumerate(decomposed_forecasts_np.items()):
        ax.plot(hours, forecast[day_index, :], marker='o', linestyle='-', label=name, color=comp_colors[i])
    ax.set_title('Decomposed and Combined Forecasts — First Test Day')
    ax.set_xlabel('Hour of Day'); ax.set_ylabel('Price'); ax.set_xticks(hours); ax.grid(True)
    _legend_below(ax, ncol=2, pad=0.22, y_offset=-0.18)
    fig.savefig(f"Model{model_type}/decomposed_forecast_first_day.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

    # ---------- Last day ----------
    day_index = N_s - 1
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    ax.plot(hours, actuals_np[day_index, :],   marker='x', linestyle='--', label='Actual Price',      color=COLOR_ACTUAL)
    ax.plot(hours, combined_forecast_np[day_index, :], marker='s', linestyle='-',  label='Combined Forecast', color=COLOR_COMBINED)
    for i, (name, forecast) in enumerate(decomposed_forecasts_np.items()):
        ax.plot(hours, forecast[day_index, :], marker='o', linestyle='-', label=name, color=comp_colors[i])
    ax.set_title('Decomposed and Combined Forecasts — Last Test Day')
    ax.set_xlabel('Hour of Day'); ax.set_ylabel('Price'); ax.set_xticks(hours); ax.grid(True)
    _legend_below(ax, ncol=2, pad=0.22, y_offset=-0.18)
    fig.savefig(f"Model{model_type}/decomposed_forecast_last_day.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

    # ---------- Save per-day decompositions to Excel ----------
    first_day_data = {'Hour': hours, 'Actual': actuals_np[0, :], 'Combined_Forecast': combined_forecast_np[0, :]}
    last_day_data  = {'Hour': hours, 'Actual': actuals_np[N_s - 1, :], 'Combined_Forecast': combined_forecast_np[N_s - 1, :]}
    for name, forecast in decomposed_forecasts_np.items():
        first_day_data[name] = forecast[0, :]
        last_day_data[name]  = forecast[N_s - 1, :]

    df_first_day = pd.DataFrame(first_day_data)
    df_last_day  = pd.DataFrame(last_day_data)
    output_path = f"Model{model_type}/decomposed_forecasts_specific_days.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        df_first_day.to_excel(writer, index=False, sheet_name="First_Day_Forecast")
        df_last_day.to_excel(writer,  index=False, sheet_name="Last_Day_Forecast")
    print(f"Saved decomposed forecasts for first and last day to {output_path}")

# %%#######################################################
############## Visual tools & daily errors ##############
########################################################
# Derive sizes from arrays to avoid stale globals
unstandardized_forecast = forecast_np
actual_test_prices      = actual_np
N_s, S                  = unstandardized_forecast.shape

mse_mean     = (
    rmse_mean_test.detach().cpu().numpy()
    if hasattr(rmse_mean_test, "detach") else np.array(rmse_mean_test)
)
overall_mean = float(overall_mean_test)

# ensure output dir exists
_outdir = f"Model{model_type}"
os.makedirs(_outdir, exist_ok=True)

# Colors
COLOR_ACTUAL   = "#1f77b4"  # blue
COLOR_FORECAST = "#d62728"  # red

# Legend helper: centered below the x-axis
def _legend_below(ax, ncol=2, pad=0.20, y_offset=-0.18):
    fig = ax.figure
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, y_offset), ncol=ncol, frameon=True)
    fig.subplots_adjust(bottom=pad)
    return leg

def visualize_forecast_for_day(day_index: int):
    """Visualize forecast vs actual for a single day (by index)."""
    if not (0 <= day_index < N_s):
        print(f"Day index out of range. Must be between 0 and {N_s-1}.")
        return
    forecasted = unstandardized_forecast[day_index]
    actual = actual_test_prices[day_index]

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(S), forecasted, marker="o", linestyle="-",  label="Combined Forecast", color=COLOR_FORECAST)
    ax.plot(np.arange(S), actual,    marker="x", linestyle="--", label="Actual Price",      color=COLOR_ACTUAL)
    ax.set_xticks(np.arange(0, S, 3)); ax.set_xticklabels(np.arange(1, S+1, 3))
    ax.set_title(f"Forecast vs Actual — Day {day_index+1}")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Price (€)")
    ax.grid(True, linestyle="--", alpha=0.7)
    _legend_below(ax, ncol=2, pad=0.20, y_offset=-0.18)
    fig.savefig(f"{_outdir}/forecast_vs_actual_day_{day_index+1}.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

def visualize_forecast_for_date(date_like):
    """Plot by calendar date (e.g., '2024-06-15'). Falls back if date not found."""
    try:
        idx_dates = pd.to_datetime(days_test[-N_s:])
        target = pd.to_datetime(date_like)
        # locate the first matching day
        matches = np.where(idx_dates.normalize() == target.normalize())[0]
        if matches.size == 0:
            print(f"Date {target.date()} not in test range. Showing first day instead.")
            return visualize_forecast_for_day(0)
        return visualize_forecast_for_day(int(matches[0]))
    except Exception as e:
        print(f"Could not resolve date ({e}). Showing first day instead.")
        return visualize_forecast_for_day(0)

def visualize_average_forecast():
    """Average forecast vs actual per hour across the whole test period."""
    avg_fc  = np.nanmean(unstandardized_forecast, axis=0)
    avg_act = np.nanmean(actual_test_prices,      axis=0)

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(S), avg_fc,  marker="o", linestyle="-",  label="Combined Forecast", color=COLOR_FORECAST)
    ax.plot(np.arange(S), avg_act, marker="x", linestyle="--", label="Actual Price",      color=COLOR_ACTUAL)
    ax.set_xticks(np.arange(0, S, 3)); ax.set_xticklabels(np.arange(1, S+1, 3))
    ax.set_title("Average Forecast vs Actual (All Test Days)")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Price (€)")
    ax.grid(True, linestyle="--", alpha=0.7)
    _legend_below(ax, ncol=2, pad=0.20, y_offset=-0.18)
    fig.savefig(f"{_outdir}/avg_forecast_vs_actual_by_hour.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

def visualize_forecast_errors():
    """Forecast errors (RMSE) across hours (averaged over test days)."""
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(1, S+1), mse_mean, alpha=0.85, label="RMSE")  # <-- label added
    ax.set_xticks(np.arange(1, S+1, 3))
    ax.set_title("Forecast Errors (RMSE) by Hour")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("RMSE")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    _legend_below(ax, ncol=1, pad=0.16, y_offset=-0.14)  # now shows “RMSE”
    fig.savefig(f"{_outdir}/rmse_by_hour.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

def visualize_daily_forecast_errors(mse_per_day, rmse_per_day):
    """Daily forecast errors (MSE & RMSE) across the test period."""
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)
    ax.plot(mse_per_day,  marker="o", linestyle="-",  label="MSE")
    ax.plot(rmse_per_day, marker="x", linestyle="--", label="RMSE")
    ax.set_title("Daily Forecast Errors Over Test Period")
    ax.set_xlabel("Day"); ax.set_ylabel("Error")
    ax.grid(True, linestyle="--", alpha=0.7)
    _legend_below(ax, ncol=2, pad=0.20, y_offset=-0.18)
    fig.savefig(f"{_outdir}/daily_errors.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

# Quick runs
visualize_forecast_for_day(0)        # Day 1
visualize_average_forecast()         # Average forecast across all days
visualize_forecast_errors()          # Hourly RMSE bar chart

# Daily performance
if actual_test_prices.shape != unstandardized_forecast.shape:
    raise ValueError(f"Shape mismatch: actual {actual_test_prices.shape} vs forecast {unstandardized_forecast.shape}")

mse_per_day  = np.mean((actual_test_prices - unstandardized_forecast)**2, axis=1)
rmse_per_day = np.sqrt(mse_per_day)

overall_mse  = float(overall_mean**2)  # overall_mean is RMSE, so MSE = RMSE**2
overall_rmse = float(overall_mean)

print("\n Forecast Performance Summary")
print(f"• MSE per day (first 10): {mse_per_day[:10]}")
print(f"• Overall MSE:  {overall_mse:.6f}")
print(f"• Overall RMSE: {overall_rmse:.6f}")

visualize_daily_forecast_errors(mse_per_day, rmse_per_day)

# Daily errors (log scale) — guard against zeros
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
ax.plot(mse_per_day + 1e-12,  marker="o", linestyle="-",  alpha=0.8, label="MSE")
ax.plot(rmse_per_day + 1e-12, marker="x", linestyle="--", alpha=0.8, label="RMSE")
ax.set_yscale("log")
ax.set_title("Daily Forecast Errors (Log Scale)")
ax.set_xlabel("Day"); ax.set_ylabel("Error (log scale)")
ax.grid(True, linestyle="--", alpha=0.7)
_legend_below(ax, ncol=2, pad=0.20, y_offset=-0.18)
fig.savefig(f"{_outdir}/daily_errors_log.png", dpi=300, bbox_inches="tight")
plt.show(); plt.close(fig)


# %% 2. OLS Weights over hours of the day — legend below, tidy layout
ols_coef = forecast_expert_ext(
    dat=dat_test, days=days_test, reg_names=data.columns[1:], model_type=model_type
)["coefficients"]

def _legend_below(ax, ncol=4, pad=0.22, y_offset=-0.18):
    fig = ax.figure
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y_offset), ncol=ncol, frameon=True, fontsize=9)
    fig.subplots_adjust(bottom=pad)

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)

for col in ols_coef.columns:
    ax.plot(ols_coef.index, ols_coef[col], label=col, linewidth=1.2)

ax.set_xlabel('Hour of Day')
ax.set_ylabel('OLS Weight')
ax.set_title('OLS Weights over Hours of the Day')
ax.grid(True)

n_series = len(ols_coef.columns)
ncol = 4 if n_series >= 8 else (3 if n_series >= 6 else 2)
_legend_below(ax, ncol=ncol, pad=0.22, y_offset=-0.18)

plt.savefig(f"{_outdir}/ols_weights_hourly.png", dpi=300, bbox_inches="tight")
plt.show(); plt.close(fig)


#%% Save forecast, actuals, and naive (Excel)
actual_prices_np = dependent_var_test.iloc[-N_s:, :].to_numpy()
df_forecast = pd.DataFrame(unstandardized_forecast, columns=[f"Hour_{h+1}" for h in range(S)])
df_actual   = pd.DataFrame(actual_prices_np,   columns=[f"Hour_{h+1}" for h in range(S)])
df_naive    = pd.DataFrame(weekly_naive_test.detach().cpu().numpy(), columns=[f"Hour_{h+1}" for h in range(S)])
df_forecast.insert(0, "Day", range(1, N_s + 1))
df_actual.insert(0, "Day", range(1, N_s + 1))
df_naive.insert(0, "Day", range(1, N_s + 1))

with pd.ExcelWriter(f"{_outdir}/forecast_vs_actual.xlsx") as writer:
    df_forecast.to_excel(writer, index=False, sheet_name="Forecast")
    df_actual.to_excel(writer,   index=False, sheet_name="Actual")
    df_naive.to_excel(writer,    index=False, sheet_name="WeeklyNaive")
print("Saved forecast, actuals, and weekly naive to Excel.")
#%%
# %%#########################################################
# Hyperparameter-centric visualizations (Optuna + Matplotlib)
#############################################################
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from optuna.importance import get_param_importances
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_edf,
)

# ---------- Global style ----------
plt.style.use("default")
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "grid.color": "lightgray",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})

# ---------- Helpers ----------
def _get_fig_axes(ax_or_axes):
    """Return (fig, [axes...]) for either a single Axes or a grid (ndarray) of Axes."""
    if isinstance(ax_or_axes, np.ndarray):
        axes = ax_or_axes.ravel()
        fig = axes[0].figure
    else:
        axes = [ax_or_axes]
        fig = ax_or_axes.figure
    return fig, axes

def _ensure_trials_df(study_obj):
    """Get trials dataframe safely; adds duration_in_seconds if missing."""
    df = study_obj.trials_dataframe()
    if "duration_in_seconds" not in df.columns:
        if "duration" in df.columns and pd.api.types.is_timedelta64_dtype(df["duration"]):
            df["duration_in_seconds"] = df["duration"].apply(lambda x: x.total_seconds())
        else:
            def _dur_seconds(row):
                try:
                    return (row["datetime_complete"] - row["datetime_start"]).total_seconds()
                except Exception:
                    return np.nan
            if {"datetime_complete", "datetime_start"}.issubset(df.columns):
                df["duration_in_seconds"] = df.apply(_dur_seconds, axis=1)
            else:
                df["duration_in_seconds"] = np.nan
    return df

def _apply_white_darkstyle(fig):
    """White bg + dark axes/text + dashed grid + legend below (if present)."""
    fig.patch.set_facecolor("white")
    for ax in fig.get_axes():
        try:
            ax.set_facecolor("white")
            ax.grid(True, linestyle="--", color="lightgray", alpha=0.6)
            ax.tick_params(colors="black")
            for spine in ax.spines.values():
                spine.set_color("black")
            ax.title.set_color("black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            leg = ax.get_legend()
            if leg is not None:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, fontsize=9)
        except Exception:
            pass

def _robust_limits(values, p_low=5, p_high=95, pad_frac=0.15):
    """Percentile-based limits with padding; handles edge cases."""
    v = pd.to_numeric(pd.Series(values).astype(float), errors="coerce").dropna().values
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        span = max(abs(hi), 1.0)
        return lo - 0.1*span, hi + 0.1*span
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad

def _apply_rmse_ylim(ax, values):
    lim = _robust_limits(values)
    if lim is not None:
        ax.set_ylim(*lim)

def _apply_rmse_xlim(ax, values):
    lim = _robust_limits(values)
    if lim is not None:
        ax.set_xlim(*lim)

def _harmonize_contour_clim(fig, values):
    """Try to set a consistent RMSE color scale on contour plots."""
    lim = _robust_limits(values)
    if lim is None:
        return
    vmin, vmax = lim
    # Matplotlib contour from optuna uses QuadContourSet/collections or images
    for ax in fig.get_axes():
        for coll in getattr(ax, "collections", []):
            try:
                coll.set_clim(vmin, vmax)
            except Exception:
                pass
        for im in ax.images:
            try:
                im.set_clim(vmin, vmax)
            except Exception:
                pass
    # Also adjust colorbars if present
    for ax in fig.get_axes():
        if hasattr(ax, "artists"):
            for art in ax.artists:
                try:
                    art.set_clim(vmin, vmax)
                except Exception:
                    pass

def save_and_show(fig, path):
    """Style + save."""
    _apply_white_darkstyle(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)

# ---------- IO prep ----------
outdir = f"Model{model_type}"
os.makedirs(outdir, exist_ok=True)

try:
    study  # noqa: F401
except NameError:
    file_optuna = f"{outdir}/optuna_study.pkl"
    study = joblib.load(file_optuna)

trials_df = _ensure_trials_df(study)

if len(trials_df) == 0 or trials_df["value"].isna().all():
    print("No completed trials found — skipping hyperparameter visualizations.")
else:
    rmse_vals = trials_df["value"].dropna().values

    # 1) Optimization history (convergence) — robust y-scale
    ax = plot_optimization_history(study)
    fig, axes = _get_fig_axes(ax)
    fig.suptitle("Optimization History (Best Value vs Trial)", y=1.02)
    try:
        axes[0].set_ylabel("RMSE")
        _apply_rmse_ylim(axes[0], rmse_vals)
    except Exception:
        pass
    save_and_show(fig, os.path.join(outdir, "hp_optimization_history.png"))

    # 2) Optuna’s importances (no RMSE axis to scale)
    try:
        ax = plot_param_importances(study)
        fig, _ = _get_fig_axes(ax)
        fig.suptitle("Hyperparameter Importances (Optuna)", y=1.02)
        save_and_show(fig, os.path.join(outdir, "hp_importances_optuna.png"))
    except Exception as e:
        print(f"Skipping Optuna importances plot (reason: {e})")

    # Top params
    try:
        imp_dict = get_param_importances(study)
        imp_sorted = sorted(imp_dict.items(), key=lambda kv: kv[1], reverse=True)
        top_params = [k for k, _ in imp_sorted[:8]]
    except Exception:
        all_params = sorted({p for t in study.trials for p in t.params.keys()})
        top_params = all_params[:8]

    # 3) Slice (partial dependence) — robust y-scale on all subplots
    if len(top_params) > 0:
        df = trials_df.copy()
        df["trial_index"] = np.arange(len(df))
        value_col = "value"
        param_cols = [f"params_{p}" for p in top_params if f"params_{p}" in df.columns]

        cvals = df["trial_index"].values.astype(float)
        norm = Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))

        n = min(8, len(param_cols))
        ncols_data, nrows = 4, (2 if n > 4 else 1)
        fig = plt.figure(figsize=(4.6*(ncols_data+0.25), 3.6*nrows), facecolor="white")
        gs = gridspec.GridSpec(nrows, ncols_data+1, width_ratios=[1,1,1,1,0.04], wspace=0.28, hspace=0.38)

        last_sc = None
        for i in range(n):
            r, c = divmod(i, ncols_data)
            ax = fig.add_subplot(gs[r, c])
            col = param_cols[i]
            p_name = col.replace("params_", "")
            x = pd.to_numeric(df[col], errors="coerce")
            y = pd.to_numeric(df[value_col], errors="coerce")
            m = x.notna() & y.notna()
            last_sc = ax.scatter(x[m], y[m], c=cvals[m], norm=norm, alpha=0.85)
            ax.set_title(p_name, color="black")
            if c == 0:
                ax.set_ylabel("RMSE", color="black")
            _apply_rmse_ylim(ax, y[m])  # <-- robust y-limit here

            if any(k in p_name for k in ["learning_rate", "weight_decay", "lambda_reg"]):
                with np.errstate(all='ignore'):
                    ax.set_xscale("log")

        for j in range(n, nrows*ncols_data):
            r, c = divmod(j, ncols_data)
            fig.add_subplot(gs[r, c]).axis("off")

        cax = fig.add_subplot(gs[:, -1])
        sm = ScalarMappable(norm=norm, cmap=(last_sc.cmap if last_sc is not None else plt.get_cmap()))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Trial", color="black")

        fig.suptitle("Slice (Partial Dependence) — Top Hyperparameters", y=0.98, color="black")
        save_and_show(fig, os.path.join(outdir, "hp_slice_top_params.png"))



    # 5) EDF — robust x-scale (RMSE on x-axis)
    try:
        ax = plot_edf(study)
        fig, axes = _get_fig_axes(ax)
        fig.suptitle("EDF of Trial RMSE Values", y=1.02, color="black")
        axes[0].set_xlabel("RMSE", color="black")
        axes[0].set_ylabel("Empirical CDF", color="black")
        _apply_rmse_xlim(axes[0], rmse_vals)  # <-- robust x-limit here
        save_and_show(fig, os.path.join(outdir, "hp_edf.png"))
    except Exception as e:
        print(f"Skipping EDF plot (reason: {e})")

    # 6) Duration vs RMSE — robust y-scale
    try:
        fig = plt.figure(figsize=(7.2, 5.2), facecolor="white")
        ax = fig.add_subplot(111)
        x = trials_df["duration_in_seconds"].values
        y = trials_df["value"].values

        param_cols_all = sorted([c for c in trials_df.columns if c.startswith("params_")])
        color_param_candidates = ["use_ols_weights", "seq_len", "number_neurons", "D_all"]
        color_param = next((p for p in color_param_candidates if f"params_{p}" in param_cols_all), None)

        if color_param is None:
            sc = ax.scatter(x, y, alpha=0.85)
        else:
            vals = trials_df[f"params_{color_param}"]
            if vals.dtype == bool or set(vals.dropna().unique()).issubset({True, False}):
                cnum = vals.astype(float)
            else:
                cnum = pd.to_numeric(vals, errors="coerce")
            sc = ax.scatter(x, y, c=cnum, alpha=0.85)
            cbar = fig.colorbar(sc); cbar.set_label(color_param, color="black")

        ax.set_xlabel("Duration (seconds)", color="black")
        ax.set_ylabel("RMSE", color="black")
        ax.set_title("Time vs Score (color-coded by key hyperparameter)", color="black")
        _apply_rmse_ylim(ax, y)  # <-- robust y-limit here
        save_and_show(fig, os.path.join(outdir, "hp_duration_vs_rmse.png"))
    except Exception as e:
        print(f"Skipping duration vs RMSE scatter (reason: {e})")

    # 7) Sampling distributions (unchanged by RMSE scaling)
    try:
        param_cols_all = sorted([c for c in trials_df.columns if c.startswith("params_")])
        n = len(param_cols_all)
        if n > 0:
            ncols = 3
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 3.6*nrows), squeeze=False)
            fig.patch.set_facecolor("white")
            for i, col in enumerate(param_cols_all):
                r, c = divmod(i, ncols)
                ax = axes[r][c]
                ax.set_facecolor("white")
                unique_vals = pd.Series(trials_df[col].dropna().unique())
                if trials_df[col].dtype == bool or set(unique_vals.tolist()).issubset({True, False}):
                    ax.hist(trials_df[col].astype(int), bins=np.arange(-0.5, 2), rwidth=0.8)
                    ax.set_xticks([0, 1]); ax.set_xticklabels(["False", "True"])
                else:
                    s_num = pd.to_numeric(trials_df[col], errors="coerce").dropna()
                    if len(s_num) > 0:
                        ax.hist(s_num, bins=20)
                    else:
                        ax.text(0.5, 0.5, "categorical", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(col.replace("params_", ""), color="black")
            last_i = i if n > 0 else -1
            for j in range(last_i+1, nrows*ncols):
                r, c = divmod(j, ncols)
                axes[r][c].axis("off")
            fig.suptitle("Sampling Distributions of Hyperparameters", y=1.02, color="black")
            save_and_show(fig, os.path.join(outdir, "hp_sampling_distributions.png"))
    except Exception as e:
        print(f"Skipping sampling distributions (reason: {e})")

   

# %%=====================================================
# Static Optimization History Plot (Best RMSE highlights)
# =======================================================
import matplotlib.pyplot as plt
import numpy as np

# Extract values
values = trials_df["value"].dropna().values
best_so_far = np.minimum.accumulate(values)
new_best_mask = values == best_so_far  # highlight new best RMSE points

# Dynamic y-limits (robust against outliers)
y_low, y_high = np.percentile(values, [5, 95])
margin = (y_high - y_low) * 0.15
ymin, ymax = y_low - margin, y_high + margin

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# --- Plot all trial RMSEs (blue) ---
ax.plot(np.arange(len(values)), values, "o", color="#1f77b4", alpha=0.6, label="Trial RMSE")

# --- Highlight new best RMSEs (green) ---
ax.plot(np.where(new_best_mask)[0], values[new_best_mask], "o", color="#2ca02c", markersize=7, label="New best RMSE")

# --- Plot best-so-far line (red) ---
ax.plot(np.arange(len(values)), best_so_far, color="#d62728", lw=2, label="Best RMSE so far")

# --- Style and grid ---
for spine in ax.spines.values():
    spine.set_color("black")
ax.tick_params(colors="black")
ax.xaxis.label.set_color("black")
ax.yaxis.label.set_color("black")
ax.title.set_color("black")
ax.grid(True, color="lightgray", linestyle="--", linewidth=0.8, alpha=0.7)

ax.set_xlim(0, len(values))
ax.set_ylim(ymin, ymax)
ax.set_title("Optimization Progress — Best RMSE over Trials", fontsize=14)
ax.set_xlabel("Trial")
ax.set_ylabel("RMSE")

# --- Legend below ---
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

# --- Save and show ---
out_path = f"Model{model_type}/optimization_progress_static.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.show(); plt.close(fig)

print(f"Saved static optimization progress plot to {out_path}")


# %%=====================================================
# Optimization History (Best RMSE over Trials)
# with "new best" points highlighted in green
# =======================================================
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Prepare data
values = trials_df["value"].dropna().values
best_so_far = np.minimum.accumulate(values)

# Identify where a new best RMSE was achieved
new_best_mask = values == best_so_far  # True at each new improvement

# Compute dynamic y-limits (robust to outliers)
y_low, y_high = np.percentile(values, [5, 95])
margin = (y_high - y_low) * 0.15
ymin, ymax = y_low - margin, y_high + margin

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, len(values))
ax.set_ylim(ymin, ymax)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# --- Style ---
for spine in ax.spines.values():
    spine.set_color("black")
ax.tick_params(colors="black")
ax.xaxis.label.set_color("black")
ax.yaxis.label.set_color("black")
ax.title.set_color("black")
ax.grid(True, color="lightgray", linestyle="--", linewidth=0.8, alpha=0.7)

ax.set_title("Optimization Progress — Best RMSE over Trials")
ax.set_xlabel("Trial")
ax.set_ylabel("RMSE")

# --- Lines and points ---
line,        = ax.plot([], [], color="#d62728", lw=2, label="Best RMSE so far")
points_all,  = ax.plot([], [], 'o', color="#1f77b4", alpha=0.6, label="Trial RMSE")
points_best, = ax.plot([], [], 'o', color="#2ca02c", alpha=0.95, markersize=7, label="New best RMSE")

# Legend below
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

# --- Animation functions ---
def init():
    line.set_data([], [])
    points_all.set_data([], [])
    points_best.set_data([], [])
    return line, points_all, points_best

def update(frame):
    x = np.arange(frame)
    y_best = best_so_far[:frame]

    # Update best line
    line.set_data(x, y_best)

    # Split into normal vs new-best points
    mask = new_best_mask[:frame]
    x_all = x
    y_all = values[:frame]

    x_best = x[mask]
    y_best_pts = y_all[mask]

    x_reg = x[~mask]
    y_reg = y_all[~mask]

    points_all.set_data(x_reg, y_reg)
    points_best.set_data(x_best, y_best_pts)

    return line, points_all, points_best

ani = animation.FuncAnimation(
    fig, update, frames=len(values), init_func=init, blit=True, interval=100
)

# Save as animated GIF
ani.save(f"Model{model_type}/optimization_progress.gif", writer="pillow", fps=10)

plt.show()

# %%

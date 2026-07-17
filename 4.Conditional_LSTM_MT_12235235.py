# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:44:44 2026

@author: Shringi Vaibhav (12235235)
"""

# %% 1. Imports
from pathlib import Path
import json
import random
import re
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


# %% 2. Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Follo_Finalv2.parquet"
RESULTS_DIR = BASE_DIR / "results_conditional_lstm_test"
RESULTS_DIR.mkdir(exist_ok=True)

CHAINAGE_COLUMN = "Station_meter"

STANDSTILL_COLUMNS = [
    "CH Penetration [mm/rot]",
    "CH Torque [MNm]",
    "Thrust Force [kN]",
    "CH Rotation [rpm]",
]

TARGET_COLUMNS = STANDSTILL_COLUMNS

BLOCK_START = 5541
BLOCK_END = 5545
BLOCKS = [(BLOCK_START, BLOCK_END)]
GAP_LENGTH = BLOCK_END - BLOCK_START + 1

# Training length is the total training station range around the test gap.
TRAINING_LENGTH = 600

INPUT_LEN = 50
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.1
BATCH_SIZE = 128
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-6
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)


# %% 3. Helpers
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", text)


def eta_value() -> float:
    return round(TRAINING_LENGTH / GAP_LENGTH, 2)


def run_id(target_col: str) -> str:
    return (
        f"Target_{clean_filename(target_col)}_"
        f"BlockStart{BLOCK_START}_"
        f"Train{TRAINING_LENGTH}_"
        f"Gap{GAP_LENGTH}_"
        f"Eta{eta_value()}"
    )


def make_working_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the useful local area.

    We split TRAINING_LENGTH approximately half before and half after the gap.
    Extra margin is added so candidate artificial gaps still have left/right context.
    """
    half_train = TRAINING_LENGTH / 2.0
    margin = INPUT_LEN + GAP_LENGTH + 5

    min_station = BLOCK_START - half_train - margin
    max_station = BLOCK_END + half_train + margin

    df_work = df.loc[
        (df[CHAINAGE_COLUMN] >= min_station)
        & (df[CHAINAGE_COLUMN] <= max_station)
    ].copy()

    df_work = df_work.sort_values(CHAINAGE_COLUMN).reset_index(drop=True)

    if df_work.empty:
        raise RuntimeError("Working dataframe is empty. Check block and training length.")

    return df_work


def allowed_training_mask(stations: np.ndarray) -> np.ndarray:
    half_train = TRAINING_LENGTH / 2.0

    return (
        (stations >= BLOCK_START - half_train)
        & (stations <= BLOCK_END + half_train)
    )


def actual_gap_mask(stations: np.ndarray) -> np.ndarray:
    return (stations >= BLOCK_START) & (stations <= BLOCK_END)


# %% 4. Load data
df_full = pd.read_parquet(DATA_FILE, engine="pyarrow")
df_full = df_full.sort_values(CHAINAGE_COLUMN).reset_index(drop=True)

missing_cols = [c for c in [CHAINAGE_COLUMN] + STANDSTILL_COLUMNS if c not in df_full.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df_work = make_working_dataframe(df_full)

print(f"Loaded: {DATA_FILE}")
print(f"Full shape: {df_full.shape}")
print(f"Working shape: {df_work.shape}")
print(f"Working station range: {df_work[CHAINAGE_COLUMN].min()} to {df_work[CHAINAGE_COLUMN].max()}")
print(f"Gap: {BLOCK_START}-{BLOCK_END}, length={GAP_LENGTH}")
print(f"Training length: {TRAINING_LENGTH}, eta={eta_value()}")
print(f"Device: {DEVICE}")


# %% 5. Sequence builder
def build_conditional_sequences(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    covariate_cols = [c for c in STANDSTILL_COLUMNS if c != target_col]

    stations = df[CHAINAGE_COLUMN].values
    all_vals = df[STANDSTILL_COLUMNS].values.astype(float)
    gap_cov_vals = df[covariate_cols].values.astype(float)
    y_vals = df[target_col].values.astype(float)

    allowed = allowed_training_mask(stations)
    in_actual_gap = actual_gap_mask(stations)

    left_list = []
    right_list = []
    gap_cov_list = []
    y_list = []

    first_gap_start = INPUT_LEN
    last_gap_start = len(df) - INPUT_LEN - GAP_LENGTH

    for gap_start in range(first_gap_start, last_gap_start + 1):
        gap_end_exclusive = gap_start + GAP_LENGTH
        gap_idx = np.arange(gap_start, gap_end_exclusive)
        full_window_idx = np.arange(gap_start - INPUT_LEN, gap_end_exclusive + INPUT_LEN)

        center_idx = gap_start + GAP_LENGTH // 2

        if not allowed[center_idx]:
            continue

        # Avoid leaking the actual test gap into training.
        if in_actual_gap[full_window_idx].any():
            continue

        left_context = all_vals[gap_start - INPUT_LEN:gap_start]
        right_context = all_vals[gap_end_exclusive:gap_end_exclusive + INPUT_LEN][::-1]
        gap_covariates = gap_cov_vals[gap_idx]
        y_target = y_vals[gap_idx]

        if (
            np.isnan(left_context).any()
            or np.isnan(right_context).any()
            or np.isnan(gap_covariates).any()
            or np.isnan(y_target).any()
        ):
            continue

        left_list.append(left_context)
        right_list.append(right_context)
        gap_cov_list.append(gap_covariates)
        y_list.append(y_target)

    if not left_list:
        raise RuntimeError("No valid training sequences. Increase TRAINING_LENGTH or reduce INPUT_LEN.")

    return (
        np.asarray(left_list, dtype=np.float32),
        np.asarray(right_list, dtype=np.float32),
        np.asarray(gap_cov_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
    )


# %% 6. Model
class ConditionalGapLSTM(nn.Module):
    def __init__(
        self,
        context_input_size: int,
        gap_cov_input_size: int,
        hidden_size: int,
        output_len: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.left_encoder = nn.LSTM(
            input_size=context_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.right_encoder = nn.LSTM(
            input_size=context_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.gap_encoder = nn.LSTM(
            input_size=gap_cov_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_len),
        )

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        x_gap_cov: torch.Tensor,
    ) -> torch.Tensor:

        _, (h_left, _) = self.left_encoder(x_left)
        _, (h_right, _) = self.right_encoder(x_right)
        _, (h_gap, _) = self.gap_encoder(x_gap_cov)

        h = torch.cat(
            [
                h_left[-1],
                h_right[-1],
                h_gap[-1],
            ],
            dim=1,
        )

        out = self.head(h)
        return out.unsqueeze(-1)


# %% 7. Training
def train_conditional_model(target_col: str):
    left_arr, right_arr, gap_cov_arr, y_arr = build_conditional_sequences(df_work, target_col)

    n_samples, _, n_context_features = left_arr.shape
    _, _, n_gap_features = gap_cov_arr.shape

    scaler_context = MinMaxScaler()
    scaler_gap = MinMaxScaler()
    scaler_y = MinMaxScaler()

    context_flat = np.concatenate(
        [
            left_arr.reshape(-1, n_context_features),
            right_arr.reshape(-1, n_context_features),
        ],
        axis=0,
    )

    gap_flat = gap_cov_arr.reshape(-1, n_gap_features)
    y_flat = y_arr.reshape(-1, 1)

    scaler_context.fit(context_flat)
    scaler_gap.fit(gap_flat)
    scaler_y.fit(y_flat)

    left_scaled = scaler_context.transform(
        left_arr.reshape(-1, n_context_features)
    ).reshape(left_arr.shape).astype(np.float32)

    right_scaled = scaler_context.transform(
        right_arr.reshape(-1, n_context_features)
    ).reshape(right_arr.shape).astype(np.float32)

    gap_cov_scaled = scaler_gap.transform(
        gap_cov_arr.reshape(-1, n_gap_features)
    ).reshape(gap_cov_arr.shape).astype(np.float32)

    y_scaled = scaler_y.transform(y_flat).reshape(n_samples, GAP_LENGTH, 1).astype(np.float32)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(left_scaled),
        torch.tensor(right_scaled),
        torch.tensor(gap_cov_scaled),
        torch.tensor(y_scaled),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = ConditionalGapLSTM(
        context_input_size=n_context_features,
        gap_cov_input_size=n_gap_features,
        hidden_size=HIDDEN_SIZE,
        output_len=GAP_LENGTH,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for xb_left, xb_right, xb_gap, yb in loader:
            xb_left = xb_left.to(DEVICE).float()
            xb_right = xb_right.to(DEVICE).float()
            xb_gap = xb_gap.to(DEVICE).float()
            yb = yb.to(DEVICE).float()

            optimizer.zero_grad()
            pred = model(xb_left, xb_right, xb_gap)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"{target_col} | epoch {epoch:03d}/{EPOCHS} | loss={np.mean(epoch_losses):.6f}")

    return model, scaler_context, scaler_gap, scaler_y, n_samples


# %% 8. Prediction
def predict_target_gap(
    target_col: str,
    model: ConditionalGapLSTM,
    scaler_context: MinMaxScaler,
    scaler_gap: MinMaxScaler,
    scaler_y: MinMaxScaler,
) -> pd.DataFrame:

    covariate_cols = [c for c in STANDSTILL_COLUMNS if c != target_col]

    df_pred = df_work.copy()

    stations = df_pred[CHAINAGE_COLUMN].values
    gap_idx = np.where(actual_gap_mask(stations))[0]

    if len(gap_idx) != GAP_LENGTH:
        raise RuntimeError(f"Expected {GAP_LENGTH} gap rows, found {len(gap_idx)}.")

    start_idx = gap_idx.min()
    end_idx = gap_idx.max()

    left_start = start_idx - INPUT_LEN
    left_end = start_idx

    right_start = end_idx + 1
    right_end = end_idx + 1 + INPUT_LEN

    if left_start < 0 or right_end > len(df_pred):
        raise RuntimeError("Not enough context in working dataframe. Increase local margin or reduce INPUT_LEN.")

    # Hide only the target parameter in the actual gap.
    df_pred.loc[gap_idx, target_col] = np.nan

    left_context = df_pred.loc[left_start:left_end - 1, STANDSTILL_COLUMNS].values.astype(float)
    right_context = df_pred.loc[right_start:right_end - 1, STANDSTILL_COLUMNS].values.astype(float)[::-1]
    gap_covariates = df_pred.loc[gap_idx, covariate_cols].values.astype(float)

    if np.isnan(left_context).any() or np.isnan(right_context).any() or np.isnan(gap_covariates).any():
        raise RuntimeError("Prediction inputs contain NaNs. Check gap/context definition.")

    left_scaled = scaler_context.transform(left_context).astype(np.float32)
    right_scaled = scaler_context.transform(right_context).astype(np.float32)
    gap_scaled = scaler_gap.transform(gap_covariates).astype(np.float32)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(
            torch.tensor(left_scaled).unsqueeze(0).to(DEVICE),
            torch.tensor(right_scaled).unsqueeze(0).to(DEVICE),
            torch.tensor(gap_scaled).unsqueeze(0).to(DEVICE),
        ).cpu().numpy()[0]

    pred = scaler_y.inverse_transform(pred_scaled)[:, 0]

    df_pred.loc[gap_idx, target_col] = pred

    return df_pred


# %% 9. Metrics and plotting
def compute_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame, target_col: str) -> Dict[str, float]:
    stations = df_true[CHAINAGE_COLUMN].values
    mask = actual_gap_mask(stations)

    y_true = df_true.loc[mask, target_col].values.astype(float)
    y_pred = df_pred.loc[mask, target_col].values.astype(float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_pred) > 0 else np.nan
    deriv_rmse = float(np.sqrt(np.mean((np.diff(y_true) - np.diff(y_pred)) ** 2))) if len(y_true) > 1 else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "deriv_rmse": deriv_rmse,
    }


def save_outputs(
    target_col: str,
    df_pred: pd.DataFrame,
    metrics: Dict[str, float],
    train_sequences: int,
) -> None:

    stations = df_work[CHAINAGE_COLUMN].values
    gap_mask = actual_gap_mask(stations)

    comparison = pd.DataFrame(
        {
            CHAINAGE_COLUMN: df_work.loc[gap_mask, CHAINAGE_COLUMN].values,
            "Ground Truth": df_work.loc[gap_mask, target_col].values,
            "Conditional LSTM": df_pred.loc[gap_mask, target_col].values,
        }
    )

    comparison_file = RESULTS_DIR / f"reconstruction_values_{run_id(target_col)}.csv"
    comparison.to_csv(comparison_file, index=False)

    start_idx = np.where(gap_mask)[0].min()
    end_idx = np.where(gap_mask)[0].max()

    plot_start = max(0, start_idx - GAP_LENGTH)
    plot_end = min(len(df_work) - 1, end_idx + GAP_LENGTH)

    x = df_work.loc[plot_start:plot_end, CHAINAGE_COLUMN].values

#    label = f"RMSE{metrics['rmse']:.2f}_Corr{metrics['corr']:.2f}"

    plt.figure(figsize=(12, 5))
    plt.plot(
        x,
        df_work.loc[plot_start:plot_end, target_col].values,
        label="Ground truth",
        linewidth=2,
        )

    # Only show the model prediction inside the artificial gap.
    pred_plot = np.full(len(x), np.nan)
    gap_plot_mask = (x >= BLOCK_START) & (x <= BLOCK_END)
    pred_plot[gap_plot_mask] = df_pred.loc[plot_start:plot_end, target_col].values[gap_plot_mask]

    plt.plot(
        x,
        pred_plot,
        "--",
        label="Conditional LSTM",
        linewidth=2,
        )

    plt.axvspan(BLOCK_START, BLOCK_END, alpha=0.18, color="gray", label="Artificial gap")
    plt.xlabel(CHAINAGE_COLUMN)
    plt.ylabel(target_col)
    plt.title(
        f"Training length = {TRAINING_LENGTH}, "
        f"Gap length = {GAP_LENGTH}, "
        f"Eta = {eta_value()}"
    )
    plt.legend()
    plt.tight_layout()

    plot_file = RESULTS_DIR / f"reconstruction_{run_id(target_col)}.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()

    summary_row = {
        "target": target_col,
        "block_start": BLOCK_START,
        "training_length": TRAINING_LENGTH,
        "gap_length": GAP_LENGTH,
        "eta": eta_value(),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "corr": metrics["corr"],
        "deriv_rmse": metrics["deriv_rmse"],
    }

    summary_file = RESULTS_DIR / "conditional_lstm_summary_compact.csv"

    if summary_file.exists():
        old = pd.read_csv(summary_file)
        summary_df = pd.concat([old, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])

    summary_df.to_csv(summary_file, index=False)

    config_file = RESULTS_DIR / f"config_{clean_filename(target_col)}.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(summary_row, f, indent=2)

    print("\nResult:")
    print(pd.DataFrame([summary_row]).to_string(index=False))
    print(f"\nSaved comparison: {comparison_file}")
    print(f"Saved plot: {plot_file}")
    print(f"Saved summary: {summary_file}")
    print(f"Saved config: {config_file}")


# %% 10. Run
set_global_seed(SEED)

for target_col in TARGET_COLUMNS:
    print(f"\n===== Conditional LSTM reconstruction: {target_col} =====")

    model, scaler_context, scaler_gap, scaler_y, train_sequences = train_conditional_model(target_col)

    df_pred = predict_target_gap(
        target_col=target_col,
        model=model,
        scaler_context=scaler_context,
        scaler_gap=scaler_gap,
        scaler_y=scaler_y,
    )

    metrics = compute_metrics(df_work, df_pred, target_col)

    save_outputs(
        target_col=target_col,
        df_pred=df_pred,
        metrics=metrics,
        train_sequences=train_sequences,
    )
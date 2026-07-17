# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 14:12:19 2025

@author: Shringi Vaibhav (12235235)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# %% User inputs
df_final = pd.read_parquet("Follo_Finalv2.parquet", engine="pyarrow")

chainage_column = "Station_meter"

standstill_columns = [
    "CH Penetration [mm/rot]",
    "CH Torque [MNm]",
    "Thrust Force [kN]",
    "CH Rotation [rpm]",
]

blocks = [(5541, 5545)]
a, b = blocks[0]
gap_length = b - a + 1

window = 20
margin = window

results_dir = Path("results_linear_regression")
results_dir.mkdir(exist_ok=True)

summary_file = results_dir / "linear_regression_summary.csv"


# %% Helpers
def clean_col_name(col):
    return (
        col.replace("/", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "_")
    )


def compute_gap_metrics(signal):
    signal = np.array(signal, dtype=float)

    std = np.std(signal)

    diff = np.diff(signal)
    diff_std = np.std(diff) if len(diff) > 0 else np.nan

    return {
        "standard_deviation": std,
        "diff_std": diff_std,
    }


def run_lr_reconstruction(df_local, df_gap_local, col):
    x = df_local[[chainage_column]].values
    y_gap = df_gap_local[col].values.copy()

    train_mask = ~np.isnan(y_gap)

    model = LinearRegression()
    model.fit(x[train_mask], y_gap[train_mask])

    y_pred = y_gap.copy()
    gap_mask = np.isnan(y_pred)

    y_pred[gap_mask] = model.predict(x[gap_mask])

    return y_pred


def compute_metrics_local(df_local, pred_series, col, a, b):
    station = df_local[chainage_column].values
    mask = (station >= a) & (station <= b)

    true_vals = df_local[col].values[mask].astype(float)
    pred_vals = pred_series[mask].astype(float)

    valid = ~np.isnan(pred_vals)

    true_vals = true_vals[valid]
    pred_vals = pred_vals[valid]

    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    mae = np.mean(np.abs(true_vals - pred_vals))

    if len(true_vals) > 1 and np.std(pred_vals) > 0:
        corr = np.corrcoef(true_vals, pred_vals)[0, 1]
    else:
        corr = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "corr": corr,
    }


def plot_results(df_local, series_pred, col, a, b, window):
    x = df_local[chainage_column].values
    y_true = df_local[col].values

    plt.figure(figsize=(12, 5))

    plt.plot(
        x,
        y_true,
        label="Ground Truth",
        linewidth=2,
    )

    # Only show LR prediction inside the artificial gap.
    y_pred_gap_only = np.full_like(series_pred, np.nan, dtype=float)
    gap_mask_plot = (x >= a) & (x <= b)
    y_pred_gap_only[gap_mask_plot] = series_pred[gap_mask_plot]

    plt.plot(
        x,
        y_pred_gap_only,
        label="LR Prediction",
        linestyle="--",
        linewidth=2,
    )

    plt.axvspan(a, b, alpha=0.25, label="Gap")

    plt.axvspan(
        a - window,
        a,
        alpha=0.1,
        color="green",
        label="Training Window",
    )

    plt.axvspan(
        b,
        b + window,
        alpha=0.1,
        color="green",
    )

    plt.xlabel("Chainage [m]")
    plt.ylabel(col)
    plt.legend()
    plt.title(
        f"{col} - Linear Regression | "
        f"Block start = {a}, Gap length = {gap_length}"
    )

    plt.tight_layout()

    plot_file = (
        results_dir
        / f"linear_regression_{clean_col_name(col)}"
          f"_BlockStart{a}_Gap{gap_length}.png"
    )

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()


def append_result_to_csv(row, summary_file):
    row_df = pd.DataFrame([row])

    if summary_file.exists():
        row_df.to_csv(summary_file, mode="a", header=False, index=False)
    else:
        row_df.to_csv(summary_file, index=False)


# %% Gap complexity metrics
gap_data = df_final.loc[
    (df_final[chainage_column] >= a)
    & (df_final[chainage_column] <= b),
    standstill_columns,
]

metrics_per_param = {}

for col in standstill_columns:
    metrics_per_param[col] = compute_gap_metrics(gap_data[col])


# %% Create artificial gap
df_original = df_final.copy()
df_gap = df_final.copy()

for col in standstill_columns:
    df_gap.loc[
        (df_gap[chainage_column] >= a)
        & (df_gap[chainage_column] <= b),
        col,
    ] = np.nan


# %% Local dataframe for LR baseline
mask_local = (
    (df_original[chainage_column] >= a - margin)
    & (df_original[chainage_column] <= b + margin)
)

df_local = df_original.loc[mask_local].copy()
df_gap_local = df_gap.loc[mask_local].copy()


# %% Run reconstruction
results = {}

for col in standstill_columns:
    pred = run_lr_reconstruction(df_local, df_gap_local, col)

    eval_metrics = compute_metrics_local(df_local, pred, col, a, b)

    row = {
        "target": col,
        "block_start": a,
        "gap_length": gap_length,
        "RMSE": eval_metrics["RMSE"],
        "corr": eval_metrics["corr"],
    }

    append_result_to_csv(row, summary_file)

    results[col] = {
        "RMSE": eval_metrics["RMSE"],
        "MAE": eval_metrics["MAE"],
        "corr": eval_metrics["corr"],
        "standard_deviation": metrics_per_param[col]["standard_deviation"],
        "diff_std": metrics_per_param[col]["diff_std"],
    }

    plot_results(df_local, pred, col, a, b, window)

    print(f"Saved result for {col}")


results_df = pd.DataFrame(results).T

print("\nCurrent run results:")
print(results_df)

print("\nAppended results to:")
print(summary_file)

print("\nNaN count in last prediction:")
print(np.isnan(pred).sum())
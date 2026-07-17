# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:32:56 2025

@author: Shringi Vaibhav (12235235)
"""

from pathlib import Path
import numpy as np
import pandas as pd
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

results_dir = Path("results_linear_interpolation")
results_dir.mkdir(exist_ok=True)

summary_file = results_dir / "linear_interpolation_summary.csv"

# %% Gap complexity metrics
def compute_gap_metrics(signal):
    signal = np.array(signal, dtype=float)

    std = np.std(signal)

    diff = np.diff(signal)
    diff_std = np.std(diff) if len(diff) > 0 else np.nan

    return {
        "standard_deviation": std,
        "diff_std": diff_std,
    }


# %% Linear interpolation baseline
def run_linear_interpolation(df_gap, col):
    df_interp = df_gap.copy()

    df_interp[col] = df_interp[col].interpolate(
        method="linear",
        limit_direction="both",
    )

    return df_interp[col].values


def compute_metrics(df_original, pred_series, col, a, b):
    station = df_original[chainage_column].values
    mask = (station >= a) & (station <= b)

    true_vals = df_original[col].values[mask].astype(float)
    pred_vals = pred_series[mask].astype(float)

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


def plot_results(df_original, pred_series, col, a, b):
    station = df_original[chainage_column].values

    plot_mask = (
        (station >= a - 5)
        & (station <= b + 5)
    )

    x = station[plot_mask]
    y_true = df_original[col].values[plot_mask]
    y_pred = pred_series[plot_mask]

    y_pred_gap_only = np.full_like(y_pred, np.nan, dtype=float)
    gap_mask_plot = (x >= a) & (x <= b)
    y_pred_gap_only[gap_mask_plot] = y_pred[gap_mask_plot]

    plt.figure(figsize=(12, 5))

    plt.plot(
        x,
        y_true,
        label="Ground Truth",
        linewidth=2,
    )

    plt.plot(
        x,
        y_pred_gap_only,
        label="Linear Interpolation",
        linestyle="--",
        linewidth=2,
    )

    plt.axvspan(a, b, alpha=0.25, label="Gap")

    plt.xlabel("Chainage [m]")
    plt.ylabel(col)
    plt.legend()
    plt.title(
        f"{col} - Linear Interpolation | "
        f"Block start = {a}, Gap length = {gap_length}"
    )

    plt.tight_layout()

    plot_file = (
        results_dir
        / f"linear_interpolation_{col.replace('/', '_').replace('[', '').replace(']', '').replace(' ', '_')}"
          f"_BlockStart{a}_Gap{gap_length}.png"
    )

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
#    plt.show()

def append_result_to_csv(row, summary_file):
    row_df = pd.DataFrame([row])

    if summary_file.exists():
        row_df.to_csv(summary_file, mode="a", header=False, index=False)
    else:
        row_df.to_csv(summary_file, index=False)


# %% Prepare artificial gap
df_original = df_final.copy()
df_gap = df_final.copy()

for col in standstill_columns:
    df_gap.loc[
        (df_gap[chainage_column] >= a)
        & (df_gap[chainage_column] <= b),
        col,
    ] = np.nan


# %% Compute gap complexity metrics
gap_data = df_original.loc[
    (df_original[chainage_column] >= a)
    & (df_original[chainage_column] <= b),
    standstill_columns,
]

metrics_per_param = {}

for col in standstill_columns:
    metrics_per_param[col] = compute_gap_metrics(gap_data[col])


# %% Run reconstruction
results = {}

for col in standstill_columns:
    pred = run_linear_interpolation(df_gap, col)

    eval_metrics = compute_metrics(df_original, pred, col, a, b)

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

    plot_results(df_original, pred, col, a, b)

    print(f"Saved result for {col}")


results_df = pd.DataFrame(results).T

print("\nCurrent run results:")
print(results_df)

print("\nAppended results to:")
print(summary_file)
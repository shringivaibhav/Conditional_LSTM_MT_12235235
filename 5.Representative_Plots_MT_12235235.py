# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 08:47:15 2026

@author: Shringi Vaibhav (12235235)
"""

import numpy as np
import matplotlib.pyplot as plt

# X-axis (chainage)
chainage = [i for i in range(5536,5551)]

# Ground truth
ground_truth = [21427.3461755955,21350.0295855526,21007.2039628056,21212.4283865977,20278.1932634857,19971.2389773172,20196.7180462847,20674.9397149719,20218.7893459571,19572.7961936661,19875.4441905272,18997.0970003381,16975.1803356834,18390.776206099,16146.4074781148]

# Reconstructions
conditional_lstm = [20129.01171875,19978.14453125,20012.384765625,20082.19140625,20004.951171875]
linear_interpolation = [20211.1,20143.9,20076.8,20009.7,19942.6]
linear_regression = [19643,19555.9,19468.8,19381.7,19294.6]

# Artificial gap boundaries
BLOCK_START = 5541
BLOCK_END = 5545

# Values for the title
TRAINING_LENGTH = 1200
GAP_LENGTH = 5

# ==========================================================
# Convert to numpy arrays
# ==========================================================

chainage = np.array(chainage)
ground_truth = np.array(ground_truth)

conditional_lstm = np.array(conditional_lstm)
linear_interpolation = np.array(linear_interpolation)
linear_regression = np.array(linear_regression)

# ==========================================================
# Find the gap indices
# ==========================================================

gap_mask = (chainage >= BLOCK_START) & (chainage <= BLOCK_END)

# Create arrays full of NaN for plotting
lstm_plot = np.full(len(chainage), np.nan)
interp_plot = np.full(len(chainage), np.nan)
reg_plot = np.full(len(chainage), np.nan)

# Fill only the gap with the reconstruction values
lstm_plot[gap_mask] = conditional_lstm
interp_plot[gap_mask] = linear_interpolation
reg_plot[gap_mask] = linear_regression

# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(12, 5))

# Ground truth
plt.plot(
    chainage,
    ground_truth,
    label="Ground truth",
    linewidth=2,
)

# Conditional LSTM
plt.plot(
    chainage,
    lstm_plot,
    "--",
    linewidth=2,
    label="Conditional LSTM",
)

# Linear interpolation
plt.plot(
    chainage,
    interp_plot,
    "-.",
    linewidth=2,
    label="Linear interpolation",
)

# Linear regression
plt.plot(
    chainage,
    reg_plot,
    ":",
    linewidth=2,
    label="Linear regression",
)

# Artificial gap shading
plt.axvspan(
    BLOCK_START,
    BLOCK_END,
    alpha=0.18,
    color="gray",
    label="Artificial gap",
)

plt.xlabel("Station_meter")
plt.ylabel("Thrust Force [kN]")

plt.title(
    f"Training length = {TRAINING_LENGTH}, "
    f"Gap length = {GAP_LENGTH}"
)

plt.legend()
plt.tight_layout()

# Save
plt.savefig("comparison_plot.png", dpi=300)

plt.show()
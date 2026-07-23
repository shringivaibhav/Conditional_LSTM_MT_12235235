# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 08:47:15 2026

@author: Shringi Vaibhav (12235235)
"""

import numpy as np
import matplotlib.pyplot as plt

# Artificial gap boundaries
BLOCK_START = 2666
BLOCK_END = 2685

# TRAINING_LENGTH = 6000
GAP_LENGTH = 20

# X-axis (chainage)
chainage = [i for i in range(BLOCK_START - GAP_LENGTH, BLOCK_END + GAP_LENGTH + 1)]

# Ground truth
ground_truth = [6.9259129643239, 6.43952856715731, 6.88387655433684, 8.04847392265168, 9.00310744926237, 7.39821698495817, 6.53941954808289, 5.92546872213482, 6.28985380797593, 6.31850078621601, 6.39987350309873, 6.48309584344542, 6.57627530324128, 6.22748886176852, 5.83440401042575, 6.05418207606446, 5.70883910899156, 5.99477275699486, 6.05917761943128, 6.16311628414307, 6.35019711502385, 6.09378928337344, 6.31565081061272, 6.20029826020039, 5.97716136436817, 6.35973838595857, 6.18398999702739, 6.69534255024402, 7.39068487266358, 7.37693826588046, 7.02182303922059, 7.18307346161319, 6.82806887124959, 6.75229001544317, 6.83164708892194, 6.68829533534966, 6.14154199858397, 5.9465502668499, 6.14003271898519, 6.00095192086471, 5.94513274191178, 5.77463836606899, 5.7812592621114, 5.79702016442247, 5.86935384842812, 6.73941229425158, 6.86785571791106, 7.2131656522405, 7.07790250312073, 7.29170038666707, 7.15416767038432, 7.24663284633932, 6.6691041202837, 6.52808965565502, 6.86861994246143, 6.16253855392976, 6.00015139759306, 5.76853968391425, 6.07689505031415, 6.17169248457471]

# Reconstructions
conditional_lstm = [10.04110909, 9.438611984, 10.01184177, 8.979600906, 9.788964272, 8.3686409, 8.046165466, 8.339483261, 6.761279106, 7.181014061, 8.384841919, 8.750817299, 8.955215454, 8.926251411, 9.949859619, 9.056106567, 10.28178215, 9.689274788, 9.064763069, 8.954341888]
linear_interpolation = [6.15274, 6.14236, 6.13198, 6.1216, 6.11122, 6.10084, 6.09046, 6.08007, 6.06969, 6.05931, 6.04893, 6.03855, 6.02817, 6.01779, 6.00741, 5.99703, 5.98665, 5.97627, 5.96589, 5.95551]
linear_regression = [6.5857, 6.5793, 6.57289, 6.56649, 6.56009, 6.55368, 6.54728, 6.54087, 6.53447, 6.52807, 6.52166, 6.51526, 6.50885, 6.50245, 6.49605, 6.48964, 6.48324, 6.47683, 6.47043, 6.46403]

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

# Reference values
plt.plot(
    chainage,
    ground_truth,
    label="Reference values",
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

# Local linear regression
plt.plot(
    chainage,
    reg_plot,
    ":",
    linewidth=2,
    label="Local linear regression",
)

# Artificial gap shading
plt.axvspan(
    BLOCK_START,
    BLOCK_END,
    alpha=0.18,
    color="gray",
    label="Artificial gap",
)

plt.xlabel("Chainage [m]")
plt.ylabel("CH Penetration [mm/rot]")

plt.legend()
plt.tight_layout()


plt.savefig(
    "case_E.pdf",
    format="pdf",
    bbox_inches="tight",
)
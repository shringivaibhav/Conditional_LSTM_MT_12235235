# Conditional LSTM Reconstruction of Missing TBM Operational Data

## Overview

This repository contains the Python scripts developed for a master's thesis at Graz University of Technology.

The study investigates the reconstruction of contiguous missing segments in chainage-indexed tunnel boring machine (TBM) operational data. The evaluated methods are:

- linear interpolation;
- local linear regression; and
- a target-specific long short-term memory (LSTM) model, termed the Conditional LSTM.

The Conditional LSTM combines multivariate observations from the left and right contexts of a gap with the three operational parameters that remain available inside the gap. A separate model is trained for each target parameter.

The four investigated TBM parameters are:

- CH Penetration [mm/rot];
- CH Torque [MNm];
- Thrust Force [kN]; and
- CH Rotation [rpm].

---

## Repository Structure

### `1.Data_Preprocessing_MT_12235235.py`

Prepares the raw TBM measurements for reconstruction.

Main operations include:

- reading and combining the raw CSV files;
- removing invalid observations and standstill conditions;
- wavelet-based denoising;
- aggregation to one observation per meter of chainage;
- hybrid infilling of missing chainage positions; and
- export of the processed dataset as `Follo_Finalv2.parquet`.

The script also creates intermediate Parquet files and diagnostic figures.

### `2.Linear_Interpolation_MT_12235235.py`

Creates an artificial gap and reconstructs each selected TBM parameter using linear interpolation.

Outputs include:

- reconstruction figures; and
- `results_linear_interpolation/linear_interpolation_summary.csv`.

### `3.Linear_Regression_MT_12235235.py`

Creates an artificial gap and reconstructs each selected parameter using a local linear regression between the parameter value and chainage.

Outputs include:

- reconstruction figures; and
- `results_linear_regression/linear_regression_summary.csv`.

### `4.Conditional_LSTM_MT_12235235.py`

Implements the target-specific Conditional LSTM.

The model contains:

- a left-context LSTM encoder;
- a right-context LSTM encoder receiving the context in reversed order;
- an in-gap covariate LSTM encoder; and
- a fully connected prediction head that reconstructs the missing target sequence in one forward pass.

The script trains one model for each target parameter and saves:

- reconstructed and reference values;
- reconstruction figures;
- a compact results summary;
- run-configuration files; and
- RMSE, MAE, Pearson correlation coefficient, and derivative-RMSE values.

### `5.Representative_Plots_MT_12235235.py`

Generates a comparison plot for a selected representative reconstruction case. The plotted reference and reconstruction values are specified directly in the script. The figure is exported in PDF format for inclusion in the thesis.


---

## Requirements

The implementation was developed using Python 3.10.

Install the principal dependencies with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn PyWavelets pyarrow torch
```

CUDA is used automatically by the Conditional LSTM script when a compatible GPU and CUDA-enabled PyTorch installation are available. Otherwise, the model runs on the CPU.

---

## Data

The raw and processed TBM datasets are not included because of data availability and confidentiality restrictions.

The preprocessing script expects the raw CSV files in the following relative directory:

```text
Follobanen/
└── S980_All_CSV_files/
    ├── file_001.csv
    ├── file_002.csv
    └── ...
```

The reconstruction scripts require:

```text
Follo_Finalv2.parquet
```

The processed dataset must contain the following columns:

```text
Station_meter
CH Penetration [mm/rot]
CH Torque [MNm]
Thrust Force [kN]
CH Rotation [rpm]
```

---

## Usage

The scripts are configuration-based and do not use command-line arguments. Parameters such as the gap location, gap length, training length, local regression window, and output directory are defined near the beginning of each script.

Run the preprocessing script first:

```bash
python 1.Data_Preprocessing_MT_12235235.py
```

After `Follo_Finalv2.parquet` has been created, the reconstruction scripts can be executed independently:

```bash
python 2.Linear_Interpolation_MT_12235235.py
python 3.Linear_Regression_MT_12235235.py
python 4.Conditional_LSTM_MT_12235235.py
```

The representative plotting script can be executed after inserting the selected ground-truth and reconstruction values:

```bash
python 5.Representative_Plots_MT_12235235.py
```

Paths and configuration values may require adjustment depending on the local directory structure and operating system.

---

## Reproducibility Note

The repository contains the principal computational implementation used in the thesis. However, it is not a fully automated reproduction package because the TBM dataset cannot be distributed and the final aggregation and formatting of the thesis tables were performed separately.

Results may also vary depending on the software environment, hardware, and selected configuration. The Conditional LSTM script uses a fixed random seed to improve repeatability.

---

## Thesis

**Title:** Reconstruction of Missing TBM Operational Data Using a Conditional LSTM

**Author:** Vaibhav Shringi, BEng  
**Master's Programme:** Geotechnical and Hydraulic Engineering  
**Institution:** Graz University of Technology  
**Institute:** Institute of Rock Mechanics and Tunnelling

---

## Disclaimer

This repository is intended for academic and research purposes. Application to other TBM datasets may require modifications to the input paths, column names, preprocessing procedure, and model configuration.

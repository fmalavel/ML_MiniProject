# ML_MiniProject

Machine Learning Foundations (2026) mini project focused on predicting ozone concentration fields from meteorological data.

## Notebook Workflow (Repository Root)

The notebooks in the repository root cover the full workflow from preprocessing to model training and evaluation.

### Preprocessing notebooks

- `Preproc_step1_Extract_model_features_and_save_daily_means.ipynb`
- `Preproc_step1_Extract_model_features_and_save_hourly_means.ipynb`
- `Preproc_step2_Interpolate_model_at_AURN_sites_and_Plot_daily.ipynb`
- `Preproc_step2_Interpolate_model_at_AURN_sites_and_Plot_hourly.ipynb`
- `Preproc_step3_Rasterize_ozone_at_aurn_site_and_save_daily_means_snapshots.ipynb`
- `Preproc_step3_Rasterize_ozone_at_aurn_site_and_save_hourly_means_snapshots.ipynb`

### Training notebook

- `Training_generic.ipynb`

How to run it:

1. Open `Training_generic.ipynb`.
2. In the parameter cell, set:
	- `model_type`
	- `training_frequency`
	- `with_rasterized_ozone`
	- `ndays`
3. Run all cells from top to bottom.
4. Trained models are written to `Trained_models/`.

### Pre-trained model visualization notebook

- `Load_preTrained_models.ipynb`

How to use it:

1. Set the same configuration (`model_type`, frequency, rasterized option) used for training.
2. Load the corresponding model from `Trained_models/`.
3. Run evaluation and visualization cells to inspect:
	- prediction maps
	- residual maps
	- related diagnostics

## Model Library (model_lib)

Models in `model_lib/` are designed to predict ozone fields from meteorological predictors, with optional inclusion of rasterized ozone as an additional input feature.

- `MLP.py`: Fully connected baseline on flattened spatial input.
- `MLP2.py`: Pixel-wise shared MLP using `TimeDistributed` dense layers.
- `2DCNN.py`: Spatial-only CNN for single-time snapshots.
- `3DCNN.py`: Spatiotemporal model using 3D convolutions.
- `CNN+LSTM.py`: TimeDistributed CNN encoder plus LSTM temporal modeling.
- `convLSTM.py`: ConvLSTM sequence model for joint spatial-temporal dynamics.
- `UNet.py`: Encoder-decoder with skip connections and temporal ConvLSTM bottleneck.
- `ml_utils.py`: Shared utilities for data loading/preprocessing, model loading, notebook detection, and CLI parsing.

## Met Office SPICE / SLURM Submission

For SPICE users, convert the generic training notebook into a Python script before submission.

### Step 1: Convert notebook to Python

```bash
bash convert_notebook_to_python.sh
```

This generates `Training_generic.py` from `Training_generic.ipynb`.

### Step 2: Submit SLURM jobs

```bash
bash submit_training_to_spice.sh
```

Notes:

- `submit_training_to_spice.sh` controls model/frequency/rasterized combinations.
- Jobs are submitted through `submit_training_to_spice.sbatch`.
- The job script activates `MLFoundationsEnv`.
- Training CLI arguments passed to `Training_generic.py` include:
  - `--model_type`
  - `--training_frequency`
  - `--ndays`
  - optional `--with_rasterized_ozone`

Outputs:

- Logs: `logs/`
- Trained models: `Trained_models/`

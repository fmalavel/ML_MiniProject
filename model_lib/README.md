# model_lib: Model Architecture Guide

This folder contains the machine learning model definitions used to predict ozone concentration fields from meteorological inputs. Each model exposes a `build_model(...)` function and a `hyperparameters` dictionary, and is designed to output a 2D ozone field with shape `(nlat, nlon, 1)`.

## Common Design Pattern

- Objective: forecast ozone concentration over the spatial grid.
- Loss/metrics: compiled with MSE loss and MAE metric by default.
- Regularization: dropout and L2 regularization are used across models.
- Configuration: model defaults can be overridden through `build_model(...)` arguments.
- Temporal handling:
  - `temporal_sequencing=False`: single snapshot input `(nlat, nlon, nfeature)`.
  - `temporal_sequencing=True`: sequence input `(ntime, nlat, nlon, nfeature)`.

## Models

### `MLP.py`

Purpose:
A dense baseline that maps a single meteorological snapshot to an ozone field. It is useful as a simple benchmark when testing feature sets and training workflows.

Architecture overview:
1. Input `(nlat, nlon, n_channels)` is flattened into one vector.
2. Stack of Dense layers (default hidden units: 512, 256, 128), each followed by BatchNorm and Dropout.
3. Final Dense layer predicts `nlat*nlon` values.
4. Reshape to `(nlat, nlon, 1)`.

### `MLP2.py` (⚠️ not working)

Purpose:
A pixel-wise MLP baseline where each grid point is processed independently with shared weights across pixels. This keeps per-pixel learning while avoiding full flattening interactions.

Architecture overview:
1. Input reshaped from `(nlat, nlon, n_channels)` to `(nlat*nlon, n_channels)`.
2. TimeDistributed Dense blocks (Dense + BatchNorm + Dropout) applied per pixel.
3. TimeDistributed Dense(1) gives one ozone value per pixel.
4. Reshape back to `(nlat, nlon, 1)`.

### `2DCNN.py`

Purpose:
A spatial convolutional model for single-time inputs, designed to capture local and regional spatial structures in meteorological fields.

Architecture overview:
1. Input `(nlat, nlon, n_channels)`.
2. Multiple Conv2D blocks (default filters: 32, 64, 128), each with BatchNorm and Dropout.
3. Final 1x1 Conv2D head projects features to one ozone channel.
4. Output remains `(nlat, nlon, 1)`.

### `3DCNN.py`

Purpose:
A sequence model that explicitly separates spatial and temporal feature extraction for meteorological time windows.

Architecture overview:
1. Input sequence `(ntime, nlat, nlon, n_channels)`.
2. Spatial Conv3D stage using kernels `(1, 3, 3)` to learn within each time slice.
3. Temporal Conv3D stage using kernels `(3, 1, 1)` to mix information across time.
4. Last time slice features are selected.
5. 1x1 Conv2D head outputs ozone field `(nlat, nlon, 1)`.

### `CNN+LSTM.py`

Purpose:
A hybrid model combining CNN spatial encoding at each time step with LSTM temporal modeling across encoded feature sequences.

Architecture overview:
1. Input sequence `(ntime, nlat, nlon, n_channels)`.
2. TimeDistributed Conv2D blocks extract spatial features per time step.
3. TimeDistributed Flatten converts each time step to a feature vector.
4. One or more LSTM layers model temporal evolution.
5. Dense projection to `nlat*nlon`, then reshape to `(nlat, nlon, 1)`.

### `convLSTM.py`

Purpose:
A spatiotemporal recurrent model that keeps spatial structure while learning temporal dependencies, suitable for sequence-based ozone forecasting.

Architecture overview:
1. Input sequence `(ntime, nlat, nlon, n_channels)`.
2. Stacked ConvLSTM2D layers (default filters: 32, 64) with BatchNorm.
3. Conv2D refinement head (default 32 filters) plus Dropout.
4. Final 1x1 Conv2D produces ozone output `(nlat, nlon, 1)`.

### `UNet.py`

Purpose:
A high-capacity encoder-decoder model with skip connections for multi-scale spatial reconstruction, extended with temporal modeling at the bottleneck.

Architecture overview:
1. Input sequence `(ntime, nlat, nlon, n_channels)`.
2. TimeDistributed U-Net encoder across time (Conv2D blocks + pooling) with stored skip connections.
3. TimeDistributed bottleneck Conv2D features.
4. Temporal ConvLSTM2D layer(s) at bottleneck to model sequence dynamics.
5. Decoder with upsampling and skip concatenation (using final-time skip maps).
6. Final 1x1 Conv2D head outputs `(nlat, nlon, 1)`.

## Notes

- `ml_utils.py` in this folder is a shared utility module (data loading/preprocessing, pretrained model loading, CLI parsing), not a model architecture file.
- Training defaults (epochs, batch size, learning rate, patience values) are model-specific and defined in each file's `hyperparameters` dictionary.

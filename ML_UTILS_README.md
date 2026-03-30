# ML Utils Library

A Python library containing shared utilities for machine learning training pipelines. This library centralizes common functions used across multiple training notebooks to reduce code duplication and improve maintainability.

## Overview

`ml_utils.py` provides reusable functions for:
- **Environment detection** — Check if code is running in Jupyter or from shell
- **Model loading** — Load pretrained Keras models
- **Data preprocessing** — Load and preprocess training data from NetCDF files
- **CLI argument parsing** — Parse command-line arguments for training scripts

## Installation

The library is a single Python file: `ml_utils.py`

No external dependencies beyond those already used in the training notebooks:
- `numpy`
- `tensorflow`
- `iris`

## API Reference

### `is_running_in_notebook()`

**Purpose:** Detect if code is executing inside a Jupyter notebook kernel.

**Returns:** `bool` — `True` if running in notebook, `False` if running from shell/script.

**Example:**
```python
from ml_utils import is_running_in_notebook

if not is_running_in_notebook():
    print("Running from command line")
else:
    print("Running in Jupyter notebook")
```

---

### `load_pretrained_model()`

**Purpose:** Load a pre-trained Keras model from disk with automatic file naming.

**Signature:**
```python
load_pretrained_model(
    model_path: str,
    model_type: str,
    frequency: str = "daily",
    with_rasterized_ozone: bool = False
) -> tf.keras.Model
```

**Parameters:**
- `model_path` (str): Directory containing the saved model file
- `model_type` (str): Model type for filename construction (e.g., "CNN", "MLP", "UNET")
- `frequency` (str, optional): Data frequency ("daily" or "hourly"). Default: "daily"
- `with_rasterized_ozone` (bool, optional): Whether model includes rasterized ozone feature. Default: False

**Returns:** Loaded Keras model ready for inference or further training.

**Filename convention:** 
- With rasterized ozone: `{model_type}_model_{frequency}_with_rasterized_o3.keras`
- Without: `{model_type}_model_{frequency}_met_only.keras`

**Example:**
```python
from ml_utils import load_pretrained_model

model = load_pretrained_model(
    model_path="Trained_models",
    model_type="CNN",
    frequency="daily",
    with_rasterized_ozone=True
)
```

---

### `load_and_preprocess_training_data()`

**Purpose:** Load and preprocess X (features) and Y (targets) training data from NetCDF files.

**Signature:**
```python
load_and_preprocess_training_data(
    training_frequency: str,
    ndays: int | None,
    feature_selection_list: list,
    target_selection_list: list | None = None,
    verbose: bool = True
) -> dict
```

**Parameters:**
- `training_frequency` (str): "daily" or "hourly"
- `ndays` (int or None): Number of days to load, or None to load all
- `feature_selection_list` (list): Variable names to use as features
- `target_selection_list` (list, optional): Variable names to predict. Default: `["mass_concentration_of_ozone_in_air"]`
- `verbose` (bool, optional): Print detailed processing messages. Default: True

**Returns:** Dictionary with keys:
- `xtrain_data_normalised` — Input features [ntime, nlat, nlon, nfeature*multiplier]
- `ytrain_data_normalised` — Target data [ntime, nlat, nlon, ntarget]
- `feature_names` — List of feature variable names
- `ntime`, `nlat`, `nlon` — Data dimensions
- `nfeature` — Number of original features
- `nfeature_multiplier` — 2 if mask channel added, 1 otherwise
- `xtrain_cube_list_regridded` — Iris CubeList of input features
- `ytrain_cube_list` — Iris CubeList of target data

**Processing steps:**
1. Loads Y-train (target) data from NetCDF files
2. Loads X-train (feature) data from NetCDF files
3. Filters by selected features/targets
4. Regrid B-grid wind data to common grid if needed
5. Handle NaN values with optional mask channel
6. Normalize data to [0, 1] range

**Example:**
```python
from ml_utils import load_and_preprocess_training_data

data = load_and_preprocess_training_data(
    training_frequency="daily",
    ndays=93,
    feature_selection_list=[
        "air_pressure_at_sea_level",
        "air_temperature",
        "relative_humidity",
        "x_wind",
        "y_wind",
        "precipitation_amount",
        "photolysis_rate_of_nitrogen_dioxide",
        "rasterized_aqum_O3_at_AURN_sites"
    ],
    target_selection_list=["mass_concentration_of_ozone_in_air"],
    verbose=True
)

xtrain = data['xtrain_data_normalised']
ytrain = data['ytrain_data_normalised']
feature_names = data['feature_names']
```

---

### `parse_arguments()`

**Purpose:** Parse command-line arguments for training scripts with sensible defaults.

**Signature:**
```python
parse_arguments(
    model_type_default: str = "UNET",
    model_type_help: str = "Type of model to train"
) -> argparse.Namespace
```

**Parameters:**
- `model_type_default` (str): Default model type. Default: "UNET"
- `model_type_help` (str): Help text for model_type argument. Default: "Type of model to train"

**Returns:** Namespace with attributes:
- `training_frequency` — "daily" or "hourly"
- `with_rasterized_ozone` — bool flag
- `model_type` — str
- `ndays` — int (93 for daily, 30 for hourly if not specified)
- `sequence_length` — int (3 for daily, 48 for hourly if not specified)

**CLI Arguments:**
```
--training_frequency {daily,hourly}
--with_rasterized_ozone
--model_type MODEL_TYPE
--ndays NDAYS
--sequence_length SEQUENCE_LENGTH
```

**Example:**
```python
from ml_utils import parse_arguments

# In Training_CNN.ipynb:
args = parse_arguments(
    model_type_default="CNN",
    model_type_help="Type of CNN model to train"
)

training_frequency = args.training_frequency
with_rasterized_ozone = args.with_rasterized_ozone
model_type = args.model_type
ndays = args.ndays
sequence_length = args.sequence_length
```

## Usage in Notebooks

### 1. Add Import

Add to your notebook's imports cell:

```python
from ml_utils import (
    is_running_in_notebook,
    load_pretrained_model,
    load_and_preprocess_training_data,
    parse_arguments
)
```

### 2. Use in Parameter Cell

```python
# User parameters (overridden by CLI args when running as script)
training_frequency = "daily"
with_rasterized_ozone = True
model_type = "CNN"  # or your model type
ndays = 93 if training_frequency == "daily" else 30
sequence_length = 3 if training_frequency == "daily" else 48

# Apply CLI arguments only when run from shell (skip in notebook)
if not is_running_in_notebook():
    args = parse_arguments(
        model_type_default=model_type,
        model_type_help="Type of CNN model"
    )
    training_frequency = args.training_frequency
    with_rasterized_ozone = args.with_rasterized_ozone
    model_type = args.model_type
    ndays = args.ndays
    sequence_length = args.sequence_length
```

### 3. Load Data

```python
data = load_and_preprocess_training_data(
    training_frequency=training_frequency,
    ndays=ndays,
    feature_selection_list=[
        "air_pressure_at_sea_level",
        "air_temperature",
        "relative_humidity",
        "x_wind",
        "y_wind",
        "precipitation_amount",
        "photolysis_rate_of_nitrogen_dioxide",
    ] + (["rasterized_aqum_O3_at_AURN_sites"] if with_rasterized_ozone else []),
    target_selection_list=["mass_concentration_of_ozone_in_air"],
    verbose=True,
)

# Extract arrays and metadata
xtrain_data_normalised = data['xtrain_data_normalised']
ytrain_data_normalised = data['ytrain_data_normalised']
feature_names = data['feature_names']
ntime, nlat, nlon = data['ntime'], data['nlat'], data['nlon']
nfeature = data['nfeature']
nfeature_multiplier = data['nfeature_multiplier']
```

### 4. Load Pretrained Model

```python
model = load_pretrained_model(
    model_path="Trained_models",
    model_type=model_type,
    frequency=training_frequency,
    with_rasterized_ozone=with_rasterized_ozone
)
```

## Files Modified

### Updated Notebooks
All 5 notebooks have been updated to import from `ml_utils.py`:
- ✅ `Training_SVM.ipynb`
- ✅ `Training_MLP.ipynb`
- ✅ `Training_CNN.ipynb`
- ✅ `Training_CNN+LSTM.ipynb`
- ✅ `Training_UNet.ipynb`

**Note:** The notebooks still contain the old function definitions. These can be safely deleted since they're now imported from `ml_utils.py`.

### Backups
Backup copies were created before updating:
- `Training_*.ipynb.bak` — Original notebook for reference

## Verification

Use the provided verification script to check which notebooks have been updated:

```bash
python check_notebook_updates.py
```

## Next Steps

1. **Remove old function definitions** from notebooks (optional but recommended for cleanliness)
2. **Test notebooks** to ensure they work correctly with the new imports
3. **Use as template** when creating new training notebooks

## Benefits

✅ **DRY Principle** — No duplicate code across 5 notebooks
✅ **Consistency** — All notebooks use identical data loading logic
✅ **Maintainability** — Fix bugs once, affects all notebooks
✅ **Testability** — Easier to unit test shared functions
✅ **Extensibility** — Add new utilities and they're available to all notebooks
✅ **Reusability** — Use `ml_utils` in new training scripts or notebooks

## Troubleshooting

### Import Error: "No module named 'ml_utils'"
Ensure `ml_utils.py` is in the same directory as your notebook or add the directory to Python path:
```python
import sys
sys.path.insert(0, '/path/to/ml_utils')
```

### Function not working as expected
Check the function's docstring and parameters:
```python
help(load_and_preprocess_training_data)
```

### Notebooks with conflicting definitions
Both the import and the local definition will exist temporarily. Delete the old function definition cells from the notebook.

"""
Shared utilities for ML training pipelines.

This module provides reusable functions for:
- Checking execution environment (notebook vs. script)
- Loading pretrained models
- Loading and preprocessing training data
- Parsing command-line arguments
"""

import os
import argparse

import numpy as np
import tensorflow as tf
import iris


def is_running_in_notebook():
    """
    Return True when executed inside a Jupyter notebook kernel.
    
    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def load_pretrained_model(model_path, model_type=None, frequency="daily", with_rasterized_ozone=False):
    """
    Load a pretrained Keras model from the specified path.

    Parameters:
    -----------
    model_path : str
        Path to the directory containing the saved Keras model
    model_type : str
        Type of the model (e.g., "CNN", "MLP", "CNN+LSTM", "UNET", "SVM", etc.)
    frequency : str
        Data frequency (e.g., "hourly", "daily"). Default is "daily".
    with_rasterized_ozone : bool
        Whether the model includes rasterized ozone as an input feature. Default is False.

    Returns:
    --------
    keras.Model
        The loaded model ready for inference or further training.

    Raises:
    -------
    ValueError
        If model_type is not specified.
    FileNotFoundError
        If the model file is not found at the specified path.
    Exception
        If there is an error loading the model.
    """
    if model_type is None:
        raise ValueError("model_type must be specified (e.g., 'CNN', 'MLP', 'CNN+LSTM', etc.)") 
    
    # Construct the full model name based on the provided parameters
    if with_rasterized_ozone:
        model_name = f"{model_type}_model_{frequency}_with_rasterized_o3.keras"
    else:
        model_name = f"{model_type}_model_{frequency}_met_only.keras"

    model_file = os.path.join(model_path, model_name)
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at: {model_file}")
    
    try:
        model = tf.keras.models.load_model(model_file)
        print(f"Successfully loaded model from {model_file}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def load_and_preprocess_training_data(training_frequency, ndays, feature_selection_list, 
                                       target_selection_list=None, verbose=True):
    """
    Load and preprocess X and Y training data from NetCDF files.
    
    Parameters:
    -----------
    training_frequency : str
        Data frequency ("daily" or "hourly")
    ndays : int or None
        Number of days to load (set to None to load all data)
    feature_selection_list : list
        List of feature variable names to include
    target_selection_list : list, optional
        List of target variable names. Defaults to ["mass_concentration_of_ozone_in_air"]
    verbose : bool, optional
        Whether to print processing messages. Default is True
    
    Returns:
    --------
    dict : Dictionary containing:
        - xtrain_data_normalised: normalized input features array [ntime, nlat, nlon, nfeature*multiplier]
        - ytrain_data_normalised: normalized target data array [ntime, nlat, nlon, ntarget]
        - feature_names: list of feature names
        - ntime, nlat, nlon: spatial/temporal dimensions
        - nfeature: number of original features
        - nfeature_multiplier: 2 if mask was added, 1 otherwise
        - xtrain_cube_list_regridded: iris CubeList of processed input features
        - ytrain_cube_list: iris CubeList of target data
    """
    
    # Set defaults
    if target_selection_list is None:
        target_selection_list = ["mass_concentration_of_ozone_in_air"]
    
    # Determine data path based on frequency
    if training_frequency == "hourly":
        training_data_path = "processed_data/Hourly_snapshots"
        file_multiplier = 24
    elif training_frequency == "daily":
        training_data_path = "processed_data/Daily_snapshots"
        file_multiplier = 1
    else:
        raise ValueError("Invalid training frequency. Please choose 'hourly' or 'daily'.")
    
    # Adjust ndays if None
    ndays_to_load = ndays if ndays is not None else float('inf')
    
    print(f"Loading training data from: {training_data_path}")
    print(f"Number of days to load: {ndays_to_load}")
    print(f"Feature selection list: {feature_selection_list}")
    print(f"Target selection list: {target_selection_list}")
    print(f"File multiplier based on frequency: {file_multiplier}")
    print(f"Expected number of files to load for Y-train: {int(ndays_to_load * file_multiplier)}")
    print(f"Expected number of files to load for X-train: {int(ndays_to_load * file_multiplier)}")
    print(f"Verbose mode is {'ON' if verbose else 'OFF'}: Detailed processing messages will {'be printed' if verbose else 'not be printed'}.\n")
    
    # ========================
    # 1 - Loading Y-TRAIN data
    # ========================
    if verbose:
        print("\n" + "="*60)
        print("STEP 1: Loading Y-TRAIN data")
        print("="*60)
        print("1.1 - Finding y_train files in the data path...")
    
    ytrain_files = sorted([f for f in os.listdir(training_data_path) if "y_train" in f and f.endswith(".nc")])
    ytrain_files = ytrain_files[:int(ndays_to_load * file_multiplier)]
    ytrain_cube_list = iris.load([os.path.join(training_data_path, f) for f in ytrain_files])
    
    if verbose:
        print(f"\nIdentified y_train files to load ({len(ytrain_files)} files):")
        # print first 10 files identified to load
        print(f"Showing first 10 files:")
        for f in ytrain_files[:10]:
            print(f" - {f}")

    if verbose:
        print(f"\nLoaded ytrain_cube_list:\n{ytrain_cube_list}")
    
    # Remove cubes that are not in target_selection_list
    if verbose:
        print("\nRemoving cubes from ytrain_cube_list that are not in target_selection_list...")
    
    for cube in list(ytrain_cube_list):
        if verbose:
            print(f"Checking cube variable name: {cube.var_name}")
        if cube.var_name not in target_selection_list:
            if verbose:
                print(f"Removing cube with variable name: {cube.var_name} as it is not in target_selection_list")
            ytrain_cube_list.remove(cube)
    
    if verbose:
        print(f"\nRemaining cubes in ytrain_cube_list after filtering:\n{ytrain_cube_list}")
    
    ntarget = len(target_selection_list)
    ntime = ytrain_cube_list[0].coord('time').points.size
    nlat = ytrain_cube_list[0].coord('grid_latitude').points.size
    nlon = ytrain_cube_list[0].coord('grid_longitude').points.size
    
    # Create a numpy array of dimension [ntime, nlat, nlon, ntarget] to contain ytrain data
    if verbose:
        print("\n1.2 - Creating ytrain_data array with dimensions [ntime, nlat, nlon, ntarget]...")
    
    ytrain_data = np.empty((ntime, nlat, nlon, ntarget))
    for i, cube in enumerate(ytrain_cube_list):
        ytrain_data[:, :, :, i] = cube.data
    
    if verbose:
        print(f"Y-train data range: min={np.min(ytrain_data)}, max={np.max(ytrain_data)}")
    
    # ========================
    # 2 - Loading X-TRAIN data
    # ========================
    if verbose:
        print("\n" + "="*60)
        print("STEP 2: Loading X-TRAIN data")
        print("="*60)
        print("2.1 - Finding x_train files in the data path...")
    
    xtrain_files = sorted([f for f in os.listdir(training_data_path) if "rasterized" in f and f.endswith(".nc")])
    xtrain_files = xtrain_files[:int(ndays_to_load * file_multiplier)]
    xtrain_cube_list = iris.load([os.path.join(training_data_path, f) for f in xtrain_files])
        
    if verbose:
        print(f"\nIdentified x_train files to load ({len(xtrain_files)} files):")
        # print first 10 files identified to load
        print(f"Showing first 10 files:")
        for f in xtrain_files[:10]:
            print(f" - {f}")

    if verbose:
        print(f"\nLoaded xtrain_cube_list:\n{xtrain_cube_list}")
    
    # Remove cubes that are not in feature_selection_list
    if verbose:
        print("\nRemoving cubes from xtrain_cube_list that are not in feature_selection_list...")
    
    for cube in list(xtrain_cube_list):
        if verbose:
            print(f"Checking cube variable name: {cube.var_name}")
        if cube.var_name not in feature_selection_list:
            if verbose:
                print(f"Removing cube with variable name: {cube.var_name} as it is not in feature_selection_list")
            xtrain_cube_list.remove(cube)
    
    if verbose:
        print(f"\nRemaining cubes in xtrain_cube_list after filtering:\n{xtrain_cube_list}")
    
    # Mask where cube.data is NaN
    if verbose:
        print("\n2.2 - Checking for NaN values in xtrain cubes and masking if necessary...")
    
    for cube in xtrain_cube_list:
        if np.isnan(cube.data).any():
            if verbose:
                print(f"... NaN values found in {cube.var_name}, masking invalid data...")
            cube.data = np.ma.masked_invalid(cube.data)
        if verbose:
            print(f"Data range for {cube.var_name}: min={cube.data.min()}, max={cube.data.max()}")
    
    # If cube is x_wind or y_wind (B-GRID), interpolate to the same grid as the other cubes
    if verbose:
        print("\n2.3 - Checking for x_wind and y_wind cubes to interpolate to the same grid as air_pressure_at_sea_level...")
    
    xtrain_cube_list_regridded = iris.cube.CubeList()
    for cube in xtrain_cube_list:
        if cube.var_name in ['x_wind', 'y_wind']:
            if verbose:
                print(f"... Interpolating {cube.var_name} to the same grid as 'air_pressure_at_sea_level'")
            target_cube = xtrain_cube_list.extract_cube(iris.Constraint('air_pressure_at_sea_level'))
            cube_new = cube.copy().interpolate([('grid_latitude', target_cube.coord('grid_latitude').points),
                                     ('grid_longitude', target_cube.coord('grid_longitude').points)],
                                    iris.analysis.Linear())
            if verbose:
                print(f"Interpolated {cube_new.var_name}: {cube_new.summary(True)}")
            xtrain_cube_list_regridded.append(cube_new)
        else:
            if verbose:
                print(f"No interpolation needed for {cube.var_name}.")
            xtrain_cube_list_regridded.append(cube)
    
    if verbose:
        print("\nAfter interpolation, checking data ranges for all cubes:")
        for cube in xtrain_cube_list_regridded:
            print(f"Data range for {cube.var_name} after interpolation: min={cube.data.min()}, max={cube.data.max()}")
        print(f"\nRegridded xtrain_cube_list:\n{xtrain_cube_list_regridded}")
    
    # =========================================================================
    # 3 - Clean NaN from X train and append a mask as an extra channel
    # =========================================================================
    if verbose:
        print("\n" + "="*60)
        print("STEP 3: Clean NaN from X train and add mask")
        print("="*60)
        print("3.1 - Creating xtrain_data array with dimensions [ntime, nlat, nlon, nfeature]...")
    
    nfeature = len(xtrain_cube_list_regridded)
    xtrain_data = np.empty((ntime, nlat, nlon, nfeature))
    
    feature_names = []
    if verbose:
        print(f"\nInitialized xtrain_data array with shape {xtrain_data.shape} to hold data for {nfeature} features, {ntime} time steps, {nlat} latitudes, and {nlon} longitudes.\n")
    
    for nf, cube in enumerate(xtrain_cube_list_regridded):
        feature_names.append(cube.var_name)
        if verbose:
            print(f"-> Assigning data for feature {nf} ({cube.var_name}) to xtrain_data array...")
            print(f"Data range for {cube.var_name} being assigned: min={cube.data.min()}, max={cube.data.max()}")
        xtrain_data[:, :, :, nf] = cube.data
        if verbose:
            print(f"After assignment, xtrain_data range for feature {nf} ({cube.var_name}): min={np.min(xtrain_data[:, :, :, nf])}, max={np.max(xtrain_data[:, :, :, nf])}")
    
    # Use a Masking Layer (recommended if NaNs represent missing pixels)
    if verbose:
        print(f"\n3.2 - Checking for NaN values in xtrain_data array...")
    
    if np.isnan(xtrain_data).any():
        if verbose:
            print("NaN values found in xtrain_data, applying Masking Layer to handle missing data.")
        mask = (~np.isnan(xtrain_data)).astype("float32")
        xtrain_data = np.nan_to_num(xtrain_data, nan=0.0)
        if verbose:
            print(f"Mask shape: {mask.shape}, Mask value range: min={mask.min()}, max={mask.max()}")
        xtrain_data_with_mask = np.concatenate([xtrain_data, mask], axis=-1)
        xtrain_data = xtrain_data_with_mask
        if verbose:
            print(f"xtrain_data with mask shape: {xtrain_data.shape}, value range: min={xtrain_data.min()}, max={xtrain_data.max()}")
        nfeature_multiplier = 2
    else:
        if verbose:
            print("No NaN values found in xtrain_data, no masking needed.")
        nfeature_multiplier = 1
    
    if verbose:
        print(f"Number of features after masking (if applied): {nfeature * nfeature_multiplier}")
    
    # =========================================
    # 4 - Normalise x_train and y_train data
    # =========================================
    if verbose:
        print("\n" + "="*60)
        print("STEP 4: Normalize x_train and y_train data")
        print("="*60)
    
    xtrain_data_normalised = xtrain_data.astype(np.float32)
    ytrain_data_normalised = ytrain_data.astype(np.float32)
    
    if verbose:
        print(f"\nBefore normalization, checking min/max values for each non-mask feature in X-train data:")
    
    for nf in range(nfeature):
        if verbose:
            print(f"Feature {nf} min/max values: {np.min(xtrain_data_normalised[:, :, :, nf])}, {np.max(xtrain_data_normalised[:, :, :, nf])}")
        xtrain_data_normalised[:, :, :, nf] /= np.max(xtrain_data_normalised[:, :, :, nf])
    
    if verbose:
        print(f"\nAfter normalization, checking min/max values for each feature in X-train data:")
        for nf in range(nfeature * nfeature_multiplier):
            print(f"Feature {nf} min/max values after normalization: min={np.min(xtrain_data_normalised[:, :, :, nf])}, max={np.max(xtrain_data_normalised[:, :, :, nf])}")
    
    if verbose:
        print(f"\nY-train data min/max values before normalization: min={np.min(ytrain_data_normalised[:, :, :, :])}, max={np.max(ytrain_data_normalised[:, :, :, :])}")
    
    ytrain_data_normalised[:, :, :, :] /= np.max(ytrain_data_normalised[:, :, :, :])
    
    if verbose:
        print(f"Y-train data min/max values after normalization: min={np.min(ytrain_data_normalised[:, :, :, :])}, max={np.max(ytrain_data_normalised[:, :, :, :])}")
    
    # Return results as a dictionary
    return {
        'xtrain_data_normalised': xtrain_data_normalised,
        'ytrain_data_normalised': ytrain_data_normalised,
        'feature_names': feature_names,
        'ntime': ntime,
        'nlat': nlat,
        'nlon': nlon,
        'nfeature': nfeature,
        'nfeature_multiplier': nfeature_multiplier,
        'xtrain_cube_list_regridded': xtrain_cube_list_regridded,
        'ytrain_cube_list': ytrain_cube_list,
    }


def parse_arguments(model_type_default="MLP", model_type_help="Type of model to train"):
    """
    Parse optional CLI arguments for training scripts.
    
    Parameters:
    -----------
    model_type_default : str
        Default model type (e.g., "UNET", "MLP", "CNN", etc.). Default is "UNET".
    model_type_help : str
        Help text for the model_type argument. Default is "Type of model to train".
    
    Returns:
    --------
    argparse.Namespace : Parsed arguments with the following attributes:
        - training_frequency: "daily" or "hourly"
        - with_rasterized_ozone: bool
        - model_type: str
        - ndays: int (number of days to load)
        - sequence_length: int (sequence length for model input)
    """
    parser = argparse.ArgumentParser(
        description=f"Train {model_type_default} model for ozone prediction"
    )
    parser.add_argument(
        "--training_frequency",
        type=str,
        default="daily",
        choices=["hourly", "daily"],
        help="Data frequency for training (default: daily)",
    )
    parser.add_argument(
        "--with_rasterized_ozone",
        action="store_true",
        help="Include rasterized ozone as an input feature (default: False)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=model_type_default,
        help=f"{model_type_help} (default: {model_type_default})",
    )
    parser.add_argument(
        "--ndays",
        type=int,
        default=None,
        help="Number of days to load for training (default: all)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=None,
        help="Sequence length for model input (default: based on frequency)",
    )

    args = parser.parse_args()

    # Set defaults based on training frequency if not provided
    if args.ndays is None:
        args.ndays = 93 if args.training_frequency == "daily" else 30
    if args.sequence_length is None:
        args.sequence_length = 3 if args.training_frequency == "daily" else 48

    return args

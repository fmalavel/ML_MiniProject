"""CNN+LSTM architecture for ozone prediction from meteorological sequences."""

# pylint: disable=invalid-name,import-error

import tensorflow as tf

hyperparameters = {
    "model_name": "CNN_LSTM_ozone_forecaster",
    "temporal_sequencing": True,
    "spatial_filters": (32, 64),
    "lstm_units": (64, 32),
    "dropout_rate": 0.3,
    "l2_lambda": 1e-5,
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "es_patience": 6,
    "lr_patience": 4,
    "min_lr": 1e-6,
    "reduce_factor": 0.3,
    "monitor_metric": "val_loss",
    "compile_loss": "mse",
    "compile_metrics": ["mae"],
}


def build_cnn_lstm_ozone_forecaster(
    ntime, nlat, nlon, n_channels, config=None
):
    """Build a CNN+LSTM that learns spatial then temporal patterns.

    Input shape is (ntime, nlat, nlon, n_channels) and output is a 2D ozone
    field (nlat, nlon, 1) at the target time step.

    Process:
      1. Apply CNN to each spatial snapshot via TimeDistributed
      2. Flatten spatial features -> (batch, ntime, features)
      3. Pass through LSTM layers for temporal modeling
      4. Dense layer maps LSTM output to 2D ozone prediction
    """
    hp = hyperparameters if config is None else config
    kernel_reg = tf.keras.regularizers.L2(hp["l2_lambda"])

    inputs = tf.keras.Input(
        shape=(ntime, nlat, nlon, n_channels),
        name="meteo_sequence",
    )

    # Stage 1: TimeDistributed CNN to extract spatial features from each
    # time step independently
    x = inputs
    for filters in hp["spatial_filters"]:
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=kernel_reg,
            )
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization()
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(hp["dropout_rate"])
        )(x)

    # Flatten spatial dimensions for LSTM input
    # (batch, ntime, nlat, nlon, filters) -> (batch, ntime, nlat*nlon*filters)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten()
    )(x)

    # Stage 2: LSTM layers for temporal modeling
    for i, lstm_units in enumerate(hp["lstm_units"]):
        return_seq = i < len(hp["lstm_units"]) - 1
        x = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=return_seq,
            dropout=hp["dropout_rate"],
            kernel_regularizer=kernel_reg,
        )(x)

    # Stage 3: Map LSTM output to 2D ozone field
    # (batch, lstm_units) -> (batch, nlat*nlon)
    x = tf.keras.layers.Dense(
        nlat * nlon,
        activation="linear",
        kernel_regularizer=kernel_reg,
    )(x)
    outputs = tf.keras.layers.Reshape(
        (nlat, nlon, 1),
        name="ozone_forecast",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=hp["model_name"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
        loss=hp["compile_loss"],
        metrics=hp["compile_metrics"],
    )
    return model


def build_model(
    input_shape,
    hidden_units=None,
    dropout_rate=None,
    spatial_filters=None,
    lstm_units=None,
    batch_size=None,
    epochs=None,
    learning_rate=None,
    es_patience=None,
    lr_patience=None,
    min_lr=None,
    reduce_factor=None,
    monitor_metric=None,
    l2_lambda=None,
    compile_loss=None,
    compile_metrics=None,
    model_name=None,
):
    """Compatibility wrapper used by Training_generic notebook.

    Any non-None argument overrides the corresponding default in
    module-level hyperparameters for this model build call.
    """
    config = hyperparameters.copy()

    overrides = {
        "hidden_units": hidden_units,
        "dropout_rate": dropout_rate,
        "spatial_filters": spatial_filters,
        "lstm_units": lstm_units,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "es_patience": es_patience,
        "lr_patience": lr_patience,
        "min_lr": min_lr,
        "reduce_factor": reduce_factor,
        "monitor_metric": monitor_metric,
        "l2_lambda": l2_lambda,
        "compile_loss": compile_loss,
        "compile_metrics": compile_metrics,
        "model_name": model_name,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if len(input_shape) != 4:
        raise ValueError(
            "CNN+LSTM expects input_shape=(ntime, nlat, nlon, nfeature). "
            f"Received input_shape={input_shape}. "
            "Set hyperparameters['temporal_sequencing']=True and provide "
            "sequence data."
        )

    ntime, nlat, nlon, nfeatures = input_shape
    return build_cnn_lstm_ozone_forecaster(
        ntime,
        nlat,
        nlon,
        nfeatures,
        config=config,
    )

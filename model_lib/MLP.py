"""MLP architecture for ozone prediction from meteorological fields."""

# pylint: disable=invalid-name,import-error

import tensorflow as tf

hyperparameters = {
    "model_name": "MLP_ozone_forecaster",
    "temporal_sequencing": False,
    "hidden_units": (512, 256, 128),
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


def build_mlp_ozone_forecaster(nlat, nlon, n_channels, config=None):
    """Build the requested flattened MLP ozone forecaster."""
    hp = hyperparameters if config is None else config
    kernel_reg = tf.keras.regularizers.L2(hp["l2_lambda"])

    inputs = tf.keras.Input(
        shape=(nlat, nlon, n_channels),
        name="meteo_snapshot",
    )
    x = tf.keras.layers.Flatten()(inputs)

    for units in hp["hidden_units"]:
        x = tf.keras.layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=kernel_reg,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(hp["dropout_rate"])(x)

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

    nlat, nlon, nfeatures = input_shape
    return build_mlp_ozone_forecaster(nlat, nlon, nfeatures, config=config)

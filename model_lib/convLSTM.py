"""ConvLSTM architecture for ozone prediction from meteorological sequences."""

# pylint: disable=invalid-name,import-error

import tensorflow as tf

hyperparameters = {
    "model_name": "ConvLSTM_ozone_forecaster",
    "temporal_sequencing": True,
    "convlstm_filters": (32, 64),
    "head_filters": 32,
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


def build_convlstm_ozone_forecaster(
    ntime, nlat, nlon, n_channels, config=None
):
    """Build a ConvLSTM model for ozone field prediction at time t.

    Input shape: (ntime, nlat, nlon, n_channels)
    Output shape: (nlat, nlon, 1)
    """
    hp = hyperparameters if config is None else config
    kernel_reg = tf.keras.regularizers.L2(hp["l2_lambda"])

    inputs = tf.keras.Input(
        shape=(ntime, nlat, nlon, n_channels),
        name="meteo_sequence",
    )

    x = inputs
    convlstm_filters = tuple(hp["convlstm_filters"])
    for i, filters in enumerate(convlstm_filters):
        return_seq = i < len(convlstm_filters) - 1
        x = tf.keras.layers.ConvLSTM2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=return_seq,
            dropout=hp["dropout_rate"],
            recurrent_dropout=0.0,
            kernel_regularizer=kernel_reg,
            recurrent_regularizer=kernel_reg,
            name=f"convlstm_{i + 1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"convlstm_bn_{i + 1}")(x)

    x = tf.keras.layers.Conv2D(
        hp["head_filters"],
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=kernel_reg,
        name="head_conv",
    )(x)
    x = tf.keras.layers.Dropout(hp["dropout_rate"], name="head_drop")(x)

    outputs = tf.keras.layers.Conv2D(
        1,
        kernel_size=(1, 1),
        padding="same",
        activation="linear",
        kernel_regularizer=kernel_reg,
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
    convlstm_filters=None,
    head_filters=None,
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
        "convlstm_filters": convlstm_filters,
        "head_filters": head_filters,
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
            "ConvLSTM expects input_shape=(ntime, nlat, nlon, nfeature). "
            f"Received input_shape={input_shape}. "
            "Set hyperparameters['temporal_sequencing']=True and provide "
            "sequence data."
        )

    ntime, nlat, nlon, nfeatures = input_shape
    return build_convlstm_ozone_forecaster(
        ntime,
        nlat,
        nlon,
        nfeatures,
        config=config,
    )

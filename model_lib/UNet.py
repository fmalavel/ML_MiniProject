"""UNet architecture for ozone prediction from meteorological sequences."""

# pylint: disable=invalid-name,import-error

import tensorflow as tf


def extract_last_timestep(tensor):
    """Return features at the final time step: (batch, lat, lon, channels)."""
    return tensor[:, -1, :, :, :]


hyperparameters = {
    "model_name": "UNet_ozone_forecaster",
    "temporal_sequencing": True,
    "unet_filters": (32, 64, 128),
    "temporal_filters": (128,),
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


def build_unet_ozone_forecaster(ntime, nlat, nlon, n_channels, config=None):
    """Build a UNet that learns spatial patterns first, then temporal ones.

    Input shape is (ntime, nlat, nlon, n_channels) and output is a 2D ozone
    field (nlat, nlon, 1) at the target time step.
    """
    hp = hyperparameters if config is None else config
    kernel_reg = tf.keras.regularizers.L2(hp["l2_lambda"])
    filters = tuple(hp["unet_filters"])
    temporal_filters = tuple(hp["temporal_filters"])

    if len(filters) < 2:
        raise ValueError("unet_filters must contain at least 2 levels.")

    inputs = tf.keras.Input(
        shape=(ntime, nlat, nlon, n_channels),
        name="meteo_sequence",
    )

    # Spatial encoder per time step.
    x = inputs
    skips = []
    for i, n_filters in enumerate(filters[:-1], start=1):
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
                n_filters,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=kernel_reg,
            ),
            name=f"enc{i}_conv1_td",
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization(),
            name=f"enc{i}_bn1_td",
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
                n_filters,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=kernel_reg,
            ),
            name=f"enc{i}_conv2_td",
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(hp["dropout_rate"]),
            name=f"enc{i}_drop_td",
        )(x)
        skips.append(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((2, 2)),
            name=f"enc{i}_pool_td",
        )(x)

    # Spatial bottleneck per time step.
    bottleneck_filters = filters[-1]
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            bottleneck_filters,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=kernel_reg,
        ),
        name="bottleneck_conv1_td",
    )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            bottleneck_filters,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=kernel_reg,
        ),
        name="bottleneck_conv2_td",
    )(x)

    # Temporal learning at bottleneck.
    for i, n_filters in enumerate(temporal_filters, start=1):
        return_sequences = i < len(temporal_filters)
        x = tf.keras.layers.ConvLSTM2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=return_sequences,
            dropout=hp["dropout_rate"],
            kernel_regularizer=kernel_reg,
            recurrent_regularizer=kernel_reg,
            name=f"temporal_convlstm_{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"temporal_bn_{i}")(x)

    # Use skip maps at final time t in the decoder.
    skip_t = [
        tf.keras.layers.Lambda(
            extract_last_timestep,
            name=f"skip_t_{i + 1}",
        )(s)
        for i, s in enumerate(skips)
    ][::-1]

    for i, (n_filters, s) in enumerate(zip(filters[:-1][::-1], skip_t), start=1):
        x = tf.keras.layers.UpSampling2D((2, 2), name=f"dec{i}_up")(x)
        x = tf.keras.layers.Concatenate(name=f"dec{i}_concat")([x, s])
        x = tf.keras.layers.Conv2D(
            n_filters,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=kernel_reg,
            name=f"dec{i}_conv1",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"dec{i}_bn1")(x)
        x = tf.keras.layers.Conv2D(
            n_filters,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=kernel_reg,
            name=f"dec{i}_conv2",
        )(x)
        x = tf.keras.layers.Dropout(hp["dropout_rate"], name=f"dec{i}_drop")(x)

    outputs = tf.keras.layers.Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="linear",
        kernel_regularizer=kernel_reg,
        name="ozone_forecast",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=hp["model_name"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
        loss=hp["compile_loss"],
        metrics=hp["compile_metrics"],
    )
    return model


def build_model(
    input_shape,
    dropout_rate=None,
    unet_filters=None,
    temporal_filters=None,
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
        "dropout_rate": dropout_rate,
        "unet_filters": unet_filters,
        "temporal_filters": temporal_filters,
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
            "UNet expects input_shape=(ntime, nlat, nlon, nfeature). "
            f"Received input_shape={input_shape}. "
            "Set hyperparameters['temporal_sequencing']=True and provide "
            "sequence data."
        )

    ntime, nlat, nlon, nfeatures = input_shape
    return build_unet_ozone_forecaster(
        ntime,
        nlat,
        nlon,
        nfeatures,
        config=config,
    )

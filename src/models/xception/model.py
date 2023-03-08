import tensorflow as tf


keras = tf.keras
xception = tf.keras.applications.xception.Xception


def xception_model(learning_rate, droprate, input_shape):
    base_model = xception (
        weights='imagenet',
        input_shape=input_shape,
        include_top=False
    )

    base_model.trainable = False

    inputs = keras.Input(input_shape)

    base = base_model(inputs, training=False)
    vector = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(100, activation='relu')(vector)
    drop = keras.layers.Dropout(droprate)(inner)

    outputs = keras.layers.Dense(10)(drop)

    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    return model
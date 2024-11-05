import tensorflow as tf
from tensorflow import keras as k


def create_model(input_shape, num_classes):
    model = k.Sequential()
    model.add(
        k.layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", input_shape=input_shape
        )
    )
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(k.layers.Dropout(0.10))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(k.layers.Dropout(0.10))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Conv2D(256, (3, 3), activation="relu"))
    model.add(k.layers.Dropout(0.10))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(128, activation="relu"))
    model.add(k.layers.Dropout(0.10))

    model.add(k.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=k.optimizers.SGD(),
        metrics=["accuracy"],
    )
    return model


def my_callbacks():
    checkpoint_cb = k.callbacks.ModelCheckpoint(
        "cla720.keras",
        save_best_only=True,
        monitor="val_loss",
    )
    early_stopping_cb = k.callbacks.EarlyStopping(
        patience=42, restore_best_weights=True, monitor="val_loss"
    )
    reduce_lr_cb = k.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )
    return [checkpoint_cb, early_stopping_cb, reduce_lr_cb]


def split_data(x, y, percent=0.1):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=percent, random_state=42
    )
    return X_train, X_test, y_train, y_test


def create_multihead_model(input_shape, num_classes):

    input_ = k.layers.Input(shape=input_shape[:2])
    # convolutional layers
    c1 = k.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
    )(input_)
    m1 = k.layers.MaxPooling2D(pool_size=(2, 2))(c1)
    b1 = k.layers.BatchNormalization()(m1)

    # 2nd layers
    c2 = k.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
    )(b1)
    d2 = k.layers.Dropout(0.10)(c2)
    m2 = k.layers.MaxPooling2D(pool_size=(2, 2))(d2)
    b2 = k.layers.BatchNormalization()(m2)

    # 3rd layers
    c3 = k.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
    )(b2)
    d3 = k.layers.Dropout(0.10)(c3)
    m3 = k.layers.MaxPooling2D(pool_size=(2, 2))(d3)
    b3 = k.layers.BatchNormalization()(m3)

    # 4th layers
    c4 = k.layers.Conv2D(
        256,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
    )(b3)
    d4 = k.layers.Dropout(0.10)(c4)
    m4 = k.layers.MaxPooling2D(pool_size=(2, 2))(d4)
    b4 = k.layers.BatchNormalization()(m4)

    # final layer begins
    f = k.layers.Flatten()(b4)

    # for hours final layers
    dense_hour1 = k.layers.Dense(128, activation="relu")(f)
    dense_dropout1 = k.layers.Dropout(0.10)(dense_hour1)
    final_hour = k.layers.Dense(num_classes, activation="softmax")(dense_dropout1)

    # for minutes final layers
    minute_dense1 = k.layers.Dense(512, activation="relu")(f)

    minute_dropout1 = k.layers.Dropout(0.5)(minute_dense1)
    minute_dense2 = k.layers.Dense(128, activation="relu")(minute_dropout1)
    minute_dropout2 = k.layers.Dropout(0.3)(minute_dense2)

    minute_dense3 = k.layers.Dense(64, activation="relu")(minute_dropout2)
    minute_dropout3 = k.layers.Dropout(0.1)(minute_dense3)

    final_minute = k.layers.Dense(1, activation="linear")(minute_dropout3)

    model = k.Model(inputs=input_, outputs=[final_hour, final_minute])
    model.summary()

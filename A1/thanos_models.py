import tensorflow as tf
from tensorflow import keras as k

def create_model(input_shape,num_classes):
    model = k.Sequential()
    model.add(k.layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(k.layers.Dropout(0.10))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(k.layers.Dropout(0.10))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(k.layers.Dropout(0.10))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(128, activation='relu'))
    model.add(k.layers.Dropout(0.10))

    model.add(k.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                optimizer=k.optimizers.SGD(),
                metrics=['accuracy'])
    return model

def my_callbacks():
    checkpoint_cb = k.callbacks.ModelCheckpoint(
    "cla720.keras", 
    save_best_only=True,
    monitor="val_loss",
    )
    early_stopping_cb = k.callbacks.EarlyStopping(  
    patience=42, restore_best_weights=True,monitor="val_loss"
    )
    return [checkpoint_cb,early_stopping_cb]


def split_data(x,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )
    X, X_val, y, y_val= train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    return X,X_test,        
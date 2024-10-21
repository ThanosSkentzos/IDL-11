from keras import layers
import keras_tuner
import keras
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import pandas as pd


class Dataset_prep():
    def __init__(self, X_train,X_valid, y_train, y_valid, input_shape):
        self.input_shape = input_shape
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        

def load_data(num_classes):
    
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    plt.imshow(x_train[0], cmap='gray')
    plt.title(f"Label: {y_train[0]}")
    print(y_train[0])
    plt.axis('off')  # Remove the axes for better visualization
    plt.show()
    X = x_train
    y = y_train

    X = (X - X.mean())/X.std()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=42)
    print(y_train.shape)
    print(X_train.shape)
    print(y_train[0])

    
    #make this as pipeline for model
    
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
        
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)


    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_valid /= 255


    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'valid samples')
    print(y_train[0])
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    dataset_p = Dataset_prep(X_train,X_valid, y_train, y_valid, input_shape)
    
    return dataset_p

num_classes = 10
dataset_p = load_data(num_classes)


from tensorflow import keras as k
def build_model(hp, input_shape):
    
    #regularizer
    #l1_strength = hp.Float('l1_strength', min_value = 1e-5, max_value = 1e-3, sampling='log')
    kernel_regularizer = k.regularizers.L2(0.01)
    
    kernel_initializer = hp.Choice("kernel_initializer", values = ["he_normal","random_normal"])
    activation = hp.Choice("activation",values= ["relu","tanh"])
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation =activation, input_shape = input_shape, kernel_initializer = kernel_initializer))
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation, kernel_initializer = kernel_initializer))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    
    # model.add(Conv2D(64, kernel_size=(3,3), activation =activation, input_shape = input_shape, kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
    # model.add(Conv2D(64, kernel_size=(3, 3),
    #              activation=activation, kernel_initializer = kernel_initializer))
    # model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer))
    
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (6, 6), activation=activation, kernel_initializer = kernel_initializer))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=activation,kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


hp = keras_tuner.HyperParameters()

# This will override the `learning_rate` in tuner
hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
input_shape = dataset_p.input_shape
tuner = keras_tuner.RandomSearch(
    hypermodel=lambda hp: build_model(hp, input_shape),
    hyperparameters=hp,
    # Prevents unlisted parameters from being tuned
    tune_new_entries=False,
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="search_a_few",
    
)


# Run the search
tuner.search(dataset_p.X_train, dataset_p.y_train, epochs=5, validation_data=(dataset_p.X_valid, dataset_p.y_valid))


tuner.search_space_summary()

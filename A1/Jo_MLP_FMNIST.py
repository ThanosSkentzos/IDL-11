# %% 
import os

import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.model_selection import train_test_split

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image
from datetime import datetime


# %%
def set_fmnst_data():
    # Use FMINST DATA
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    
    num_classes = 10
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    y_train_scaled = keras.utils.to_categorical(y_train, num_classes)
    y_test_scaled = keras.utils.to_categorical(y_test, num_classes)
    y_val_scaled = keras.utils.to_categorical(y_val, num_classes)
    
    return x_train, x_val, x_test, y_train_scaled, y_val_scaled, y_test_scaled


# %%
def model_mlp(batch_size, activation, optimizer, dropout1, dropout2, dropout3):
    x_train, x_val, x_test, y_train_scaled, y_val_scaled, y_test_scaled = set_fmnst_data()
    
    print(str(optimizer))
    print(str(activation))
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))

    model.add(keras.layers.Dropout(dropout1))
    model.add(keras.layers.Dense(64, activation=activation, kernel_initializer=keras.initializers.HeNormal()))
    model.add(keras.layers.Dropout(dropout2))
    model.add(keras.layers.Dense(64, activation=activation, kernel_initializer=keras.initializers.HeNormal(), kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dropout(dropout3))
    model.add(keras.layers.Dense(64, activation=activation, kernel_initializer=keras.initializers.HeNormal()))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # model.summary()

    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    history = model.fit(x_train, y_train_scaled,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_val, y_val_scaled),
                    callbacks=[early_stopping_cb])
    
    
    score = model.evaluate(x_test, y_test_scaled, verbose=0)
    
    return history, score


# %%
batch_size = [
    # 32, 
    # 64,
    128,
    # 512
]
epochs = 50

activations = [
    'relu',
    # 'sigmoid',
    # 'tanh',
    # 'softmax',
    # keras.layers.LeakyReLU(alpha=0.01),
    # 'selu',
    # 'elu',
    # keras.activations.swish,
]

images = []

kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)

dropouts  = [
    # [0, 0.3, 0],
    # [0.3, 0, 0.3],
    [0.1, 0.3, 0],
    # [0.3, 0.5, 0.3],
    # [0.7, 0.3, 0.7]
]
dropout2  = [
    # 0,
    # 0.1,
    # 0.2,
    # 0.3,
    # 0.5,
    # 0.7
]
dropout3  = [
    # 0,
    # 0.1,
    # 0.2,
    # 0.3,
    # 0.5,
    # 0.7
]
# %%

for each_dp1 in dropouts:
    for each_size in batch_size:
        print(f"Batch Size: {each_size}")
        for each_act in activations:
            optimizers = [
                # keras.optimizers.SGD(),
                # keras.optimizers.SGD(momentum=0.9),
                # keras.optimizers.SGD(nesterov=True),
                # keras.optimizers.Adam(),
                # keras.optimizers.AdamW(weight_decay=1e-4),
                keras.optimizers.RMSprop(),
                # keras.optimizers.Adagrad(),
                # keras.optimizers.Adadelta(),
                # keras.optimizers.Nadam(),
                # keras.optimizers.Ftrl()
            ]
            oprimizer_str = [
                # 'keras.optimizers.SGD()',
                # 'keras.optimizers.SGD(momentum=0.9)',
                # 'keras.optimizers.SGD(nesterov=True)',
                # 'keras.optimizers.Adam()',
                # 'keras.optimizers.AdamW(weight_decay=1e-4)',
                'keras.optimizers.RMSprop()',
                # 'keras.optimizers.Adagrad()',
                # 'keras.optimizers.Adadelta()',
                # 'keras.optimizers.Nadam()',
                # 'keras.optimizers.Ftrl()'
                ]
            for i in range(len(optimizers)):
                startTime = datetime.now()
                history, score = model_mlp(batch_size=each_size, activation=each_act, 
                                        optimizer=optimizers[i], dropout1=each_dp1[0], 
                                        dropout2=each_dp1[1], dropout3=each_dp1[2])
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])

                loss = history.history['loss']
                val_loss = history.history['val_loss']

            x_train, x_val, y_train, y_val = train_test_split(x_train_scaled, y_train, 
                                                            test_size=0.1, random_state=42)

            y_train_scaled = keras.utils.to_categorical(y_train, num_classes)
            y_test_scaled = keras.utils.to_categorical(y_test, num_classes)
            y_val_scaled = keras.utils.to_categorical(y_val, num_classes)
            
            print(str(optimizers[i]))
            print(str(each_act))
            model = keras.models.Sequential()
            model.add(keras.layers.Flatten(input_shape=[28, 28]))

            # model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(64, activation="relu"))
            # model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(64, activation=each_act))
            # model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(64, activation=each_act))
            model.add(keras.layers.Dense(10, activation='softmax'))
            
            early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, 
                                                              restore_best_weights=True)

            # model.summary()

            model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers[i],
                    metrics=['accuracy'])

            history = model.fit(x_train, y_train_scaled,
                            batch_size=each_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(x_val, y_val_scaled),
                            callbacks=[early_stopping_cb])

            score = model.evaluate(x_test, y_test_scaled, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            # Plot the loss
            plt.figure(figsize=(10, 6))
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Model Loss over Epochs')
            plt.legend()
            plt.show()
            plt.savefig("plot_image.png")  # Saves as a PNG image
            
            cell_name = get_cell_name(row_count, 10)
            
            sheet.append((3, "No", 3, each_size, str(each_act), oprimizer_str[i], score[0], score[1]))
            img = Image("plot_image.png")
            
            img.width, img.height = 40, 40         # Resize image if necessary
            sheet.add_image(img, cell_name)
            row_count += 1

workbook.save("sample_with_image.xlsx")


# %%
str(optimizers[0])

# # %%
# batch_size = 128
# num_classes = 10
# epochs = 12

# # input image dimensions
# img_rows, img_cols = 28, 28


# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# # %%
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# print(y_train.shape)
# print(y_test.shape)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.summary()

# # %%
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

# # %%
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])



# %%

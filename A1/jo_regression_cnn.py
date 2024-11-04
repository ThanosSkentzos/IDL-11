# %%
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split


# %%
images = np.load("A1/data/images.npy")
labels = np.load("A1/data/labels.npy")


# %%
# Define the function to map (hour, minute) tuples to categories
def time_to_regression(row):
    return row[0] + round(row[1]/60, 2)

img_rows, img_cols = 75, 75  #pixel size

new_labels = np.apply_along_axis(time_to_regression, axis=1, arr=labels)


#%%
images = images.astype('float32')/255.0

x_train, x_test, y_train, y_test = train_test_split(images, new_labels, test_size=0.10, random_state=42) 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=33) 

# %%
batch_size = 72
epochs = 200

if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
print(x_val.shape)

# %%

def reg_loss_funtion(y_true, y_pred):
    # Calculate the absolute error
    abs_diff = tf.abs(y_true - y_pred)
    wrapped_diff = tf.abs(12 - abs_diff)
    print(abs_diff, wrapped_diff)
    
    below_min_penalty = tf.reduce_sum(tf.square(tf.maximum(0.0, 0 - y_pred)))
    above_max_penalty = tf.reduce_sum(tf.square(tf.maximum(0.0, y_pred - 12)))
    
    range_penalty = below_min_penalty + above_max_penalty

    # Calculate the minimum error
    min_error = tf.minimum(abs_diff, wrapped_diff)

    # Return the mean squared of the minimum error
    return tf.reduce_mean(tf.square(min_error) + 0.1 * range_penalty)


# %%

def build_model(input_shape):

    # model = keras.models.Sequential(
    #     [ 
    #         keras.layers.Conv2D(128, 7, activation="relu", padding="same", input_shape=input_shape), 
    #         keras.layers.MaxPooling2D(2), 
    #         keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
    #         keras.layers.Conv2D(512, 3, activation="relu", padding="same"), 
    #         keras.layers.MaxPooling2D(2), 
    #         keras.layers.Conv2D(512, 3, activation="relu", padding="same"), 
    #         keras.layers.Conv2D(512, 3, activation="relu", padding="same"), 
    #         keras.layers.MaxPooling2D(2), keras.layers.Flatten(), 
    #         keras.layers.Dense(256, activation="relu"), 
    #         keras.layers.Dropout(0.5), 
    #         keras.layers.Dense(128, activation="relu"), 
    #         keras.layers.Dropout(0.5), 
    #         keras.layers.Dense(1, activation="linear")
    #     ]
    # )
    
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu', padding='same',
                    input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(1, activation='linear'))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, 
                                                  weight_decay=1e-5), 
                  loss=reg_loss_funtion, metrics=['mae'])

    return model

# %%

model = build_model(input_shape=input_shape)
model.summary()

#%%
# Initialize EarlyStopping with patience
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # metric to monitor
    patience=20,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)



history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping]
)


# %%
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss[:-1], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.title(f'Model Loss over Epochs 3 Layers batch size of {each_size} for {oprimizer_str[i]} and {each_act}')
plt.legend()
plt.show()
plt.close()

# %%
t = model.predict(x_test)
# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error


score = mean_absolute_error(y_test, t)



# %%
score
# %%

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
# Define the function to map (hour, minute) tuples to float values upto 
def time_to_regression(row):
    return row[0] + round(row[1]/60, 2)

img_rows, img_cols = 75, 75  #pixel size

new_labels = np.apply_along_axis(time_to_regression, axis=1, arr=labels)


#%%
images = images.astype('float32')/255.0

x_train, x_test, y_train, y_test = train_test_split(images, new_labels, test_size=0.10, random_state=42) 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=33) 

# %%
batch_size = 64
epochs = 500

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
    
    below_min_penalty = tf.reduce_sum(tf.square(tf.maximum(0.0, 0 - y_pred)))
    above_max_penalty = tf.reduce_sum(tf.square(tf.maximum(0.0, y_pred - 12)))
    
    range_penalty = below_min_penalty + above_max_penalty

    # Calculate the minimum error
    min_error = tf.minimum(abs_diff, wrapped_diff)

    # Return the mean squared of the minimum error
    return tf.reduce_mean(tf.square(min_error) + 0.1 * range_penalty)

# def reg_custom_mae(y_true, y_pred):
#     abs_diff = tf.abs(y_true - y_pred)
#     wrapped_diff = tf.abs(12 - abs_diff)
#     common_sense_error = tf.minimum(abs_diff, wrapped_diff)
#     return common_sense_error


# class CustomMAE(tf.keras.metrics.Metric):
#     def __init__(self, name="custom_mae", **kwargs):
#         super(CustomMAE, self).__init__(name=name, **kwargs)
#         # Initialize variables to keep track of the total absolute error and count
#         self.total_error = self.add_weight(name="total_error", initializer="zeros")
#         self.count = self.add_weight(name="count", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # Calculate the absolute difference between the true and predicted values

#         abs_diff = tf.abs(y_true - y_pred)
#         wrapped_diff = tf.abs(12 - abs_diff)
#         common_sense_error = tf.minimum(abs_diff, wrapped_diff)        
#         # Sum the absolute errors and update the total error
#         self.total_error.assign_add(tf.reduce_sum(common_sense_error))
        
#         # Update the count with the batch size
#         self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
        
#     def result(self):
#         # Calculate the mean absolute error
#         return self.total_error / self.count

#     def reset_states(self):
#         # Reset the metric's state (total error and count) between epochs
#         self.total_error.assign(0.0)
#         self.count.assign(0.0)


# %%
def build_model(input_shape):
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
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.1))

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
    patience=40,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)



history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping, reduce_lr]
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
plt.title(f'{img_rows} x {img_cols} - Regression loss history with diminishing learning rate')
plt.legend()
plt.show()
plt.close()

# %%
y_test_pred = model.predict(x_test)

# %%
abs_values = abs(y_test_pred.flatten()-y_test)
common_sense_error = np.minimum(12-abs_values, abs_values)

score = np.mean(common_sense_error)

# %%
score

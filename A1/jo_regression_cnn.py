# %%
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split


# %%
images = np.load("data/images_150.npy")
labels = np.load("data/labels_150.npy")

# %%
# function to map (hour, minute) tuples to float values upto 12
def time_to_regression(row):
    return round((row[0] + row[1]/60.0), 3)

img_rows = images.shape[1]
img_cols = images.shape[2]

new_labels = np.apply_along_axis(time_to_regression, axis=1, arr=labels)


#%%
images = images.astype('float32')/255.0

x_train, x_test, y_train, y_test = train_test_split(images, new_labels, test_size=0.20, random_state=42) 
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.50, random_state=33) 

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
# Custom loss function for common sense difference
def reg_loss_funtion(y_true, y_pred):
    # Calculate the absolute error
    abs_diff = tf.abs(y_true - y_pred)
    
    min_error = tf.where((y_pred >= 0.) & (y_pred < 12.), tf.minimum(abs_diff, 12. - abs_diff), tf.abs(y_pred - y_true))

    # Return the mean squared of the minimum error
    return tf.reduce_mean(tf.square(min_error))

def reg_custom_mae(y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    wrapped_diff = tf.abs(12. - abs_diff)
    common_sense_error = tf.minimum(abs_diff, wrapped_diff)
    return tf.reduce_mean(common_sense_error, axis=0)


def custom_huber_loss(y_true, y_pred, delta=1.0):
    abs_diff = tf.abs(y_true - y_pred)
    error = tf.where((y_pred >= 0.) & (y_pred < 12.), tf.minimum(abs_diff, 12. - abs_diff), tf.abs(y_pred - y_true))
    is_small_error = tf.abs(error) <= delta
    small_error_loss = 0.5 * tf.square(error)
    large_error_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.reduce_mean(tf.where(is_small_error, small_error_loss, large_error_loss))

# %%
optimizer = keras.optimizers.Adam(learning_rate=1e-3,  weight_decay=1e-5)
# optimizer = "adam"
loss_function = "huber"
metrics = ["mae"]

# %%
def build_model(input_shape):
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(32, kernel_size=(5, 5),
                    activation='relu', padding='same',
                    input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(32, kernel_size=(5, 5),
                    activation='relu', padding='same',
                    input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(keras.layers.Dropout(0.1))
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

    # model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(keras.layers.Dropout(0.3))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    
    model.add(keras.layers.Dense(1, activation='linear'))
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    return model


# %%

model = build_model(input_shape=input_shape)
model.summary()

#%%
# Initialize EarlyStopping with patience
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # metric to monitor
    patience=50,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.8, patience=10, min_lr=1e-6
)

#%%
history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping, reduce_lr]
)

#%%
# model.save_weights('75_regression_without_rlrop.weights.h5')


# %%
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
# plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'{img_rows} x {img_cols} - Regression loss history with diminishing learning rate and custom huber loss')
plt.legend()
plt.show()
plt.close()

# %%
y_test_pred = model.predict(x_test).flatten()

# %%
abs_values = abs(y_test_pred.flatten()-y_test)
# common_sense_error = np.minimum(12-abs_values, abs_values)
common_sense_error = np.where((y_test_pred.flatten() >= 0.) & (y_test_pred.flatten() < 12.), np.minimum(abs_values, 12. - abs_values), np.minimum(np.abs(y_test_pred.flatten() - y_test), np.abs(y_test_pred.flatten() - (12. - y_test))))

score = np.mean(common_sense_error)
score

#%%
plt.scatter(y_test_pred, y_test)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Regression with diminishing learning rate with {loss_function}')

# %%
sorted_indices = np.argsort(y_test)
sorted_y_test = y_test[sorted_indices]
sorted_y_pred = y_test_pred[sorted_indices]

same_indices = np.where(sorted_y_test == sorted_y_pred)[0]

#%%
# Plot y_true and y_pred
plt.figure(figsize=(10, 6))
plt.plot(sorted_y_test, label='y_true', marker='o', color='blue', linestyle='None')
plt.plot(sorted_y_pred, label='y_pred', marker='o', color='orange', linestyle='None')

# Highlight points where y_true and y_pred are the same
plt.scatter(same_indices, sorted_y_test[same_indices], color='green', s=100, label='Matching Points')

# Draw dotted lines between y_true and y_pred points at each index
for i in range(len(sorted_y_test)):
    plt.plot([i, i], [sorted_y_test[i], sorted_y_pred[i]], 'k:', alpha=0.6)  # 'k:' makes a black dotted line

plt.xlabel('Index')
plt.ylabel('Values')
plt.title(f'Plot of y_true and y_pred connected for with diminishing learning rate and {loss_function}')
plt.legend()

# Show the plot
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

values = -common_sense_error[sorted_indices]  # Random values for each angle

# Create polar plot
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.scatter(2 * np.pi * sorted_y_test/12, values, marker='o')

# Add title and show plot
ax.set_title(f"Regression Common sense Error with Dim LR and {loss_function}")
theta_ticks = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # 12 hours around the clock
ax.set_xticks(theta_ticks)
ax.set_yticks(np.arange(-6, 0))

ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])

plt.show()


# %%
y_train_pred = model.predict(x_train.reshape(x_train.shape[0], img_rows, img_cols)).flatten()
abs_values = abs(y_train_pred.flatten()-y_train)
common_sense_error = np.minimum(abs_values, 12. - abs_values)
cse_train = np.mean(common_sense_error)

#%%
y_val_pred = model.predict(x_val.reshape(x_val.shape[0], img_rows, img_cols)).flatten()
abs_values = abs(y_val_pred.flatten()-y_val)
common_sense_error = np.minimum(abs_values, 12. - abs_values)
cse_val = np.mean(common_sense_error)

#%%

min(history.history['val_loss'])

#%%
min_val_loss_index = history.history['val_loss'].index(min(history.history['val_loss']))
history.history['loss'][min_val_loss_index]
# %%
history.history['mae'][min_val_loss_index]

#%%
history.history['val_mae'][min_val_loss_index]

#%%
print(f"Min Val loss : {min(history.history['val_loss'])}")
print(f"Loss at this : {history.history['loss'][min_val_loss_index]}")
print(f"MAE : {history.history['mae'][min_val_loss_index]}")
print(f"MAE Val : {history.history['val_mae'][min_val_loss_index]}")
print(f"Common Sense MAE Train : {cse_train}")
print(f"Common Sense MAE Val : {cse_val}")
print(f"Common Sense MAE Test : {score}")
# %%

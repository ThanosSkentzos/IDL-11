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
    return round(row[1]/60, 2)

img_rows, img_cols = 75, 75  #pixel size

new_labels = np.apply_along_axis(time_to_regression, axis=1, arr=labels)


#%%
images = images.astype('float32')/255.0

x_train, x_test, y_train, y_test = train_test_split(images, new_labels, test_size=0.10, random_state=42) 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=33) 

# %%
batch_size = 64
epochs = 12

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
    # Wrap-around error for periodic values with period 12
    wrapped_diff = tf.math.round((1.0 - abs_diff)*100)/100
    print(abs_diff, wrapped_diff)
    # Calculate the minimum error
    min_error = tf.minimum(abs_diff, wrapped_diff)
    penalty = tf.maximum(0.0, y_pred) ** 2  
    # Return the mean squared of the minimum error
    return tf.reduce_mean(tf.square(min_error))


# %%

def build_model(input_shape):

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.10))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.10))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.10))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.10))

    model.add(keras.layers.Dense(1, activation='linear'))
    
    model.compile(
        optimizer="adam",
        loss=reg_loss_funtion,
        metrics=['mae']
    )

    return model

#%%

model = build_model(input_shape=input_shape)
model.summary()

#%%
history = model.fit(x_train, y_train,
                batch_size=64,
                epochs=100,
                verbose=1,
                validation_data=(x_val, y_val))


# %%
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
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

import tensorflow as tf
import keras
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.initializers import HeUniform, GlorotUniform
import pandas as pd

from tensorflow.keras.layers import BatchNormalization
#Starting with classification
images = np.load("/kaggle/input/75-pix/images.npy")
labels = np.load("/kaggle/input/75-pix/labels.npy")



class Dataset_prep():
    def __init__(self, X_train,X_valid, y_train, y_valid, input_shape)-> None:
        self.input_shape = input_shape
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        
        
# Define the function to map (hour, minute) tuples to categories
def time_to_category(time_tuple):
    hour, minute = time_tuple
    # Determine the category based on the minute
    if minute < 30:
        category = hour * 2  # First half-hour of the hour
    else:
        category = hour * 2 + 1  # Second half-hour of the hour

    # Return the category in range [0, 23]
    return category % 24

img_rows, img_cols = 75, 75  #pixel size

# Example usage
list_of_times = labels
categories = [time_to_category(time) for time in list_of_times]
classes = list(set(categories))  #classes 
new_labels = np.array(categories)


X_train, X_test,y_train,  y_test = train_test_split(images, new_labels, test_size=0.10, random_state=42) 

split_index = int(0.2 * len(X_train))  # 20% of the data
X_train = X_train.astype('float32')

X_valid = X_train[:split_index]/255.0
X_train = X_train[split_index:]/255.0

y_valid = y_train[:split_index]
y_train = y_train[split_index:]



# lets start the initilization for layer, come on!

batch_size = 64
num_classes = len(classes)
epochs = 12

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    
print(X_train.shape)
print(X_valid.shape)
 

dataset_p = Dataset_prep(X_train,X_valid, y_train, y_valid, input_shape)
    
    


def custom_cyclic_loss(y_true, y_pred, num_classes=24):
    # to be improved
    
    rps_weight=0.01
    cyclic_weight=0.05
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.nn.softmax(y_pred)
    
    # Calculate cumulative predicted probabilities for RPS
    cumulative_pred = tf.cumsum(y_pred, axis=1)
    
    # Create cumulative true distribution
    mask = tf.sequence_mask(y_true + 1, maxlen=num_classes, dtype=tf.float32)
    cumulative_true = tf.cumsum(mask, axis=1)
    
    # Calculate RPS with scaling
    rps = tf.reduce_mean(tf.square(cumulative_pred - cumulative_true), axis=1)
    scaled_rps = rps_weight * rps  # Scale down RPS to make it less dominant
    #tf.print("Scaled RPS component:", scaled_rps)
    
    # Cyclic loss calculation
    class_indices = tf.range(num_classes, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    direct_diff = tf.abs(class_indices - tf.expand_dims(y_true, axis=-1))
    wrap_around_diff = num_classes - direct_diff
    cyclic_distances = tf.minimum(direct_diff, wrap_around_diff)
    
    cyclic_loss = tf.reduce_sum(y_pred * cyclic_distances, axis=1)
    scaled_cyclic_loss = cyclic_weight * cyclic_loss
    #tf.print("Scaled Cyclic Loss component:", scaled_cyclic_loss)
    
    # Combine scaled RPS and cyclic loss
    final_loss = scaled_rps + scaled_cyclic_loss
    final_mean_loss = tf.reduce_mean(final_loss)
    #tf.print("Final Loss:", final_mean_loss)
    
    return final_mean_loss

class DisplayPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # Run predictions on a small batch of validation data
        sample_batch, _ = self.validation_data  # Get validation images and labels
        predictions = self.model.predict(sample_batch[:5])  # Predict on first 5 samples
        print(f"Epoch {epoch + 1} predictions:", predictions)  # Display first 5 predictions


def custom_loss(y_true, y_pred):
    loss = min((24-y_pred), (y_pred-y_true))
    return loss


def build_model(input_shape):
    num_classes = 24
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, padding = "same", kernel_initializer=GlorotUniform()))
    BatchNormalization(),
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', padding = "same", kernel_initializer=GlorotUniform()))
    BatchNormalization(),
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeUniform()))
    BatchNormalization(),
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeUniform()))
    BatchNormalization(),
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=GlorotUniform()))
    BatchNormalization(),
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer=GlorotUniform()))
    BatchNormalization(),
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=GlorotUniform()))
    BatchNormalization(),
    """
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  
        loss="sparse_categorical_crossentropy",
                  #loss=lambda y_true, y_pred: custom_cyclic_loss(y_true, y_pred, num_classes=24),
                  #optimizer = Adam(learning_rate=0.001),

                  metrics=['accuracy'])
    #model.compile(
    #               optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #               #optimizer = Adam(learning_rate=0.001),
    #               loss = lambda y_true, y_pred: custom_loss(y_true, y_pred),
    #               metrics=['accuracy'])
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        #loss=lambda y_true, y_pred: custom_cyclic_loss(y_true, y_pred, num_classes=num_classes),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    return model


def train_model(model):
    validation_sample = (X_valid[:5], y_valid[:5])  # Use first 5 samples for demonstration
    

    # history = model.fit(X_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1,
    #           validation_data=(X_valid, y_valid))




    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "classifier_model.keras", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs=100,validation_data = (X_valid, y_valid), callbacks=[checkpoint_cb, early_stopping_cb, DisplayPredictionsCallback(validation_sample)])
    
    data_history = pd.DataFrame(history.history)
    #history_model_data = pd.DataFrame(data_history).plot(figsize= (8,5))
#     plt.grid(True)
#     plt.gca().set_ylim(0,1) #set the limit on y axis values - we are cheecking loss and accuracy hence from 0 to 1
#     plt.xlabel('epochs')
#     plt.ylabel('score')
#     plt.tight_layout()
#     plt.show()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.legend(loc="best")

    # Adjust y-axis limits if necessary
    # plt.ylim(0, 1)  # Uncomment if you want to set limits for accuracy between 0 and 1

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    data_history.to_csv("train_report.csv")
    return data_history

model = build_model(input_shape= input_shape)
model.summary()


history = train_model(model)


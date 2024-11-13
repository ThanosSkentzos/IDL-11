from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.initializers import HeUniform, GlorotUniform
from tensorflow.keras.layers import BatchNormalization
import pandas as pd
import keras
import matplotlib.pyplot as plt
import numpy as np




images = np.load("/data/75-pix/images.npy")
labels = np.load("/data/75-pix/labels.npy")

#Starting with classification


# Define the function to map (hour, minute) tuples to categories
def time_to_category_24(time_tuple):
    hour, minute = time_tuple
    # Determine the category based on the minute
    if minute < 30:
        category = hour * 2  # First half-hour of the hour
    else:
        category = hour * 2 + 1  # Second half-hour of the hour

    # Return the category in range [0, 23]
    return category % 24

img_rows, img_cols = images.shape[1], images.shape[2]  #pixel size

# Example usage
list_of_times = labels
categories = [time_to_category_24(time) for time in list_of_times]
classes = list(set(categories))  #classes 
new_labels = np.array(categories)

#images = (images - images.mean())/images.std()
X_train, X_test,y_train,  y_test = train_test_split(images, new_labels, stratify=new_labels, test_size=0.10, random_state=42) 
X_train, X_valid,y_train,  y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.10, random_state=42) 

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')

# lets start the initilization for layer, come on!

batch_size = 64
num_classes = len(classes)
epochs = 12

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



def circular_loss(y_true, y_pred, num_classes=24):
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Circular loss component for temporal/cyclic structure
    theta_true = 2 * np.pi * tf.cast(y_true, tf.float32) / num_classes
    class_indices = tf.range(num_classes, dtype=tf.float32)
    theta_pred = 2 * np.pi * tf.reduce_sum(class_indices * y_pred, axis=-1) / num_classes
    cos_similarity = tf.cos(theta_true - theta_pred)
    circular_loss = 1 - cos_similarity
    
    # Calculate mean values to get the relative scale of each loss component
    ce_loss_mean = tf.reduce_mean(ce_loss)
    circular_loss_mean = tf.reduce_mean(circular_loss)
    total_mean = ce_loss_mean + circular_loss_mean

    # Use tf.cond to handle the alpha calculation dynamically within the graph
    alpha = tf.cond(total_mean > 0, 
                    lambda: ce_loss_mean / total_mean,  # Proportional weight for ce_loss
                    lambda: 0.5)  # Fallback value if total_mean is zero
    
    # Combine the two losses with dynamically calculated alpha
    combined_loss = alpha * ce_loss + (1 - alpha) * circular_loss
    return combined_loss


####### prep for training
def common_sense_error(y_true, y_pred, num_classes=24):
    y_true = tf.cast(y_true, tf.float32)
    
    y_pred = tf.argmax(y_pred, axis=1)
    y_pred = tf.cast(y_pred, tf.float32)
    abs_diff = tf.abs(y_true - y_pred)
    wrapped_diff = tf.abs(tf.cast(num_classes, tf.float32) - abs_diff)
    common_sense_error = tf.minimum(abs_diff, wrapped_diff)
    return tf.reduce_mean(common_sense_error, axis=0)

    
def build_model(input_shape, num_classes):
 
    
    model = Sequential()

    # Convolutional Block 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding="same", kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Convolutional Block 2
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=HeUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=HeUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Convolutional Block 3
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=HeUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=HeUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Optional: Convolutional Block 4
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer=HeUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=GlorotUniform()))

   
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=lambda y_true, y_pred: circular_loss(y_true, y_pred, num_classes=num_classes),
       
        metrics=['accuracy', common_sense_error]
    )

    return model



def train_model(model):
    


    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "classifier_model.keras", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss',
        patience=10, restore_best_weights=True
    )
    history = model.fit(X_train, y_train, batch_size =64, epochs=4,validation_data = (X_valid, y_valid), callbacks=[checkpoint_cb, early_stopping_cb])
    model.save('final_model.keras')
    data_history = pd.DataFrame(history.history)
    #history_model_data = pd.DataFrame(data_history).plot(figsize= (8,5))
#     plt.grid(True)
#     plt.gca().set_ylim(0,1) #set the limit on y axis values - we are cheecking loss and accuracy hence from 0 to 1
#     plt.xlabel('epochs')
#     plt.ylabel('score')
#     plt.tight_layout()
#     plt.show()
#     plt.plot(history.history['accuracy'], label='accuracy')
#     plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['loss'], label='training custom loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.plot(history.history['common_sense_error'], label='common sense error')
    
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.legend(loc="best")

    # Adjust y-axis limits if necessary
    # plt.ylim(0, 1)  # Uncomment if you want to set limits for accuracy between 0 and 1

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    data_history.to_csv("train_report.csv")
    return model, data_history
print(num_classes)
model = build_model(input_shape= input_shape, num_classes=num_classes)
model.summary()


model, history = train_model(model)



#get predictions
predictions =  model.predict(X_test)
predicted_classes = predictions.argmax(axis =1)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_classes)
print("Accuracy:", accuracy)

# Detailed classification report (precision, recall, F1-score for each class)
print("Classification Report:\n", classification_report(y_test, predicted_classes))

# Confusion matrix for further insight
conf_matrix = confusion_matrix(y_test, predicted_classes)
print("Confusion Matrix:\n", conf_matrix)

abs_values = abs(predicted_classes-y_test)
common_sense_error = np.minimum(abs_values, num_classes - abs_values)
cse_test = np.mean(common_sense_error)

print("Common sense error in Test data", cse_test)

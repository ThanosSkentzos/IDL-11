from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.initializers import HeUniform, GlorotUniform
import pandas as pd
import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as k
from tensorflow.keras.losses import categorical_crossentropy, mse


images = np.load("/data/150-pix/images.npy")
labels = np.load("/data/150-pix/labels.npy")

#make labels for hour and minutes


def preprocess_labels_com(labels, num_classes):
    
    
    hour_labels = labels[:,0]
    hour_labels = k.utils.to_categorical(hour_labels, num_classes)
    minute_labels = labels[:,1]/60
  
    return hour_labels, minute_labels


# dataset preparation
num_classes = len(np.unique(labels[:,0]))

images = images/255
X_train1, X_test,y_train1,  y_test = train_test_split(images, labels, test_size=0.10, random_state=42) 
X_train, X_valid, y_train,  y_valid = train_test_split(X_train1, y_train1, test_size=0.10, random_state=42) 


X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')



 
img_rows = X_train.shape[1]
img_cols = X_train.shape[2]

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
    X_test= X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    X_test= X_test.reshape(X_test.shape[0],  img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    
y_train_hours, y_train_minutes = preprocess_labels_com(y_train, num_classes)

y_valid_hours, y_valid_minutes = preprocess_labels_com(y_valid, num_classes)
y_test_hours, y_test_minutes = preprocess_labels_com(y_test, num_classes)



input_ = k.layers.Input(shape=X_train.shape[1:])

# convolutional layers
c1 = k.layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu',padding='same',
                    input_shape=input_shape)(input_)
m1 = k.layers.MaxPooling2D(pool_size=(2, 2))(c1)
b1 = k.layers.BatchNormalization()(m1)


#2nd layers
c2 = k.layers.Conv2D(64, kernel_size=(3, 3),
                    activation='relu',padding='same',
                    input_shape=input_shape)(b1)
d2=   k.layers.Dropout(0.10)(c2)
m2 = k.layers.MaxPooling2D(pool_size=(2, 2))(d2)
b2 = k.layers.BatchNormalization()(m2)

#3rd layers
c3 = k.layers.Conv2D(128, kernel_size=(4, 4),
                    activation='relu',padding='same',
                    input_shape=input_shape)(b2)
d3 = k.layers.Dropout(0.10)(c3)
m3 = k.layers.MaxPooling2D(pool_size=(2, 2))(d3)
b3 = k.layers.BatchNormalization()(m3)

                           
#4th layers
c4 = k.layers.Conv2D(256, kernel_size=(5, 5),
                    activation='relu',padding='same',
                    input_shape=input_shape)(b3)
d4 = k.layers.Dropout(0.10)(c4)
m4 = k.layers.MaxPooling2D(pool_size=(2, 2))(d4)
b4 = k.layers.BatchNormalization()(m4)
                           

#final layer begins
f = k.layers.Flatten()(b4)
                           
                           
#for hours final layers
dense_hour1 = k.layers.Dense(128, activation='relu')(f)
dense_dropout1 = k.layers.Dropout(0.10)(dense_hour1)
final_hour = k.layers.Dense(num_classes, activation='softmax')(dense_dropout1)


# for minutes final layers
minute_dense1 = k.layers.Dense(512, activation='relu')(f)

minute_dropout1 = k.layers.Dropout(0.5)(minute_dense1)
minute_dense2 = k.layers.Dense(128, activation='relu')(minute_dropout1)
minute_dropout2 =  k.layers.Dropout(0.3)(minute_dense2)
    
minute_dense3 = k.layers.Dense(64, activation='relu')(minute_dropout2)
minute_dropout3 =  k.layers.Dropout(0.1)(minute_dense3)

final_minute = k.layers.Dense(1, activation='linear')(minute_dropout3)
                          
model = k.Model(inputs=input_, outputs=[final_hour, final_minute])
model.summary()


model.compile(loss=['categorical_crossentropy', 'mse'], 
              optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              metrics=['accuracy', 'mae'])


checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"combined_model.keras", save_best_only=True
    )
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # metric to monitor
    patience=30,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)

history = model.fit( X_train, [y_train_hours, y_train_minutes],       
        epochs=200,
        verbose = 1,
        validation_data=(X_valid, [y_valid_hours, y_valid_minutes]),
        callbacks=[checkpoint_cb, early_stopping, reduce_lr])


total_loss, hour_loss, minute_loss = model.evaluate(
        X_test, [y_test_hours, y_test_minutes])

model.save(f'submission1_model_combined_{X_train.shape[1]}.keras')
data_history = pd.DataFrame(history.history)
data_history.to_csv("better_multihead_version.csv")


y_pred_main, y_pred_aux = model.predict(X_test)
prediction_hours = y_pred_main.argmax(axis=-1)
y_pred_aux = y_pred_aux.flatten()
prediction_mins = y_pred_aux*60

abs_values = abs(prediction_hours-y_test[:,0])
common_sense_error = np.minimum(abs_values, 12 - abs_values)
cse_test = np.mean(common_sense_error)
print(cse_test)

abs_values = abs(prediction_mins-y_test[:,1])
common_sense_error = np.minimum(abs_values, 60 - abs_values)
cse_test = np.mean(common_sense_error)
print(cse_test)



total_loss, hour_loss, minute_loss = model.evaluate(
        X_test, [y_test_hours, y_test_minutes])

# %%
# IMPORTS
import os, pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_AFFINITY"] = "noverbose"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras as k
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%% 
# READ  & SCALE
import numpy as np
images = np.load("data/images_small.npy")
labels = np.load("data/labels_small.npy")
images = images.astype(float)/255


#%% 
# PREPARE LABELS
num_classes=24 # trying every half hour
labels_24 = 2*labels[:,0] + labels[:,1]//30
assert num_classes == len(set(labels_24))
labels_24 = labels_24.astype(float)
encoded_labels = k.utils.to_categorical(labels_24, num_classes)
# %% 
# SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    images, encoded_labels, test_size=0.1, random_state=42
)
X, X_val, y, y_val= train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)
img_rows, img_cols = images.shape[1:]
input_shape=(img_rows,img_cols,1)

#%% 
# MODEL
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

model.summary
#%% 
# CALLBACKS
checkpoint_cb = k.callbacks.ModelCheckpoint(
    "cla24.keras", 
    save_best_only=True,
    monitor="val_loss",
)
early_stopping_cb = k.callbacks.EarlyStopping(
    patience=42, restore_best_weights=True,monitor="val_loss"
)
#%%
# TRAIN
batch_size=64
epochs=500
history = model.fit(X, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=[checkpoint_cb, early_stopping_cb])
# %% 
# Plot results
import pandas as pd
import matplotlib.pyplot as plt
hist = pd.DataFrame(history.history)
acc_col = ["accuracy","val_accuracy"]
loss_col = ["loss","val_loss"]
hist[acc_col].plot(figsize=(8, 5))
hist[loss_col].plot(figsize=(8,5))
hist.to_csv('cla24.csv',index=False)
plt.grid(True)
# %%
# calculate accuract
y_pred  = model.predict(X_test)
y_pred_class = np.argmax(y_pred,axis=1)
y_true_class= np.argmax(y_test,axis=1)
"test accuracy",sum(y_pred_class==y_true_class)/len(y_true_class)
# %%

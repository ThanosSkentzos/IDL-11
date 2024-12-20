# %%
# IMPORTS
import os, pickle

import pandas as pd
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
images = np.load("data/images.npy")
labels = np.load("data/labels.npy")
images = images.astype(float)/255


#%% 
# PREPARE LABELS
num_classes=24 # trying every half hour
labels_24 = 2*labels[:,0] + labels[:,1]//30
assert num_classes == len(set(labels_24))
labels_24 = labels_24.astype(float)
encoded_labels = k.utils.to_categorical(labels_24, num_classes)
counts = pd.DataFrame(labels_24).value_counts()

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
from thanos_models import create_model
model = create_model(input_shape=input_shape,num_classes=num_classes)
model.summary()

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
import matplotlib.pyplot as plt
hist = pd.DataFrame(history.history)
acc_col = ["accuracy","val_accuracy"]
loss_col = ["loss","val_loss"]
title = f'{X.shape[1]}x{X.shape[2]} {num_classes} classes'
hist[acc_col].plot(figsize=(8, 5),xlabel='epochs',ylabel='accuracy',title=title+' acc history')
plt.savefig('cla24_acc.png')

hist[loss_col].plot(figsize=(8,5),xlabel='epochs',ylabel='loss',title=title+' loss history')
plt.savefig('cla24_loss.png')

hist.to_csv('cla24.csv',index=False)
# %%
# calculate test accuracy ==> 88.3%
y_pred  = model.predict(X_test)
y_pred_class = np.argmax(y_pred,axis=1)
y_true_class= np.argmax(y_test,axis=1)
"test accuracy",sum(y_pred_class==y_true_class)/len(y_true_class)
# %%

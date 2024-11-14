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
import pandas as pd

#%% 
# READ  & SCALE
import numpy as np
images = np.load("data/images.npy")
labels = np.load("data/labels.npy")
images = images.astype(float)/255


#%% 
# PREPARE LABELS
num_classes=720//5 # trying every half hour
labels_144 = (60//5)*labels[:,0] + labels[:,1]//5
assert num_classes == len(set(labels_144))
labels_144= labels_144.astype(float)
encoded_labels = k.utils.to_categorical(labels_144, num_classes)

counts = pd.DataFrame(labels_144).value_counts()
print(f"counts are: {counts.mean()} per class")
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
    f"cla{num_classes}.keras", 
    save_best_only=True,
    monitor="val_loss",
    )
early_stopping_cb = k.callbacks.EarlyStopping(  
    patience=42,
    restore_best_weights=True,
    monitor="val_loss",
    )
cb =  [checkpoint_cb,early_stopping_cb]
#%%
# TRAIN
batch_size=64
epochs=500
history = model.fit(X, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=cb)
# %% 
# Plot results
import pandas as pd
import matplotlib.pyplot as plt
hist = pd.DataFrame(history.history)
acc_col = ["accuracy","val_accuracy"]
loss_col = ["loss","val_loss"]
title = f'{X.shape[1]}x{X.shape[2]} {num_classes} classes'
hist[acc_col].plot(figsize=(8, 5),xlabel='epochs',ylabel='accuracy',title=title+' acc history')
plt.savefig(f'cla{num_classes}_acc.png')

hist[loss_col].plot(figsize=(8,5),xlabel='epochs',ylabel='loss',title=title+' loss history')
plt.savefig(f'cla{num_classes}_loss.png')

hist.to_csv(f'cla{num_classes}.csv',index=False)
# %%
# calculate accuracy -->? 
y_pred  = model.predict(X_test)
y_pred_class = np.argmax(y_pred,axis=1)
y_true_class= np.argmax(y_test,axis=1)
"test accuracy",sum(y_pred_class==y_true_class)/len(y_true_class)

# %%

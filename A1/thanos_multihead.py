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

# %%
# READ  & SCALE
import numpy as np

images = np.load("data/images.npy")
labels = np.load("data/labels.npy")
images = images.astype(float) / 255


# %%
# PREPARE CATEGORICAL LABELS
num_classes = 12  # trying every half hour
hour_labels = labels[:, 0]
assert num_classes == len(set(hour_labels))
hour_labels = hour_labels.astype(float)
encoded_labels = k.utils.to_categorical(hour_labels, num_classes)
counts = pd.DataFrame(hour_labels).value_counts()

# %%
# PREPARE NUMERICAL LABELS
minute_labels = labels[:, 1]/60.0
# %%
# JOIN LABELS IN DF
labels = pd.DataFrame(encoded_labels)
labels["minutes"] = minute_labels
# %%
from thanos_models import split_data

split_h_m = lambda x: (x.values[:, 0:num_classes], x.values[:, num_classes])
X_train, X_test, y_train, y_test = split_data(images, labels.copy(), percent=0.1)
X, X_val, y, y_val = split_data(X_train, y_train, percent=0.1)

y_test_h, y_test_m = split_h_m(y_test)
y_val_h, y_val_m = split_h_m(y_val)
y_h, y_m = split_h_m(y)
# %%
# MODEL
from thanos_models import create_multihead_model

img_rows, img_cols = images.shape[1:]
input_shape = (img_rows, img_cols, 1)
model = create_multihead_model(input_shape=input_shape, num_classes=num_classes)
model.summary()


# %%
# TRAIN VARIABLES
model.compile(loss=['categorical_crossentropy', 'mse'], 
              optimizer=k.optimizers.Adam(learning_rate=0.0001), 
              metrics=['accuracy', 'mae'])
checkpoint_cb = k.callbacks.ModelCheckpoint(
        f"combined_model.keras", save_best_only=True
    )
early_stopping = k.callbacks.EarlyStopping(
    monitor='val_loss',  # metric to monitor
    patience=42,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)
reduce_lr = k.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)
#%%
# TRAINING
batch_size = 64
epochs=500
history = model.fit( X, [y_h, y_m], 
        batch_size=batch_size,
        epochs=epochs,
        verbose = 1,
        validation_data=(X_val, [y_val_h, y_val_m]),
        callbacks=[checkpoint_cb, early_stopping, reduce_lr])
# %%
# Plot results
import pandas as pd
import matplotlib.pyplot as plt
hist = pd.DataFrame(history.history)
acc_col = ["dense_1_accuracy","val_dense_1_accuracy"]
cla_loss_col = ["dense_1_loss","val_dense_1_loss"]
loss_col = ["dense_5_loss","val_loss"]
title = f'{X.shape[1]}x{X.shape[2]} {num_classes} classes'
hist[acc_col].plot(figsize=(8, 5),xlabel='epochs',ylabel='accuracy',title=title+' acc history')
plt.savefig('multi_cla_acc.png')

hist[cla_loss_col].plot(figsize=(8,5),xlabel='epochs',ylabel='loss',title=title+' loss history')
plt.savefig('multi_cla_loss.png')
hist[loss_col].plot(figsize=(8,5),xlabel='epochs',ylabel='loss',title=title+' loss history')
plt.savefig('multi_loss.png')

hist.to_csv('multi.csv',index=False)

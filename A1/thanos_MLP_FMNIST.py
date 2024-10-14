# %%
# IMPORTS
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_AFFINITY"] = "noverbose"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras as k
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# DATA
fashion_mnist = k.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full_scaled = (X_train_full - X_train_full.mean()) / X_train_full.std()

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full_scaled, y_train_full, test_size=0.1, random_state=42
)


# %%
# MODEL
reg = {
    "kernel_regularizer": k.regularizers.L2(0.1),
    "kernel_initializer": k.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
    "bias_initializer": "zeros",
}


def NN():
    model = k.models.Sequential()
    model.add(k.layers.Flatten(input_shape=[28, 28]))
    model.add(k.layers.Dense(64, activation="relu", **reg))
    # model.add(k.layers.Dropout(0.1))
    model.add(k.layers.Dense(64, activation="relu", **reg))
    # model.add(k.layers.Dropout(0.1))
    model.add(k.layers.Dense(64, activation="relu", **reg))
    # model.add(k.layers.Dropout(0.1))
    model.add(k.layers.Dense(10, activation="softmax"))
    return model


model = NN()
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.summary()

# %%
# TRAIN
history = model.fit(
    X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val)
)

# %%
# HISTORY
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
# set the vertical range to [0-1]
plt.show()

# %%

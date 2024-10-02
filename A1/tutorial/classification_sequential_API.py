# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_AFFINITY"] = "noverbose"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras as k

tf.__version__
# %%
# tf.debugging.set_log_device_placement(True)

# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

# print(c)
# gpus = tf.config.list_physical_devices('GPU')

# %% DATASET
fashion_mnist = k.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

X_train_full.shape, X_train_full.dtype


# %%
def NN():
    model = k.models.Sequential()
    model.add(k.layers.Flatten(input_shape=[28, 28]))
    model.add(k.layers.Dense(300, activation="relu"))
    model.add(k.layers.Dense(100, activation="relu"))
    model.add(k.layers.Dense(10, activation="softmax"))
    return model


model = NN()
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
)
model.summary()

# %% TRAIN
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# %% PLOT METRICS
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

# %% EVALUATE
model.evaluate(X_test, y_test)
# %% PREDICT
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
# %%
# %%

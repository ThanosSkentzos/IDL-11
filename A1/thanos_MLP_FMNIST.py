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

best_params_filename = "fmnist_mlp_best_params.pickle"

# %%
# DATA
fashion_mnist = k.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
m, s = X_train_full.mean(), X_train_full.std()
# X_train_full_scaled = (X_train_full - m) / s
# X_test = (X_test - m) / s  # should we scale with its own values
X_train_full_scaled = X_train_full / 255
X_test_scaled = X_test / 255

y_train_full = tf.keras.utils.to_categorical(y_train_full, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full_scaled, y_train_full, test_size=0.1, random_state=42
)


# %%
# MODEL
import keras_tuner


def NN(hp):
    global_params = {
        "kernel_regularizer": k.regularizers.L2(0.01),
        "kernel_initializer": k.initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=None
        ),
        "bias_initializer": "zeros",
        # "units": hp.Int("units", min_value=32, max_value=512, step=32),
        # "activation": hp.Choice("activation", ["relu", "tanh"]),
    }
    l1_params = {
        **global_params,
        "units": hp.Int("units1", min_value=256, max_value=512, step=32),
        "activation": hp.Choice("activation1", ["relu", "tanh"]),
    }

    l2_params = {
        **global_params,
        "units": hp.Int("units2", min_value=128, max_value=512, step=32),
        "activation": hp.Choice("activation2", ["relu", "tanh"]),
    }

    l3_params = {
        **global_params,
        "units": hp.Int("units3", min_value=64, max_value=256, step=32),
        "activation": hp.Choice("activation3", ["relu", "tanh"]),
    }
    model = k.models.Sequential()
    model.add(k.layers.Flatten(input_shape=[28, 28]))

    model.add(k.layers.Dense(**l1_params))
    # model.add(k.layers.Dropout(0.1))
    model.add(k.layers.Dense(**l2_params))
    # model.add(k.layers.Dropout(0.1))
    model.add(k.layers.Dense(**l3_params))
    # model.add(k.layers.Dropout(0.1))
    model.add(k.layers.Dense(10, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


model = NN(keras_tuner.HyperParameters())

model.summary()


# %%
# TUNE

tuner = keras_tuner.RandomSearch(
    hypermodel=NN,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=1,
    overwrite=True,
    directory="models",
    project_name="idl",
)
tuner.search_space_summary()
# %%
# TUNE search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
models = tuner.get_best_models(5)
best_model = models[0]
best_model.summary()

# get best params and fit model
best_hps = tuner.get_best_hyperparameters(5)
# (384,relu),(224,tanh),(224,relu)
with open(best_params_filename, "wb") as f:
    pickle.dump(best_hps[0], f)

# %%
# GET TUNING RESULTS & TRAIN
with open(best_params_filename, "rb") as f:
    best_params = pickle.load(f)
model = NN(best_params)
model.summary()
history = model.fit(X_train_full_scaled, y_train_full, batch_size=64, epochs=50)
model.save_weights("model.weights.h5")

# HISTORY
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
# set the vertical range to [0-1]
plt.show()

pred = model.predict(X_test).argmax(axis=1)
y_test_vals = y_test.argmax(axis=1)
print(sum(pred==y_test_vals)/len(pred),"test accuracy")

# %%
# CIFAR 10 DATA and MODEL
c10 = k.datasets.cifar10
(X_train_full_c, y_train_full_c), (X_test_c, y_test_c) = c10.load_data()
y_train_full_c = tf.keras.utils.to_categorical(y_train_full_c, 10)
y_test_c = tf.keras.utils.to_categorical(y_test_c, 10)
X_train_full_c = X_train_full_c.mean(axis=-1)
mc, sc = X_train_full_c.mean(), X_train_full_c.std()
X_train_full_c_scaled = (X_train_full_c - mc) / sc
X_test_c = (X_test_c - mc) / sc  # should we scale with its own values
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full_c_scaled, y_train_full_c, test_size=0.1, random_state=42
)


def NN_cifar(hp):
    global_params = {
        "kernel_regularizer": k.regularizers.L2(0.01),
        "kernel_initializer": k.initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=None
        ),
        "bias_initializer": "zeros",
    }
    l1_params = {
        **global_params,
        "units": hp.Int("units1", min_value=512, max_value=512, step=32),
        "activation": hp.Choice("activation1", ["relu", "tanh"]),
    }

    l2_params = {
        **global_params,
        "units": hp.Int("units2", min_value=512, max_value=512, step=32),
        "activation": hp.Choice("activation2", ["relu", "tanh"]),
    }

    # l3_params = {**global_params,
    #     "units": hp.Int("units3", min_value=64, max_value=256, step=32),
    #     "activation": hp.Choice("activation3", ["relu", "tanh"]),
    # }
    model = k.models.Sequential()
    model.add(k.layers.Flatten(input_shape=[32, 32]))

    model.add(k.layers.Dense(**l1_params))
    model.add(k.layers.Dropout(0.2))
    model.add(k.layers.Dense(**l2_params))
    model.add(k.layers.Dropout(0.2))
    # model.add(k.layers.Dense(**l3_params))
    # model.add(k.layers.Dropout(0.1))
    model.add(k.layers.Dense(10, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


# %%
# TRAIN ON CIFAR
with open(best_params_filename, "rb") as f:
    best_params = pickle.load(f)
model_cifar = NN_cifar(best_params)
model_cifar.summary()
history = model_cifar.fit(
    X_train_full_c_scaled, y_train_full_c, batch_size=64, epochs=500
)
# %%

# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_AFFINITY"] = "noverbose"

from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
# %%
model = keras.models.Sequential(
    [
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1),
    ]
)
model.compile(loss="mean_squared_error", optimizer="sgd")
model.summary()
# %%
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "my_keras_model.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)


class CustomCallbackClass(keras.callbacks.Callback):
    # Training callbacks
    def on_train_begin(self, *args, **kwargs):  # or _end
        print("Hi I am a callback written by thanos...")

    def on_epoch_begin(self, *args, **kwargs):  # or _end
        print("I wonder what epic journey I am going to have again!")

    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))

    def on_batch_begin(self, *args, **kwargs):  # or _end
        pass

    # Evaluation callbacks
    def on_test_begin(self, *args, **kwargs):  # or _end
        pass

    def on_test_batch_begin(self, *args, **kwargs):  # or _end
        pass

    # Prediction callbacks
    def on_predict_begin(self, *args, **kwargs):  # or _end
        pass

    def on_predict_batch_begin(self, *args, **kwargs):  # or _end
        pass


my_cb = CustomCallbackClass()

# %% TENSORBOARD
import os, time

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time

    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()  # e.g., './my_logs/run_2019_06_07-15_15_22'
# %% tb callback from keras
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    callbacks=[checkpoint_cb, early_stopping_cb, my_cb, tensorboard_cb],
    validation_data=(X_valid, y_valid),
)

# %% get best model after training
model = keras.models.load_model("my_keras_model.h5")

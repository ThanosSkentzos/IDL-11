#%%
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras_tuner import Hyperband, HyperParameters


# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Define a model-building function for Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(28, 28, 1)))

    # Conv layers with activation selection
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(layers.Conv2D(
            filters=hp.Choice('conv_filters_' + str(i), [32, 64, 128]),
            kernel_size=hp.Choice('conv_kernel_size_' + str(i), [3, 5]),
            padding='same',
            activation=hp.Choice('conv_activation_' + str(i), ['relu', 'tanh', 'swish']),
            kernel_initializer=hp.Choice('kernel_initializer', ['he_uniform', 'he_normal', 'glorot_uniform']),
            kernel_regularizer=regularizers.l1_l2(
                l1=hp.Float('l1_reg', 1e-5, 1e-2, sampling='log'),
                l2=hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')
            )
        ))

    # Pooling layer
    model.add(layers.MaxPooling2D((2, 2)))

    # Dropout layer
    model.add(layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))

    # Flatten and Dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Choice('dense_units', [64, 128, 256]),
        activation=hp.Choice('dense_activation', ['relu', 'tanh', 'swish']),
        kernel_regularizer=regularizers.l1_l2(
            l1=hp.Float('l1_reg', 1e-5, 1e-2, sampling='log'),
            l2=hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')
        )
    ))

    # Output layer
    model.add(layers.Dense(10, activation='softmax'))

    # Optimizer mapping
    optimizer_choice = hp.Choice('optimizer', ['SGD', 'SGD_momentum', 'SGD_nesterov', 'Adam', 'AdamW', 'RMSprop'])
    if optimizer_choice == 'SGD':
        optimizer = keras.optimizers.SGD()
    elif optimizer_choice == 'SGD_momentum':
        optimizer = keras.optimizers.SGD(momentum=0.9)
    elif optimizer_choice == 'SGD_nesterov':
        optimizer = keras.optimizers.SGD(nesterov=True)
    elif optimizer_choice == 'Adam':
        optimizer = keras.optimizers.Adam()
    elif optimizer_choice == 'AdamW':
        optimizer = keras.optimizers.AdamW(weight_decay=1e-4)
    elif optimizer_choice == 'RMSprop':
        optimizer = keras.optimizers.RMSprop()

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Set up the tuner with Hyperband strategy
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=1000,
    factor=3,
    directory='my_dir_cnn',
    project_name='fashion_mnist_tuning'
)

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Perform the search
tuner.search(x_train, y_train, epochs=1000, validation_split=0.2, batch_size=HyperParameters().Int('batch_size', 32, 512, step=32),
             callbacks=[early_stopping])

# %%
results = []
for trial in tuner.oracle.get_best_trials(num_trials=10):  # Adjust num_trials as desired
    trial_summary = {param: trial.hyperparameters.values[param] for param in trial.hyperparameters.values}
    trial_summary['val_accuracy'] = trial.metrics.get_best_value('val_accuracy')
    results.append(trial_summary)

# Convert to DataFrame and save
df = pd.DataFrame(results)
df.to_excel("tuning_results_cnn.xlsx", index=False)
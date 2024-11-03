# %%

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras_tuner import Hyperband, HyperParameters

# Load and preprocess Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

num_classes = 10

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train_scaled = keras.utils.to_categorical(y_train, num_classes)
y_test_scaled = keras.utils.to_categorical(y_test, num_classes)


# Define a model-building function for Keras Tuner
def build_model(hp: HyperParameters):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    
    # Add tunable dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Choice('units_' + str(i), [64, 128]),
            activation=hp.Choice('activation_' + str(i), ['relu', 'tanh', 'swish']),
            kernel_regularizer=regularizers.l1_l2(
                l1=hp.Float('l1_reg', 1e-5, 1e-2, sampling='log'),
                l2=hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')
            ),
            kernel_initializer=hp.Choice('kernel_initializer', ['he_uniform', 'he_normal', 'glorot_uniform'])
        ))
        
        # Dropout layer with tunable dropout rate
        model.add(layers.Dropout(hp.Float('dropout_' + str(i), 0.0, 0.5, step=0.1)))

    # Output layer
    model.add(layers.Dense(10, activation='softmax'))
    
    # Select optimizer and tune its hyperparameters
    optimizer_choice = hp.Choice('optimizer', ['SGD', 'Adam', 'AdamW', 'RMSprop'])
    if optimizer_choice == 'SGD':
        optimizer = keras.optimizers.SGD(
            learning_rate=hp.Float('sgd_lr', 1e-4, 1e-2, sampling='log'),
            momentum=hp.Choice('sgd_momentum', [0.0, 0.9]),
            nesterov=hp.Boolean('sgd_nesterov')
        )
    elif optimizer_choice == 'Adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=hp.Float('adam_lr', 1e-4, 1e-2, sampling='log')
        )
    elif optimizer_choice == 'AdamW':
        optimizer = keras.optimizers.AdamW(
            learning_rate=hp.Float('adamw_lr', 1e-4, 1e-2, sampling='log'),
            weight_decay=1e-4
        )
    elif optimizer_choice == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(
            learning_rate=hp.Float('rmsprop_lr', 1e-4, 1e-2, sampling='log')
        )
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
    directory='my_dir',
    project_name='fashion_mnist_mlp_tuning'
)

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Perform the search
tuner.search(x_train, y_train_scaled, epochs=1000, validation_split=0.1, 
             batch_size=HyperParameters().Int('batch_size', 32, 512, step=32), callbacks=[early_stopping])

results = []
for trial in tuner.oracle.get_best_trials(num_trials=10):  # Adjust num_trials as desired
    trial_summary = {param: trial.hyperparameters.values[param] for param in trial.hyperparameters.values}
    trial_summary['val_accuracy'] = trial.metrics.get_best_value('val_accuracy')
    results.append(trial_summary)

# Convert to DataFrame and save
df = pd.DataFrame(results)
df.to_excel("tuning_results.xlsx", index=False)


# Get the best model and evaluate it
# best_model = tuner.get_best_models(num_models=1)[0]
# best_model.evaluate(x_test, y_test)

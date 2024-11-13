from keras import layers
import keras_tuner
import keras
from tensorflow.keras.datasets import fashion_mnist, cifar10
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import pandas as pd
import pickle
import json

class Dataset_prep():
    def __init__(self, X_train,X_valid, y_train, y_valid,x_test,y_test, input_shape):
        self.input_shape = input_shape
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test

def load_data(num_classes):
    
    img_rows, img_cols = 32, 32
    #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    plt.imshow(x_train[0])
    plt.title(f"Label: {y_train[0]}")
    unique_classes = np.unique(y_train)
    print(unique_classes)
    print(y_train[0])
    plt.axis('off')  # Remove the axes for better visualization
    plt.show()
    X_train = x_train
    Y_train = y_train

   
    print(y_train.shape)
    print(X_train.shape)
    print(y_train[0])



    X_train = X_train.astype('float32')
    #X_valid = X_valid.astype('float32')
    X_train /= 255
    #X_valid /= 255
    x_test = x_test.astype('float32')
    x_test /= 255
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    #print(X_valid.shape[0], 'valid samples')
    print(y_train[0])
    y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    input_shape = X_train.shape[1:]
    print("Input shape is ",input_shape)
    X_valid = None
    y_valid = None
    dataset_p = Dataset_prep(X_train,X_valid, y_train, y_valid,x_test,y_test, input_shape)
    
    return dataset_p

num_classes = 10
dataset_p = load_data(num_classes)


from tensorflow import keras as k
def build_model3( input_shape, kernel_initializer, activation):
    
    #regularizer
    #l1_strength = hp.Float('l1_strength', min_value = 1e-5, max_value = 1e-3, sampling='log')
    kernel_regularizer = k.regularizers.L2(0.01) 
    #kernel_initializer = hp.Choice("kernel_initializer", values = ["he_normal","random_normal",'glorot_uniform'])
    
    #activation = hp.Choice("activation",values= ["relu","tanh"])
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation =activation, input_shape = input_shape, kernel_initializer = kernel_initializer, padding = 'same'))
    
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation, kernel_initializer = kernel_initializer, padding='same'))

    model.add(MaxPooling2D(pool_size=(3, 3)))
    
  
    model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer,kernel_regularizer = kernel_regularizer))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer))
    
    #model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.1))
    
  
    model.add(Flatten())
    model.add(Dense(128, activation=activation,kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_classes, activation='softmax'))

    #learning_rate = hp.Float("learning_rate", min_value=0.001, max_value=0.1, sampling="log")
    #momentum = hp.Float("momentum", min_value=0.09, max_value=0.9, step=0.1)
    
    # Create the optimizer with the tunable parameters

    #optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001, weight_decay = 1e-4
        ),
        #optimizer = optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model

def build_model2( input_shape, kernel_initializer, activation):
    
    #regularizer
    #l1_strength = hp.Float('l1_strength', min_value = 1e-5, max_value = 1e-3, sampling='log')
    kernel_regularizer = k.regularizers.L2(0.01) 
    #kernel_initializer = hp.Choice("kernel_initializer", values = ["he_normal","random_normal",'glorot_uniform'])
    
    #activation = hp.Choice("activation",values= ["relu","tanh"])
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2,2), activation =activation, input_shape = input_shape, kernel_initializer = kernel_initializer, padding = 'same'))
    
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation, kernel_initializer = kernel_initializer, padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
  
    model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer,kernel_regularizer = kernel_regularizer))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer))
    
    #model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.1))
    
  
    model.add(Flatten())
    model.add(Dense(128, activation=activation,kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_classes, activation='softmax'))

    #learning_rate = hp.Float("learning_rate", min_value=0.001, max_value=0.1, sampling="log")
    #momentum = hp.Float("momentum", min_value=0.09, max_value=0.9, step=0.1)
    
    # Create the optimizer with the tunable parameters

    optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.8)
    model.compile(
#         optimizer=keras.optimizers.Adam(
#             learning_rate=0.0001, weight_decay = 1e-4
#         ),
        optimizer = optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model

def build_model1( input_shape, kernel_initializer, activation):
    
    #regularizer
    #l1_strength = hp.Float('l1_strength', min_value = 1e-5, max_value = 1e-3, sampling='log')
    kernel_regularizer = k.regularizers.L2(0.01) 
    #kernel_initializer = hp.Choice("kernel_initializer", values = ["he_normal","random_normal",'glorot_uniform'])
    
    #activation = hp.Choice("activation",values= ["relu","tanh"])
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation =activation, input_shape = input_shape, kernel_initializer = kernel_initializer, padding = 'same'))
    
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation, kernel_initializer = kernel_initializer, padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
  
    model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer))
    #model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation=activation, kernel_initializer = kernel_initializer))
    model.add(Dropout(0.2))
    #model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer))
    
    #model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = kernel_initializer))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(Dropout(0.1))
    
  
    model.add(Flatten())
    model.add(Dense(128, activation=activation,kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_classes, activation='softmax'))

    #learning_rate = hp.Float("learning_rate", min_value=0.001, max_value=0.1, sampling="log")
    #momentum = hp.Float("momentum", min_value=0.09, max_value=0.9, step=0.1)
    
    # Create the optimizer with the tunable parameters

    #optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.8)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001, weight_decay = 1e-4
        ),
        #optimizer = optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model



input_shape = dataset_p.input_shape
kernel_initializer = 'he_normal'
activation = 'relu'

model1 = build_model1(input_shape, kernel_initializer, activation)

history1 = model1.fit(dataset_p.X_train, dataset_p.y_train,
          batch_size=64,
          epochs=20
                     )
          #validation_data=(dataset_p.X_valid, dataset_p.y_valid))

plt.plot(history1.history['loss'], label=f'loss with Adam, #params:{model1.count_params()}')

model2 = build_model2(input_shape, kernel_initializer, activation)

history2 = model2.fit(dataset_p.X_train, dataset_p.y_train,
          batch_size=64,
          epochs=20)
          
          #validation_data=(dataset_p.X_valid, dataset_p.y_valid))

plt.plot(history2.history['loss'], label=f'loss with SGD, #params:{model2.count_params()}')

model3 = build_model3(input_shape, kernel_initializer, activation)

history3 = model3.fit(dataset_p.X_train, dataset_p.y_train,
          batch_size=64,
          epochs=20)
          
          #validation_data=(dataset_p.X_valid, dataset_p.y_valid))
plt.plot(history3.history['loss'], label=f'loss with Adam, #params:{model3.count_params()}')

score3 = model3.evaluate(dataset_p.x_test, dataset_p.y_test, verbose=0)        
score2 = model2.evaluate(dataset_p.x_test, dataset_p.y_test, verbose=0)
score1 = model1.evaluate(dataset_p.x_test, dataset_p.y_test, verbose=0)

print(f"score for Adam, #params:{model1.count_params()}", score1)
print(f"score for SGD, #params:{model2.count_params()}", score2)
print(f"score for Adam, #params:{model3.count_params()}", score3)



plt.xlabel('epochs')
plt.ylabel('score')
plt.legend(loc="best")

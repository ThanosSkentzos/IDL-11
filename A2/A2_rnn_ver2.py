# %% [markdown]
# <div style="text-align: right">   </div>
# Introduction to Deep Learning (2024)
# 
# **Assignment 2 - Sequence processing using RNNs**
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/UniversiteitLeidenLogo.svg/1280px-UniversiteitLeidenLogo.svg.png" width="300">
# 
# # Introduction
# 
# The goal of this assignment is to learn how to use encoder-decoder recurrent neural networks (RNNs). 
# Specifically we will be dealing with a sequence to sequence problem and try to build recurrent models that can learn the principles behind simple arithmetic operations (**integer addition, subtraction and multiplication.**).
# 
# <img src="https://i.ibb.co/5Ky5pbk/Screenshot-2023-11-10-at-07-51-21.png" alt="Screenshot-2023-11-10-at-07-51-21" border="0" width="500"></a>
# 
# In this assignment you will be working with three different kinds of models, based on input/output data modalities:
# 
# 1. **Text-to-text**: given a text query containing two integers and an operand between them (+ or -) the model's output should be a sequence of integers that match the actual arithmetic result of this operation
# 
# 2. **Image-to-text**: same as above, except the query is specified as a sequence of images containing individual digits and an operand.
# 
# 3. **Text-to-image**: the query is specified in text format as in the text-to-text model, however the model's output should be a sequence of images corresponding to the correct result.
# 
# ### Description
# 
# Let us suppose that we want to develop a neural network that learns how to add or subtract
# 
# two integers that are at most two digits long. For example, given input strings of 5 characters: ‘81+24’ or
# 
# ’41-89’ that consist of 2 two-digit long integers and an operand between them, the network should return a
# 
# sequence of 3 characters: ‘105 ’ or ’-48 ’ that represent the result of their respective queries. Additionally,
# 
# we want to build a model that generalizes well - if the network can extract the underlying principles behind
# 
# the ’+’ and ’-’ operands and associated operations, it should not need too many training examples to generate
# 
# valid answers to unseen queries. To represent such queries we need 13 unique characters: 10 for digits (0-9),
# 
# 2 for the ’+’ and ’-’ operands and one for whitespaces ’ ’ used as padding.
# 
# The example above describes a text-to-text sequence mapping scenario. However, we can also use different
# 
# modalities of data to represent our queries or answers. For that purpose, the MNIST handwritten digit
# 
# dataset is going to be used again, however in a slightly different format. The functions below will be used to create our datasets.
# 
# *To work on this notebook you should create a copy of it.*
# 

# %% [markdown]
# # Function definitions for creating the datasets
# 
# First we need to create our datasets that are going to be used for training our models.
# 
# In order to create image queries of simple arithmetic operations such as '15+13' or '42-10' we need to create images of '+' and '-' signs using ***open-cv*** library. 
# We will use these operand signs together with the MNIST dataset to represent the digits.

# %%
import matplotlib.pyplot as plt
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed # type: ignore
from tensorflow.keras.layers import RepeatVector, Conv2D, ConvLSTM2D # type: ignore


# %%
from scipy.ndimage import rotate


# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):
    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates
        if sign == '*':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
            
            # Rotate 45 degrees
            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)

            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)

    return blank_images


def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))

# %%
def create_data(highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:
    
    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=False)
                query_image = []

                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=False)
                result_image = []
                
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())
                    
                X_text.append(query_string)
                X_img.append(np.stack(query_image))

                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '

    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)


# %% [markdown]
# # Creating our data
#
# The dataset consists of 20000 samples that (additions and subtractions between all 2-digit integers) and they have two kinds of inputs and label modalities:
#  
#   **X_text**: strings containing queries of length 5: ['  1+1  ', '11-18', ...]
#   **X_image**: a stack of images representing a single query, dimensions: [5, 28, 28]
# 
#   **y_text**: strings containing answers of length 3: ['  2', '156']
#   **y_image**: a stack of images that represents the answer to a query, dimensions: [3, 28, 28]

# %%
# Illustrate the generated query/answer pairs

unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer)
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)

## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
        
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])


# %% [markdown]
# ## Helper functions
# 
# 
# 
# The functions below will help with input/output of the data.

# %%
# One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs

# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
    n = len(labels)
    length = len(labels[0])
    char_map = dict(zip(unique_characters, range(len(unique_characters))))
    one_hot = np.zeros([n, length, len(unique_characters)])
    
    for i, label in enumerate(labels):
        m = np.zeros([length, len(unique_characters)])
        for j, char in enumerate(label):
            m[j, char_map[char]] = 1
        one_hot[i] = m

    return one_hot

def decode_labels(labels):
    pred = np.argmax(labels, axis=1)
    predicted = ''.join([unique_characters[i] for i in pred])
    
    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

print(X_text_onehot.shape, y_text_onehot.shape)


# %% [markdown]
# ---
# 
# ## I. Text-to-text RNN model
# 
# The following code showcases how Recurrent Neural Networks (RNNs) are built using Keras. Several new layers are going to be used:
# 
# 1. LSTM
# 
# 2. TimeDistributed
# 
# 3. RepeatVector
# 
# The code cell below explains each of these new components.
# 
# <img src="https://i.ibb.co/NY7FFTc/Screenshot-2023-11-10-at-09-27-25.png" alt="Screenshot-2023-11-10-at-09-27-25" border="0" width="500"></a>
# 

# %%
def build_text2text_model():
    # We start by initializing a sequential model
    text2text = keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. 
    # Each of these 5 elements in the query will be fed to the network one by one, as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. 
    # Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). 
    # This is necessary as TimeDistributed in the below expects the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text

# %%
## Your code (look at the assignment description for your tasks for text-to-text model):
##( Your first task is to fit the text2text model using X_text and y_text)
import tensorflow.keras as keras # type: ignore
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


data_percentage = [[80.0, 10.0, 10.0, 50], [50.0, 0.0, 50.0, 50], [25.0, 0.0, 75.0, 75], [10.0, 0.0, 90.0, 100]]
models=[]
for each in data_percentage:
    print("TRAIN, VALID, TEST Percentage", each[0], each[1], each[2])
    X_train, X_test, y_train, y_test = train_test_split(X_text_onehot, y_text_onehot, 
                                                        test_size=(each[1]+each[2])/100.0, random_state=42) 

    
    print(decode_labels(X_test[0]))
    print(decode_labels(y_test[0]))
    # Fit the model
    model = build_text2text_model()
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
            f"text_to_text_best.keras", save_best_only=True
        )
    
    if each[1]:
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, 
                                                            test_size=each[1]/(each[1]+each[2]), random_state=42) 
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',  # metric to monitor
            patience=10,          # number of epochs to wait for improvement
            restore_best_weights=True  # restore the best weights after stopping
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            batch_size=32,
            epochs=50,
            callbacks=[checkpoint_cb, early_stopping],
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=each[-1],
            callbacks=[checkpoint_cb]
        )
    model.save(f'submission1_text_to_text.keras')

    data_history = pd.DataFrame(history.history)
    data_history.to_csv('text_to_text_history.csv')
    
    plt.plot(history.history['loss'], label='loss')
    if each[1]:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('score')
    plt.legend(loc="best")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    predictions = model.predict(X_test)
    y_pred = [decode_labels(y) for y in predictions]
    y_actual = [decode_labels(y) for y in y_test]

    accuracy = accuracy_score(y_actual, y_pred)
    print("Result for TRAIN, VALID, TEST Percentage", each[0], each[1], each[2])
    print("Train Accuracy for text to text model: ", model.evaluate(X_train, y_train))
    print("Test Accuracy for text to text model: ", model.evaluate(X_test, y_test))
    print("Test String Accuracy: ", accuracy)

    #to clear cache
    models.append(model)
    continue
    import gc

    # Delete unnecessary variables
    del model

    # Force garbage collection
    gc.collect()
#%%
# TODO check that we get correct results -> done
preds = [list(map(decode_labels,m.predict(X_test))) for m in models]
trues = list(map(decode_labels,y_test))
scores = [accuracy_score(trues,i) for i in preds+[trues]]
evals = [m.evaluate(X_test,y_test)[1] for m in models] + [1]
#%%
# TODO when do we get a decreased accuracy and why
# entire output numbers
columns = [",".join([str(i) for i in l]) for l in data_percentage] + ['true']
score_df = pd.DataFrame([scores,evals])
score_df.index=["test_string_accuracy","test_character_accuracy"]
score_df.columns = columns
df = pd.DataFrame(preds+[trues]).T
df.columns = columns
df.plot.scatter(columns[1],columns[-1])
df.plot.scatter(columns[2],columns[-1])
df.plot.scatter(columns[3],columns[-1])
#%%
df.plot.scatter(columns[0],columns[-1])
#%%
# symbol by symbol
wrong_positions = [np.argwhere(np.array(p)!=trues) for p in preds]
# TODO visualize the differences perhaps scatterplot
#find out what kind of mistakes your models make on the misclassified samples.
wrong_data = [X_test[pos] for pos in wrong_positions]

# decoded_wrong_inputs = [[list(map(decode_labels,data)) for data in model_wrong_data] for model_wrong_data in wrong_data]
wrong_outputs = [np.array(p)[idx] for idx,p in zip(wrong_positions,preds)]
wrong_out_characters = ["".join([str(i) for i in w.ravel()]) for w in wrong_outputs]
correct_characters = ["".join([str(i) for i in np.array(trues)[idx].ravel()]) for idx in wrong_positions ]

pred_characters = ["".join([str(i) for i in y]) for y in preds]
true_characters = "".join([str(char) for y in trues for char in y])
mapping = {i:n for i,n in zip(unique_characters,range(len(unique_characters)))}

from collections import Counter
counts = [Counter([(true,pred) for true,pred in zip(true_characters,chars)]) for chars in pred_characters]


for i in range(len(models)):
    x=[each[0] for each in counts[i].keys()]
    y=[each[1] for each in counts[i].keys()]
    z=[each for each in counts[i].values()]
    plt.figure(figsize=(8,6))
    sc = plt.scatter(x, y, c=z, cmap='hot', s=50, edgecolor='none')
    plt.colorbar(sc, label='Intensity')
    plt.title("Heated Scatter Plot (Color by Value)")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()


# for i in range(len(models)):
#     plt.figure()
#     x = list(map(mapping.get,true_characters)),
#     y = list(map(mapping.get,pred_characters[i]))

#     plt.xticks(range(len(unique_characters)),unique_characters)
#     plt.yticks(range(len(unique_characters)),unique_characters)
# %% [markdown]
# 
# ## II. Image to text RNN Model
# 
# Hint: There are two ways of building the encoder for such a model - again by using the regular LSTM cells (with flattened images as input vectors) or 
# recurrect convolutional layers [ConvLSTM2D](https://keras.io/api/layers/recurrent_layers/conv_lstm2d/).
# 
# The goal here is to use **X_img** as inputs and **y_text** as outputs.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_img, y_text_onehot, test_size=0.20, random_state=42) 
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=42) 

# %%
print(decode_labels(y_test[0]))

plt.imshow(np.hstack(X_test[0]), cmap='gray')  # Display the frame in grayscale
plt.title(f"X")
plt.axis("off")
plt.show()

# %%
## Your code
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D, ConvLSTM2D,MaxPooling2D, BatchNormalization, Dropout,LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


def build_img2text_model():
    # We start by initializing a sequential model
    img2text = keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. 
    # Each of these 5 elements in the query will be fed to the network one by one, as shown in the image above (except with 5 elements).

    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).

    img2text.add(
        ConvLSTM2D(filters=16,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu", 
        input_shape=(5, 28, 28, 1), 
        kernel_regularizer=l2(0.01))
    )
    img2text.add(BatchNormalization())
    img2text.add(Dropout(0.2))
    
    img2text.add(
        ConvLSTM2D(filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu", 
        kernel_regularizer=l2(0.01))
    )
    img2text.add(BatchNormalization())

    #img2text.add(Flatten())
    img2text.add(TimeDistributed(GlobalAveragePooling2D()))
    img2text.add(Dense(256, activation='relu'))
    img2text.add(Dropout(0.2))
    
    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    img2text.add(GlobalAveragePooling1D())
    img2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). 
    # This is necessary as TimeDistributed in the below expects the first dimension to be the timesteps.
    img2text.add(LSTM(128, return_sequences=True))
    img2text.add(LSTM(64, return_sequences=True))
    img2text.add(LayerNormalization())

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    img2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax',kernel_regularizer=l2(0.01))))

    # Next we compile the model using categorical crossentropy as our loss function.
    optimizer = Adam(learning_rate=0.001) 
    img2text.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    img2text.summary()

    return img2text

def build_img2text_model2():
    img2text = keras.Sequential()
    img2text.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu'), input_shape=(5, 28, 28, 1)))
    img2text.add(TimeDistributed(MaxPooling2D((2, 2))))
    img2text.add(TimeDistributed(Dropout(0.2)))
    img2text.add(TimeDistributed(Flatten()))
    img2text.add(TimeDistributed(Dense(200, activation='relu')))
    img2text.add(LSTM(256, return_sequences=True))
    img2text.add(LSTM(256))
    
    img2text.add(RepeatVector(max_answer_length))
    img2text.add(LSTM(256, return_sequences=True))
    img2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))
    img2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    img2text.summary()

    return img2text

model = build_img2text_model2()

# %%
import tensorflow.keras as keras
# Fit the model

checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"image_to_text_best.keras", save_best_only=True
    )
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # metric to monitor
    patience=8,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)

#%%
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    batch_size=32,
    epochs=100,
    callbacks = [checkpoint_cb, early_stopping, lr_scheduler]
)

#%%
model.save(f'submission1_image_to_text.keras')

#%%
import pandas as pd
data_history = pd.DataFrame(history.history)
data_history.to_csv('image_to_text_history.csv')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epochs')
plt.ylabel('score')
plt.legend(loc="best")

plt.grid(True)
plt.tight_layout()
plt.show()

print("Train Accuracy for text to text model:", model.evaluate(X_train, y_train))
print("Test Accuracy for text to text model:", model.evaluate(X_test, y_test))


# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
predictions = model.predict(X_test)
y_pred = [decode_labels(y) for y in predictions]
y_actual = [decode_labels(y) for y in y_test]
accuracy = accuracy_score(y_actual, y_pred)
print("Accuracy for image to text model:", accuracy)

# %%
from tensorflow.keras.layers import GlobalAveragePooling2D,GlobalAveragePooling1D,ConvLSTM2D, BatchNormalization, Dropout,LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

def build_img2text_withConv():
    ## Your code
    # We start by initializing a sequential model
    img2text = keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.

    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,

    # as shown in the image above (except with 5 elements).

    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).

    img2text.add(
        TimeDistributed(
            Conv2D(
                filters=512,
                kernel_size=(3, 3),
                padding="same",
                return_sequences=True,
                activation="relu", 
                input_shape=(5, 28, 28, 1), 
                kernel_regularizer=l2(0.01)
            )
        )
    )
    img2text.add(TimeDistributed(GlobalAveragePooling2D()))
    img2text.add(BatchNormalization())
    img2text.add(Dropout(0.2))
 
    #img2text.add(Flatten())
    
    img2text.add(Dense(256, activation='relu'))
    img2text.add(Dropout(0.2))
    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')

    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    img2text.add(GlobalAveragePooling1D())
    img2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). 
    # This is necessary as TimeDistributed in the below expects the first dimension to be the timesteps.
    img2text.add(LSTM(128, return_sequences=True))
    img2text.add(LSTM(64, return_sequences=True))
    img2text.add(LayerNormalization())

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.    
    img2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax',kernel_regularizer=l2(0.01))))

    # Next we compile the model using categorical crossentropy as our loss function.    
    optimizer = Adam(learning_rate=0.001) 
    img2text.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    img2text.summary()

    return img2text


# %%
import pandas as pd

data_history = pd.DataFrame(history.history)
data_history.to_csv('image_to_text_history.csv')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epochs')
plt.ylabel('score')
plt.legend(loc="best")


plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# 
# ---
# 
# 
# 
# ## III. Text to image RNN Model
# 
# 
# 
# Hint: to make this model work really well you could use deconvolutional layers in your decoder (you might need to look up ***Conv2DTranspose*** layer). However, regular vector-based decoder will work as well.
# 
# 
# 
# The goal here is to use **X_text** as inputs and **y_img** as outputs.

# %%
# Your code

#MNIST CLASSIFICATION

import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd

class DatasetPrep():
    def __init__(self, X_train, X_valid, y_train, y_valid, X_test, y_test, input_shape):
        self.input_shape = input_shape
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        
def encode_out(labels):
    n = len(labels)
    characters = "1234567890- "
    char_map = dict(zip(characters, range(len(characters))))
    one_hot = np.zeros([n, len(characters)])
    
    for i, label in enumerate(labels):
        m = np.zeros([len(characters)])
        for j, char in enumerate(label):
            m[char_map[char]] = 1
        one_hot[i] = m
    return one_hot

def decode_out(labels):
    pred = np.argmax(labels, axis=1)
    predicted = ''.join(["1234567890- "[i] for i in pred])
    
    return predicted

def load_data(num_classes):
    img_rows, img_cols = 28, 28
    item_size = 7000
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Concatenate Train and Test
    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train.astype(str), y_test.astype(str)), axis=0)
    
    # Add Minus Images
    X = np.concatenate((X, generate_images(item_size, '-')), axis=0)
    Y = np.concatenate((Y, np.full((item_size), '-')), axis=0)
    
    #Add Empty Images
    X = np.concatenate((X, np.zeros([item_size, img_rows, img_cols])), axis=0)
    Y = np.concatenate((Y, np.full((item_size), ' ')), axis=0)
    
    print(X.shape, Y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.50, random_state=42)
    
    if keras.backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
        
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    
    plt.imshow(X_train[0])
    plt.title(f"Label: {y_train[0]}")
    unique_classes = np.unique(y_train)
    print(unique_classes)
    print(y_train[0])
    plt.axis('off')  # Remove the axes for better visualization
    plt.show()
   
    print(y_train.shape)
    print(X_train.shape)
    print(y_train[0])

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_valid /= 255
    X_test /= 255
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    #print(X_valid.shape[0], 'valid samples')
    print(y_train[0])
    

        
    y_train = encode_out(y_train)
    y_valid = encode_out(y_valid)
    y_test = encode_out(y_test)
    input_shape = X_train.shape[1:]
    print("Input shape is ", input_shape)
  
    dataset_p = DatasetPrep(X_train, X_valid, y_train, y_valid, X_test, y_test, input_shape)
    
    return dataset_p

num_classes = 12
dataset_p = load_data(num_classes)


#%%
def build_image_classifier(input_shape, kernel_initializer, activation):
    #regularizer
    kernel_regularizer = keras.regularizers.L2(0.01)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation=activation, input_shape=input_shape, kernel_initializer=kernel_initializer, padding='same'))
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same'))

    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128, activation=activation,kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
    
    model.add(Dropout(0.2))    
    model.add(Dense(num_classes, activation='softmax'))

    # Create the optimizer with the tunable parameters
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

#%%
input_shape = dataset_p.input_shape
kernel_initializer = 'he_normal'
activation = 'relu'

image_classifier_model = build_image_classifier(input_shape, kernel_initializer, activation)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"mnist_classification_best.keras", save_best_only=True
    )
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # metric to monitor
    patience=8,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)

history = image_classifier_model.fit(
    dataset_p.X_train,
    dataset_p.y_train,
    validation_data=(dataset_p.X_valid, dataset_p.y_valid),
    batch_size=64,
    epochs=20,
    callbacks = [checkpoint_cb, early_stopping, lr_scheduler]
)


import pandas as pd
data_history = pd.DataFrame(history.history)
data_history.to_csv('mnist_classification.csv')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epochs')
plt.ylabel('score')
plt.legend(loc="best")


plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import tensorflow as tf


score3 = image_classifier_model.evaluate(dataset_p.X_test, dataset_p.y_test) 
print(score3)
images = tf.unstack(y_img[100], axis=0)
single_image = images[1]  # Shape: (28, 28, 1)
plt.imshow(single_image, cmap='gray')

# to check for blank image 
if np.all(single_image==0):
    print(True)
    
single_image_batch = tf.expand_dims(single_image, axis=0)  # Shape: (1, 28, 28, 1)
print(single_image_batch.shape)

predictions = image_classifier_model.predict(single_image_batch)

predicted_classes = predictions.argmax(axis=1)
print(predicted_classes)


# %%
# TEXT TO IMAGE PREPARATION
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_text_onehot, y_img, test_size=0.20, random_state=42) 
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=42) 

_, _, y_train_text, y_test_text = train_test_split(X_text_onehot, y_text, test_size=0.20, random_state=42)
_, _, y_test_text, y_valid_text = train_test_split(_, y_test_text, test_size=0.5, random_state=42) 


# %%
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (ConvLSTM2D, TimeDistributed, GlobalMaxPooling2D, Conv2DTranspose,
                                     Dense, LSTM, RepeatVector, Dropout, BatchNormalization, Reshape,
                                     LayerNormalization, Flatten, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_text2img_model():

    text2img = tf.keras.Sequential()

    text2img.add(LSTM(256, input_shape=(None, len(unique_characters))))
    text2img.add(Dropout(0.1))
    text2img.add(RepeatVector(max_answer_length))

    # add reshape 
    text2img.add(LSTM(256, return_sequences=True))
    text2img.add(Dropout(0.1))
    
    # text2img.add(TimeDistributed(Dense( 28 * 28, activation='sigmoid', kernel_regularizer=l2(0.001))))
    
    # lets slowly increase the dense 
    text2img.add(TimeDistributed(Dense( 7 * 7 * 28, activation='sigmoid', kernel_regularizer=l2(0.001))))
    text2img.add(Reshape((3, 7, 7, 28)))

    #default_args=dict(kernel_size=(3,3),  padding='same', activation='relu')

    text2img.add(TimeDistributed(Conv2DTranspose(filters=7, kernel_size=(2, 2), strides=2, padding='same', 
                                                 activation='relu')))
    
    text2img.add(TimeDistributed(Conv2DTranspose(filters=7, kernel_size=(2, 2) ,strides=2, padding='same', 
                                                 activation = 'relu')))
    text2img.add(BatchNormalization())
    text2img.add(TimeDistributed(Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid')))

    # Compile the model
    text2img.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    text2img.summary()
    #text2img.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    

    return text2img

model = build_text2img_model()

# %%
print(y_train)

plt.imshow(np.hstack(y_train[200]), cmap='gray')  # Display the frame in grayscale
plt.title(f"Y")
plt.axis("off")
plt.show()

# %%
checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"text_to_image_best_previous.keras", save_best_only=True
    )
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # metric to monitor
    patience=8,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    batch_size=32,
    epochs=50,
    callbacks = [checkpoint_cb, early_stopping, lr_scheduler]
)

#%%
model.save(f'submission1_text_to_image_previous.keras')

#%%
from tensorflow.keras.models import load_model

# Path to the saved model file
model_path = 'submission1_text_to_image_previous.keras'

# Load the model
model = load_model(model_path)

#%%
import pandas as pd
data_history = pd.DataFrame(history.history)
data_history.to_csv('text_to_image_history_previous.csv')
plt.plot(data_history['loss'], label='loss')
plt.plot(data_history['val_loss'], label='val_loss')
plt.xlabel('epochs')
plt.ylabel('score')
plt.legend(loc="best")


plt.grid(True)
plt.tight_layout()
plt.show()

#%%
import pandas as pd
data_history = pd.DataFrame(history.history)
data_history.to_csv('text_to_image_history_previous.csv')
plt.plot(data_history['loss'], label='loss')
plt.plot(data_history['val_loss'], label='val_loss')
plt.xlabel('epochs')
plt.ylabel('score')
plt.legend(loc="best")


plt.grid(True)
plt.tight_layout()
plt.show()

#%%
output_images = model.predict(X_test).reshape((X_test.shape[0] * 3, 28, 28))

predicted_nhot = image_classifier_model.predict(output_images)
predicted_images_values = []
for i in range(X_test.shape[0]):
    predicted_images_values.append(decode_out(predicted_nhot[i*3:i*3+3]))


print(len(predicted_images_values))

print("Accuracy Score: ", accuracy_score(y_test_text, predicted_images_values))


#%%%

# def predict_model(model, text):
#     # Encodes the text and predicts the output image
#     encoded_text = encode_labels([text], max_len=5)
#     predicted_image = model.predict(encoded_text).reshape((3, 28, 28))
#     return predicted_image

# Input texts
# inputs = ['10-30', '42+25', '48+81']

# Predict the images for each input
# output_images = [predict_model(model, text) for text in inputs]


#%%

# def display_images(images, labels=None):
#     # Displays multiple sets of images with optional labels
#     fig, axes = plt.subplots(len(images), 3, figsize=(6, 4))
    
#     for i, (image_set, ax_row) in enumerate(zip(images, axes)):
#         for j, ax in enumerate(ax_row):
#             ax.imshow(image_set[j], cmap='gray')
#             ax.axis('off')
        
#         # Add label text if provided
#         if labels and i < len(labels):
#             ax_row[0].text(
#                 -0.5, 0.5, labels[i],
#                 verticalalignment='center',
#                 horizontalalignment='right',
#                 fontsize=10,
#                 transform=ax_row[0].transAxes
#             )
    
#     plt.tight_layout()
#     plt.show()

# # Labels for the inputs
# labels = [f"Input Text: {text}" for text in inputs]

# # Display the images with the corresponding labels
# display_images(output_images, labels=labels)

# # %%
# predicted_images = model.predict(X_test).reshape((X_test.shape[0], 3, 28, 28))
# # %%

# %%
import math
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.model_selection import train_test_split


# %%
images = np.load("A1/data/images.npy")
labels = np.load("A1/data/labels.npy")

num_classes = len(np.unique(labels[:,0]))
# %%
# Define the function to map minute to sin and cos polar coordinate values
def time_to_polar(row):
    hour_cosine = math.cos(row[0] * 2 * math.pi / 12)
    hour_sine = math.sin(row[0] * 2 * math.pi / 12)
    
    minute_cosine = math.cos(row[1] * 2 * math.pi / 60)
    minute_sine = math.sin(row[1] * 2 * math.pi / 60)
    return [hour_cosine, hour_sine, minute_cosine, minute_sine]

img_rows = images.shape[1]
img_cols = images.shape[2]

#%%
images = images.astype('float32')/255.0

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=45) 
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.50, random_state=33) 

#%%

y_train_hours = keras.utils.to_categorical(y_train[:,0], num_classes)
y_train_minutes = np.apply_along_axis(time_to_polar, axis=1, arr=y_train)

y_test_hours = keras.utils.to_categorical(y_test[:,0], num_classes)
y_test_minutes = np.apply_along_axis(time_to_polar, axis=1, arr=y_test)

y_val_hours = keras.utils.to_categorical(y_val[:,0], num_classes)
y_val_minutes = np.apply_along_axis(time_to_polar, axis=1, arr=y_val)


# %%
batch_size = 64
epochs = 500

if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# %%
def build_model(input_shape):
    input_ = keras.layers.Input(shape=x_train.shape[1:])

    # convolutional layers
    #1st Layer
    c1 = keras.layers.Conv2D(32, kernel_size=(5, 5),
                        activation='relu',padding='same',
                        input_shape=input_shape)(input_)
    m1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)
    b1 = keras.layers.BatchNormalization()(m1)

    #2nd layers
    c11 = keras.layers.Conv2D(32, kernel_size=(5, 5),
                        activation='relu',padding='same',
                        input_shape=input_shape)(b1)
    m11 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c11)
    b11 = keras.layers.BatchNormalization()(m11)
    
    #3nd layers
    c2 = keras.layers.Conv2D(64, kernel_size=(5, 5),
                        activation='relu',padding='same',
                        input_shape=input_shape)(b11)
    d2 = keras.layers.Dropout(0.10)(c2)
    m2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(d2)
    b2 = keras.layers.BatchNormalization()(m2)

    #4rd layers
    c3 = keras.layers.Conv2D(128, kernel_size=(3, 3),
                        activation='relu',padding='same',
                        input_shape=input_shape)(b2)
    d3 = keras.layers.Dropout(0.10)(c3)
    m3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(d3)
    b3 = keras.layers.BatchNormalization()(m3)

    #5th layers
    c4 = keras.layers.Conv2D(256, kernel_size=(3, 3),
                        activation='relu',padding='same',
                        input_shape=input_shape)(b3)
    d4 = keras.layers.Dropout(0.10)(c4)
    m4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(d4)
    b4 = keras.layers.BatchNormalization()(m4)
    
    
    #4th layers
    # c5 = keras.layers.Conv2D(512, kernel_size=(3, 3),
    #                     activation='relu',padding='same',
    #                     input_shape=input_shape)(b4)
    # d5 = keras.layers.Dropout(0.10)(c5)
    # m5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(d5)
    # b5 = keras.layers.BatchNormalization()(m5)

    #final layer begins
    f = keras.layers.Flatten()(b4)
     
    # #for hours final layers
    # dense_hour1 = keras.layers.Dense(128, activation='relu')(f)
    # dense_dropout1 = keras.layers.Dropout(0.10)(dense_hour1)
    
    # final_hour = keras.layers.Dense(num_classes, activation='softmax')(dense_dropout1)

    # for minutes final layers
    minute_dense1 = keras.layers.Dense(512, activation='relu')(f)
    minute_dropout1 = keras.layers.Dropout(0.5)(minute_dense1)
    
    minute_dense2 = keras.layers.Dense(128, activation='relu')(minute_dropout1)
    minute_dropout2 =  keras.layers.Dropout(0.3)(minute_dense2)
    
    minute_dense3 = keras.layers.Dense(64, activation='relu')(minute_dropout2)
    minute_dropout3 =  keras.layers.Dropout(0.1)(minute_dense3)
    
    final_minute = keras.layers.Dense(4, activation='tanh')(minute_dropout3)

    # final model
    # model = keras.Model(inputs=input_, outputs=[final_hour, final_minute])
    model = keras.Model(inputs=input_, outputs=final_minute)
    
    
    # model.compile(optimizer="adam", loss=['categorical_crossentropy', 'huber'], metrics=["accuracy", 'mae'])
    
    model.compile(optimizer="adam", loss='huber', metrics='mae')

    return model
 
 
#%%
model = build_model(input_shape=input_shape)
model.summary()

#%%
# Initialize EarlyStopping with patience
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # metric to monitor
    patience=25,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6, cooldown=5
)

# history = model.fit(x_train, [y_train_hours, y_train_minutes],
#                 batch_size=batch_size,
#                 epochs=epochs,
#                 verbose=1,
#                 validation_data=(x_val, [y_val_hours, y_val_minutes]),
#                 callbacks=[early_stopping, reduce_lr]
# )

history = model.fit(x_train, y_train_minutes,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_val, y_val_minutes),
                callbacks=[early_stopping, reduce_lr]
)

#%%
loss = history.history['loss']
val_loss = history.history['val_loss']


#%%
# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'{img_rows} x {img_cols} - Regression loss history with diminishing learning rate')
plt.legend()
plt.show()
plt.close()

#%%
y_pred = model.predict(x_test)

# %%
y_pred_hours = np.argmax(y_pred[0], axis=1)
y_test_hours_real = y_test[:,0]

abs_hr = np.abs(y_pred_hours - y_test_hours_real)
cse_hr = np.minimum(abs_hr, 12 - abs_hr)

# %%
# Scatter plot to show deviation for hour from Real to Pred

plt.scatter(y_pred_hours, y_test_hours_real)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Scatter for hour - Classification')


# %%
y_test_minutes_real =  y_test[:,1]

# Convert the sin and cos values back to minutes
y_pred_mins = (np.arctan2(y_pred[1][:,1], y_pred[1][:,0]) % (2 * np.pi)) *  60 / (2 * np.pi)

abs_mins = abs(y_pred_mins - y_test_minutes_real)
cse_min = np.minimum(abs_mins, 60. - abs_mins)


# %%
# Scatter plot to show deviation for minutes from Real to Pred
plt.scatter(y_pred_mins, y_test[:,1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Scatter for minutes - Polar')

# %%
sorted_indices_hr = np.argsort(y_test_hours_real)
sorted_y_test_hr = y_test_hours_real[sorted_indices_hr]

# Create polar plot to show common sense error for hour from Real to Pred
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(2 * np.pi * sorted_y_test_hr/12, -cse_hr[sorted_indices_hr], marker='o')

# Add title and show plot
ax.set_title("Polar Classification for hours")
theta_ticks = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # 12 hours around the clock
ax.set_xticks(theta_ticks)
ax.set_yticks(np.arange(-6, 0))
ax.set_xticklabels(range(12))

plt.show()

# %%
sorted_indices_min = np.argsort(y_test_minutes_real)
sorted_y_test_min = y_test_minutes_real[sorted_indices_min]

# Create polar plot to show common sense error for minutes from Real to Pred
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(2 * np.pi * sorted_y_test_min/60, -cse_min[sorted_indices_min], marker='o')

# Add title and show plot
ax.set_title("Polar Common sense Error for minutes")
theta_ticks = np.linspace(0, 2 * np.pi, 60, endpoint=False)  # 60 mins around the clock
ax.set_xticks(theta_ticks)
# ax.set_yticks(np.arange(-10, 0))
ax.set_xticklabels(range(60))

plt.show()


#%%

x_test_reshaped = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
score = model.evaluate(x_test_reshaped, [y_test_hours, y_test_minutes], verbose=False) 


#%%
for each in history.history.keys():
    print(f"Train {each}: {history.history[each][-1]}")

print(f"\nTest dense_19_accuracy: {score[3]}")
print(f"Test dense_19_loss: {score[0]}")
print(f"Test dense_23_loss: {score[1]}")
print(f"Test dense_23_mae: {score[2]}")
print(f"Test loss: {score[4]}\n")

print(f"Common Sense MAE Min Test: {np.mean(cse_min/60)}")
# %%

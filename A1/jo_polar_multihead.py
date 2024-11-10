# %%
import math
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.model_selection import train_test_split


# %%
images = np.load("data/images_150.npy")
labels = np.load("data/labels_150.npy")

# num_classes = len(np.unique(labels[:,0]))

# %%
# Define the function to map minute to sin and cos polar coordinate values
def time_to_polar(row):
    hour_cosine = math.cos(2 * np.pi * ((row[0] % 12) + row[1] / 60) / 12)
    hour_sine = math.sin(2 * np.pi * ((row[0] % 12) + row[1] / 60) / 12)
    
    minute_cosine = math.cos(row[1] * 2 * math.pi / 60)
    minute_sine = math.sin(row[1] * 2 * math.pi / 60)
    return [hour_cosine, hour_sine, minute_cosine, minute_sine]

img_rows = images.shape[1]
img_cols = images.shape[2]

#%%
images = images.astype('float32')/255.0

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=45) 
# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.50, random_state=33) 

#%%
# y_train_hours = keras.utils.to_categorical(y_train[:,0], num_classes)
y_train_minutes = np.apply_along_axis(time_to_polar, axis=1, arr=y_train)

# y_test_hours = keras.utils.to_categorical(y_test[:,0], num_classes)
y_test_minutes = np.apply_along_axis(time_to_polar, axis=1, arr=y_test)

# y_val_hours = keras.utils.to_categorical(y_val[:,0], num_classes)
# y_val_minutes = np.apply_along_axis(time_to_polar, axis=1, arr=y_val)


# %%
batch_size = 64
epochs = 500

if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    # x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    # x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# %%
def build_model(input_shape):
    input_ = keras.layers.Input(shape=x_train.shape[1:])

    # convolutional layers
    #1st Layer
    c1 = keras.layers.Conv2D(32, kernel_size=(5, 5),
                        activation='relu',padding='same',
                        input_shape=input_shape)(input_)
    m1 = keras.layers.MaxPooling2D(pool_size=(4, 4))(c1)
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
    # hour_dense1 = keras.layers.Dense(512, activation='relu')(f)
    # hour_dropout1 = keras.layers.Dropout(0.5)(hour_dense1)
    
    # hour_dense2 = keras.layers.Dense(128, activation='relu')(hour_dropout1)
    # hour_dropout2 =  keras.layers.Dropout(0.3)(hour_dense2)
    
    # hour_dense3 = keras.layers.Dense(128, activation='relu')(hour_dropout2)
    # hour_dropout3 =  keras.layers.Dropout(0.1)(hour_dense3)
    
    # final_hour = keras.layers.Dense(num_classes, activation='softmax')(hour_dropout3)

    # for minutes final layers
    minute_dense1 = keras.layers.Dense(512, activation='relu')(f)
    minute_dropout1 = keras.layers.Dropout(0.5)(minute_dense1)
    
    minute_dense2 = keras.layers.Dense(128, activation='relu')(minute_dropout1)
    minute_dropout2 =  keras.layers.Dropout(0.3)(minute_dense2)
    
    minute_dense3 = keras.layers.Dense(128, activation='relu')(minute_dropout2)
    minute_dropout3 =  keras.layers.Dropout(0.1)(minute_dense3)
    
    final_minute = keras.layers.Dense(4, activation='tanh')(minute_dropout3)

    # final model
    # model = keras.Model(inputs=input_, outputs=[final_hour, final_minute])
    model = keras.Model(inputs=input_, outputs=final_minute)
    
    # model.compile(optimizer="adam", loss=['categorical_crossentropy', 'huber'], metrics=["accuracy", 'mae'])
    
    model.compile(optimizer="adam", loss='huber', metrics=['mae'])

    return model
 
 
#%%
model = build_model(input_shape=input_shape)
model.summary()

#%%
# Initialize EarlyStopping with patience
early_stopping = keras.callbacks.EarlyStopping(
    monitor='loss',  # metric to monitor
    patience=25,          # number of epochs to wait for improvement
    restore_best_weights=True  # restore the best weights after stopping
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.8, patience=10, min_lr=1e-6, cooldown=5
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
                # validation_data=(x_val, y_val_minutes),
                callbacks=[early_stopping, reduce_lr]
)


#%%
loss = history.history['loss']
# val_loss = history.history['val_loss']


#%%
# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'{img_rows} x {img_cols} - Regression loss history with diminishing learning rate (Lograthmic Scale)')
plt.legend()
plt.show()
plt.close()

#%%
y_pred = model.predict(x_test)

#%%
hour_angles = np.arctan2(y_pred[:,1], y_pred[:,0])
hour_angles = np.where(hour_angles < 0, hour_angles + 2 * np.pi, hour_angles)

y_pred_hours = np.round(hour_angles * 12 / (2 * np.pi), 1) % 12

# %%
# y_pred_hours = np.argmax(y_pred[0], axis=1)
y_test_hours_real = y_test[:,0]
y_test_minutes_real =  y_test[:,1]

y_test_real = np.round(y_test_hours_real + y_test_minutes_real/60, 4)

# %%
# Convert the sin and cos values back to minutes
y_pred_mins = np.round((np.arctan2(y_pred[:,3], y_pred[:,2]) % (2 * np.pi)) *  60 / (2 * np.pi), 2) % 60

# %%
new_pred_hr = np.where(y_pred_hours - np.floor(y_pred_hours) < 0.25, 
                       np.where(y_pred_mins > 40, 
                                np.floor(y_pred_hours) - 1,                     # Round the hour down
                                np.floor(y_pred_hours)),                        # Floor the hour
                       np.where(y_pred_hours - np.floor(y_pred_hours) > 0.7,
                                np.where(y_pred_mins < 15,
                                         np.floor(y_pred_hours) + 1,            # Round the hour up
                                         np.floor(y_pred_hours)),               # Floor the hour
                                np.floor(y_pred_hours))) % 12                   # Floor the hour

#%%
hr_diff = (new_pred_hr - y_test_hours_real)
cse_hr = np.where(hr_diff > 6, hr_diff - 12, np.where(hr_diff < -6, hr_diff + 12,  hr_diff))

cse_diff = abs(cse_hr * 60 + (y_pred_mins - y_test_minutes_real))

#%%
import pandas as pd

df = pd.DataFrame({'PredHour': y_pred_hours,
                   "NewPred Hour": new_pred_hr,
                   'TestHour': y_test_hours_real,
                   'PredMin': y_pred_mins,  
                   "TestMin": y_test_minutes_real,
                   "cse_diff": cse_diff
                   })
df[df['cse_diff'] > 25]

# %%
# Scatter plot to show deviation for hour from Real to Pred

plt.scatter(new_pred_hr, y_test_hours_real)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Scatter for hour - Classification')

# %%
abs_mins = abs(y_pred_mins - y_test_minutes_real)
cse_min = np.minimum(abs_mins, 60. - abs_mins)

# %%
# Scatter plot to show deviation for minutes from Real to Pred
plt.scatter(y_pred_mins, y_test[:,1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Scatter for minutes - Polar')

# %%
cse_diff = np.minimum(abs(cse_hr * 60 + (y_pred_mins - y_test_minutes_real)), 60 - 
                      abs(cse_hr * 60 + (y_pred_mins - y_test_minutes_real)))

#%%

sorted_indices_hr = np.argsort(y_test_hours_real)
sorted_y_test_hr = y_test_hours_real[sorted_indices_hr] + y_test_minutes_real[sorted_indices_hr]/60

# Create polar plot to show common sense error for hour from Real to Pred
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.scatter(2 * np.pi * sorted_y_test_hr/12, -cse_diff[sorted_indices_hr], marker='o')

# Add title and show plot
ax.set_title("Polar Classification for Time")
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)      # Set clockwise direction

theta_ticks = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # 12 hours around the clock
ax.set_xticks(theta_ticks)
ax.set_yticks(np.arange(-20, 0, 5))
ax.set_xticklabels([12] + list(range(1, 12)))

plt.show()

# %%
# sorted_indices_min = np.argsort(y_test_minutes_real)
# sorted_y_test_min = y_test_minutes_real[sorted_indices_min]

# # Create polar plot to show common sense error for minutes from Real to Pred
# plt.figure(figsize=(6, 6))
# ax = plt.subplot(111, polar=True)
# ax.plot(2 * np.pi * sorted_y_test_min/60, -cse_min[sorted_indices_min], marker='o')

# # Add title and show plot
# ax.set_title("Polar Common sense Error for minutes")
# # theta_ticks = np.linspace(0, 2 * np.pi, 60, endpoint=False)  # 60 mins around the clock
# # ax.set_xticks(theta_ticks)
# ax.set_yticks(np.arange(-100, 0, 25))
# ax.set_xticklabels(range(60))

# plt.show()


#%%
x_test_reshaped = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
score = model.evaluate(x_test_reshaped, y_test_minutes, verbose=False)

#%%
for each in history.history.keys():
    print(f"Train {each}: {history.history[each][-1]}")

print(f"\nTest loss: {score[0]}")
print(f"Test mae: {score[1]}")

print(f"Common Sense MAE Min Test: {np.mean(cse_min/60)}")

# %%

#%%
import pandas as pd
import matplotlib.pyplot as plt
h24= pd.read_csv("data/150_cla24.csv")
h48= pd.read_csv("data/cla48.csv")
h144= pd.read_csv("data/cla144.csv")
h720= pd.read_csv("data/cla720.csv")

# %%
acc_col = ["accuracy","val_accuracy"]
loss_col = ["loss","val_loss"]
#&&

h24[acc_col].plot(figsize=(8, 5),xlabel='epochs',ylabel='accuracy')
h48[acc_col].plot(figsize=(8, 5),xlabel='epochs',ylabel='accuracy')
h144[acc_col].plot(figsize=(8, 5),xlabel='epochs',ylabel='accuracy')
h720[acc_col].plot(figsize=(8, 5),xlabel='epochs',ylabel='accuracy')
# %%
h24.columns = ["24"+i for i in h24.columns]
h48.columns = ["48"+i for i in h48.columns]
h144.columns = ["144"+i for i in h144.columns]
h720.columns = ["720"+i for i in h720.columns]
# %%
df = pd.concat([h24,h48,h144,h720],axis=1)
# %%
acc_cols = [i for i in df.columns if "acc" in i ]
fig = plt.figure()
ax = fig.add_subplot(111)
import matplotlib.colors as colors
import numpy as np
cmap = plt.get_cmap('tab10')
cmap = cmap(np.linspace(0.0,0.39,8))
cmap = colors.ListedColormap(cmap)
plt.title('Classification output size comparison')
df[acc_cols].plot(figsize=(8, 5),xlabel='epochs',ylabel='accuracy',ax=ax,colormap=cmap)
# %%

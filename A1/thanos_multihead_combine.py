
#%%
import pandas as pd
import matplotlib.pyplot as plt
l= pd.read_csv("data/lower_multihead_version1.csv",index_col=0)
m= pd.read_csv("data/medium_multihead_version1.csv",index_col=0)
b= pd.read_csv("data/better_multihead_version.csv",index_col=0)

# %%
acc_col = ["accuracy","val_accuracy"]
# loss_col = ["loss","val_loss"]
#&&
# %%
l.columns = ["l_"+i for i in l.columns]
m.columns = ["m_"+i for i in m.columns]
b.columns = ["b_"+i for i in b.columns]
# %%
df = pd.concat([l,m,b],axis=1)
# %%
acc_cols = [i for i in df.columns if "acc" in i ]
fig = plt.figure()
ax = fig.add_subplot(111)
import matplotlib.colors as colors
import numpy as np
cmap = plt.get_cmap('tab10')
cmap = cmap(np.linspace(0.0,0.39,6))
cmap = colors.ListedColormap(cmap)
plt.title('Classification output size comparison')
df[acc_cols].plot(figsize=(8, 5),xlabel='epochs',ylabel='accuracy',ax=ax,colormap=cmap)
# %%

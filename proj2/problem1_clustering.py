import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns


# data import
data_raw = np.fromfile('./data1.dat')
data = data_raw.reshape((401, 401))

# plot data
#plt.imshow(np.flipud(data), cmap='gray_r')

# clean up the data and extract the x, y indices of the points that have value of 1
# write your code
[x, y] = np.nonzero(data)
arr = []
for i in range(len(x)):
    arr.append([x[i], y[i]])
# clustering algorithm
# write your code
clustering = DBSCAN(eps=15, min_samples=8).fit(arr)
length = len(clustering.labels_)
for i in range(len(clustering.labels_)-1,-1,-1):
    if clustering.labels_[i] == -1:
        x = np.delete(x, i)
        y = np.delete(y, i)
        clustering.labels_ = np.delete(clustering.labels_, i)
# plot the results
# write your code


df = pd.DataFrame({
    'x': y,
    'y': x,
    'label': clustering.labels_
})
fg = sns.FacetGrid(data=df, hue='label')
fg.map(plt.scatter, 'x', 'y', s=6).add_legend()

plt.xlim(0, 401)
plt.ylim(100, 401)
plt.show()



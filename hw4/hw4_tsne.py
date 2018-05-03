import numpy as np
import pandas as pd
import sys
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

src = np.load('visualization.npy')
stdsc = StandardScaler()
x_train = stdsc.fit_transform(src)

with open('./save/pca_401.pickle','rb') as f:
  pca = pickle.load(f)

x_train_pca = pca.fit_transform(x_train)
kmeans = KMeans(n_clusters=2, max_iter=5000)
kmeans.fit(x_train_pca)
labels = kmeans.labels_
ind_0 = [index for index in range(len(labels)) if labels[index] == 0]
ind_1 = [index for index in range(len(labels)) if labels[index] == 1]

X_embedded = TSNE(n_components=2).fit_transform(x_train_pca)

# plt.scatter(X_embedded[:5000,0], X_embedded[:5000,1], c='b', label='dataset A', s = 0.2)
# plt.scatter(X_embedded[5000:,0], X_embedded[5000:,1], c='r', label='dataset B', s = 0.2)
# plt.legend()
# plt.savefig('tsne.png')


plt.scatter(X_embedded[ind_0,0], X_embedded[ind_0,1], c='b', label='dataset A', s = 0.2)
plt.scatter(X_embedded[ind_1,0], X_embedded[ind_1,1], c='r', label='dataset B', s = 0.2)
plt.legend()
plt.savefig('mine.png')

import numpy as np
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

np.random.seed(5)

path = sys.argv[1]
test_path = sys.argv[2]

src = np.load(path)
stdsc = StandardScaler()
x_train = stdsc.fit_transform(src)

pca = PCA(n_components=401, whiten=True) # svd_solver ='full'
x_train_pca = pca.fit_transform(x_train)

# save model
with open('./save/pca_401.pickle', 'wb') as f:
    pickle.dump(pca, f)

kmeans = KMeans(n_clusters=2, max_iter= 5000)
kmeans.fit(x_train_pca)
labels = kmeans.labels_

# reading test data
test = pd.read_csv(test_path)
data = test[['image1_index','image2_index']].values

result = []
for i in data:
  if labels[i[0]] == labels[i[1]]: 
    result.append(1)
  else: 
    result.append(0)

# output result
f = open(sys.argv[3],'w')
f.write('ID' + ',' + 'Ans\n')
for i in range(len(result)):
  f.write(str(i) + ',' + str(int(result[i])) + '\n')
f.close()

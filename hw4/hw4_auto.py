import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans


train_num = 130000

path = 'ml2018spring-hw4-v2/image.npy'
test_path = 'ml2018spring-hw4-v2/test_case.csv'
src = np.load(path)

src = src.astype('float32')/255.

x_train = src[:train_num]
x_val = src[train_num:]

# input(x_train.shape) (130000,784)
# input(x_val.shape) (10000,784)

input_img = Input(shape=(784,))
# encoder
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# build encoder
encoder = Model(input=input_img, output=encoded)

# build autoencoder
adam = Adam(lr=5e-4)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.summary()

autoencoder.fit(x_train, x_train,
		epochs=50,
		batch_size=256,
		shuffle=True,
		validation_data=(x_val, x_val))
autoencoder.save('./save/autoencoder.h5')
encoder.save('./save/encoder.h5')

encoded_imgs = encoder.predict(src)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
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
f = open('./result/hw4_auto.csv','w')
f.write('ID' + ',' + 'Ans\n')
for i in range(len(result)):
  f.write(str(i) + ',' + str(int(result[i])) + '\n')
f.close()

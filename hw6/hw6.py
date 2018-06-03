# ref1: https://beckernick.github.io/matrix-factorization-recommender/
# ref2: https://nipunbatra.github.io/blog/2017/recommend-keras.html

import keras
import pandas as pd
import numpy as np
from keras.layers import *
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
users_list =  [i.strip().split("::") for i in open('./ml2018-spring-hw6/users.csv' , 'r', encoding='ISO-8859-1').readlines()][1:]
# movies_list = [i.strip().split("::") for i in open('./ml2018-spring-hw6/movies.csv', 'r', encoding='ISO-8859-1').readlines()][1:]
# UserID::Gender::Age::Occupation::Zip-code
users_df = pd.DataFrame(users_list, columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], dtype = int)
users_df['Gender'].replace(['F', 'M'], [0, 1],inplace=True)

def get_model(n_users, n_items, latent_dim=666):
  user_input = Input(shape=[1])
  item_input = Input(shape=[1])
  user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
  user_vec = Flatten()(user_vec)
  item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
  item_vec = Flatten()(item_vec)
  user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
  user_bias = Flatten()(user_bias)
  item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
  item_bias = Flatten()(item_bias)
  r_hat = Dot(axes=1,normalize=True)([user_vec, item_vec])
  r_hat = Add()([r_hat, user_bias, item_bias])
  model = keras.models.Model([user_input, item_input], r_hat)
  model.compile(loss='mse', optimizer=Adam())
  model.summary()
  return model

ratings_df = pd.read_csv('./ml2018-spring-hw6/train.csv', usecols=['UserID','MovieID','Rating'])
ratings_df['Rating'] = ratings_df['Rating'].astype('float')
# input(ratings_df['Rating'])
n_users, n_movies= len(ratings_df.UserID.unique()), len(ratings_df.MovieID.unique())
ratings_df = ratings_df.merge(users_df, how='inner')
# parameter = np.array([ratings_df.Rating.mean(), ratings_df.Rating.std()])
# np.save('parameter.npy', parameter)
# ratings_df.Rating = (ratings_df.Rating - ratings_df.Rating.mean()) / ratings_df.Rating.std()
train ,valid = train_test_split(ratings_df, test_size=0.33)

# Compile model
checkpoint = ModelCheckpoint('MF.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
callbacks_list = [checkpoint, early_stop]
model = get_model(n_users, n_movies)
history = model.fit([train.Age, train.MovieID], train.Rating, validation_data=([valid.Age, valid.MovieID],valid.Rating),batch_size =4096, epochs=400, verbose = 1, callbacks=callbacks_list)

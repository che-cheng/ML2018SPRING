import os
import math
import csv
import numpy as np
import pandas as pd
import random
import sys
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import applications

def load_data():
  path_train = sys.argv[1]
  train = pd.read_csv(path_train)
  x_train = train['feature'].str.split(' ',expand=True).values.astype('float')
  y_train = train['label'].values.astype('float')
  mu = np.mean(x_train, axis = 0)
  sigma = np.std(x_train, axis = 0)
  x_train = (x_train-mu) / (sigma+1e-21)
  Para = np.hstack((mu, sigma))
  np.save('Para.npy',Para)
  return x_train.reshape(len(x_train),48,48,1) , to_categorical(y_train)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

filepath="weights.best.hdf5"

x_train, y_train = load_data()

# split train in train and validation:
my_valid=[]
validation=[]
labelsvalidation=[]
for i in range(len(x_train)//10):
  a = random.randrange(1,len(x_train),1)  
  my_valid.append(a)
  validation.append(x_train[a])
  labelsvalidation.append(y_train[a])

validation = np.array(validation)
labelsvalidation = np.array(labelsvalidation)
train = np.delete(x_train, my_valid, 0)
labels = np.delete(y_train, my_valid, 0)

# image generator
traindatagenerator = ImageDataGenerator(
	featurewise_center=False,
	samplewise_center=False,
	featurewise_std_normalization=False,
	samplewise_std_normalization=False, 
	width_shift_range=0.2,
	height_shift_range=0.2,
	rotation_range=10,
	zoom_range=0.1,
	zca_whitening=False, 
	horizontal_flip=True,
	vertical_flip=False)

batchsize=128
validationdatagenerator = ImageDataGenerator()
train_generator=traindatagenerator.flow(train,labels,batch_size=batchsize) 
validation_generator=validationdatagenerator.flow(validation, labelsvalidation,batch_size=batchsize)

# create model
model = Sequential()
model.add( Conv2D(64,(3,3), padding='same',input_shape=(48,48,1)))
model.add( PReLU())
model.add( BatchNormalization())
model.add( Conv2D(64,(3,3), padding='same'))
model.add( PReLU())
model.add( BatchNormalization())
model.add( MaxPooling2D((2,2), strides=(2, 2)))

model.add( Conv2D(128,(3,3), padding='same'))
model.add( PReLU())
model.add( BatchNormalization())
model.add( Conv2D(128,(3,3), padding='same'))
model.add( PReLU())
model.add( BatchNormalization())
model.add( MaxPooling2D((2,2), strides=(2, 2)))

model.add( Conv2D(256,(3,3), padding='same'))
model.add( PReLU())
model.add( BatchNormalization())
model.add( Conv2D(256,(3,3), padding='same'))
model.add( PReLU())
model.add( BatchNormalization())
model.add( MaxPooling2D((2,2), strides=(2, 2)))

model.add( Conv2D(512,(3,3), padding='same'))
model.add( PReLU())
model.add( BatchNormalization())
model.add( Conv2D(512,(3,3), padding='same'))
model.add( PReLU())
model.add( BatchNormalization())
model.add( MaxPooling2D((2,2), strides=(2, 2)))

model.add( Flatten())
model.add( Dense(32))
model.add( PReLU())
model.add( Dense(units=16))
model.add( PReLU())
model.add( Dense(units = 7, activation='softmax'))

# Compile model
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_acc', patience=40, mode='max') 
callbacks_list = [checkpoint, early_stop]

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=int(len(train)/batchsize), epochs=100, validation_data=validation_generator, validation_steps=int(len(validation)/batchsize), verbose=2, callbacks=callbacks_list)

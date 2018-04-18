import os
import argparse
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical
# from utils import *
# from termcolor import colored,cprint

def load_data():
	path_train = os.getcwd() + '/ml-2018spring-hw3/train.csv'
	train = pd.read_csv(path_train)
	x_train = train['feature'].str.split(' ',expand=True).values.astype('float')
	mu = np.mean(x_train, axis = 0)
	sigma = np.std(x_train, axis = 0)
	x_train = (x_train-mu) / (sigma+1e-21)
	return x_train


model_name = "weights.best_67957.hdf5"

emotion_classifier = load_model(model_name)
input_img = emotion_classifier.input

x_train = load_data()

sess = K.get_session()
for idx in range(10):
    val_proba = emotion_classifier.predict(x_train[idx].reshape(1, 48, 48, 1))
    pred = val_proba.argmax(axis=-1)
    target = K.mean(emotion_classifier.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])
    heatmap = None
    thres = 0.5
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    heatmap = grads.eval(session=sess, feed_dict={input_img:x_train[idx].reshape(1, 48, 48, 1)}).reshape(48,48)
    see = x_train[idx].reshape(48, 48)
    see[np.where(heatmap <= thres)] = np.mean(see)
    plt.figure()
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('color'+str(idx)+'.jpg', dpi=100)

    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('gray'+str(idx)+'.jpg', dpi=100)   
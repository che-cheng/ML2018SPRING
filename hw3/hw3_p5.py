#!/usr/bin/env python
# -- coding: utf-8 --

import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from keras import backend as K
# from utils import *
# from marcos import *
import numpy as np

def load_data():
    path_train = os.getcwd() + '/ml-2018spring-hw3/train.csv'
    train = pd.read_csv(path_train)
    x_train = train['feature'].str.split(' ',expand=True).values.astype('float')
    mu = np.mean(x_train, axis = 0)
    sigma = np.std(x_train, axis = 0)
    x_train = (x_train-mu) / (sigma+1e-21)
    return x_train.reshape(-1,48,48,1)

private_pixels = load_data()

model_path = "weights.best_67957.hdf5"
emotion_classifier = load_model(model_path)


layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)
input_img = emotion_classifier.input
name_ls = ['conv2d_1', 'p_re_lu_1']
collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]


choose_id = 17
photo = private_pixels[choose_id].reshape(1,48,48,1)
for cnt, fn in enumerate(collect_layers):
    im = fn([photo, 0]) #get the output of that layer
    fig = plt.figure(figsize=(14, 8))
    nb_filter = 32
    for i in range(nb_filter):
        ax = fig.add_subplot(nb_filter/8, 8, i+1)
        ax.imshow(im[0][0,:,:,i], cmap='BuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
    img_path = './Q5'
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))
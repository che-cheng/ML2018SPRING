import os
import math
import csv
import sys
import numpy as np
import pandas as pd
import random
from keras.models import load_model
from keras.layers import *
from keras.utils import np_utils, to_categorical
from keras import applications

def load_data():
  Para = np.load(os.getcwd() + '/Para/Para.npy')
  path_test  = sys.argv[1]
  test = pd.read_csv(path_test)
  x_test  = test['feature'].str.split(' ',expand=True).values.astype('float')
  mu = Para[:len(Para)//2]
  sigma = Para[len(Para)//2:]
  x_test =  (x_test -mu) / (sigma+1e-21)
  return x_test.reshape(len(x_test),48,48,1)

if sys.argv[3] == 'private':
  filepath = "./model/weights.best_68375.hdf5"
elif sys.argv[3] == 'public':
  filepath = "./model/weights.best_67957.hdf5"

x_test = load_data()

model = load_model(filepath)
predict = model.predict(x_test)
result = predict.argmax(axis=1)

# output result
f = open(sys.argv[2],'w') 
f.write('id' + ',' + 'label\n')
for i in range(len(result)):
  f.write(str(i) + ',' + str(int(result[i])) + '\n')
f.close()

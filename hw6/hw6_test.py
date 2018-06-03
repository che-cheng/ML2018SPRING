import sys
import pandas as pd
import numpy as np
from keras.models import load_model

model_path = './model/model1.hdf5'
test_df = pd.read_csv(sys.argv[1], usecols=['UserID','MovieID'])

users_list =  [i.strip().split("::") for i in open(sys.argv[4] , 'r', encoding='ISO-8859-1').readlines()][1:]
users_df = pd.DataFrame(users_list, columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], dtype = int)
movies_list = [i.strip().split("::") for i in open(sys.argv[3], 'r', encoding='ISO-8859-1').readlines()][1:]

# test_df = test_df.merge(users_df, how='inner')

model = load_model(model_path)
predict = model.predict([test_df.UserID, test_df.MovieID])

f = open(sys.argv[2],'w')
f.write('TestDataID' + ',' + 'Rating\n')
for i in range(len(predict)):
  if predict[i] > 5.:
    predict[i] = 5.
  if predict[i] < 1.:
    predict[i] = 1.
  f.write(str(i+1) + ',' + str(predict[i][0]) + '\n')
f.close()

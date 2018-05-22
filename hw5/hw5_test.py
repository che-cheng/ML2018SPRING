import sys 
import keras
import _pickle as pk
import numpy as np
from keras.models import Model, Sequential, load_model

from util import DataManager

# argv settings
test_path = sys.argv[1]
output_path = sys.argv[2]
mode = sys.argv[3]

# load data
dm = DataManager()
dm.add_data('test_data',test_path,False)


if mode=='private':
  # tokenizer
  dm.load_tokenizer('./token/token.pk')
  # load model
  model = load_model('./model/model1.hdf5')
elif mode=='public':
  # tokenizer
  dm.load_tokenizer('./token/token_filter.pk')
  # load model
  model = load_model('./model/model2.hdf5')

dm.to_sequence(40,'test')
test_all_x = dm.get_data('test_data')

print(model.summary())
predict = model.predict(test_all_x,batch_size = 1024, verbose=1)
predict[predict <=  0.5] = 0
predict[predict  >  0.5] = 1

f = open(output_path,'w')
f.write('id' + ',' + 'label\n')
for i in range(len(predict)):
  f.write(str(i) + ',' + str(int(predict[i])) + '\n')
f.close()

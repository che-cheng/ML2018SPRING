import sys
import keras
import _pickle as pk
import readline
import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Input, GRU, LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from util import DataManager

train_path = sys.argv[1]
semi_path  = sys.argv[2]
action = 'train'

def RNN( embedding_matrix):
    embedding_layer = Embedding( embedding_matrix.shape[0], 
                                 embedding_matrix.shape[1],
                                 weights=[embedding_matrix],
                                 trainable=False, 
                                 input_length = 40)
    dropout_rate = 0.4
    RNN_cell1 = GRU(512, 
                   return_sequences=True, 
                   recurrent_dropout=dropout_rate,
                   dropout=dropout_rate)
    RNN_cell2 = GRU(256,
                   return_sequences=False,
                   recurrent_dropout=dropout_rate,
                   dropout=dropout_rate)
    model = Sequential()
    model.add(embedding_layer)
    model.add(RNN_cell1)
    model.add(RNN_cell2)
#    model.add(Dense(128,activation='relu'))
#    model.add(Dropout(dropout_rate))
#    model.add(Dense(64,activation='relu'))
#    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    print ('compile model...')
    model.compile( loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy']) 
    return model

def main():
    dm = DataManager()
    dm.add_data('train_data',train_path,True)
    dm.add_data('semi_data',semi_path,False)
    
    print ('Get Tokenizer...')
    dm.load_tokenizer('./token/token.pk')
    
    embedding_mat = dm.to_sequence(40,action)
   
    print ('Initial model...')
    if action == 'train':
      model = RNN(embedding_mat)  
      print (model.summary())
    elif action == 'semi':
      model = load_model('./model/model1.hdf5')
      print(model.summary())

    if action == 'train' :
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', 0.2)
        earlystopping = EarlyStopping(monitor='val_acc', patience = 30, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(filepath='./model/model.hdf5',verbose=1,save_best_only=True,monitor='val_acc',mode='max')
        model.fit(X, Y,validation_data=(X_val, Y_val),epochs=80,batch_size=512,callbacks=[checkpoint, earlystopping] )

    elif action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', 0.2)
        [semi_all_X] = dm.get_data('semi_data')
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(filepath='./model/model_semi.hdf5',verbose=1,save_best_only=True,monitor='val_acc',mode='max' )
        for i in range(10):
            semi_pred = model.predict(semi_all_X, batch_size=2048, verbose=1)
            semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred,0.1)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            model.fit(semi_X, semi_Y,validation_data=(X_val, Y_val),epochs=2,batch_size=512,callbacks=[checkpoint, earlystopping])
            print ('load model from')
            model = load_model('./model/model_semi.hdf5')
if __name__ == '__main__':
  main()


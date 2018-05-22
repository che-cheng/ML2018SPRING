import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import _pickle as pk

class DataManager:
    def __init__(self):
        self.data = {}
    
    def add_data(self,name, data_path, with_label=True):
        print ('read data from %s...'%data_path)
        X, Y = [], []
        with open(data_path,'r') as f:
            firstline = True
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                else:
                    if name == 'test_data':
                      if firstline == True:
                        firstline = False
                        continue
                      lines = line.strip().split(",", 1)[1]
                      X.append(lines)
                    else:
                      X.append(line.strip())
        if with_label:
            self.data[name] = [X,Y]
        else:
            self.data[name] = [X]
    
    def tokenize(self):
        print ('create new tokenizer')
        self.tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
        for key in self.data:
            print ('tokenizing %s'%key)
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)
        
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump(self.tokenizer, open(path, 'wb'))
            
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pk.load(open(path, 'rb'))

    def to_sequence(self, maxlen, name):
        all_doc = []
        for key in self.data:
            print ('Converting %s to sequences'%key)
            docu = [doc.split(" ") for doc in self.data[key][0]]
            all_doc = all_doc + docu
            tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen, padding='post')) 
        
        word_index = self.tokenizer.word_index
        embedding_matrix = np.zeros([len(word_index) + 1, 200])
        if name == 'train':
          word_vectors = Word2Vec(all_doc, size=200, window=10, min_count=3, sg=1)
          for w, i in word_index.items():
            if w in word_vectors:
              embedding_matrix[i] = word_vectors[w]
        return embedding_matrix
    
    def get_semi_data(self,name,label,threshold) : 
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        return semi_X[index,:], semi_Y[index]

    def get_data(self,name):
        return self.data[name]

    def split_data(self, name, ratio):
        data = self.data[name]
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])

import os
import math
import csv 
import sys
import numpy as np

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

path_t = sys.argv[1]
test 	 = open(path_t, 'r')
testX  = [];
row_test = csv.reader(test , delimiter= "\n")

para = np.load('normal_logistic.npy')
w = np.load('w_logistic.npy')

mu = para[0:123]
sigma = para[123:]

# testing data
firstline = True;
for r in row_test:
	if firstline:
		firstline = False;
		continue;
	splt = r[0].split(",");
	c = [int(i) for i in splt]
	testX.append(c);
testX = np.array(testX);

# normalization
testX = (testX-mu) / (sigma+1e-21)
testX = np.concatenate((np.ones((testX.shape[0],1)),testX), axis=1)

# prediction
sig =  sigmoid(np.dot(testX,w))
result = []
for i in range(len(sig)):
	r = 0;
	if sig[i] > 0.5: r = 1;
	result = np.append(result,r)

f = open(sys.argv[2],'w') 
f.write('id' + ',' + 'label\n')
for i in range(len(result)):
	f.write(str(i+1) + ',' + str(int(result[i])) + '\n')
f.close()
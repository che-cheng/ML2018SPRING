import os
import math
import csv 
import sys
import numpy as np

lr = 0.05;
la = 0.1;

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def accuracy(predicted_labels, actual_labels):
	diff = predicted_labels - actual_labels
	return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

def logistic(X, y, iteration):
	X_inv = np.linalg.pinv(X)
	w = np.dot(X_inv,y)
	error = []
	for it in range(iteration):
		sig  =  sigmoid(np.dot(X,w))
		cost = - np.dot( y, np.log(sig+1e-21) ) - np.dot((1-y), np.log(1-sig+1e-21))
		gra  = np.dot( X.transpose(), (sig-y)) + la*w
		w = w - lr * gra / len(X)
		predict = np.where(sig > 0.5, 1, 0)
		print('Iteration: ' + str(it) + '. Categorization Accuracy: ' + str(accuracy(predict, y.flatten())))
	np.save('w_logistic.npy',w)
	return w

path_x = sys.argv[1]
path_y = sys.argv[2]
text_x = open(path_x, 'r')
text_y = open(path_y, 'r')

trainX = [];
trainY = [];
row_x  = csv.reader(text_x , delimiter="\n")
row_y  = csv.reader(text_y , delimiter="\n")


firstline = True;
for r in row_x:
	if firstline:
		firstline = False;
		continue;
	splt = r[0].split(",");
	c = [int(i) for i in splt]
	trainX.append(c);
for r in row_y:
	trainY.append(int(r[0]))

trainX = np.array(trainX)
trainY = np.array(trainY)

# normalization
mu = np.mean(trainX, axis = 0)
sigma = np.std(trainX, axis = 0)
trainX = (trainX-mu) / (sigma+1e-21)
trainX = np.concatenate((np.ones((trainX.shape[0],1)),trainX), axis=1)

normal = np.hstack((mu, sigma))
np.save('normal_logistic.npy',normal)
w = logistic(trainX,trainY, 10000)
import sys
import os
import math
import csv 
import numpy as np

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def accuracy(predicted_labels, actual_labels):
	diff = predicted_labels - actual_labels
	return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

def generative(X,y):
	# calculate mu1 and mu2 (without normalization)
	mu1 = np.zeros(len(X[0]))
	mu2 = np.zeros(len(X[0]))
	# calculate sigma1 and sigma2 (without normalization)
	sigma1 = np.zeros((len(X[0]),len(X[0])))
	sigma2 = np.zeros((len(X[0]),len(X[0])))
	cut1 = 0
	cut2 = 0
	for i in range(len(X)):
		if y[i] == 1:
			mu1 += X[i]
			cut1 += 1
		else:
			mu2 += X[i]
			cut2 += 1
	mu1 /= cut1
	mu2 /= cut2
	for i in range(len(X)):
		if y[i] == 1:
			Submean = X[i] - mu1
			sigma1 += np.dot(np.transpose([Submean]), [Submean])
		else:
			Submean = X[i] - mu2
			sigma2 += np.dot(np.transpose([Submean]), [Submean])
	sigma = (sigma1 / len(X)) + (sigma2 / len(X))
	return mu1,cut1,mu2,cut2,sigma;

def predict(X,u1,num1,u2,num2,sigma):
	sigma_inv = np.linalg.pinv(sigma)
	w = np.dot((u1-u2), sigma_inv)
	b = -0.5 * np.dot(np.dot([u1], sigma_inv), u1) + 0.5*np.dot(np.dot([u2], sigma_inv), u2) + np.log(float(num1) / num2)
	z = np.dot(w, X.T) + b;
	sig = sigmoid(z)
	result = []
	for i in range(len(sig)):
		r = 0;
		if sig[i] > 0.5: r = 1;
		result = np.append(result,r)
	return result
	

path_x = os.getcwd() + '/' + sys.argv[1]
path_y = os.getcwd() + '/' + sys.argv[2]
path_t = os.getcwd() + '/' + sys.argv[3]
text_x = open(path_x, 'r')
text_y = open(path_y, 'r')
test 	 = open(path_t, 'r')

trainX = [];
trainY = [];
testX  = [];
row_x  = csv.reader(text_x , delimiter="\n")
row_y  = csv.reader(text_y , delimiter="\n")
row_test = csv.reader(test , delimiter= "\n")

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
sigma_train = np.std(trainX, axis = 0)
trainX = (trainX-mu) / (sigma_train+1e-4)
trainX = np.concatenate((np.ones((trainX.shape[0],1)),trainX), axis=1)

mu1, num1, mu2, num2, sigma = generative(trainX, trainY);

# accuracy
result_train = predict(trainX, mu1, num1, mu2, num2, sigma);
accu = accuracy(result_train, trainY);
print('accuracy= ' + str(accu))

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
testX = (testX-mu) / (sigma_train+1e-4)
testX = np.concatenate((np.ones((testX.shape[0],1)),testX), axis=1)

# prediction
result = predict(testX, mu1, num1, mu2, num2, sigma);

f = open(sys.argv[4],'w') 
f.write('id' + ',' + 'label\n')
for i in range(len(result)):
	f.write(str(i+1) + ',' + str(int(result[i])) + '\n')
f.close()
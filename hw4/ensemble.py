import os
import csv 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


# load data
path_x = os.getcwd() + '/ntu-ml2018spring-hw2/train_X'
path_y = os.getcwd() + '/ntu-ml2018spring-hw2/train_Y'
path_t = os.getcwd() + '/ntu-ml2018spring-hw2/test_X'
text_x = open(path_x, 'r')
text_y = open(path_y, 'r')
test 	 = open(path_t, 'r')

trainX = [];
trainY = [];
testX  = [];
row_x  = csv.reader(text_x , delimiter="\n")
row_y  = csv.reader(text_y , delimiter="\n")
row_test = csv.reader(test , delimiter="\n")

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
trainX = (trainX-mu) / sigma
trainX = np.concatenate(((trainX[:,0]**2).reshape(trainX.shape[0],1),trainX), axis=1)
trainX = np.concatenate(((trainX[:,77:82]**2).reshape(trainX.shape[0],5),trainX), axis=1)

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
testX = (testX-mu) / sigma
testX = np.concatenate(((testX[:,0]**2).reshape(testX.shape[0],1),testX), axis=1)
testX = np.concatenate(((testX[:,77:82]**2).reshape(testX.shape[0],5),testX), axis=1)

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
bdt.fit(trainX, trainY)
sig = bdt.predict(testX)
result = []
for i in range(len(sig)):
  r = 0;
  if sig[i] > 0.5: r = 1;
  result = np.append(result,r)

f = open('./result/hw4_adaboost.csv','w') 
f.write('id' + ',' + 'label\n')
for i in range(len(result)):
	f.write(str(i+1) + ',' + str(int(result[i])) + '\n')
f.close()
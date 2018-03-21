import os
import numpy as np
import csv 

def gradient_descent(X, y, lr, iteration):
	
	feature_num = len(X[0])
	err = []
	b = 0
	w = np.zeros(feature_num).astype('float')
	lr_b = 0
	lr_w = np.zeros(feature_num).astype('float')
	Lambda = 100.

	for it in range(iteration):
		b_grad = 0.0
		w_grad = np.zeros(feature_num).astype('float')
		Sum = 0.0
		for n in range(len(X)):
			error = y[n] - b - np.dot(w,X[n])
			Sum = Sum + error**2
			b_grad = b_grad - 2.0*error*1.0
			w_grad = w_grad - 2.0*error*X[n] + 2*Lambda*w
			
		lr_b += b_grad**2
		lr_w += np.square(w_grad)
		# update parameters
		b = b - lr * b_grad/np.sqrt(lr_b)
		w = w - lr * w_grad/(np.sqrt(lr_w) + 1e-21)

		root_mean = np.sqrt(Sum/len(X))
		if it % 10 == 0 :
			err = np.append( err, root_mean)
			Save = np.hstack((w, b))
		print('Iteration: ' + str(it) + '. RMSE: ' + str(root_mean))

	np.save('error.npy', err)
	np.save('parameters.npy', Save)	
	return b, w

path = os.getcwd() + '/ml-2018spring-hw1/train.csv'
data = []    
for i in range(18):
	data.append([])

# store data according to the feature
n_row = 0
text = open(path, 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row != 0:
		for i in range(3,27):
			if r[i] != "NR":
				data[(n_row-1)%18].append(float(r[i]))
			else:
				data[(n_row-1)%18].append(float(0)) 
	n_row = n_row+1
text.close()


x = []
y = []
for i in range(12):
	for j in range(471):
		x.append([])
		for t in range(18):
			for s in range(9):
				x[471*i+j].append(data[t][480*i+j+s])
		y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# Data pre-processing for training data
del_num = np.array([])
for i in range(len(x)):
	tmp1 = x[i][0:90]
	tmp2 = x[i][99:]
	PM25 = x[i][81:90]
	if (len(tmp1[tmp1 <= 0]) > 0) or (y[i] >= 300 or y[i] <= 0) or (len(tmp2[tmp2 <= 0]) > 0) or (len(PM25[PM25 >= 300]) > 0):
		del_num = np.append( del_num, i)


train_X = np.delete(x, del_num, axis = 0)
train_Y = np.delete(y, del_num, axis = 0)
b, w = gradient_descent(train_X, train_Y, 10 ,100)
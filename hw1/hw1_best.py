import os
import sys
import numpy as np
import csv 

def gradient_descent(X, y, lr, iteration):
	
	feature_num = len(X[0])
	err = []
	b = 0
	w = np.zeros(feature_num).astype('float')
	lr_b = 0
	lr_w = np.zeros(feature_num).astype('float')
	Lambda = 10.

	for it in range(iteration):
		b_grad = 0.0
		w_grad = np.zeros(feature_num).astype('float')
		
		Sum = 0.0
		for n in range(len(X)):
			error = y[n] - b - np.dot(w,X[n])
			Sum = Sum + error**2
			b_grad = b_grad - 2.0*error
			w_grad = w_grad - 2.0*error * X[n] + 2*Lambda*w
			
		lr_b  += b_grad**2
		lr_w  += w_grad**2
		# update parameters
		b = b - lr * b_grad/(lr_b)**(0.5)
		w = w - lr * w_grad/((lr_w)**(0.5) + 1e-21)

		root_mean = (Sum/len(X))**(0.5)
		if it % 10 == 0 :
			err = np.append( err, root_mean)
			Save = np.hstack((w, b))
		print('[BEST] Iteration: ' + str(it) + '. RMSE: ' + str(root_mean))

	np.save('error_best.npy', err)
	np.save('parameters_best.npy', Save)
	return b, w

def get_test_result(data,para):
	# w1 = para[0:9]
	w = para[:len(para)-1]
	b = para[len(para)-1]
	y = []
	for i in data:
		result = np.dot(i,w) + b
		y = np.append(y,result)
	return y


# path_train = os.getcwd() + '/ml-2018spring-hw1/train.csv'
# data = []    
# for i in range(18):
# 	data.append([])

# # store data according to the feature
# n_row = 0
# text = open(path_train, 'r', encoding='big5') 
# row = csv.reader(text , delimiter=",")
# for r in row:
# 	if n_row != 0:
# 		for i in range(3,27):
# 			if r[i] != "NR":
# 				data[(n_row-1)%18].append(float(r[i]))
# 			else:
# 				data[(n_row-1)%18].append(float(0)) 
# 	n_row = n_row+1
# text.close()

# x = []
# y = []
# # (4) 8=PM10 9=PM2.5 12=SO2 15=WIND_DIREC 16=WIND_SPEED
# # (5) 7=O3, 8=PM10 9=PM2.5 15=WIND_DIREC 16=WIND_SPEED
# # (6) 8=PM10 9=PM2.5
feature = [2,5,8,9]

# for i in range(12):
# 	for j in range(471):
# 		x.append([])
# 		for f in feature:
# 			for s in range(9):
# 				x[471*i+j].append(data[f][480*i+j+s])
# 		y.append(data[9][480*i+j+9])
# x = np.array(x)
# y = np.array(y)

# # Data pre-processing for training data
# del_num = np.array([])
# for i in range(len(x)):
# 	tmp = x[i]
# 	PM25 = tmp[27:]
# 	if (len(tmp[tmp <= 0]) > 0) or (y[i] >= 300 or y[i] <= 0) or (len(PM25[PM25 >= 300]) > 0):
# 		del_num = np.append( del_num, i)


# train_X = np.delete(x, del_num, axis = 0)
# train_Y = np.delete(y, del_num, axis = 0)
# b, w = gradient_descent(train_X, train_Y, 10 ,50000)


pa_path = os.getcwd() + '/parameters_best.npy'
# path_test = os.getcwd() + '/ml-2018spring-hw1/test.csv' # argv[1]
para = np.load(pa_path)
test_x = []
n_row = 0
text = open(sys.argv[1],"r")
row = csv.reader(text , delimiter= ",")

# Data pre-processing for testing data
for r in row:
	if n_row%18 == 2 or n_row%18 == 5 or n_row%18 == 8 or n_row%18 == 9:
		if n_row %18 == feature[0]: test_x.append([])
		for i in range(2,11):
			if float(r[i]) == 0:
				if i==2: test_x[n_row//18].append(float(r[i+1]))
				elif i==10: test_x[n_row//18].append(float(r[i-1]))
				else: test_x[n_row//18].append((float(r[i+1])+float(r[i-1]))/2.)
			elif r[i] =="NR":
				test_x[n_row//18].append(0)
			else:
				test_x[n_row//18].append(float(r[i]))
	n_row = n_row+1
text.close()
test_x = np.array(test_x)


result = get_test_result(test_x,para)
f = open(sys.argv[2],'w')  # argv[2]
f.write('id' + ',' + 'value\n')
for i in range(len(result)):
	f.write('id_' + str(i) + ',' + str(result[i]) + '\n')
f.close()
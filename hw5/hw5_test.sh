#!/bin/bash
mkdir model
wget https://www.dropbox.com/s/9i4t3f3niis7n1j/model1.hdf5 -P ./model/
wget https://www.dropbox.com/s/nf6tz73gjo6jjff/model2.hdf5 -P ./model/
python3 hw5_test.py $1 $2 $3
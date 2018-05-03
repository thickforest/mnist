#!/usr/bin/python
#coding:utf-8
########################################################################
# File Name: keras_show.py
# Author: forest
# Mail: thickforest@126.com 
# Created Time: 2018年05月02日 星期三 11时49分37秒
########################################################################
from keras.models import load_model
from matplotlib import pyplot as plt

MODEL_FILENAME = "number_model.hdf5"

model = load_model(MODEL_FILENAME)
model.summary()

n = 10
row = 2
col = n/row
if (row * col < n):
	col += 1
for i in range(10):
	plt.subplot(row, col, i+1)
	#plt.imshow(model.layers[1].get_weights()[0][:,i].reshape(28, 28), cmap='gray')
	plt.imshow(model.layers[1].get_weights()[0][:,i].reshape(28, 28))

plt.show()

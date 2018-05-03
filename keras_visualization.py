#!/usr/bin/python
#coding:utf-8
# apt-get install graphviz
# pip install pydot-ng
from keras.models import load_model
from keras.utils import plot_model

MODEL_FILENAME = "number_model2.hdf5"

# Load the trained neural network
model = load_model(MODEL_FILENAME)

model.summary()
# 绘制模型结构图
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

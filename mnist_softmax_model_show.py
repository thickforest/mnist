#!/usr/bin/python
#coding:utf-8
import tensorflow as tf
import matplotlib.pyplot as plt

def debug():
    import pdb
    pdb.set_trace()

sess = tf.Session()
load = tf.train.import_meta_graph("mnist_softmax/model-999.meta")
load.restore(sess, tf.train.latest_checkpoint("mnist_softmax/"))
graph = tf.get_default_graph()

# graph.get_operations()
weights = graph.get_tensor_by_name("weights/Variable:0")
W = weights.eval(session=sess)

row = 2
col = 5
for i in range(min(W.shape[1], row*col)):
    plt.subplot(row, col, i+1)
    img = W[:,i].reshape(28, 28)
    plt.imshow(img)
plt.show()

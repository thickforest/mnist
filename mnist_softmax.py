#!/usr/bin/env python
#coding:utf-8
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
#from tensorflow.examples.tutorials.mnist import input_data
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

sess = tf.InteractiveSession()

def varible_summaries(var):
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)
# 随着迭代的加深，最大值越来越大，最小值越来越小，与此同时，也伴随着方差越来越大，这样的情况是我们愿意看到的，神经元之间的参数差异越来越大。因为理想的情况下每个神经元都应该去关注不同的特征，所以他们的参数也应有所不同。

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
with tf.name_scope('weights'):
  W = tf.Variable(tf.zeros([784, 10]))
  varible_summaries(W)
with tf.name_scope('biases'):
  b = tf.Variable(tf.zeros([10]))
  varible_summaries(b)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/mnist_deep_logs', sess.graph_def)
# tensorboard --logdir /tmp/mnist_deep_logs

saver = tf.train.Saver(max_to_keep=1)

# Train
tf.initialize_all_variables().run()
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  summary_str,_ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
  writer.add_summary(summary_str, i)

saver.save(sess, "mnist_softmax/model", global_step=i)

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

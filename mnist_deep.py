#!/usr/bin/env python
# load MNIST data
import input_data
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

# start tensorflow interactiveSession
import tensorflow as tf
sess = tf.InteractiveSession()

# weight initialization
def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name = name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial, name = name)

# convolution
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Create the model
# placeholder
x = tf.placeholder("float", [None, 784], name = "x-input")
y_ = tf.placeholder("float", [None, 10], name = "y_output")

# first convolutinal layer
w_conv1 = weight_variable([5, 5, 1, 32], name = "conv_W1")
b_conv1 = bias_variable([32], name = "conv_B1")

x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
	h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64], name = 'conv_W2')
b_conv2 = bias_variable([64], name = 'conv_B2')

with tf.name_scope('conv2'):
	h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([7*7*64, 1024], name = 'dense_W1')
b_fc1 = bias_variable([1024], name = 'dense_B1')

with tf.name_scope('dense1'):
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
with tf.name_scope('dropout'):
	keep_prob = tf.placeholder("float", name = 'dropout_prob')
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, 10], name = 'dense_W2')
b_fc2 = bias_variable([10], name = 'dense_B2')

with tf.name_scope('softmax'):
	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# train and evaluate the model
with tf.name_scope('xent'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    _ = tf.summary.scalar('cross entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(1e-5).minimize(cross_entropy)

with tf.name_scope('test'):
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	_ = tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/mnist_deep_logs', sess.graph_def)

sess.run(tf.initialize_all_variables())
test_batch = mnist.test.next_batch(10)
for i in range(100):
	batch = mnist.train.next_batch(20)
	if i%20 == 0:
		'''
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
		print("step %d, train accuracy %g" %(i, train_accuracy))
		'''
		test_feed = {x: test_batch[0], y_: test_batch[1], keep_prob:1.0}
		result = sess.run([merged, accuracy], feed_dict=test_feed)
		summary_str = result[0]
		acc = result[1]
		writer.add_summary(summary_str, i)
		print('Accuracy at step %s: %s' % (i, acc))
	else:
		train_feed={x:batch[0], y_:batch[1], keep_prob:0.5}
		#train_step.run(feed_dict=train_feed)
		sess.run(train_step, feed_dict=train_feed)

print("test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

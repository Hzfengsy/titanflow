import numpy as np
import titanflow as tf

X = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0]).reshape((1, 4, 4, 1))
W = np.array([1, 1, 0, 1]).reshape((2, 2, 1, 1))

x = tf.placeholder(tf.float32)
w = tf.placeholder(tf.float32)

conv = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID')
max_pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
grad_x, grad_w = tf.gradients(conv, [x, w])
print X[0, :, :, 0]
print conv.eval(feed_dict = {x: X, w: W})[0, :, :, 0]
print grad_x.eval(feed_dict = {x: X, w: W})[0, :, :, 0]
print grad_w.eval(feed_dict = {x: X, w: W})[:, :, 0, 0]
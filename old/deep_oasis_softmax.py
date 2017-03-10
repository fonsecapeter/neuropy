# currently way too slow to run on mac

import tensorflow as tf
from sample_data import oasis

BATCH_SIZE = 50

# oasis.images are the mris flattened from
# 91 * 109 * 91 = 902629
oasis.load_data()

sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 902629])
y_ = tf.placeholder(tf.float32, shape=[None, 2])


# 1st convolutional layer
# -----------------------------------------------------------------------------------------------
# convolution, slide a 30x30x30 cube around each image, with 1 color channel, producing 32 features
W_conv1 = weight_variable([30, 30, 30, 1, 32])

b_conv1 = bias_variable([32])
# reshape x to be a 5d tensof
x_image = tf.reshape(x, [-1, 91, 109, 91, 1])

# convolute x with weight tensor, add bias, apply ReLU
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
# apply max_pool to reduce to 14x14 image
h_pool1 = max_pool_2x2x2(h_conv1)

# 2nd convolutional layer
# -----------------------------------------------------------------------------------------------
# deep network, stack next layer with 64 features for each 30x30x30 cube
W_conv2 = weight_variable([30, 30, 30, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
# downsample
h_pool2 = max_pool_2x2x2(h_conv2)

# Densely Connected Layer
# -----------------------------------------------------------------------------------------------
# we add a fully-connected layer with 1024 neurons to allow processing on the entire image
# reshape pooling layer tensor into a batch of vectors
# mult by a weight matrix, add bias, & apply ReLu
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# -----------------------------------------------------------------------------------------------
# to reduce overfitting, apply a droupout before the readout layer
# create placeholder for prob that a neuron's output is kept so we can turn drouput on/off
# (train/test) - auto with tf
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Redout Layer
# -----------------------------------------------------------------------------------------------
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Train and Evaluate the Model
# -----------------------------------------------------------------------------------------------
# WARNING: training takes a while, maybe 0.5hr
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(2000):
    batch = oasis.train.next_batch(BATCH_SIZE)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0
        })
        print('step %d, training accuracy: %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print('test accuracy: %g' % accuracy.eval(feed_dict={
    x: oasis.test.images, y_: mnissisaot.labels, keep_prob: 1.0
}))


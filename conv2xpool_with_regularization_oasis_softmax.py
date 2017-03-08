# wayyyyyy too slow

import numpy as np
import tensorflow as tf
import time
from sample_data import oasis

BATCH_SIZE = 16
NUM_TRAINS = 1
PATCH_SIZE = 5
DEPTH = 16
H_SIZE = 64
REG_BETA = 3e-4

oasis.load_data()

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, *oasis.image_shape, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, oasis.num_labels])

    # -------------------------------------------------------------------------------
    # Layer 1 
    W_1 = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1, DEPTH], stddev=0.1))
    b_1 = tf.Variable(tf.zeros([DEPTH]))
    conv = tf.nn.conv3d(x, W_1, [1, 2, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + b_1)
    pool = tf.nn.max_pool3d(hidden, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='SAME')

    # -------------------------------------------------------------------------------
    # Layer 2 
    W_2 = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
    b_2 = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
    conv = tf.nn.conv3d(pool, W_2, [1, 1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv * b_2)
    pool = tf.nn.max_pool3d(hidden, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='SAME')

    # -------------------------------------------------------------------------------
    # Layer 3
    flat_shape = np.prod(pool.get_shape().as_list()[1:])
    W_3 = tf.Variable(tf.truncated_normal([flat_shape, H_SIZE], stddev=0.1))
    b_3 = tf.Variable(tf.constant(1.0, shape=[H_SIZE]))
    reshape = tf.reshape(pool, [-1, flat_shape])
    hidden = tf.nn.relu(tf.matmul(reshape, W_3) + b_3)

    # -------------------------------------------------------------------------------
    # Output Layer
    W_o = tf.Variable(tf.truncated_normal([H_SIZE, oasis.num_labels]))
    b_o = tf.Variable(tf.constant(1.0, shape=[oasis.num_labels]))

    # -------------------------------------------------------------------------------
    # Train
    logits = tf.matmul(hidden, W_o) + b_o
    y = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    regularizers = (
            tf.nn.l2_loss(W_1) + tf.nn.l2_loss(b_1) +
            tf.nn.l2_loss(W_2) + tf.nn.l2_loss(b_2) +
            tf.nn.l2_loss(W_3) + tf.nn.l2_loss(b_3) +
            tf.nn.l2_loss(W_o) + tf.nn.l2_loss(b_o)
    )
    loss += (REG_BETA * regularizers)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph=graph) as sess:
    print('initializing graph...')
    tf.global_variables_initializer().run()
    for i in range(NUM_TRAINS):
        start = time.time()
        batch_xs, batch_ys = oasis.train.next_batch(BATCH_SIZE)
        batch_xs = batch_xs.reshape(-1, *oasis.image_shape, 1)
        # if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_xs, y_: batch_ys
        })
        print('step %d, training accuracy: %g in %.2fs' % (i, train_accuracy, time.time() - start))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # -------------------------------------------------------------------------------
    # Display Validation Accuracy
    print('test accuracy:', sess.run(accuracy, feed_dict={x: oasis.test.images.reshape(-1, *oasis.image_shape, 1), y_: oasis.test.labels}))

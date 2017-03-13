# https://github.com/WeiTang114/MVCNN-TensorFlow
import numpy as np
import math
import tensorflow as tf
from sample_data import oasis
from util.progress import print_progress

LEARNING_RATE = 0.000001
MOMENTUM = 0.3

BATCH_SIZE = 60
NUM_EPOCHS = 800

PATCH_1 = 9 
DEPTHi_1 = 96

PATCH_2 = 5
DEPTH_2 = 256

PATCH_3 = 3
DEPTH_3 = 384

PATCH_4 = 3
DEPTH_4 = 384

PATCH_5 = 3
DEPTH_5 = 256

CONNECTED_SIZE = 49
CONNECTED_DEPTH = 4096

REG_BETA = 0.0005
KEEP_PROB = 0.5

oasis.load_data()
shape = oasis.image_shape

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, shape[0], shape[1], shape[2],  1])
    y_ = tf.placeholder(tf.float32, shape=[None, oasis.num_labels])
    # -------------------------------------------------------------------------------
    # Convolutional Layers
    W_1 = tf.Variable(tf.truncated_normal(np.asarray([PATCH_1, PATCH_1, PATCH_1, 1, DEPTH_1], dtype='int64'), stddev=0.1))
    b_1 = tf.Variable(tf.zeros([DEPTH_1]))
    
    W_2 = tf.Variable(tf.truncated_normal([PATCH_2, PATCH_2, PATCH_2, DEPTH_1, DEPTH_2], stddev=0.1))
    b_2 = tf.Variable(tf.constant(1.0, shape=[DEPTH_2]))
    
    W_3 = tf.Variable(tf.truncated_normal([PATCH_3, PATCH_3, PATCH_3, DEPTH_2, DEPTH_3], stddev=0.1))
    b_3 = tf.Variable(tf.constant(1.0, shape=[DEPTH_3]))

    W_4 = tf.Variable(tf.truncated_normal([PATCH_4, PATCH_4, PATCH_4, DEPTH_3, DEPTH_4], stddev=0.1))
    b_4 = tf.Variable(tf.constant(1.0, shape=[DEPTH_4]))

    W_5 = tf.Variable(tf.truncated_normal([PATCH_5, PATCH_5, PATCH_5, DEPTH_4, DEPTH_5], stddev=0.1))
    b_5 = tf.Variable(tf.constant(1.0, shape=[DEPTH_5]))

    W_6 = tf.Variable(tf.truncated_normal([CONNECTED_SIZE*CONNECTED_SIZE*DEPTH_5, CONNECTED_DEPTH]], stddev=0.1))
    b_6 = tf.Variable(tf.constant(1.0, shape=[CONNECTED_DEPTH]))

    W_6 = tf.Variable(tf.truncated_normal([CONNECTED_DEPTH, CONNECTED_DEPTH]], stddev=0.1))
    b_6 = tf.Variable(tf.constant(1.0, shape=[CONNECTED_DEPTH]))
    # -------------------------------------------------------------------------------
    # Output Layer
    W_o = tf.Variable(tf.truncated_normal([CONNECTED_DEPTH, oasis.num_labels]))
    b_o = tf.Variable(tf.zeros([oasis.num_labels]))

    def model(data, train=False):
        conv = tf.nn.conv3d(x, W_1, [1, 4, 4, 4, 1], padding='SAME')
        hidden = tf.nn.relu(conv + b_1)
        pool = tf.nn.max_pool3d(
            hidden,
            ksize=[1, 3, 3, 3, 1],
            stride=[1, 2, 2, 2, 1],
            padding='SAME'
        )

        conv = tf.nn.conv3d(pool, W_2, [1, 1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + b_2)

        conv = tf.nn.conv3d(conv, W_3, [1, 1, 1, 1, 1], padding='SAME')

        conv = tf.nn.conv3d(conv, W_4, [1, 1, 1, 1, 1], padding='SAME') 

        conv = tf.nn.conv3d(conv, W_5, [1, 1, 1, 1, 1], padding='SAME') 
        pool = tf.nn.max_pool3d(
            conv,
            ksize=[1, 3, 3, 3, 1],
            stride=[1, 2, 2, 2, 1],
            padding='SAME'
        )
        # -------------------------------------------------------------------------------
        # Reshape & Fully Connected
        # flat_shape = np.prod(pool.get_shape().as_list()[1:])
        # W_3 = tf.Variable(tf.truncated_normal([flat_shape, H_SIZE], stddev=0.1))
        # b_3 = tf.Variable(tf.constant(1.0, shape=[H_SIZE]))
        # reshape = tf.reshape(pool, [-1, flat_shape])
        reshape = tf.reshape(pool, [-1, CONNECTED_SIZE*CONNECTED_SIZE*DEPTH_5])
        fc = tf.nn.tanh(tf.matmul(reshape, W_6) + b_6)
        fc = tf.dropout(fc, KEEP_PROB)

        fc = tf.nn.tanh(tf.matmul(fc, W_7) + b_7)
        fc = tf.dropout(fc, KEEP_PROB)

        return tf.matmul(fc, W_o) + b_o
    # -------------------------------------------------------------------------------
    # Train
    logits = model(x, train=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    regularizers = (
        tf.nn.l2_loss(W_1) + tf.nn.l2_loss(b_1) +
        tf.nn.l2_loss(W_2) + tf.nn.l2_loss(b_2) +
        # tf.nn.l2_loss(W_3) + tf.nn.l2_loss(b_3) +
        tf.nn.l2_loss(W_o) + tf.nn.l2_loss(b_o)
    )
    loss = loss + (REG_BETA * regularizers)

    optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=MOMENTUM).minimize(loss)

    # -------------------------------------------------------------------------------
    # Predictions for training, validation, and test
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(x))
    # test_prediction = tf.nn.softmax(model(oasis.test.images.reshape(-1, *oasis.image_shape, 1)))

with tf.Session(graph=graph) as sess:
    print('initializing graph...')
    tf.global_variables_initializer().run()
    for epoch in range(NUM_EPOCHS):
        steps_per_epoch = math.ceil(oasis.train.length / BATCH_SIZE)
        for step in range(int(steps_per_epoch)):
            batch_xs, batch_ys = oasis.train.next_batch(BATCH_SIZE)
            _, l, predictions = sess.run(
                [optimizer, loss, train_prediction],
                feed_dict={x: batch_xs.reshape(-1, shape[0], shape[1], shape[2], 1), y_: batch_ys}
            )
            if step >= steps_per_epoch - 1 and epoch % 20 == 0:
                print_progress(
                    step, steps_per_epoch - 1, prefix='epoch %3d' % epoch, length=40,
                    fill='-', blank=' ', left_cap='', right_cap=''
                )
                print('  minibatch loss: %.4f' % l)
                print('  minibatch accuracy: %.2f%%' % accuracy(predictions, batch_ys))
                print('  validation accuracy: %.2f%%' % accuracy(sess.run(
                    test_prediction,
                    {x: oasis.validation.images.reshape(-1, shape[0], shape[1], shape[2], 1), y_: oasis.validation.labels}
                ), oasis.validation.labels))

    # -------------------------------------------------------------------------------
    # Display Validation Accuracy
    print('===========================================')
    print('Test accuracy: %.2f%%' % accuracy(sess.run(
        test_prediction,
        {x: oasis.test.images.reshape(-1, shape[0], shape[1], shape[2], 1), y_: oasis.test.labels}
    ), oasis.test.labels))

# batch_size: 4...
# epoch  0 ----------------------------------------  100%
#   minibatch loss: 0.0339
#   minibatch accuracy: 100.00%
#   validation accuracy: 61.90%
# epoch  0 ----------------------------------------  100%
#   minibatch loss: 0.0552
#   minibatch accuracy: 100.00%
#   validation accuracy: 57.14%
# ===========================================
# Test accuracy: 56.45%

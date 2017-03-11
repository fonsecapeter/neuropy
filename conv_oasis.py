import numpy as np
import math
import tensorflow as tf
from sample_data import oasis
from util.progress import print_progress

LEARNING_RATE = 0.000001
MOMENTUM = 0.3

BATCH_SIZE = 16
NUM_EPOCHS = 2

PATCH_SIZE = math.ceil(oasis.image_shape[0] * 0.1)
STRIDE_SIZE = math.ceil(oasis.image_shape[0] * 0.1)
DEPTH = 16

H_SIZE = 64
REG_BETA = 0.0005
KEEP_PROB = 0.5

oasis.load_data()
shape = oasis.image_shape

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, shapep[0], shape[1], shape[2],  1])
    y_ = tf.placeholder(tf.float32, shape=[None, oasis.num_labels])
    # -------------------------------------------------------------------------------
    # First Convolutional Layer
    W_1 = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1, DEPTH], stddev=0.1))
    b_1 = tf.Variable(tf.zeros([DEPTH]))
    # -------------------------------------------------------------------------------
    # Second Convolutional Layer
    W_2 = tf.Variable(tf.truncated_normal([1, 1, 1, DEPTH, DEPTH], stddev=0.1))
    b_2 = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
    # -------------------------------------------------------------------------------
    # Output Layer
    W_o = tf.Variable(tf.truncated_normal([H_SIZE, oasis.num_labels]))
    b_o = tf.Variable(tf.zeros([oasis.num_labels]))

    def model(data, train=False):
        conv = tf.nn.conv3d(x, W_1, [1, STRIDE_SIZE, STRIDE_SIZE, STRIDE_SIZE, 1], padding='VALID')
        hidden = tf.nn.relu(conv + b_1)
        pool = tf.nn.max_pool3d(
            hidden,
            [1, STRIDE_SIZE, STRIDE_SIZE, STRIDE_SIZE, 1],
            [1, STRIDE_SIZE, STRIDE_SIZE, STRIDE_SIZE, 1],
            padding='VALID'
        )

        conv = tf.nn.conv3d(pool, W_2, [1, 1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv * b_2)
        pool = tf.nn.max_pool3d(
            hidden,
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            padding='VALID'
        )
        # -------------------------------------------------------------------------------
        # Reshape
        flat_shape = np.prod(pool.get_shape().as_list()[1:])
        W_3 = tf.Variable(tf.truncated_normal([flat_shape, H_SIZE], stddev=0.1))
        b_3 = tf.Variable(tf.constant(1.0, shape=[H_SIZE]))
        reshape = tf.reshape(pool, [-1, flat_shape])
        hidden = tf.nn.relu(tf.matmul(reshape, W_3) + b_3)

        hidden = tf.nn.dropout(hidden, KEEP_PROB)
        return tf.matmul(hidden, W_o) + b_o
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
        for step in range(steps_per_epoch):
            batch_xs, batch_ys = oasis.train.next_batch(BATCH_SIZE)
            _, l, predictions = sess.run(
                [optimizer, loss, train_prediction],
                feed_dict={x: batch_xs.reshape(-1, shape[0], shape[1], shape[2], 1), y_: batch_ys}
            )
            print_progress(
                step, steps_per_epoch - 1, prefix='epoch %3d' % epoch, length=40,
                fill='-', blank=' ', left_cap='', right_cap=''
            )
            if step >= steps_per_epoch - 1:
                print('')
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

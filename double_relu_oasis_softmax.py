import numpy as np
import math
import tensorflow as tf
from sample_data import oasis

LEARNING_RATE = 0.00001
BATCH_SIZE = 50
NUM_EPOCHS = 800
KEEP_PROB = 0.5
H_SIZE = 64
REG_BETA = 0.0005

oasis.load_data()


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, oasis.flat_image_size])
    y_ = tf.placeholder(tf.float32, shape=[None, oasis.num_labels])

    # -------------------------------------------------------------------------------
    # First hidden RELU Layer
    W_h1 = tf.Variable(tf.truncated_normal([oasis.flat_image_size, H_SIZE]))
    b_h1 = tf.Variable(tf.zeros([H_SIZE]))
    # -------------------------------------------------------------------------------
    # Second hidden RELU Layer
    W_h2 = tf.Variable(tf.truncated_normal([H_SIZE, H_SIZE]))
    b_h2 = tf.Variable(tf.zeros([H_SIZE]))
    # -------------------------------------------------------------------------------
    # Output Layer
    W_o = tf.Variable(tf.truncated_normal([H_SIZE, oasis.num_labels]))
    b_o = tf.Variable(tf.zeros([oasis.num_labels]))

    def model(data, train=False):
        relu = tf.nn.relu(tf.matmul(data, W_h1) + b_h1)
        relu = tf.nn.relu(tf.matmul(relu, W_h2) + b_h2)
        relu = tf.nn.dropout(relu, KEEP_PROB)
        return tf.matmul(relu, W_o) + b_o

    # -------------------------------------------------------------------------------
    # Train
    logits = model(x, train=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    regularizers = (
        tf.nn.l2_loss(W_h1) + tf.nn.l2_loss(b_h1) +
        tf.nn.l2_loss(W_h2) + tf.nn.l2_loss(b_h2) +
        tf.nn.l2_loss(W_o) + tf.nn.l2_loss(b_o)
    )
    loss = loss + (REG_BETA * regularizers)

    optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.5).minimize(loss)

    # -------------------------------------------------------------------------------
    # Predictions for training, validation, and test
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(oasis.validation.images))
    test_prediction = tf.nn.softmax(model(oasis.test.images))

with tf.Session(graph=graph) as sess:
    print('initializing graph...')
    tf.global_variables_initializer().run()
    for epoch in range(NUM_EPOCHS):
        steps_per_epoch = math.ceil(oasis.train.length / BATCH_SIZE)
        for step in range(steps_per_epoch):
            batch_xs, batch_ys = oasis.train.next_batch(BATCH_SIZE)
            _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict={x: batch_xs, y_: batch_ys})
        if epoch % 10 == 0:
            oasis.validation.shuffle()
            print('epoch %d -----------------------------' % epoch)
            print('  minibatch loss: %.4f' % l)
            print('  minibatch accuracy: %.2f%%' % accuracy(predictions, batch_ys))
            print('  validation accuracy: %.2f%%' % accuracy(valid_prediction.eval(), oasis.validation.labels))

    # -------------------------------------------------------------------------------
    # Display Validation Accuracy
    print('===============================')
    print('Test accuracy: %.2f%%' % accuracy(test_prediction.eval(), oasis.test.labels))

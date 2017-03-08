import tensorflow as tf
from sample_data import oasis

BATCH_SIZE = 100
NUM_TRAINS = 2000
H_SIZE = 64
REG_BETA = 5e-4

oasis.load_data()

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, oasis.flat_image_size])
    y_ = tf.placeholder(tf.float32, shape=[None, oasis.num_labels])

    # -------------------------------------------------------------------------------
    # Single hidden RELU Layer
    W_h = tf.Variable(tf.truncated_normal([oasis.flat_image_size, H_SIZE]))
    b_h = tf.Variable(tf.zeros([H_SIZE]))
    relu_h = tf.nn.relu(tf.matmul(x, W_h) + b_h)

    # -------------------------------------------------------------------------------
    # Output Layer
    W_o = tf.Variable(tf.truncated_normal([H_SIZE, oasis.num_labels]))
    b_o = tf.Variable(tf.zeros([oasis.num_labels]))

    # -------------------------------------------------------------------------------
    # Train
    logits = tf.matmul(relu_h, W_o) + b_o
    y = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    regularizers = (tf.nn.l2_loss(W_h) + tf.nn.l2_loss(b_h) + tf.nn.l2_loss(W_o) + tf.nn.l2_loss(b_o))
    loss = loss + (REG_BETA * regularizers)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph=graph) as sess:
    print('initializing graph...')
    tf.global_variables_initializer().run()
    for i in range(NUM_TRAINS):
        batch_xs, batch_ys = oasis.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_xs, y_: batch_ys
            })
            print('step %d, training accuracy: %g' % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # -------------------------------------------------------------------------------
    # Display Validation Accuracy
    print('test accuracy:', sess.run(accuracy, feed_dict={x: oasis.test.images, y_: oasis.test.labels}))

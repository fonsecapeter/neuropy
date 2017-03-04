import tensorflow as tf
from sample_data import oasis

BATCH_SIZE = 100
NUM_TRAINS = 2000

oasis.load_data()

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, oasis.flat_image_size])
    y_ = tf.placeholder(tf.float32, [None, oasis.num_labels])

    W = tf.Variable(tf.zeros([oasis.flat_image_size, oasis.num_labels]))
    b = tf.Variable(tf.zeros([oasis.num_labels]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # -------------------------------------------------------------------------------
    # train
    y_ = tf.placeholder(tf.float32, [None, oasis.num_labels])

    # implement cross entropy function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # ask tf to minimize this loss
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # launch the model in an interactive session
    sess = tf.InteractiveSession()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph=graph) as sess:
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
    # display the accuracy
    print('test accuracy:', sess.run(accuracy, feed_dict={x: oasis.test.images, y_: oasis.test.labels}))

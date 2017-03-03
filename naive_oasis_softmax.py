import tensorflow as tf
from sample_data import oasis

BATCH_SIZE = 100

# mnist.train.images are the mris flattened from
# 91 * 109 * 91 = 902629
oasis.load_data()


x = tf.placeholder(tf.float32, [None, 902629])

W = tf.Variable(tf.zeros([902629, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# -------------------------------------------------------------------------------
# train
y_ = tf.placeholder(tf.float32, [None, 2])

# implement cross entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# ask tf to minimize this loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# launch the model in an interactive session
sess = tf.InteractiveSession()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.initialize_all_variables().run()
# train 2000 times
for i in range(2000):
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

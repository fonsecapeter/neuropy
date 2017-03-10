import tensorflow as tf
from sample_data import load_ucla

# ucla.train.images are the mris flattened from
ucla = load_ucla()

x = tf.placeholder(tf.float32, [None, 11141120])

W = tf.Variable(tf.zeros([11141120, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# -------------------------------------------------------------------------------
# train
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.initialize_all_variables().run()

for i in range(4):
    batch_xs, batch_ys = ucla.train.next_batch(68)
    train_accuracy = accuracy.eval(feed_dict={
        x: batch_xs, y_: batch_ys
    })
    print('step %d, training accuracy: %g' % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# -------------------------------------------------------------------------------
# display the accuracy
print('test accuracy:', sess.run(accuracy, feed_dict={x: ucla.test.images, y_: ucla.test.labels}))

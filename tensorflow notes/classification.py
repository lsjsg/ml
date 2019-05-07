import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.random_normal([1, out_size]))
    wxb = tf.add(tf.matmul(inputs, Weight), bias)
    if activation_function is None:
        return wxb
    else:
        return activation_function(wxb)


def compute_accuracy(test_x, test_y):
    global prediction
    pred_y = sess.run(prediction, feed_dict={xs: test_x})
    correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(test_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: test_x, ys: test_y})
    return result


# create placeholders do not define the sample size but we define how many data required for every sample
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys *
                                              tf.log(prediction), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)+(1-ys)*(tf.log(1-prediction))),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    # socastic gradient discent, not training on all data but only some to same training time
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

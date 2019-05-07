# 可视化数据流向和网络结构
import numpy as np
import tensorflow as tf
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1],name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1],name="y_input")
with tf.name_scope("layer1"):
    with tf.name_scope("weights1"):
        Weight1 = tf.Variable(tf.random_normal([1, 10]),name="Weight1")
    with tf.name_scope("bias1"):
        Bias1 = tf.Variable(tf.random_normal([1, 10]),name="bias1")
    with tf.name_scope("Z1"):
        Z1 = tf.add(tf.matmul(xs, Weight1),Bias1)
    A1 = tf.nn.relu(Z1)
with tf.name_scope("layer2"):
    with tf.name_scope("weights2"):
        Weight2 = tf.Variable(tf.random_normal([10, 1]))
    with tf.name_scope("bias2"):
        Bias2 = tf.Variable(tf.random_normal([1, 1]))
    with tf.name_scope("A2"):
        A2 = tf.matmul(A1, Weight2) + Bias2
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - A2), reduction_indices=[1]),name="loss")
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("logs/", sess.graph)
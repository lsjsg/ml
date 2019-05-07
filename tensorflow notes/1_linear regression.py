import tensorflow as tf
import numpy as np

x = np.random.rand(100).astype(np.float32)
y = x * 0.1 + 0.3

# create tensorflow structire
Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias = tf.Variable(tf.zeros([1]))
y_pred = Weight*x + bias
loss = tf.reduce_mean(tf.square(y-y_pred))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)  # initialize

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weight),sess.run(bias),sess.run(loss))

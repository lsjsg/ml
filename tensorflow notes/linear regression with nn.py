import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

Weight1 = tf.Variable(tf.random_normal([1, 10]))
Bias1 = tf.Variable(tf.random_normal([1, 10]))
Z1 = tf.matmul(xs, Weight1) + Bias1
A1 = tf.nn.relu(Z1)

Weight2 = tf.Variable(tf.random_normal([10, 1]))
Bias2 = tf.Variable(tf.random_normal([1, 1]))
A2 = tf.matmul(A1, Weight2) + Bias2

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - A2), reduction_indices=[1]))
# reduce_indices=1 对每行进行求和, reduce_indices=0 对每列进行求和

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

fig = plt.figure()  # 创建图片框
ax = fig.add_subplot(1, 1, 1)  # 为了连续性画图, 添加ax轴, 并编号1,1,1
ax.scatter(x_data, y_data)
plt.ylim(-1, 1)
plt.ion()  # 显示图像过后不暂停
plt.show()

for i in range(10000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(A2, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, "r-", lw=5)
        plt.pause(0.1)
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

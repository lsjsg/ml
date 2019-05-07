# placeholder, tensorflow中的占位符, 暂时储存变量, 接受外界传入的数据
# 格式为 sess.run(***,feed_dict={input:**})

import tensorflow as tf
# 定义dtype为tf.float32, 后面接着一个shape
input1 = tf.placeholder(tf.float32) 
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.2]}))
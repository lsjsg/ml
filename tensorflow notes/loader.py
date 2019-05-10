import tensorflow as tf

# remember to define the same dtype and shape when restore
W = tf.Variable(tf.zeros([2,3]),dtype=tf.float32,name="weights")
b = tf.Variable(tf.zeros([1,3]),dtype=tf.float32,name="bias")

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"my_net/save_net.ckpt")
    print("weights:",sess.run(W))
    print("bias:",sess.run(b))
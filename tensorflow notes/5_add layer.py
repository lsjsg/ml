import tensorflow as tf

def add_layer(input,insize,outsize,activation_function=None):
    Weight = tf.Variable(tf.random_normal([insize,outsize]))
    bias = tf.Variable(tf.zeros([1,outsize]) + 0.1)
    y = tf.matmul(input,Weight) + bias
    if activation_function is None:
        outputs = y
    else:
        outputs = activation_function(y)
    return outputs




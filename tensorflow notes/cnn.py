import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={x_s:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={x_s:v_xs,ys:v_ys,keep_prob:1})
    return result

x_s = tf.placeholder(tf.float32,[None,784])/255.0
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
xs = tf.reshape(x_s,[-1,28,28,1])
# conv layer 1
# patch 5*5, insize 1(image 厚度), outsize 32(转换为32层)
W1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([32],stddev=0.1))
# output shape:28x28x32
Z1 = tf.nn.conv2d(xs,W1,strides=[1,1,1,1],padding="SAME") + b1
A1 = tf.nn.relu(Z1)
# output shape:14x14x32
P1 = tf.nn.max_pool(A1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# conv layer 2
# patch 5*5, insize 1(image 厚度), outsize 32(转换为32层)
W2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([64],stddev=0.1))
# output shape:14x14x64
Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME") + b2
A2 = tf.nn.relu(Z2)
# output shape:7x7x64
P2 = tf.nn.max_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# fully connected layer 1
FW1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
Fb1 = tf.Variable(tf.truncated_normal([1024],stddev=0.1))
Flatten = tf.reshape(P2,[-1,7*7*64])
FC1 = tf.nn.relu(tf.matmul(Flatten,FW1) + Fb1)
F1 = tf.nn.dropout(FC1,keep_prob)

# fully connected layer 2
FW2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
Fb2 = tf.Variable(tf.truncated_normal([10],stddev=0.1))
prediction = tf.nn.softmax(tf.matmul(F1,FW2) + Fb2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1001):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x_s:batch_xs,ys:batch_ys,keep_prob:0.2})
    if i%20 == 0:
        # print(sess.run(cross_entropy,feed_dict={xs:batch_xs,ys:batch_ys}))
        print(compute_accuracy(mnist.test.images,mnist.test.labels))


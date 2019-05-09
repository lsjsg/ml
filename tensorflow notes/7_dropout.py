import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#load data
digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
def add_layer(input,insize,outsize,layer_name,activation_function=None):
    keep_prob = 0.5
    Weight = tf.Variable(tf.random_normal([insize,outsize]))
    bias = tf.Variable(tf.random_normal([1,outsize]))
    y = tf.matmul(input,Weight) + bias
    y = tf.nn.dropout(y,keep_prob)
    if activation_function is None:
        outputs = y
    else:
        outputs = activation_function(y)
    tf.contrib.deprecated.histogram_summary(layer_name+"/outputs",outputs)
    return outputs

xs = tf.placeholder(tf.float32,[None,64])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32,[1])

#add layer
l1 = add_layer(xs,64,50,"l1",activation_function=tf.nn.tanh)
prediction = add_layer(l1,50,10,"output",activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
tf.contrib.deprecated.scalar_summary("loss",cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
merged = tf.contrib.deprecated.merge_all_summaries()
# summery writer
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
test_writer = tf.summary.FileWriter("logs/test",sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(500):
    sess.run(train_step,feed_dict={xs:x_train,ys:y_train})
    if i % 50==0:
        # print(sess.run(cross_entropy, feed_dict={xs:x_train,ys:y_train}))
        # print(sess.run(cross_entropy, feed_dict={xs:x_test,ys:y_test}))
        train_result = sess.run(merged,feed_dict={xs:x_train,ys:y_train})
        test_result = sess.run(merged,feed_dict={xs:x_test,ys:y_test})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)

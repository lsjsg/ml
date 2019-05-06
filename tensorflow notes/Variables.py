import tensorflow as tf

state = tf.Variable(0,name="counter")
# print(state.name)
one = tf.constant(1)
new_value = tf.add(state,one) # add two values
update = tf.assign(state,new_value) # assign value to a variable
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for _ in range(3):
    sess.run(update)
    print(sess.run(state))
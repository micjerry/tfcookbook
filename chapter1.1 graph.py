import numpy as np
import tensorflow as tf

#just disable warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


sess = tf.Session()

x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)

m_const = tf.constant(3.)

my_product = tf.multiply(x_data, m_const)

for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))

#make the graph
writer = tf.summary.FileWriter('./my_graph', sess.graph)
writer.close()

sess.close()
#
#    练习2：用tensorflow中占位符、变量、常量做一个点乘与乘法计算，用tensorboard查看计算过程
#
import tensorflow as tf
import numpy as np

t = np.arange(6).reshape([3, 2])
a = tf.Variable(initial_value=t, name='a', dtype=tf.float32, shape=[3, 2])
b = tf.placeholder(name='b', shape=[2, 2], dtype=tf.float32)
c = tf.constant(value=10, name='c', dtype=tf.float32)

d = tf.multiply(a, c, 'multiply')
e = tf.matmul(d, b, name='matmul')

with tf.Session() as sess:
    tf.summary.FileWriter('../data', sess.graph)
    sess.run(tf.global_variables_initializer())
    f = np.arange(4).reshape([2, 2])
    print(sess.run(e, feed_dict={b: f}))
    # 查看图形tensorboard --logdir="D:\HDDevPY\HDDevPY\data"

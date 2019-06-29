#
#    练习2：使用上一个练习保存的结果
#
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("../data/input_data", one_hot=True)
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# 构造神经网络
# 输入层784到第一层500，[None,784] * [784,500]  》[None, 500]
W1 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal([784, 500]))
B1 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal([500]))
L1 = tf.add(tf.matmul(X, W1), B1)
L1 = tf.tanh(L1)
# 第一层500到第二层200，[None,500] * [500,200] > [None,200]
W2 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal([500, 200]))
B2 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal([200]))
L2 = tf.add(tf.matmul(L1, W2), B2)
L2 = tf.tanh(L2)
# 第二层200到输出层10，[None,200] * [200,10] > [None,10]
W3 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal([200, 10]))
B3 = tf.Variable(dtype=tf.float32, initial_value=tf.random.normal([10]))
y_pre = tf.add(tf.matmul(L2, W3), B3)

# 交叉熵得到差距
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pre))

a, b = tf.argmax(y_pre, axis=1), tf.argmax(y, axis=1)
result = tf.equal(a, b)
result = tf.cast(result, tf.float32)
result = tf.reduce_mean(result)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "../data/tmp/mnist")
    print(mnist.test.images.shape, type(mnist.test))
    for i in range(100):
        X_test, y_test = mnist.test.next_batch(55)
        # sess.run(y_pre, feed_dict={X: X_test, y: y_test})
        if i % 10 == 0:
           print(f'第{i}次，误差{sess.run(result, feed_dict={X: X_test, y: y_test})}')

#
#    练习3：sin函数生成测试数据,用tensorflow实现多项式的线性回归  y_predict = w3*X3+ w2*X2 + w1*X + w4
#    用折线图和散点图显示拟合出来的曲线与实际数据，在tensorboard中增加误差的情况
#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X = np.linspace(-3, 3, 100)
y = np.sin(X) + np.random.uniform(-1, 1, 100)

w1 = tf.Variable(initial_value=0, dtype=tf.float64, name='w1')
w2 = tf.Variable(initial_value=0, dtype=tf.float64, name='w2')
w3 = tf.Variable(initial_value=0, dtype=tf.float64, name='w3')
w4 = tf.Variable(initial_value=0, dtype=tf.float64, name='w4')
y_pre = tf.add(w4, tf.multiply(w1, X))
y_pre = tf.add(y_pre, tf.multiply(w2, tf.pow(X, 2)))
y_pre = tf.add(y_pre, tf.multiply(w3, tf.pow(X, 3)))
loss = tf.reduce_mean(tf.square(y - y_pre), name='reduce_mean')
tf.summary.scalar('abc', loss)
merge = tf.summary.merge_all()

op = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fw = tf.summary.FileWriter('../data', sess.graph)
    for i in np.arange(1000):
        sess.run(op)
        fw.add_summary(merge.eval(), i)
        if i % 10 == 0:
            print(f'均差{loss.eval()}, {w4.eval()}')
    plt.scatter(X, y)
    plt.plot(X, y_pre.eval())
    plt.show()

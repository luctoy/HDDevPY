# HDDevPY
Day02:
练习1：创建矩阵的三个写法
练习2：取矩阵中的数据6个写法
练习3：从图片中读取数据，处理写回图片
练习4：pandas 基础信息、打印前n行、排序、groupby、统计个数
练习5：可视化（散点(总金额、tip)、柱状图（female与male均值/2种方式）、饼图（female与male）统计个数）
练习6：加载house，用线性回归预测后，与实际样本进行比较
Day03:
练习1：用科学库实现对波士顿房价特征的PCA（主成分分析），进行多项式线性拟合，比较方差与正确率，
用折线显示预测目标与实际数据；随后将模型保存，待后续直接加载使用
练习2：用tensorflow中占位符、变量、常量做一个点乘与乘法计算，用tensorboard查看计算过程
练习3：sin函数生成测试数据,用tensorflow实现多项式的线性回归  y_predict = w3*X3+ w2*X2 + w1*X + w4
用折线图和散点图显示拟合出来的曲线与实际数据，在tensorboard中增加误差的情况
Day04:
练习1：从图片集合中获取图片，以两层隐藏层的DNN神经网络（深度学习）进行学习，然后分析test集合中的图片，
保存训练结果，下一次直接使用
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/input_data",one_hot=True)
练习2：使用上一个练习保存的结果
练习3：One-Hot编码: 可以把文字映射到数字的编码方式
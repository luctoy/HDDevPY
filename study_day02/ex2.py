#
#    练习2：取矩阵中的数据6个写法
#
import numpy as np

t1 = np.arange(12).reshape([3, 4])
print(t1)
print('_' * 12, '#1', '_' * 12)
print(t1[0, 1])
print('_' * 12, '#2', '_' * 12)
print(t1[0:2, 1:2])
print('_' * 12, '#3', '_' * 12)
print(t1[0:2, 1:])
print('_' * 12, '#4', '_' * 12)
print(t1[0:2, (1, 2)])
print('_' * 12, '#5', '_' * 12)
print(t1[(0, 2), (1, 3)])
print('_' * 12, '#5区别', '_' * 12)
t2 = t1[(0, 2), :]
print(t2[:, (1, 3)])

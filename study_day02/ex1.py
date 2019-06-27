#
#   练习1：创建矩阵的三个写法
#
import numpy as np

# 写法1
t1 = np.arange(12).reshape([3, 4])
print(t1)

# 写法2
l1 = [[1, 2, 3], [4, 4, 5], [6, 7, 8]]
t1 = np.array(l1)
print(t1)

# 写法3
t1 = np.random.randint(low=1, high=100, size=12).reshape([3, 4])
print(t1)

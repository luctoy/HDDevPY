#
#    练习6：加载house，用线性回归预测后，与实际样本进行比较
#
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd

# CSV读入样本
df = pd.read_csv('../data/house.csv')
# 数据清理
# 特征工程
y = df['price']
X = df.drop(['price', 'row_id'], axis=1)
# 划分训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# 训练
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pre = lr.predict(X_test)
# 验证
print(mean_squared_error(y_test,y_pre))

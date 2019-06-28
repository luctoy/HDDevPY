#
#    练习1：用科学库实现对波士顿房价特征的PCA（主成分分析），进行多项式线性拟合，比较方差与正确率，
#    用折线显示预测目标与实际数据；随后将模型保存，待后续直接加载使用
#
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

lb = load_boston()
X = lb['data']
y = lb['target']
X = pd.DataFrame(data=X, columns=lb['feature_names'])
plt.rcParams['font.sans-serif'] = ['SimHei']

# def show_scatter(Xname):
#     plt.scatter(df[Xname], y)
#     plt.xlabel(Xname)
#     plt.ylabel('price')
#     plt.title(Xname+'_price')
#     plt.show()
#
#
# show_scatter('CRIM')
# show_scatter('RM')
# show_scatter('INDUS')
# show_scatter('ZN')

X = X.drop(['NOX', 'ZN'], axis=1)
pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X = pf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
try:
    lr = joblib.load('../data/lr.pk1')
    print('成功加载')
except:
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, '../data/lr.pk1')
y_pre = lr.predict(X_test)
# 误差
print(mean_squared_error(y_pre, y_test))
print(lr.score(X_test, y_test))

x = np.arange(len(y_test))
plt.plot(x, y_pre, color='#ff0000', marker='.', label="y_predict")
plt.plot(x, y_test, color='#00ff00', marker='.', label="y_test")
plt.legend()
plt.show()

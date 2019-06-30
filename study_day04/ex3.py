#
#    练习3：One-Hot编码: 可以把文字映射到数字的编码方式
#
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
# 0 1
ohe.fit([['男', '中国', '篮球'],
         ['女', '美国', '篮球'],
         ['男', '日本', '羽毛球'],
         ['女', '中国', '乒乓球']])
array = ohe.transform([['女','中国','羽毛球']])
# [[{0. 1.} { 1. 0. 0. }  1 . 0. 0. 0.]]
print(array)
print(ohe.inverse_transform(array))

#
#    练习4：pandas 基础信息、打印前n行、排序、groupby、统计个数
#
import pandas as pd

df = pd.read_csv('../data/groupby.csv')
df.info()
print(df.head(3))
print(df.sort_values('Count', ascending=True))
gp = df.groupby(by='Brand')
for idx, val in gp:
    print(idx, val)
print(df.groupby('Brand')['Count'].count())
print(df.groupby('Brand')['Count'].sum())

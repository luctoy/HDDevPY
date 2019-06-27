#
#    练习5：可视化（散点(总金额、tip)、柱状图（female与male均值/2种方式）、饼图（female与male）统计个数）
#
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../data/tips.csv')
total_bill = df['total_bill']
tip = df['tip']
plt.scatter(total_bill, tip)
plt.xlabel = 'x'
plt.ylabel = 'y'
plt.title = 'x-y'
plt.show()

male = df[df['sex'] == 'Male']['tip'].mean()
female = df[df['sex'] == 'Female']['tip'].mean()
plt.bar(['male', 'female'], [male, female])
plt.show()

ss = df.groupby(by='sex')['tip'].mean()
plt.bar(ss.index, ss.values)
plt.show()

ss = df['sex'].value_counts()
plt.pie(x=ss.values, labels=ss.index, autopct='%.2f%%')
plt.show()

import pandas as pd
import numpy as np

ss = pd.Series(data=[1, 2, 3], index=list('abc'), dtype=np.int, name='??')

print(ss, type(ss))
print(ss.index, ss.values)

df = pd.DataFrame(data=np.arange(12).reshape(3,4), index=list('abc'), columns=list('abcd'), dtype=np.float16)
print(df)
print('_'*100)
print(type(df), type(df.values))


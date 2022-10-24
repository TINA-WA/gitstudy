import numpy as np
import pandas as pd
import math

path=r'22E题热力图-相关性矩阵.xlsx'#22E题热力图-相关性矩阵.xlsx 内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）.xlsx
data=pd.read_excel(path,names=None,sort=False,index=None)#usecols=cols,,header=None,usecols=['D:T']
print(data)
u1=data.values
print(u1)

np.corrcoef(u1.T) # 计算矩阵所有列的相关系数
np.around(np.corrcoef(u1), decimals=3) # 这里是将矩阵结果保留3位小数

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['KaiTi']   # 指定默认字体 显示中文
figure, ax = plt.subplots(figsize=(15, 15))
p1=sns.heatmap(data.corr(), square=True, annot=True,)
s1=p1.get_figure()
s1.savefig('HeapMap_+降水.jpg',dpi=256)

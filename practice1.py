# 任务目标：完成一元线性回归模型，通过已有的数据进行预测x=3.5时的y值，并且评估模型表现
#
# 1、数据加载
# 2、建立模型
# 3、模型评估

# step1

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

import numpy as np

data = pd.read_excel('./AIlearning/AI-data.xlsx')

x = data.loc[:, 'x']
y = data.loc[:, 'y']

# plt.figure(figsize=(5, 5))
# plt.scatter(x, y)
# plt.show()

# step2
lr_model = LinearRegression()
x = np.array(x)
y = np.array(y)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
# print(y.shape + x.shape)
lr_model.fit(x, y)
y_predict = lr_model.predict(x)
y_3 = lr_model.predict([[3.5]])
print(y_3)

# step3
a = lr_model.coef_
b = lr_model.intercept_
print(f'y={np.array(a)}x+{int(b)}')
MSE = mean_squared_error(y, y_predict)
R2 = r2_score(y, y_predict)

print(MSE, R2)

plt.figure(figsize=(5, 5))
plt.plot(y, y_predict)
plt.show()

# 基于usa_housing_price.csv数据，建立线性回归模型，预测合理房价：
# 1、以面积为输入变量，建立单因子模型，评估模型表现，可视化线性回归预测结果
# 2、以income、house age、numbers of rooms、population、area为输入变量，建立多因子模型，评估模型表现
# 3、预测Income=65000,House Age=5,Number of Rooms=5 Population:=30000,size=200的合理房价
# 步骤
# 1、数据加载
# 2、建立模型
# 3、模型评估

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data_build = pd.read_csv('./AIlearning/usa_housing_price.csv')
mult = data_build.drop(['Price'], axis=1)
size = data_build.loc[:, 'size']
price = data_build.loc[:, 'Price']

# region 画图显示
# fig = plt.figure(figsize=(6, 5))
# fig1 = plt.subplot(231)
# plt.scatter(income, price)
# plt.title('price vs income')
# fig2 = plt.subplot(232)
# plt.scatter(age, price)
# plt.title('price vs house_age')
# fig3 = plt.subplot(233)
# plt.scatter(rooms, price)
# plt.title('price vs avg_rooms')
# fig4 = plt.subplot(234)
# plt.scatter(population, price)
# plt.title('price vs avg_population')
# fig5 = plt.subplot(235)
# plt.scatter(size, price)
# plt.title('price vs size')
# plt.show()
# endregion

# region 单因子模型
# step2
lr_1 = LinearRegression()
size = np.array(size)
price = np.array(price)
size = size.reshape(-1, 1)
price = price.reshape(-1, 1)
lr_1.fit(size, price)
price_predict = lr_1.predict(size)

# step3
MSE_1 = mean_squared_error(price, price_predict)
R2_1 = r2_score(price, price_predict)
print(f'MSE={MSE_1},R2={R2_1}')

plt.figure(figsize=(5, 5))
plt.scatter(size, price)
plt.plot(size, price_predict, color='r')
plt.show()
# endregion

# region 多因子模型
# step2
lr_2 = LinearRegression()
lr_2.fit(mult, price)
price_predict_mult = lr_2.predict(mult)

# step3
MSE_2 = mean_squared_error(price, price_predict_mult)
R2_2 = r2_score(price, price_predict_mult)
print(f'MSE_mult={MSE_2},R2_mult={R2_2}')
fig7 = plt.figure(figsize=(5, 5))
plt.scatter(price_predict_mult, price)
plt.show()
# endregion

x_test = [[65000, 5, 5, 30000, 200]]
x_test = pd.DataFrame(x_test)
print(x_test)
price_test = lr_2.predict(x_test)
print(price_test)

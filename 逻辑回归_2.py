# 1、基于chip test.csv数据，建立逻辑回归模型（三阶边界），评估模型表现；
# 2、以函数的方式求解边界曲线
# 3、描绘出完整的决策边界伷线

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#step1 load file
data = pd.read_csv('./AIlearning/chip_test.csv')
X1 = data.loc[:, 'test1']
X2 = data.loc[:, 'test2']
Y = data.loc[:, 'pass']
mask = data.loc[:, 'pass'] == 1

# region 图形化显示
# visualize
fig1 = plt.figure()
passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask], color='r')
failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask], color='y', marker='^')
plt.title('test1 and test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()
# endregion

# step2 build lr second order
X1_2 = X1 ** 2
X2_2 = X2 ** 2
X1_X2 = X1 * X2
X_mult = pd.concat([X1, X2, X1_2, X2_2, X1_X2], axis=1, keys=['X1', 'X2', 'X1^2', 'X2^2', 'X1*X2'])
LR1 = LogisticRegression()
LR1.fit(X_mult, Y)
Y_predict = LR1.predict(X_mult)

# step3 Evaluation model
accuracy1 = accuracy_score(Y, Y_predict)
print(f'二阶模型的准确率为：{accuracy1}')
theta0 = LR1.intercept_
theta1, theta2, theta3, theta4, theta5 = LR1.coef_[0][0], LR1.coef_[0][1], LR1.coef_[0][2], LR1.coef_[0][3], \
LR1.coef_[0][4]
print(theta1, theta2, theta3, theta4, theta5)

# step4 visualize handle
X1_new = X1.sort_values()
a = theta4
b = theta5 * X1_new + theta2
c = theta0 + theta1 * X1_new + theta3 * X1_new * X1_new
X2_new1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
X2_new2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)

# region 图形化显示
# visualize
fig2 = plt.figure()
passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask], color='r')
failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask], color='y', marker='^')
plt.plot(X1_new, X2_new1)
plt.plot(X1_new, X2_new2)
plt.title('test1 and test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()
# endregion

#step5 Complete linear construction
X1_range = [-0.9 + x/10000 for x in range(0, 19000)]
X1_range = pd.DataFrame(X1_range)
a = theta4
b = theta5 * X1_range + theta2
c = theta0 + theta1 * X1_range + theta3 * X1_range * X1_range
X2_range1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
X2_range2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)

# region 图形化显示
# visualize
fig2 = plt.figure()
passed = plt.scatter(data.loc[:, 'test1'][mask], data.loc[:, 'test2'][mask], color='r')
failed = plt.scatter(data.loc[:, 'test1'][~mask], data.loc[:, 'test2'][~mask], color='y', marker='^')
plt.plot(X1_range, X2_range1, color='b')
plt.plot(X1_range, X2_range2, color='b')
plt.title('test1 and test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()
# endregion

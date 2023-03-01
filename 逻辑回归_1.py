# 基于examdata.csv数据，建立逻辑回归模型预测Exam1-75,Exam2-60时，
# 该同学在Exam3是passed or failed;建立二阶边界，提高模型准确度

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv('./AIlearning/examdata.csv')

mask = data.loc[:, 'Pass'] == 1

# region 图形化显示
# visualize
fig1 = plt.figure()
passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask], color='r')
failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask], color='y', marker='^')
plt.title('Exam1 and Exam2')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()
# endregion

X = data.drop('Pass', axis=1)
Y = data.loc[:, 'Pass']

LR = LogisticRegression()
LR.fit(X, Y)
Y_predict = LR.predict(X)

# step3
accuracy = accuracy_score(Y, Y_predict)
print(f'准确率为:{accuracy}')

theta0 = LR.intercept_
theta1, theta2 = LR.coef_[0][0], LR.coef_[0][1]
X1 = X.loc[:, 'Exam1']
X2_new = -(theta0 + theta1 * X1) / theta2

# region 图形化显示
# visualize
fig2 = plt.figure()
passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask], color='r')
failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask], color='y', marker='^')
plt.plot(X1,X2_new)
plt.title('Exam1 and Exam2')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()
# endregion

X2 = X.loc[:, 'Exam2']
X1_2 = X1 * X1
X2_2 = X2 * X2
X1_X2 = X1 * X2

X_new = pd.concat([X1, X2, X1_2, X2_2, X1_X2], axis=1, keys=['X1', 'X2', 'X1_2', 'X2_2', 'X1_X2'])
print(X_new.head())

LR2 = LogisticRegression()
LR2.fit(X_new, Y)
Y_predict2 = LR2.predict(X_new)

accuracy2 = accuracy_score(Y_predict2, Y)
print(f'\n准确率为:{accuracy2}')

X1_new = X1.sort_values()
theta3 = LR2.intercept_
theta4, theta5, theta6, theta7, theta8 = LR2.coef_[0][0], LR2.coef_[0][1], LR2.coef_[0][2], LR2.coef_[0][3], \
LR2.coef_[0][4]

a = theta7
b = theta8 * X1_new + theta5
c = theta3 + theta4 * X1_new + theta6 * X1_new * X1_new
X2_new2 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

# region 图形化显示
# visualize
fig3 = plt.figure()
passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask], color='r')
failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask], color='y', marker='^')
plt.plot(X1_new, X2_new2)
plt.title('Exam1 and Exam2')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()
# endregion
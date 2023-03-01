# 决策树实战task:
# 1基于iris data.csv数据，建立决策树模型，评估模型表现
# 2、可视化决策树结构
# 3、修改min_samples._leaf参数，对比模型结果

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn import tree

iris = load_iris()
iris_data = iris['data']
iris_target = iris['target']
iris_data = np.array(iris_data)
iris_target = np.array(iris_target).reshape(-1, 1)
iris_name = []
for i in iris_target:
    if i == 0:
        iris_name.append('setosa')
    elif i == 2:
        iris_name.append('versicolor')
    else:
        iris_name.append('virgincia')
iris_name = np.array(iris_name).reshape(-1, 1)
iris = np.concatenate((iris_data, iris_target, iris_name), axis=1)

#step2 build model
dc_tree=tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5)
dc_tree.fit(iris_data, iris_target)

#evaluate the model
Y_predict = dc_tree.predict(iris_data)
accuracy_tree = accuracy_score(iris_target,Y_predict)
print(accuracy_tree)

# visualize tree
fig0 = plt.figure(figsize=(8, 8))
tree.plot_tree(dc_tree, label='iris tree', filled='True', feature_names=['sl', 'sw', 'pl', 'pw'],
               class_names=['setosa', 'versicolor', 'virgincia'])
plt.show()


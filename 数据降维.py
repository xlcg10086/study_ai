# PCA实战task:
# 1、基于iris_data.csv数据，建立KNN模型实现数据分类(n_neighbors=3)
# 2、对数据进行标准化处理，选取一个维度可视化处理后的效果
# 3、进行与原数据等维度PA,查看各主成分的方差比例
# 4、保留合适的主成分，可视化降维后的数据
# 5、基于降维后数据建立KNN模型，与原数据表现进行对比

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# establish the model and calculate the accuracy
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(iris_data, iris_target)
iris_target_predict = KNN.predict(iris_data)
accuracy_KNN1 = accuracy_score(iris_target_predict, iris_target)
print(accuracy_KNN1)
iris_data_norm = StandardScaler().fit_transform(iris_data)
# print(iris_data_norm)

# visualize
fig0 = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(iris_data[:, 0], bins=100)
plt.title('iris_data')
plt.subplot(122)
plt.hist(iris_data_norm[:, 0], bins=100)
plt.title('iris_data_norm')
plt.show()

# calcurate mean and sigma
iris_data_mean = iris_data.mean()
iris_data_std = iris_data.std()
iris_data_norm_mean = iris_data_norm.mean()
iris_data_norm_std = iris_data_norm.std()
print(iris_data_mean, iris_data_std, iris_data_norm_mean, iris_data_norm_std)

# PCA数据降维,第一步先不降维，输入标准正态数据，进行计算权重
pca = PCA(n_components=4)
iris_data_pca = pca.fit_transform(iris_data_norm)
var_ratio = pca.explained_variance_ratio_
print(var_ratio)

fig1 = plt.figure(figsize=(10, 5))
plt.bar([1, 2, 3, 4], var_ratio)
plt.xticks([1, 2, 3, 4], ['PC1', 'PC2', 'PC3', 'PC4'])
plt.ylabel('variance ratio of each pc')
plt.show()

pca = PCA(n_components=2)
iris_data_pca = pca.fit_transform(iris_data_norm)

fig2 = plt.figure(figsize=(8, 8))
Y1 = plt.scatter(iris_data_pca[:, 0][iris_target[:, 0] == 0], iris_data_pca[:, 1][iris_target[:, 0] == 0], marker='x')
Y2 = plt.scatter(iris_data_pca[:, 0][iris_target[:, 0] == 1], iris_data_pca[:, 1][iris_target[:, 0] == 1], marker='^')
Y3 = plt.scatter(iris_data_pca[:, 0][iris_target[:, 0] == 2], iris_data_pca[:, 1][iris_target[:, 0] == 2], marker='o')
plt.title('three flowers')
plt.legend((Y1, Y2, Y3), ('setosa', 'versicolor', 'virgincia'))
plt.show()

#compare two model
KNN2 = KNeighborsClassifier(n_neighbors=3)
KNN2.fit(iris_data_pca, iris_target)
iris_target_predict2 = KNN2.predict(iris_data_pca)
accuracy_KNN2 = accuracy_score(iris_target_predict2, iris_target)
print(accuracy_KNN2)
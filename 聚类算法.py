# 1、采用Kmeans:算法实现2D数据自动聚类，预测V1=80,V2=60数据类别；
# 2、计算预测准确率，完成结果矫正
# 3、采用KNN、Meanshift算法，重复步骤1-2

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.neighbors import KNeighborsClassifier

# load file
data = pd.read_csv('./AIlearning/data.csv')
V1 = data.loc[:, 'V1']
V2 = data.loc[:, 'V2']
X = pd.concat((V1, V2), axis=1)
label = data.loc[:, 'labels']
print(pd.value_counts(label))
# region 图形化显示
# visualize
fig1 = plt.figure(figsize=(8, 6))
lab0 = plt.scatter(V1[label == 0], V2[label == 0])
lab1 = plt.scatter(V1[label == 1], V2[label == 1])
lab2 = plt.scatter(V1[label == 2], V2[label == 2])
plt.title('V1 and V2')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((lab0, lab1, lab2), ('first categroy', 'second categray', 'third categray'))
plt.show()
# endregion

'''
# region 聚类Kmeans
# build model
KM = KMeans(n_clusters=3, random_state=0)
KM.fit(X)
print(KM.cluster_centers_)

Y_predict = KM.predict(X)
Y_corrected = []
for i in Y_predict:
    if i == 0:
        Y_corrected.append(1)
    elif i == 1:
        Y_corrected.append(2)
    else:
        Y_corrected.append(0)
print(pd.value_counts(Y_corrected))
Y_corrected = np.array(Y_corrected)
# step3 test model
accuracy1 = accuracy_score(Y_corrected, label)
print(f'accuracy:{accuracy1}')

# region 图形化显示
# visualize
fig2 = plt.subplot(121)
lab0 = plt.scatter(V1[label == 0], V2[label == 0])
lab1 = plt.scatter(V1[label == 1], V2[label == 1])
lab2 = plt.scatter(V1[label == 2], V2[label == 2])
plt.title('real')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((lab0, lab1, lab2), ('first categroy', 'second categray', 'third categray'))
fig3 = plt.subplot(122)
centers = KM.cluster_centers_
lab0 = plt.scatter(V1[Y_corrected == 0], V2[Y_corrected == 0])
lab1 = plt.scatter(V1[Y_corrected == 1], V2[Y_corrected == 1])
lab2 = plt.scatter(V1[Y_corrected == 2], V2[Y_corrected == 2])
plt.scatter(centers[:, 0], centers[:, 1])
plt.title('predict')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((lab0, lab1, lab2), ('first categroy', 'second categray', 'third categray'))
plt.show()
# endregion
# endregion
'''

'''#build model KNN算法
KNN= KNeighborsClassifier(n_neighbors= 3)
KNN.fit(X,label)
Y_predict_KNN_test = KNN.predict([[80,60]])
print(f'(80,60):{Y_predict_KNN_test}')

#test model
Y_predict_KNN = KNN.predict(X)
accuracy_KNN = accuracy_score(Y_predict_KNN,label)
print(f'accuracy_KNN:{accuracy_KNN}')
print(pd.value_counts(Y_predict_KNN))

# region 图形化显示
# visualize
fig2 = plt.subplot(121)
lab0 = plt.scatter(V1[label == 0], V2[label == 0])
lab1 = plt.scatter(V1[label == 1], V2[label == 1])
lab2 = plt.scatter(V1[label == 2], V2[label == 2])
plt.title('real')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((lab0, lab1, lab2), ('first categroy', 'second categray', 'third categray'))
fig3 = plt.subplot(122)
lab0 = plt.scatter(V1[Y_predict_KNN == 0], V2[Y_predict_KNN == 0])
lab1 = plt.scatter(V1[Y_predict_KNN == 1], V2[Y_predict_KNN == 1])
lab2 = plt.scatter(V1[Y_predict_KNN == 2], V2[Y_predict_KNN == 2])
plt.title('predict')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((lab0, lab1, lab2), ('first categroy', 'second categray', 'third categray'))
plt.show()
# endregion'''

'''# build model Meanshift算法
bw = estimate_bandwidth(X, n_samples=500)
print(bw)
MS = MeanShift(bandwidth=bw)
MS.fit(X)
Y_predict_MS = MS.predict(X)

Y_corrected_MS = []
for i in Y_predict_MS:
    if i == 0:
        Y_corrected_MS.append(2)
    elif i == 1:
        Y_corrected_MS.append(1)
    else:
        Y_corrected_MS.append(0)
Y_corrected_MS = np.array(Y_corrected_MS)
print(pd.value_counts(Y_corrected_MS))

# step3 test model
accuracy_MS = accuracy_score(Y_corrected_MS, label)
print(f'accuracy_MS:{accuracy_MS}')

# region 图形化显示
# visualize
fig2 = plt.subplot(121)
lab0 = plt.scatter(V1[label == 0], V2[label == 0])
lab1 = plt.scatter(V1[label == 1], V2[label == 1])
lab2 = plt.scatter(V1[label == 2], V2[label == 2])
plt.title('real')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((lab0, lab1, lab2), ('first categroy', 'second categray', 'third categray'))
fig3 = plt.subplot(122)
lab0 = plt.scatter(V1[Y_corrected_MS == 0], V2[Y_corrected_MS == 0])
lab1 = plt.scatter(V1[Y_corrected_MS == 1], V2[Y_corrected_MS == 1])
lab2 = plt.scatter(V1[Y_corrected_MS == 2], V2[Y_corrected_MS == 2])
plt.title('predict')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((lab0, lab1, lab2), ('first categroy', 'second categray', 'third categray'))
plt.show()
# endregion
'''

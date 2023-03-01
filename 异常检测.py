import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.metrics import accuracy_score
from sklearn.covariance import EllipticEnvelope

# load the file
data = pd.read_csv('./AIlearning/chip_test.csv')
X1 = data.loc[:, 'test1']
X2 = data.loc[:, 'test2']
data = pd.concat([X1, X2], axis=1)
# visualize data
fig0 = plt.figure(figsize=(8, 8))
plt.scatter(X1, X2)
plt.title('innormal pointd test')
plt.show()

# step2
fig1 = plt.figure(figsize=(10, 5))
plt.hist(X1, bins=100)
plt.title('X1 distribution')
plt.show()

# calcurate the mean and the std
X1_mean = X1.mean()
X1_std = X1.std()
X2_mean = X2.mean()
X2_std = X2.std()

X1_range = np.linspace(-2, 2, 300)
X1_norm = norm.pdf(X1_range, X1_mean, X1_std)
# print(X1_norm)
X2_range = np.linspace(-2, 2, 300)
X2_norm = norm.pdf(X2_range, X2_mean, X2_std)
# print(X2_norm)

# visualize
fig2 = plt.figure(figsize=(10, 5))
fig3 = plt.subplot(121)
plt.plot(X1_range, X1_norm)
plt.title('X1 Normal distribution')
fig4 = plt.subplot(122)
plt.plot(X2_range, X2_norm)
plt.title('X2 Normal distribution')
plt.show()

# step2 establish the model and predict
ad_model = EllipticEnvelope(contamination=0.05)
ad_model.fit(data)

# make prediction
Y_predict = ad_model.predict(data)
print(Y_predict)
fig5 = plt.figure(figsize=(8, 8))
normal_data = plt.scatter(X1, X2, marker='x')
anomaly_data = plt.scatter(X1[Y_predict == -1], X2[Y_predict == -1], marker='o', facecolor='None', edgecolors='r',
                           s=150)
plt.title('anomaly pointd test')
plt.legend((normal_data, anomaly_data), ('normal_data', 'anomaly_data'))
plt.show()

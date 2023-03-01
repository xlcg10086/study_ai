# 酶活性预测实战task:
# 1、基于T-R-train.csv数据，建立线性回归模型，计算其在T-R-test.csv数据上的r2分数，可视化模型预测结果
# 2、加入多项式特征(2次、5次)，建立回归模型
# 3、计算多项式回归模型对测试数据进行预测的x2分数，判断哪个模型预测更准确
# 4、可视化多项式回归模型数据预测结果，判断哪个模型预测更准确

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data_train = pd.read_csv('./AIlearning/T-R-train.csv')


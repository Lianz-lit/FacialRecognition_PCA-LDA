#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：facialRecognition 
@File    ：sklearn_lda.py
@IDE     ：PyCharm 
@Author  ：Lianz
@Date    ：2023/4/2 23:28
"""

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

faces = datasets.fetch_olivetti_faces(data_home="/Users/lit/Documents/博1/proj/[dataset]/scikit_learn_data")
# 默认存放路径：/Users/lit/scikit_learn_data

print(faces.images.shape)

from matplotlib import pyplot as plt

i = 0
plt.figure(figsize=(20, 20))
for img in faces.images:
    #总共400张图，把图像分割成20X20
    plt.subplot(20, 20, i+1)
    plt.imshow(img, cmap="gray")
    #关闭x，y轴显示
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(faces.target[i])
    i = i + 1

plt.show()

#人脸数据
X = faces.data
#人脸对应的标签
y = faces.target
print(X[0])
print(y[0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.svm import SVC

#使用SVC作为模型
clf = SVC(kernel="linear")
#训练
clf.fit(X_train, y_train)
#预测
y_predict = clf.predict(X_test)
#对比实际值
print("实际标签：", y_test[0], "预测标签：", y_predict[0])

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_predict))

# 选几个分类的机器学习模型做回归
# estimators = {"knn classifier": KNeighborsClassifier(),
#               "svc kernel=rbf classifier": SVC(),
#               "svc kernel=linear classifier": SVC(kernel="linear"),
#               "randomforest classifier": RandomForestClassifier(n_estimators=10, max_depth=10)}
#
# for key, estimator in estimators.items():
#     estimator.fit(X_train, y_train)
#     y_predict = estimator.predict(X_test)
#     print(key, ":", accuracy_score(y_test, y_predict))


#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.datasets import load_digits  # 数据集
from sklearn.preprocessing import LabelBinarizer  # 标签二值化
from sklearn.cross_validation import train_test_split  # 数据集分割
import numpy as np

import pylab as pl  #  数据可视化


def sigmoid(x):  # 激活函数
    return 1/(1+np.exp(-x))


def dsigmoid(x):  # sigmoid的倒数
    return x*(1-x)


class NeuralNetwork:
    def __init__(self, layers):  # 这里是三层网络，列表[64,100,10]表示输入，隐藏，输出层的单元个数
        # 初始化权值，范围1~-1
        self.V = np.random.random((layers[0]+1, layers[1]))*2-1  # 隐藏层权值(65,100)，之所以是65，因为有偏置W0
        self.W = np.random.random((layers[1], layers[2]))*2-1  # (100,10)

    def train(self, X, y, lr=0.1, epochs=10000):
        # lr为学习率，epochs为迭代的次数
        # 为数据集添加偏置
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X
        X = temp  # 这里最后一列为偏置

        # 进行权值训练更新
        for n in range(epochs+1):
            i = np.random.randint(X.shape[0])  # 随机选取一行数据(一个样本)进行更新
            x = X[i]
            x = np.atleast_2d(x)  # 转为二维数据

            L1 = sigmoid(np.dot(x, self.V))  # 隐层输出(1,100)
            L2 = sigmoid(np.dot(L1, self.W))  # 输出层输出(1,10)

            #  delta
            L2_delta = (y[i]-L2)*dsigmoid(L2)  # (1,10)
            L1_delta = L2_delta.dot(self.W.T)*dsigmoid(L1)  # (1,100)，这里是数组的乘法，对应元素相乘

            # 更新
            self.W += lr*L1.T.dot(L2_delta)  # (100,10)
            self.V += lr*x.T.dot(L1_delta)

            # 每训练1000次预测准确率
            if n % 1000 == 0:
                predictions = []
                for j in range(X_test.shape[0]):
                    out = self.predict(X_test[j])  # 用验证集去测试
                    predictions.append(np.argmax(out))  # 返回预测结果
                accuracy = np.mean(np.equal(predictions, y_test))  # 求平均值
                print('epoch:', n, 'accuracy:', accuracy)

    def predict(self, x):
        # 添加转置,这里是一维的
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        x = temp
        x = np.atleast_2d(x)

        L1 = sigmoid(np.dot(x, self.V))  # 隐层输出
        L2 = sigmoid(np.dot(L1, self.W))  # 输出层输出
        return L2


digits = load_digits()  # 载入数据
X = digits.data  # 数据

y = digits.target   # 标签
# print y[0:10]

# 数据归一化,一般是x=(x-x.min)/x.max-x.min
X -= X.min()
X /= X.max()

# 创建神经网络
nm = NeuralNetwork([64, 100, 10])

X_train, X_test, y_train, y_test = train_test_split(X, y)  # 默认分割：3:1

# 标签二值化
labels_train = LabelBinarizer().fit_transform(y_train)
# print labels_train[0:10]
labels_test = LabelBinarizer().fit_transform(y_test)

print 'start'

nm.train(X_train, labels_train, epochs=20000)

print 'end'
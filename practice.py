#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0-np.tanh(x)*np.tanh(x)


def logistic(x):
    return 1/(1+np.exp(x))


def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # 对权重初始化。对每一层的权重都要初始化
        self.weights = []
        for i in range(1,len(layers)-1):
            # 对每一层的权重都要初始化初始值范围在-0.25~0.25之间，然后保存在weight中
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):  # 默认学习率即步长为0.2，循环最多的次数为1000
        X = np.atleast_2d(X)  # 判断输入训练集是否为二维
        temp = np.ones([X.shape[0],X.shape[1]+1])  # 列加1是因为最后一列要存入标签分类，这里标签都为1
        temp[:,0:-1] = X
        X = temp
        y = np.array(y)  # 训练真实值

        for k in range(epochs):  # 循环
            i = np.random.randint(X.shape[0])  # 随机选取训练集中的一个
            a = [X[i]]
            # 计算激活值
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))

            error = y[i] - a[-1]  # 计算偏差
            deltas = [error*self.activation_deriv(a[-1])]  # 输出层误差
            # 下面计算隐藏层
            for l in range(len(a)-2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            # 下面开始更新权重和偏向
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    # 预测函数
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        for i in range(a):
            if a[i] > 0.5:
                a[i] = 1
            else:
                a[i] = 0
        return a


nn = NeuralNetwork([2, 2, 1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, nn.predict(i))
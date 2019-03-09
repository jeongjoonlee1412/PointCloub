#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 定义存储输入数据（x）和目标数据（y）的数组
x, y = [], []
# 遍历数据集，变量sample对应的正是一个个样本
for sample in open('E:\\test.txt', 'r'):
    _x, _y = sample.split(',')
    x.append(float(_x))
    y.append(float(_y))
# 读取完数据后，将其转化为Numpy数组以方便进一步的处理
x = np.array(x)
y = np.array(y)


# 标准化
x = (x - x.mean()) / x.std()
# 将原数据以散点图的形式画出
plt.figure()
plt.scatter(x, y, c='g', s=20)

# 在（-2,4）这个区间上去100个点作为画图的基础
x0 = np.linspace(-2, 4, 100)


# 利用Numpy的函数定义训练并返回多项式回归模型的函数
# deg参数代表着模型参数中的n、亦即模型中多项式的次数
# 返回的模型能够根据输入的x（默认是x0）、返回相应的预测值y
def get_model(deg):
    return lambda input_x = x0: np.polyval(np.polyfit(x, y, deg), input_x)


# 根据参数n、输入的x、y返回相应的损失
def get_cost(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()


# 定义测试参数集并根据它进行各种实验
test_set = (1, 4, 10)
for d in test_set:
    # 输出相应的损失
    print(get_cost(d, x, y))
plt.scatter(x, y, c='g', s=20)
for d in test_set:
    plt.plot(x0, get_model(d)(), label='degree = {}'.format(d))
plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)
plt.legend()
plt.show()

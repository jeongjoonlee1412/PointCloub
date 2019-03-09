#!/usr/bin/python
# -*- coding: UTF-8 -*-
import csv
from numpy import *
import operator

def load_csv(file_path):
    data = []
    count = 0
    csv_file = csv.reader(open(file_path, 'r'))
    for rows in csv_file:
        data.append(rows)
        count = count + 1
    return count, data


def train_matrix():
    train_count, train_test = load_csv('E:/robotStudy/digit-recognizer/train.csv')
    train_matrix = zeros((42000, 784))
    train_lable = []
    for i in range(1, train_count):
        train_lable.append(int(train_test[i][0]))
        for j in range(1, 785):
            train_matrix[i-1][j-1] = int(train_test[i][j])
    return train_matrix, train_lable


def test_matrix():
    test_count, test_test = load_csv('E:/robotStudy/digit-recognizer/test.csv')
    test_matrix = zeros((28000, 784))
    for i in range(1, test_count):
        for j in range(784):
            test_matrix[i-1][j] = int(test_test[i][j])
    return test_matrix


def train_classified():
    data = []
    lable0 = []
    lable1 = []
    lable2 = []
    lable3 = []
    lable4 = []
    lable5 = []
    lable6 = []
    lable7 = []
    lable8 = []
    lable9 = []
    count = 0
    csv_file = csv.reader(open('E:/robotStudy/digit-recognizer/train.csv', 'r'))
    for rows in csv_file:
        data.append(rows)
        count = count + 1
    for i in range(1, count):
        column = int(data[i][0])
        if column == 0:
            lable0.append(data[i])
        elif column == 1:
            lable1.append(data[i])
        elif column == 2:
                lable2.append(data[i])
        elif column == 3:
            lable3.append(data[i])
        elif column == 4:
                lable4.append(data[i])
        elif column == 5:
            lable5.append(data[i])
        elif column == 6:
            lable6.append(data[i])
        elif column == 7:
            lable7.append(data[i])
        elif column == 8:
            lable8.append(data[i])
        elif column == 9:
            lable9.append(data[i])
        else:
            continue
    return [lable0, lable1, lable2, lable3, lable4, lable5, lable6, lable7, lable8, lable9]


def handWritingClassTest():
    trainMat, label = train_matrix()
    test_data = test_matrix()
    # mTest = test_matrix().shape[0]
    for i in range(28000):
        vectorUnderTest = test_data[i][:]
        classifierResult = classify0(vectorUnderTest, trainMat, label, 10)
        print "the classifierResult came back with: %d" % classifierResult


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    handWritingClassTest()

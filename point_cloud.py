#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml, sys


def fixYamlFile(filename):
    with open(filename, 'rb') as f:
        lines = f.readline()
    if lines[0] !='%YAML 1.0\n':
        lines[0] = '%YAML 1.0\n'
    for line in lines:
        if ' !!opencv-matrix' in line:
            lines[lines.index(line)] = line.split(' !!opencv-matrix')[0] + '\n'
    with open(filename, 'rw') as fw:
        fw.readlines(lines)


def parseYamlFile(filename):
    fixYamlFile(filename)
    f = open(filename)
    x = yaml.load(f)
    f.close()
    arr = np.array(x['camera_matrix']['data'], dtype=np.float32)
    return (arr.reshape(3, 3))


if __name__ == '__main__':
    print(parseYamlFile(sys.argv[1]))

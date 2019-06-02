# -*- coding:utf-8 -*-

# 数据集：《机器学习》--西瓜数据集4.0

# 算法流程：《机器学习》--学习向量化算法 图9.4



import math
import numpy as np
import pylab as pl

intialData = """
1,0.697,0.46,Y,2,0.774,0.376,Y,3,0.634,0.264,Y,4,0.608,0.318,Y,5,0.556,0.215,Y,
6,0.403,0.237,Y,7,0.481,0.149,Y,8,0.437,0.211,Y,9,0.666,0.091,N,10,0.243,0.267,N,
11,0.245,0.057,N,12,0.343,0.099,N,13,0.639,0.161,N,14,0.657,0.198,N,15,0.36,0.37,N,
16,0.593,0.042,N,17,0.719,0.103,N,18,0.359,0.188,N,19,0.339,0.241,N,20,0.282,0.257,N,
21,0.748,0.232,N,22,0.714,0.346,Y,23,0.483,0.312,Y,24,0.478,0.437,Y,25,0.525,0.369,Y,
26,0.751,0.489,Y,27,0.532,0.472,Y,28,0.473,0.376,Y,29,0.725,0.445,Y,30,0.446,0.459,Y"""


# 定义一个西瓜类，四个属性，分别是编号，密度，含糖率，是否好瓜
class Watermelon:
    def __init__(self, properties):
        self.number = properties[0]
        self.density = float(properties[1])
        self.sweet = float(properties[2])
        self.good = properties[3]


# 数据处理

_initalData = intialData.split(',')
_dataSet = []  # dataset数据集
for i in range(int(len(_initalData) / 4)):  # 将data中的数据按三个数据一行划分
    _tupleData = tuple(_initalData[i * 4: i * 4 + 4])  # b = a[i:j] 表示复制a[i]到a[j-1]，以生成新的元祖
    _dataSet.append(Watermelon(_tupleData))  # 将元祖数据定义为西瓜类放入数据集


# 计算欧几里得距离,_firstTuple,_secondTuple分别为两个元组
def dist(_firsttuple, _secondtuple):
    # 每一个样本是有两个属性“密度”“含糖率”的二维向量
    return math.sqrt(math.pow(_firsttuple[0] - _secondtuple[0], 2) + math.pow(_firsttuple[1] - _secondtuple[1], 2))


# 算法模型
def LVQ(dataset, a, max_iter):
    # 统计样本一共有多少个类别标记
    _Tab = list(set(one.good for one in dataset))
    # 随机产生原型向量
    _prototypeVector = [(_one.density, _one.sweet) for _one in np.random.choice(dataset, len(_Tab))]
    while max_iter > 0:
        X = np.random.choice(dataset, 1)[0]
        _index = np.argmin(dist((X.density, X.sweet), i) for i in _prototypeVector)
        t_index = _Tab[_index]
        if t_index == X.good:
            _prototypeVector[_index] = (
                (1 - a) * _prototypeVector[_index][0] + a * X.density,
                (1 - a) * _prototypeVector[_index][1] + a * X.sweet)
        else:
            _prototypeVector[_index] = (
                (1 + a) * _prototypeVector[_index][0] - a * X.density,
                (1 + a) * _prototypeVector[_index][1] - a * X.sweet)
        max_iter -= 1
    return _prototypeVector


def show(dataset, pV):
    prototype = [[] for i in pV]
    for i in dataset:
        prototype[i.good == 'Y'].append(i)
    return prototype


# 画图
def draw(_data, _PV):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(_data)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(_data[i])):
            coo_X.append(_data[i][j].density)
            coo_Y.append(_data[i][j].sweet)
        pl.scatter(coo_X, coo_Y, marker='.', color=colValue[i % len(colValue)], label=i, s=80)
    # 展示原型向量
    P_x = []
    P_y = []
    for i in range(len(_PV)):
        P_x.append(_PV[i][0])
        P_y.append(_PV[i][1])
        pl.scatter(_PV[i][0], _PV[i][1], marker='+', color=colValue[i % len(colValue)], label="prototypeVector", s=80)
    pl.legend(loc='upper right')
    pl.show()


prototypeVector = LVQ(_dataSet, 0.01, 60)
Data = show(_dataSet, prototypeVector)
draw(Data, prototypeVector)

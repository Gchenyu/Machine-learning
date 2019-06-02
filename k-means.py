# -*- coding:utf-8 -*-

# 数据集：《机器学习》--西瓜数据集4.0

# 算法流程：《机器学习》--k均值算法 图9.2


import math
import numpy as np
import pylab as pl

# 数据集：每三个是一组分别是西瓜的编号，密度，含糖量
intialData = """
1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""


# 定义一个西瓜类，三个属性，分别是编号，密度，含糖率

class Watermelon:
    def __init__(self, properties):
        self.number = properties[0]
        self.density = float(properties[1])
        self.sweet = float(properties[2])


# 数据处理

_initalData = intialData.split(',')  # 返回一个列表  例：“A，B，C”.split(",")结果为['A','B','C'] 即将一整个字符串根据所给的分隔符分割成多个字符串再放成列表类型返回
_dataSet = []
for i in range(int(len(_initalData) / 3)):  # 将data中的数据按三个数据一行划分
    _tupleData = tuple(_initalData[i * 3: i * 3 + 3])
    _dataSet.append(Watermelon(_tupleData))  # 将元祖数据定义为西瓜类放入数据集


# 计算欧几里得距离,_firstTuple,_secondTuple分别为两个元组
def dist(_firsttuple, _secondtuple):
    # 每一个样本是有两个属性“密度”“含糖率”的二维向量
    return math.sqrt(math.pow(_firsttuple[0] - _secondtuple[0], 2) + math.pow(_firsttuple[1] - _secondtuple[1], 2))


# k均值算法模型
def k_means(k, dataset, max_iter):
    _meanVector = np.random.choice(dataset, k)  # choice从序列中随机选择k个样本作为均值向量   例random.choice([1,2,3,4,5,6,7,8,9])
    _meanVector = [(watermelon.density, watermelon.sweet) for watermelon in _meanVector]  # 均值向量列表
    _categoryList = [[] for i in range(k)]  # 初始化分类列表
    meanVector_update = []  # 均值向量更新列表
    while max_iter > 0:
        # 分类
        for i in dataset:
            temp = np.argmin([dist((i.density, i.sweet), _meanVector[j]) for j in range(len(_meanVector))])
            _categoryList[temp].append(i)
        # 更新均值向量
        for i in range(k):
            ui_density = 0.0
            ui_sweet = 0.0
            for j in _categoryList[i]:
                ui_density += j.density
                ui_sweet += j.sweet
            meanVector_update.append((ui_density / len(_categoryList[i]), ui_sweet / len(_categoryList[i])))
        # 每五次输出一次分类图
        if max_iter % 5 == 0:
            draw(_categoryList, _meanVector)
        # 比较U和U_update，如果相同则算法停止，得到最终的簇划分
        if _meanVector == meanVector_update:
            break
        _meanVector = meanVector_update
        meanVector_update = []
        _categoryList = [[] for i in range(k)]
        max_iter -= 1

    return _categoryList, _meanVector


# 画图
def draw(_cL, _mV):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(_cL)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(_cL[i])):
            coo_X.append(_cL[i][j].density)
            coo_Y.append(_cL[i][j].sweet)
        pl.scatter(coo_X, coo_Y, marker='.', color=colValue[i % len(_cL)], s=80)  # , label=str(i)
    # 展示均值向量
    u_x = []
    u_y = []
    for i in _mV:
        u_x.append(i[0])
        u_y.append(i[1])
    pl.scatter(u_x, u_y, marker='+', color=colValue[6], s=80)  # , label="avg_vector"
    # pl.legend(loc='upper right')
    pl.show()


categoryList, meanVector = k_means(3, _dataSet, 30)
draw(categoryList, meanVector)

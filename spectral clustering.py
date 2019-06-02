from sklearn.cluster import KMeans
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from itertools import cycle, islice


# -----------------数据集生成--------------
def genTwoCircles(n_samples=1000):
    x, y = datasets.make_circles(n_samples, shuffle=True, random_state=None, factor=0.5, noise=0.05)
    # x, y = datasets.make_moons(n_samples, noise=0.05)
    return x, y


# ------------------距离矩阵---------------------
def euclidDistance(x1, x2, sqrt_flag=False):  # 欧氏距离
    res = np.sum((x1 - x2) ** 2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res


def calEuclidDistanceMatrix(X):
    X = np.array(X)
    # print(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S


# ---------------------邻接矩阵-----------------
def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N, N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k + 1)]  # xi's k nearest neighbours

        for j in neighbours_id:  # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j] / 2 / sigma / sigma)
            A[j][i] = A[i][j]  # mutually

    return A


# ------------------标准化的拉普拉斯矩阵-----------------
def calLaplacianMatrix(adjacentMatrix):
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # print degreeMatrix

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # print laplacianMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)


# ----------------画图-------------------
def plot(X, y_sp, y_km):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y_km) + 1))))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_sp])
    plt.title("Spectral Clustering")
    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_km])
    plt.title("Kmeans Clustering")
    plt.show()
    # plt.savefig("../figures/spectral_clustering.png")


# ---------------------------------
np.random.seed(1)

data, label = genTwoCircles(n_samples=500)

Similarity = calEuclidDistanceMatrix(data)

Adjacent = myKNN(Similarity, k=10)

Laplacian = calLaplacianMatrix(Adjacent)
# 特征值分解
x, V = np.linalg.eig(Laplacian)

x = zip(x, range(len(x)))
x = sorted(x, key=lambda x: x[0])

H = np.vstack([V[:, i] for (v, i) in x[:500]]).T
# Kmeans
sp_kmeans = KMeans(n_clusters=2).fit(H)

pure_kmeans = KMeans(n_clusters=2).fit(data)

plot(data, sp_kmeans.labels_, pure_kmeans.labels_)

import copy
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from matplotlib import colors
from numpy import linalg as LA, sign
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.stats import shapiro

from func import plot_graph, to_categorical

if __name__ == '__main__':
    ########
    # generated graph
    ########
    data_num = 1000
    top_num = 500
    bottom_num = data_num - top_num
    graph = nx.bipartite.random_graph(top_num, bottom_num, 0.8)
    labels = np.array([d['bipartite'] for n, d in graph.nodes(data=True)])  # 0/1 label
    print(graph.number_of_edges())
    print(nx.number_connected_components(graph))
    norm_lap = nx.normalized_laplacian_matrix(graph)

    e, U = LA.eigh(norm_lap.A)
    features = np.ones((len(graph.nodes), 1))  # N * 1 matrix

    ########
    # generated labels by norm dist
    ########
    ids = np.array(list(map(int, graph.nodes)))
    pre_labels = copy.deepcopy(labels)
    groups = to_categorical(pre_labels)

    labels[ids > top_num] = np.random.normal(loc=1, scale=1, size=(labels[ids > top_num].shape[0]))
    labels[ids <= top_num] = np.random.normal(loc=10, scale=1, size=(labels[ids <= top_num].shape[0]))
    # labels[ids > top_num] = 10
    # labels[ids <= top_num] = 10
    labels = labels.reshape((-1, 1))

    ########
    # plot label
    ########
    g1 = []
    g2 = []
    for id, v in enumerate(list(graph.nodes)):
        if int(v) > top_num:
            g1.append(labels[id][0])
        else:
            g2.append(labels[id][0])

    bins = np.linspace(-5, 16, 20)
    plt.hist([g1, g2], bins, label=['top', 'bottom'])
    plt.show()

    plt.scatter(e, np.dot(U.T, labels) / np.dot(U.T, np.ones((data_num, 1))))
    plt.plot(e, np.dot(U.T, labels) / np.dot(U.T, np.ones((data_num, 1))), '-ok')
    plt.title("g(x)")
    plt.show()

    ########
    # plot graph, selected eigenvectors
    ########
    g_func = np.abs(np.dot(U.T, labels) / np.dot(U.T, np.ones((data_num, 1))))
    max_id = (-g_func.flatten()).argsort()[:3]
    # plot_graph(graph, label=groups, max_id=max_id)

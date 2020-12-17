from numpy import linalg as LA
import matplotlib.pyplot as plt
import networkx as nx
from utils import load_data


def plot_eigenvalues(dsname):
    data = load_data(dsname)
    graph = nx.from_dict_of_lists(data)
    print('# of nodes: {}'.format(graph.number_of_nodes()))
    print('# of edges: {}'.format(graph.number_of_edges()))

    norm_lap = nx.normalized_laplacian_matrix(graph)
    e, U = LA.eigh(norm_lap.A)

    plt.hist(e, bins=200)
    plt.title(dsname + ': ' + str(graph.number_of_nodes()) + ' nodes')
    plt.xlabel('λ')
    plt.ylabel('count')
    plt.show()

plot_eigenvalues('Cora')
plot_eigenvalues('Citeseer')
plot_eigenvalues('Pubmed')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from numpy import linalg as LA
from utils import load_data_with_features


def plot_all_filters(dataset, k, root):
    data, features, labels = load_data_with_features(dataset)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=-1)

    # Sort the labels according to their counts from small to big
    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    res = sorted(zip(counts_labels, unique_labels), key=lambda x: x[0])
    res = zip(*res)
    res = [list(a) for a in res]
    sorted_counts = res[0]
    sorted_labels = res[1]
    print(dataset)
    print(f"Sorted labels: {sorted_labels}")
    print(f"Labels counts: {sorted_counts}")

    # Eigendecomposition
    graph = nx.from_dict_of_lists(data)
    norm_lap = nx.normalized_laplacian_matrix(graph)
    e, U = LA.eigh(norm_lap.A)
    print('# of nodes: {}'.format(graph.number_of_nodes()))
    print('# of edges: {}'.format(graph.number_of_edges()))

    # To-do: save e and U

    # Create folder
    if not os.path.exists(root):
        os.mkdir(root)

    # Save features
    np.savetxt(f"{root}/features.csv", features, delimiter=",")

    # Plot all filters
    for i in range(k):
        label = sorted_labels[i]
        binary_labels = [1 if x == label else 0 for x in labels]
        savedir = f"{root}/label_{label}/"
        if not os.path.exists(savedir):
            os.mkdir(savedir)
            # Save labels
            np.savetxt(f"{savedir}/labels_{label}.csv", binary_labels, delimiter=",")
        for j in range(features.shape[1]):
            filename = savedir + f"label_{label}_vs_feature_{j+1}.png"
            plot_filter(features[:, j], binary_labels, e, U, j, label, filename)

    # features = np.array(features.todense()[:, feature_id])  # One feature only
    # features = [i[0] for i in features]  # 2D array to 1D
    # np.clip(labels, 0, 1, out=labels)

    # plt.hist(features, bins=2)
    # plt.title('Cora feature {}'.format(feature_id))
    # plt.xlabel('feature value')
    # plt.ylabel('count')
    # plt.show()


def plot_filter(x, y, e, U, feature_id, label, filename):
    # if feature_id not in [30, 108]:
    #     return

    eps = 1e-4
    xh = np.dot(U.T, x)
    yh = np.dot(U.T, y)

    output = np.divide(yh, xh, out=np.zeros(yh.shape, dtype=float), where=np.abs(xh) > eps)

    fig = plt.figure()
    plt.scatter(e, output, s=10)
    plt.title("g(x) of label {} vs feature {}".format(label, feature_id+1))
    plt.savefig(filename)
    plt.close(fig)
    return


plot_all_filters('cora', 3, 'img.nosync/cora')
plot_all_filters('citeseer', 3, 'img.nosync/citeseer')
plot_all_filters('pubmed', 3, 'img.nosync/pubmed')


import copy
import pickle as pk
import random
from collections import defaultdict

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
import torch
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linalg as LA
from scipy.sparse import lil_matrix, csr_matrix

DATA_NUM = 100
FEAT_NUM = 1
CLASS_NUM = 3

HI_MEAN = [5, 10]
LO_MEAN = [0, 5]
HI_CVAR = [0, 5]
LO_CVAR = [5, 10]

boundary_node = []


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def rand_pos(v):
    return v[0] + (1 if random.random() < 0.5 else -1) * random.uniform(0.02, 0.03), \
           v[1] + (1 if random.random() < 0.5 else -1) * random.uniform(0.02, 0.03)


def plot_graph(graph, label=None, cache=False, max_id=None):
    val_map = ['cyan', 'red', 'blue', 'magenta', 'gray', 'purple', 'orange', 'yellow', 'green', 'black', 'pink']

    if label is not None:
        if len(np.shape(label)) > 1:
            values = [val_map[_] for _ in np.where(label == 1)[1].tolist()]
        else:
            values = [val_map[int(_)] for _ in label]
    else:
        values = None
    # if graph.name == 'Zachary\'s Karate Club':
    #     vals = {'Mr. Hi': 0, 'Officer': 1}
    #     values = [vals[__['club']] for _, __ in graph._node.items()]

    if cache:
        pos = pk.load(open('./pos.pk', 'rb'))
    else:
        pos = nx.fruchterman_reingold_layout(graph, k=0.1, iterations=50)
        pk.dump(pos, open('./pos.pk', 'wb'))
    # pos = nx.circular_layout(graph,scale=1)
    # pos = nx.random_layout(graph)
    # pos = nx.shell_layout(graph)
    # pos = nx.spectral_layout(graph)

    if nx.bipartite.is_bipartite(graph):
        l, r = nx.bipartite.sets(graph)
        pos = {}
        pos.update((node, (1, index)) for index, node in enumerate(l))
        pos.update((node, (2, index)) for index, node in enumerate(r))

    print("\nPlotting a graph...")

    # plot graph
    plt.axis('off')
    plt.figure(1, figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

    ax0 = plt.subplot(gs[0])
    # plot eigenvalues
    norm_lap = nx.normalized_laplacian_matrix(graph)
    eigval, eigvec = LA.eigh(norm_lap.A)
    ax0.plot(eigval, 'ro')

    ax1 = plt.subplot(gs[1])
    # nx.draw(graph, pos=pos, node_color=values, node_size=15, width=0.1)
    nx.draw(graph, pos, node_color=label[:0], node_size=20, width=0.1, cmap=plt.cm.rainbow, with_labels=False)
    # color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=np.min(label), vmax=np.max(label)))
    sm._A = []
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("bottom", size="5%")
    cbar = plt.colorbar(sm, cax=cax, ticks=[-1, -.5, -.25, -.1, 0, .1, .25, .5, 1], orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('node text in graph is index=val', rotation=360)

    plt.title('original networks w/ eigenvalues')
    plt.show()

    # visualize eigen vectors
    # for _ in range(3):  # len(graph._node)):
    for _ in list(range(3)) + max_id.flatten().tolist():  # len(graph._node)):
        plt.figure(1, figsize=(7, 8))
        fig, ax = plt.subplots(2, 1, num=1)

        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

        # plot eigen vector values
        ax0 = plt.subplot(gs[0])
        cur_eigv = eigvec[:, _]
        print("{}:{:.4f}".format(_, eigval[_]))
        print(["{}:{:.4f}".format(k, v) for k, v in enumerate(cur_eigv)])
        ax0.plot(range(cur_eigv.shape[0]), cur_eigv, 'b--')
        for i, txt in enumerate(cur_eigv):
            ax0.annotate(i, (range(cur_eigv.shape[0])[i], cur_eigv[i]))

        # ax0.set_ylabel('scale')
        ax0.set_xlabel('eigenvector of i={}'.format(_))

        # plot eigen vector as labels on graph
        ax1 = plt.subplot(gs[1])
        nx.draw(graph, pos, node_color=cur_eigv, node_size=20, width=0.1, cmap=plt.cm.rainbow, with_labels=False)

        # label draw
        labels = {k: "{}={:.3f}".format(k, v) for k, v in enumerate(cur_eigv.tolist())}
        pos_higher = {}

        for k, v in pos.items():
            pos_higher[k] = (v[0], v[1] + (1 if random.random() < 0.5 else -1) * random.uniform(0.02, 0.03))
        # nx.draw_networkx_labels(graph, pos_higher, labels, font_size=6)

        # color bar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow,
                                   norm=plt.Normalize(vmin=np.min(cur_eigv), vmax=np.max(cur_eigv)))
        sm._A = []
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="5%")
        cbar = plt.colorbar(sm, cax=cax, ticks=[-1, -.5, -.25, -.1, 0, .1, .25, .5, 1], orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_xlabel('node text in graph is index=val', rotation=360)
        plt.show()

    # plt.savefig("plot.png", dpi=600)


def gen_rand_graph(data_num, neighbors=20):
    adj_dict = {}
    for _ in range(data_num):
        neighbor_num = min(random.sample(range(1, neighbors), 1)[0], data_num - _ - 1)  # hard threshold

        adj_dict[_] = sorted(random.sample(range(_ + 1, data_num), neighbor_num))

    adj = defaultdict(int, adj_dict)

    return nx.from_dict_of_lists(adj)


def normal_rw_lap(G, nodelist=None, weight='weight'):
    import scipy.sparse
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  format='csr')
    n, m = A.shape
    diags = A.sum(axis=1).flatten()
    D = scipy.sparse.spdiags(diags, [0], m, n, format='csr')
    L = D - A
    with scipy.errstate(divide='ignore'):
        diags_sqrt = 1.0 / diags
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    DH = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    return DH.dot(L)


def normal_sym_lap(G, nodelist=None, weight='weight'):
    import scipy.sparse
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  format='csr')
    n, m = A.shape
    diags = A.sum(axis=1).flatten()
    D = scipy.sparse.spdiags(diags, [0], m, n, format='csr')
    L = D - A
    with scipy.errstate(divide='ignore'):
        diags_sqrt = 1.0 / scipy.sqrt(diags)
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    DH = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    return DH.dot(L.dot(DH))


def normal_gcn_lap(G, nodelist=None, weight='weight'):
    import scipy.sparse
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  format='csr')
    n, m = A.shape
    A += sp.eye(m)

    diags = A.sum(axis=1).flatten()
    with scipy.errstate(divide='ignore'):
        diags_sqrt = 1.0 / scipy.sqrt(diags)
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    DH = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    return DH.dot(A.dot(DH))


def normalize_lap(graph, opt=0):
    eigval, eigvec = LA.eigh(nx.laplacian_matrix(graph).toarray())
    lap = nx.laplacian_matrix(graph)
    print('Norm opt:{}'.format(opt))

    if -1 == opt:
        norm_lap = nx.laplacian_matrix(graph)
    elif 0 == opt:
        norm_lap = nx.normalized_laplacian_matrix(graph)
    elif opt == 1:
        norm_lap = normal_rw_lap(graph)
    elif opt == 2:
        norm_lap = normal_sym_lap(graph)
    elif opt == 3:
        norm_lap = normal_gcn_lap(graph)
    elif opt == 4:
        norm_lap = (1. / eigval[-1]) * lap
    elif opt == 5:
        norm_lap = (2. / eigval[-1]) * lap - sp.eye(lap.shape[0])
    elif opt == 6:
        norm_lap = np.exp(eigval)
    else:
        print('incorrect opt')

    eigval, eigvec = LA.eigh(norm_lap.toarray())

    return norm_lap, eigval, eigvec


def jump_func(x, opt=11):
    print("Jump opt: {}".format(opt))
    if opt == -1:
        return x
    elif opt == 0:
        return np.sqrt(np.abs(x))
    elif opt == 1:
        return np.sqrt(abs(x - .5))
    elif opt == 2:
        return np.minimum(abs(x + .5), np.exp(x + 1))
    elif opt == 3:  # jump
        return np.sign(x - .34) * 0.5 + 0.5
    elif opt == 4:
        return np.abs(x - .34)
    elif opt == 5:  # jump
        return 1 + x / (abs(x) - .5)
    elif opt == 6:
        return np.maximum(.85, np.sin(x + x ** 2)) - x / 20
    elif opt == 7:
        return -x - x ** 2 + np.exp(-(30 * (x - .47)) ** 2)
    elif opt == 8:
        return np.sign(x - .4) - np.sign(x - .6)
    elif opt == 9:
        data_len = x.shape[0]
        x[int(data_len / 2)] = 2 * x[int(data_len / 2)]
        return x
    elif opt == 10:  # jump
        return .5 * np.sign(x - .5) + .5
    elif opt == 11:  # jump
        ret = copy.deepcopy(x)
        ret[3] = x[3] * 100
        return ret
    elif opt == 12:

        return 1 - np.abs(x - 1)

    else:
        print('incorrect opt!')
        exit()


def gen_label_feat(data_num=DATA_NUM, feat_num=FEAT_NUM, class_num=CLASS_NUM, label=None, feat_jump=True,
                   label_jump=False, noise=False):
    if label is None:
        # randomly generate labels for data, actually around (# of data/# of class) for each class, and sort them
        # could consider possibilities as :
        # raw_labels = np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        raw_labels = sorted(np.random.choice(class_num, data_num))
    else:
        raw_labels = sorted(label)
    raw_labels_dict = Counter(raw_labels)
    feat_list = []

    # randomly choose one(default) or more classes to be added a large value
    if feat_jump:
        num_2_pick = 1
        picked_num = random.sample(range(class_num), num_2_pick)

    for _class, _num in raw_labels_dict.items():
        feat_num_selected = random.sample(range(3, feat_num), 1)[0]  # decide feature # in [3, feat_num]
        feat_col_selected = random.sample(range(feat_num), feat_num_selected)  # choose feat indexes

        feats = np.array([0.0] * feat_num)  # initialize feature for this _class
        feats[feat_col_selected] = 1.0  # fill select indexes with value

        # if this class is selected class
        if feat_jump and _class in picked_num:
            feats[feat_col_selected] = 10000.0

        for _n in range(_num):
            if noise:
                # decide feature # in [0, feat_num]
                max_noise_col = 3
                noise_num_selected = random.sample(range(0, max_noise_col), 1)[0]

                if noise_num_selected != 0:
                    noise_col_selected = random.sample(range(feat_num), noise_num_selected)  # choose feat indexes
                    feats[noise_col_selected] += 1.0

            feat_list.append(feats)

    # label jump
    if label_jump:
        # select 50 nodes to jump
        label_ind = random.sample(range(data_num), 50)
        for _ in label_ind:
            ori_label = raw_labels[_]
            candidate_class = list(range(class_num))
            candidate_class.remove(ori_label)
            raw_labels[_] = random.sample(candidate_class, 1)[0]

    # shuffle
    feat_list = np.array(feat_list)
    raw_labels = np.array(raw_labels)
    reorder = np.array(list(range(data_num)))
    old_order = copy.deepcopy(reorder)
    np.random.shuffle(reorder)
    feat_list[old_order, :] = feat_list[reorder, :]
    raw_labels[old_order] = raw_labels[reorder]

    return lil_matrix(feat_list), raw_labels


def gen_label_graph(label, data_num=DATA_NUM, neighbors_within=8, neigbors_across=3):
    # global boundary_node
    adj_dict = {}
    group_index = {}

    # from pygcn.train import args
    #
    # neighbors_within = args.inner_neigbor
    # neigbors_across = args.cross_neigbor

    # localize indexes of data for each class: 1={2,3,5}, 2={4,8,10} and so on
    for _class in range(len(set(label))):
        group_index[_class] = np.where(label == _class)[0].tolist()

    # iterate each data point
    for _ in range(data_num):
        _class = label[_]  # class of this data point

        # add neighbors within the same group
        # find the data indexes of same group
        self_group = group_index[_class]
        # choose [1, neighbor_num] as neighbor num
        # when self group is larger than threshold, select pre-defined number of neighbors at most
        # else, select all members within the same group at most
        neighbors_within_num = random.sample(range(0, min(neighbors_within, len(self_group))), 1)[0]
        # unrepeatly select neighbors from self group
        self_conn = random.sample(self_group, neighbors_within_num)
        # remove self
        if _ in self_conn:
            self_conn.remove(_)

        # add neighbors across groups
        neighbors_across_num = np.random.choice(np.arange(0, min(neigbors_across, data_num)))  # , p=[0.8, 0.1, 0.1])
        cross_conn = random.sample(range(data_num), neighbors_across_num)

        # # for stat only
        # if neighbors_across_num > 0:
        #     boundary_node.append(_)
        #     boundary_node += [__ for __ in cross_conn if __ not in self_group]

        neighbors = list(self_conn + cross_conn)
        adj_dict[_] = sorted(neighbors)

    # boundary_node = list(set(boundary_node))
    adj = defaultdict(int, adj_dict)
    return nx.from_dict_of_lists(adj)


def plot_single_signal(eigval, eigvec, label_ori, feat):
    f = eigval  # np.where(label == 1)[1]
    plt.plot(range(len(f)), f, 'ro', markersize=2)
    plt.vlines(range(len(f)), [0], f)
    plt.title('eigenvalue')
    plt.ylabel('f(i)')
    plt.xlabel('index')
    plt.show()

    f = np.dot(eigvec.T, label_ori)  # np.where(label == 1)[1]
    plt.plot(eigval, f, 'ro', markersize=1)
    plt.vlines(eigval, [0], f)
    plt.title('$U^{T} * Y$: spectral signal')
    plt.legend(['$U^{T} * Y$'], loc='upper right')
    plt.grid(True)
    plt.ylabel('f(i)')
    plt.xlabel('$\lambda$')
    plt.show()

    f = np.dot(eigvec.T, feat)  # np.where(label == 1)[1]
    plt.plot(eigval, f, 'ro', markersize=1)
    plt.vlines(eigval, [0], f)
    plt.title('$U^{T} * X$: spectral signal')
    plt.legend(['$U^{T} * X$'], loc='upper right')
    plt.grid(True)
    plt.ylabel('f(i)')
    plt.xlabel('$\lambda$')
    plt.show()

    f = np.dot(eigvec.T, label_ori) / np.dot(eigvec.T, feat)
    plt.plot(eigval, f, 'ro', markersize=1)
    plt.vlines(eigval, [0], f)
    plt.title('spectral node feat: $g(\lambda)$')
    plt.legend(['$g(\lambda)$'], loc='upper right')
    plt.grid(True)
    plt.ylabel('$g(\lambda)$')
    plt.xlabel('$\lambda$')
    plt.show()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def poly_recur(support, orders=4, poly_style='norm', lap_eig='eig'):  # norm or cheby polynomials
    t_k = list()
    if lap_eig == 'eig':
        t_k.append(np.ones(support.shape[0]))
    elif lap_eig == 'lap':
        t_k.append(np.eye(support.shape[0]))
    t_k.append(support)

    def normal_recurrence(support, t_k_minus_one):
        if lap_eig == 'eig':
            return np.multiply(support, t_k_minus_one)
        elif lap_eig == 'lap':
            return np.matmul(support, t_k_minus_one)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, support):
        if lap_eig == 'eig':
            return 2 * np.multiply(support, t_k_minus_one) - t_k_minus_two
        elif lap_eig == 'lap':
            return 2 * np.matmul(support, t_k_minus_one) - t_k_minus_two

    for i in range(2, orders + 1):
        if poly_style == 'cheby':
            t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], support))
        elif poly_style == 'norm':
            t_k.append(normal_recurrence(t_k[-1], support))

    return np.array(t_k)


def gen_laplacian(data_num=DATA_NUM, opt=27, cache=False):
    label = None

    if cache:
        print('Loading cached graph')
        graph = pk.load(open('tmp/g.pk', 'rb'))
    else:
        print('Generating graph opt {}'.format(opt))

        if 1 == opt:
            graph = gen_rand_graph(data_num=data_num)
        if 2 == opt:
            top_num = random.randint(1, data_num)
            bottom_num = data_num - top_num
            graph = nx.bipartite.random_graph(top_num, bottom_num, 0.9)
            label = [d['bipartite'] for n, d in graph.nodes(data=True)]
        elif 3 == opt:
            graph = nx.balanced_tree(4, 5)
        elif 4 == opt:
            graph = nx.complete_graph(data_num)
        elif 5 == opt:
            no1 = random.randint(1, data_num)
            no2 = random.randint(1, int(data_num / no1))
            no3 = data_num / no1 / no2
            graph = nx.complete_multipartite_graph(no1, no2, no3)
        elif 6 == opt:
            graph = nx.circular_ladder_graph(data_num)
        elif 7 == opt:
            graph = nx.cycle_graph(data_num)
        elif 8 == opt:
            graph = nx.dorogovtsev_goltsev_mendes_graph(5)
        elif 9 == opt:
            top_num = int(random.random() * data_num)
            bottom_num = data_num / top_num
            graph = nx.grid_2d_graph(top_num, bottom_num)
        elif 10 == opt:
            no1 = random.randint(1, data_num)
            no2 = random.randint(1, int(data_num / no1))
            no3 = data_num / no1 / no2
            graph = nx.grid_graph([no1, no2, no3])
        elif 11 == opt:
            graph = nx.hypercube_graph(10)
        elif 12 == opt:
            graph = nx.ladder_graph(data_num)
        elif 13 == opt:
            top_num = int(random.random() * data_num)
            bottom_num = data_num - top_num
            graph = nx.lollipop_graph(top_num, bottom_num)
        elif 14 == opt:
            graph = nx.path_graph(data_num)
        elif 15 == opt:
            graph = nx.star_graph(data_num)
        elif 16 == opt:
            graph = nx.wheel_graph(data_num)
        elif 17 == opt:
            graph = nx.margulis_gabber_galil_graph(35)
        elif 18 == opt:
            graph = nx.chordal_cycle_graph(data_num)
        elif 19 == opt:
            graph = nx.fast_gnp_random_graph(data_num, random.random())
        elif 20 == opt:  # jump eigen value
            graph = nx.gnp_random_graph(data_num, random.random())
        elif 21 == opt:  # disconnected graph
            graph = nx.dense_gnm_random_graph(data_num, data_num / 2)
        elif 22 == opt:  # disconnected graph
            graph = nx.gnm_random_graph(data_num, data_num / 2)
        elif 23 == opt:
            graph = nx.erdos_renyi_graph(data_num, data_num / 2)
        elif 24 == opt:
            graph = nx.binomial_graph(data_num, data_num / 2)
        elif 25 == opt:
            graph = nx.newman_watts_strogatz_graph(data_num, 5, random.random())
        elif 26 == opt:
            graph = nx.watts_strogatz_graph(data_num, 5, random.random())
        elif 26 == opt:  # smooth eigen
            graph = nx.connected_watts_strogatz_graph(data_num, 5, random.random())
        elif 27 == opt:  # smooth eigen
            graph = nx.random_regular_graph(5, data_num)
        elif 28 == opt:  # smooth eigen
            graph = nx.barabasi_albert_graph(data_num, 5)
        elif 29 == opt:  # smooth eigen
            graph = nx.powerlaw_cluster_graph(data_num, 5, random.random())
        elif 30 == opt:  # smooth eigen
            graph = nx.duplication_divergence_graph(data_num, random.random())
        elif 31 == opt:
            p = random.random()
            q = random.random()
            graph = nx.random_lobster(data_num, p, q)
        elif 32 == opt:
            p = random.random()
            q = random.random()
            k = random.random()

            graph = nx.random_shell_graph([(data_num / 3, 50, p), (data_num / 3, 40, q), (data_num / 3, 30, k)])
        elif 33 == opt:  # smooth eigen
            top_num = int(random.random() * data_num)
            bottom_num = data_num - top_num
            graph = nx.k_random_intersection_graph(top_num, bottom_num, 3)
        elif 34 == opt:
            graph = nx.random_geometric_graph(data_num, .1)
        elif 35 == opt:
            graph = nx.waxman_graph(data_num)
        elif 36 == opt:
            graph = nx.geographical_threshold_graph(data_num, .5)
        elif 37 == opt:
            top_num = int(random.random() * data_num)
            bottom_num = data_num - top_num
            graph = nx.uniform_random_intersection_graph(top_num, bottom_num, .5)

        elif 39 == opt:
            graph = nx.navigable_small_world_graph(data_num)
        elif 40 == opt:
            graph = nx.random_powerlaw_tree(data_num, tries=200)
        elif 41 == opt:
            graph = nx.karate_club_graph()
        elif 42 == opt:
            graph = nx.davis_southern_women_graph()
        elif 43 == opt:
            graph = nx.florentine_families_graph()
        elif 44 == opt:
            graph = nx.complete_multipartite_graph(data_num, data_num, data_num)

        # OPT 1
        # norm_lap = nx.normalized_laplacian_matrix(graph).toarray()

        # OPT 2: renormalized

        # pk.dump(graph, open('tmp/g.pk', 'wb'))

    # plot_graph(graph, label)
    # note difference: normalized laplacian and normalzation by eigenvalue
    norm_lap, eigval, eigvec = normalize_lap(graph)

    return graph, norm_lap, eigval, eigvec

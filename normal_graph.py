
import numpy as np

from func import plot_graph, gen_laplacian, jump_func, plot_single_signal


def graph_forge(opt=1, ret_recur='', orders=4, poly_style='cheby',
                load_style=1, norm_style=4, selected_index=3,
                mute_plot=True, regr=False):
    print('load_style: {}'.format(load_style))
    global DATA_NUM

    print('Generating synthetic data')
    ########
    # generate graph
    ########
    graph, norm_lap, eigval, eigvec = gen_laplacian(opt=opt, cache=False)

    ########
    # generate x/Y
    ########
    feat = np.ones((eigvec.shape[0], 1))
    y = jump_func(eigval, opt=12)  # make jump signal
    feat_spectral = np.dot(eigvec.T, feat)
    label = np.dot(eigvec, np.dot(np.diag(y), feat_spectral))

    ########
    # plot eigenvalue, filter function of eigenvalue, spectral transferred x/Y
    ########
    plot_single_signal(eigval, eigvec, label, feat)

    feat = feat if isinstance(feat, np.ndarray) else feat.A

    ########
    # max id is the id for plotting, which has largest abs value (the largest influence eigenvector)
    ########
    g_func = np.abs(np.dot(eigvec.T, label) / np.dot(eigvec.T, feat))
    max_id = (-g_func.flatten()).argsort()[:3]

    ########
    # plot graph, selected eigenvectors
    ########
    plot_graph(graph, label, cache=False, max_id=max_id)


if __name__ == '__main__':
    graph_forge()

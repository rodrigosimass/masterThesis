import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix


def plot_multiple_line_charts(
    x_l,
    y_l_l,
    label_l,
    xlabel,
    ylabel,
    title="multiple_runs",
    path="img/multiple_runs.png",
):
    for y_l, label in zip(y_l_l, label_l):
        plt.plot(x_l, y_l, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.ylim((0, 100))
    plt.legend()
    plt.savefig(path)


def retrieve_hist(codes, W, bins=100, density=True):

    s = csr_matrix.dot(codes, W)
    max = np.max(s, axis=-1)
    plt.hist(max.toarray(), bins=bins, density=density)


def dentritic_potential_hist(codes, W, bins=100, density=True):

    s = csr_matrix.dot(codes, W)
    plt.hist(s, bins=bins, density=density)

from math import log2
import numpy as np
from scipy.stats import entropy
from .plot import *

def distributions(dataset):
    activity = np.count_nonzero(dataset == 1, axis=0)
    dist = activity / np.count_nonzero(dataset == 1)
    unif = np.full_like(dist, 1 / dist.size)
    return dist, unif


def dist_difference(d1, d2):
    """
    Measures the mean difference between two distributions

    @param d1: 1d array of a distribution (should sum to 1)
    @param d2: 1d array of a distribution (should sum to 1)

    @return distance: distance between the two distributions
    """
    diff = d1 - d2
    abs = np.absolute(diff)
    total = sum(abs)
    return total / abs.size


def dist_difference_set(dataset):
    """
    Dataset interface
    """
    d1, d2 = distributions(dataset)
    return dist_difference(d1, d2)


def op1(p, q):
    """
    Operator for the KL-div.
    Does log2(p/q) if both p and q are not zero, else returns 0.
    """
    if p == 0 or q == 0:
        return 0
    else:
        return log2(p / q)


def kl_divergence(p, q, verbose=0):
    """
    Computes KL(p || q)
    """
    if verbose > 1:
        print(f"p={p}\nq={q}")

    k = sum(p[i] * op1(p[i], q[i]) for i in range(len(p)))

    if verbose > 0:
        print("KL(P || Q): %.3f bits" % k)
    return k


def kl_divergence_set(dataset, verbose=0):
    """
    Dataset interface
    """
    p, q = distributions(dataset)
    return kl_divergence(p, q, verbose)


def shannon_entropy(dist, b=2):
    e = entropy(dist, base=b)
    return e


def shannon_entropy_set(dataset, b=2):
    """
    Dataset interface
    """
    dist, _ = distributions(dataset)
    return shannon_entropy(dist, b)


def measure_data_distribution(p, q, entropy_b=2, verbose=False):
    """
    Measures how well-distributed a dataset is
    @param dataset: dataset
    @param entropy_b: base for the shannon entropy

    @return: distance to uniform, kl-divergence, and shannon entropy
    """
    d = dist_difference(p, q)
    kl = kl_divergence(p, q)
    e = shannon_entropy(p, b=entropy_b)
    if verbose:
        print(f"d_unif={d:.5f}, KL_div={kl:.5f}, shannon_e={e:.5f}")
    return (d, kl, e)


def measure_data_distribution_set(dataset, entropy_b=2, verbose=False):
    """
    Dataset interface
    """
    p, q = distributions(dataset)
    return measure_data_distribution(p, q, entropy_b, verbose)


"""
Run this file to test if methods are correctly implemented
"""
if __name__ == "__main__":

    u = np.array([0.25, 0.25, 0.25, 0.25])

    q1 = np.array([0.1, 0.1, 0.1, 0.7])
    q2 = np.array([0.15, 0.15, 0.15, 0.55])
    q3 = np.array([0.2, 0.2, 0.2, 0.4])
    q4 = np.array([0.24, 0.26, 0.24, 0.26])

    l_dists = [q1, q2, q3, q4, u]

    d1, kl1, e1 = measure_data_distribution(q1, u, verbose=True)
    d2, kl2, e2 = measure_data_distribution(q2, u, verbose=True)
    d3, kl3, e3 = measure_data_distribution(q3, u, verbose=True)
    d4, kl4, e4 = measure_data_distribution(q4, u, verbose=True)
    d5, kl5, e5 = measure_data_distribution(u, u, verbose=True)

    l_d = [d1, d2, d3, d4, d5]
    l_kl = [kl1, kl2, kl3, kl4, kl5]
    l_e = [e1, e2, e3, e4, e5]

    plot_dists(l_dists, l_d, l_kl, l_e)

    dset = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])
    print(f"dset shape= {dset.shape}")
    q, u = distributions(dset)

    print(
        """
    using p and q:
    """
    )
    print(f"q={q}")
    print(f"u={u}")

    dist = dist_difference(q, u)
    print(f"dist to unif = {dist}")

    kl = kl_divergence(q, u)
    print(f"kl = {kl}")

    e = shannon_entropy(q, 2)
    print(f"shannon entropy = {e}")

    print("--------------------------")

    dist = dist_difference_set(dset)
    print(f"dist to unif = {dist}")

    kl = kl_divergence_set(dset)
    print(f"kl = {kl}")

    e = shannon_entropy_set(dset, b=2)
    print(f"shannon entropy = {e}")

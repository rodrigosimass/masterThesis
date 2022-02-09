import numpy as np
from scipy.sparse import csr_matrix


def compute_dist(codes):
    d = np.mean(codes, axis=0)
    return np.array(d).reshape(
        -1,
    )


def compute_dists(codes, lbls, n_classes=10):
    """
    Computes de probability of activity for eapch dimention of the codes, classwise.
    @returns dists: (2d np array) with the distribution of each class
    """
    dists = np.empty((n_classes, codes.shape[1]))

    for c in range(n_classes):
        dists[c] = compute_dist(codes[lbls == c])
    return dists


def avg_bits_per_code(codes):
    nnz = codes.nnz
    n_codes = codes.shape[0]
    return nnz / n_codes


def sample_from_dist(dist, n=1):
    samples = np.random.rand(n, dist.shape[0])
    samples = samples - dist  # subtracts the dist from each row
    samples[samples >= 0] = 0
    samples[samples < 0] = 1
    return samples


def sample_from_dists(dists, gen_lbls):
    """
    @param dists: distribution for each class
    """
    n_samples = gen_lbls.shape[0]
    code_size = dists.shape[1]
    samples = np.empty((n_samples, code_size))

    for i in range(n_samples):
        lbl = gen_lbls[i]
        dist = dists[lbl]
        s = sample_from_dist(dist).flatten()
        samples[i] = s

    return csr_matrix(samples)


def create_gen_lbls(n_classes=10, n_exs=10, transpose=False):
    gen_lbls = np.tile(np.arange(n_classes), n_exs).reshape(n_exs, n_classes)
    if transpose:
        gen_lbls = gen_lbls.T
    return gen_lbls.flatten()

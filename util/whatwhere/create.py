import numpy as np
from scipy.sparse import csr_matrix


def compute_prob_dist(codes):
    d = np.mean(codes, axis=0)
    return np.array(d).reshape(
        -1,
    )


def avg_bits_per_code(codes):
    nnz = codes.nnz
    num_codes = codes.shape[0]
    return nnz / num_codes


def sample_from_dist(dist, n=1):
    samples = np.random.rand(n, dist.shape[0])
    samples = samples - dist  # subtracts the dist from each row
    samples[samples >= 0] = 0
    samples[samples < 0] = 1
    return csr_matrix(samples)

from numpy.core.numeric import False_
from scipy.sparse import csr_matrix
import numpy as np


def simple_1NN_classifier(ret, codes, trn_lbls, verbose=False):
    sim = csr_matrix.dot(ret, codes.T)
    nn = csr_matrix.argmax(sim, axis=1)

    num_stored = ret.shape[0]

    prediction = trn_lbls[nn]
    solution = trn_lbls[:num_stored]
    diff = prediction.flatten() - solution.flatten()

    errors = np.count_nonzero(diff)
    error_rate = errors / num_stored

    if verbose:
        print(
            f"1NN accuracy = {diff.shape[0] - errors}/{diff.shape[0]} ; Error rate: {error_rate}"
        )
    return error_rate
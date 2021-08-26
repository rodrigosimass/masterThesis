from numpy.core.numeric import False_
from scipy.sparse import csr_matrix
import numpy as np


def simple_1NN_classifier(ret_csr, codes_csr, trn_lbls, num_stored, verbose=False):
    sim = csr_matrix.dot(ret_csr, codes_csr.T)
    nn = csr_matrix.argmax(sim, axis=1)

    prediction = trn_lbls[nn]
    solution = trn_lbls[:num_stored]
    diff = prediction.flatten() - solution.flatten()

    errors = np.count_nonzero(diff)

    if verbose:
        print(
            f"""
        Total number of patterns: {diff.shape[0]}
        Correct: {diff.shape[0] - errors}
        Wrong: {errors}
        Error rate: {errors / diff.shape[0]}
        """
        )
    error_rate = errors / num_stored
    return error_rate
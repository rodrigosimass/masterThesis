import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csr_matrix, vstack
from .plot import *
from .classifier import *


def train(codes_csr, num_stored, verbose=False):

    n = codes_csr.shape[1]  # size of patterns to Memorize
    if verbose:
        print("codes_set shape: ", codes_csr.shape)
        print("training willshaw of size ", n, "*", n)

    addresses = codes_csr[:num_stored]
    will = np.zeros((n, n))

    for a in addresses:  # for each pattern to store in will
        num_nz = len(a.indices)
        for i in range(num_nz):
            # nz has the indexes of x that are non-zero
            for j in range(i, num_nz):
                idx_i = a.indices[i]
                idx_j = a.indices[j]
                will[idx_i, idx_j] = 1
                will[idx_j, idx_i] = 1

    return csr_matrix(will)


def incremental_train(new_data, current_W=None):

    if current_W == None:
        n = new_data.shape[1]  # size of patterns to Memorize
        will = np.zeros((n, n))
    else:
        will = current_W.toarray()

    for pattern in new_data:  # for each pattern to store in will
        num_nz = len(pattern.indices)
        for i in range(num_nz):
            # nz has the indexes of x that are non-zero
            for j in range(i, num_nz):
                idx_i = pattern.indices[i]
                idx_j = pattern.indices[j]
                will[idx_i, idx_j] = 1
                will[idx_j, idx_i] = 1

    will = csr_matrix(will)
    sparsity = will.nnz / (will.shape[0] * will.shape[1])

    return (will, sparsity)


def H(vec):
    vec[vec >= 0] = 1
    vec[vec < 0] = 0
    return vec


def incremental_retreive(new_cues, W, prev_ret):

    ret = np.zeros(new_cues.shape)

    s = csr_matrix.dot(new_cues, W)

    for i in range(new_cues.shape[0]):  # for all retreival cues
        aux = s[i].toarray()
        m = np.max(aux)
        if m != 0:
            aux1 = aux - m
            ret[i] = H(aux1)
        else:
            # zero activations
            ret[i] = 0

    ret = csr_matrix(ret)

    AS = ret.nnz / (ret.shape[0] * ret.shape[1])
    densest = np.max(csr_matrix.sum(ret, axis=1)) / ret.shape[1]

    if prev_ret != None:
        ret = vstack((prev_ret, ret))

    return ret, AS, densest


def retreive(codes, num_stored, W):

    codes = codes[:num_stored]

    ret = np.zeros(codes.shape)

    s = csr_matrix.dot(codes, W)

    for i in range(codes.shape[0]):  # for all retreival cues
        aux = s[i].toarray()
        m = np.max(aux)
        if m != 0:
            aux1 = aux - m
            ret[i] = H(aux1)
        else:
            # zero activations
            ret[i] = 0

    ret = csr_matrix(ret)
    AS = ret.nnz / (ret.shape[0] * ret.shape[1])
    densest = np.max(csr_matrix.sum(ret, axis=1)) / ret.shape[1]

    return ret, AS, densest


def performance_perfect_ret(codes, ret, verbose=False):
    hit = 0
    miss = 0
    for i in range(ret.shape[0]):
        if (codes[i] != ret[i]).nnz != 0:
            miss += 1
        else:
            hit += 1
    if verbose:
        print(f"Perfection Performance: {hit}/{hit+miss}")
    return miss / (hit + miss)


def performance_avg_error(codes, ret, verbose=False):
    errors = 0
    for i in range(ret.shape[0]):
        errors += np.sum(codes[i] != ret[i])

    avg_error = errors / (ret.shape[0] * ret.shape[1])
    if verbose:
        print(f"Avg Error: {avg_error }")

    return avg_error


def performance_loss_noise(codes, ret, verbose=False):
    loss = 0
    noise = 0
    for i in range(ret.shape[0]):
        diff = (codes[i] - ret[i]).toarray()
        loss += np.count_nonzero(diff == 1)
        noise += np.count_nonzero(diff == -1)

    total = ret.shape[0] * ret.shape[1]

    if verbose:
        print(f"Information loss: {loss} ; Added nosie: { noise}")

    return (loss / total, noise / total)


def performance(codes, ret, trn_lbls, verbose=False):
    err_avg = performance_avg_error(codes, ret, verbose)
    err_loss, err_noise = performance_loss_noise(codes, ret, verbose)
    err_perf = performance_perfect_ret(codes, ret, verbose)
    err_1nn = simple_1NN_classifier(ret, codes, trn_lbls, verbose=True)
    return (err_perf, err_avg, err_loss, err_noise, err_1nn)

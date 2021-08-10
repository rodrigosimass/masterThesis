import numpy as np
import random
import matplotlib.pyplot as plt
import time
import pickle
from scipy.sparse import csr_matrix
from plots import *


def H(vec):
    vec[vec >= 0] = 1
    vec[vec < 0] = 0
    return vec


def retreive(codes, factor_of_stored, W):

    num_stored = 784 * factor_of_stored

    codes = codes[:num_stored]

    ret = np.zeros(codes.shape)

    s = csr_matrix.dot(codes,W).astype(np.int64)

    for i in range(codes.shape[0]):  # for all retreival cues
        aux = s[i].toarray()
        m = np.max(aux)
        aux1 = aux - m
        ret[i] = H(aux1)

    return ret


def performance(cues, ret):
    hit = 0
    miss = 0
    for i in range(cues.shape[0]):
        if np.array_equal(cues[i], ret[i]):
            hit += 1
        else:
            miss += 1
    print(f"Performance: {hit}/{hit+miss}")


# X_trn: binary trainning set; M:
def train(codes_csr, num_stored, verbose=0):
    print("codes_csr shape: ", codes_csr.shape)

    n = codes_csr.shape[1]  # size of patterns to Memorize
    print("training willshaw of size ", n, "*", n)

    addresses = codes_csr[:num_stored]
    print("addresses shape: ", addresses.shape)
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

    return will


def load_or_compute_will(run_name, codes_csr, factor_of_stored):
    num_stored = codes_csr.shape[1] * factor_of_stored
    try:
        will = pickle.load(
            open(f"whatwhere/pickles/{run_name}_fac{factor_of_stored}__will.p", "rb")
        )
        will_csr = csr_matrix(will, dtype=np.ushort)
        print(
            f"loaded will from pickle: pickles/{run_name}_fac{factor_of_stored}__will.p"
        )
    except (OSError, IOError) as _:
        will_np = train(codes_csr, num_stored)
        will_csr = csr_matrix(will_np, dtype=np.ushort)
        pickle.dump(
            will_csr,
            open(f"whatwhere/pickles/{run_name}_fac{factor_of_stored}__will.p", "wb"),
        )
        print(
            f"saving trained willshaw to pickles/{run_name}_fac{factor_of_stored}__will.p"
        )

    if np.array_equal(will_csr.toarray(), (will_csr.toarray()).T):
        print("[OK] Willshaw matrix is symmetric")

    print(
        f"""W martix sparsity = {will_csr.nnz/ (will_csr.shape[0] * will_csr.shape[1])}
    """
    )

    return will_csr


def load_or_compute_ret(
    trn_imgs,
    features,
    run_name,
    codes,
    will,
    labels,
    k,
    Q,
    factor_of_stored,
    plot=False,
):
    set_id = "R" + "_fac" + str(factor_of_stored)

    try:
        ret = pickle.load(
            open(f"whatwhere/pickles/{run_name}_fac{factor_of_stored}__ret.p", "rb")
        )
        ret_csr = csr_matrix(ret, dtype=np.ushort)
        print(
            f"loaded ret from pickle: pickles/{run_name}_fac{factor_of_stored}__ret.p"
        )
    except (OSError, IOError) as _:
        ret = retreive(codes, factor_of_stored, will)
        ret_csr = csr_matrix(ret, dtype=np.ushort)
        pickle.dump(
            ret_csr,
            open(f"whatwhere/pickles/{run_name}_fac{factor_of_stored}__ret.p", "wb"),
        )
        print(f"saving ret to pickles/{run_name}_fac{factor_of_stored}__ret.p")

    if plot:
        plot_examples(trn_imgs, ret_csr, features, k, Q, run_name, set_id, num_examples=3)
        plot_mnist_codes_activity(trn_imgs, ret_csr, k, Q, run_name, set_id)
        plot_feature_maps(ret_csr, k, Q, run_name, set_id)
        plot_feature_maps_overlaped(trn_imgs, ret_csr, k, Q, run_name, set_id)
        plot_class_activity_2D(ret_csr, labels, k, Q, run_name, set_id)
        plot_class_activity_1D(ret_csr, labels, k, Q, run_name, set_id)
        plot_sparse_dense_examples(trn_imgs, ret_csr, features, k, Q, run_name, set_id)
        plot_sparsity_distribution(ret_csr, k, Q, run_name, set_id)

    print(
        f"""Retrieved set sparsity = {ret_csr.nnz/ (ret_csr.shape[0] * ret_csr.shape[1])}
    """
    )

    return ret_csr
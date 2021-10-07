from math import log2
import numpy as np

# does log2(p/q) if both p and q are not zero, else returns 0.
def op1(p, q):
    if p == 0 or q == 0:
        return 0
    else:
        return log2(p / q)


# KL(P || Q)
def kl_divergence(p, q):
    return sum(p[i] * op1(p[i], q[i]) for i in range(len(p)))


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# p distribution is NOT normalized (sum might not yield 1.0)
def kl_div_set(patterns, verbose=0):
    activity = np.count_nonzero(patterns == 1, axis=0)
    p = activity / patterns.shape[0]
    if verbose > 1:
        print("p:")
        print(p)
        print(f"sum = {np.sum(p)}")

    q = np.full(p.shape, np.sum(p) / p.size)
    if verbose > 1:
        print("q:")
        print(q)
        print(f"sum = {np.sum(q)}")

    kl_pq = kl_divergence(p, q)
    if verbose > 0:
        print("KL(P || Q): %.3f bits" % kl_pq)

    return kl_pq


# p distribution is normalized (sum yields 1.0)
def kl_div_set_2(patterns, verbose=False):
    activity = np.count_nonzero(patterns == 1, axis=0)
    p = activity / sum(activity)
    if verbose:
        print("p:")
        print(p)
        print(f"sum = {np.sum(p)}")

    sparsity = np.count_nonzero(patterns == 1) / patterns.size

    q = np.full(p.shape, sparsity)
    if verbose:
        print("q:")
        print(q)
        print(f"sum = {np.sum(q)}")

    kl_pq = kl_divergence(p, q)
    if verbose:
        print("KL(P || Q): %.3f bits" % kl_pq)

    return kl_pq
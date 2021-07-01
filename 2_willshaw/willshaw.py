import numpy as np
import random
import matplotlib.pyplot as plt


def H(vec):
    for i in range(len(vec)):
        if vec[i] < 0:
            vec[i] = 0
        else:
            vec[i] = 1
    return vec


def generate_random_sparse_patterns(M, n, sparsity):
    patterns = np.zeros((M, n))
    for i in range(M):
        for j in range(n):
            roll = random.uniform(0, 1)
            if roll <= sparsity:
                patterns[i][j] = 1
    return patterns


def plot_data_histogram(patterns, M, s):
    freqs = np.sum(patterns, axis=0)
    freqs /= M
    idx = np.arange(n)

    plt.bar(idx, freqs, align="center", alpha=0.5)
    plt.ylabel("Frequency of activation")
    plt.xlabel("Index of the pattern")
    plt.title(f"Activity in the stored patterns (sparsity={s})")
    plt.savefig("data_hist.png")


def train(M, n, patterns):
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            correlation = 0
            for p in patterns:
                correlation += p[i] * p[j]
            W[i][j] = min(1, correlation)
    return W


def retreive(cues, W):
    ret = np.empty_like(cues)

    for i in range(cues.shape[0]):  # for all retreival cues
        s = W.dot(cues[i])
        print("s:")
        print(s)
        ret[i] = H(s - max(s))
        print("y:")
        print(ret[i])
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


""" parameters """
M = 2  # number of patterns
n = 4  # size of patterns
sparsity = 0.1

p1 = np.array([0, 0, 1, 1])
p2 = np.array([1, 1, 0, 0])
patterns = np.vstack((p1, p2))
print(patterns)
cues = np.array([[1, 0, 1, 1], [1, 0, 0, 0]])
""" patterns = generate_random_sparse_patterns(M, n, sparsity)
cues = np.copy(patterns)  # auto association
plot_data_histogram(patterns, M, sparsity) """

W = train(M, n, patterns)
print("W:")
print(W)
ret = retreive(cues, W)
print(ret)

# performance(cues, ret)

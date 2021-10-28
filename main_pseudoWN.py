import numpy as np
import random
import matplotlib.pyplot as plt
import util.willshaw.memory as WN
from scipy.sparse import csr_matrix
from models.MLP import *


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


def train(M, n, patterns, verbose=False):
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            correlation = 0
            for p in patterns:
                correlation += p[i] * p[j]
            W[i][j] = min(1, correlation)
    if verbose:
        print(f"W ({W.shape}):")
        print(W)

    return W


def retreive(cues, W, verbose=False):
    ret = np.empty_like(cues)

    s = np.dot(cues, W)
    for i in range(cues.shape[0]):  # for all retreival cues
        m = np.max(s[i])
        if m != 0:
            ret[i] = H(s[i] - m)
        else:
            ret[i] = 0

        if verbose:
            print("s:")
            print(s)
            print("y:")
            print(ret[i])

    return ret, s


def plot_dendritic_potential(s):

    fig, axs = plt.subplots(1, 1)

    im1 = axs.imshow(s, cmap="viridis")
    fig.colorbar(im1, orientation="vertical")

    plt.savefig("img/willshaw/dendritic_potential.png")


def plot_dendritic_potential_hist(s):
    s = np.max(s, axis=1)
    M = np.amax(s)
    m = np.amin(s)
    bins = np.arange(m, M + 1)
    print(bins)

    plt.hist(s, bins=bins, density=True, edgecolor="k")
    plt.ylabel("Frequency")
    plt.xlabel("dendritic potential")
    plt.savefig("img/willshaw/dendritic_potential_max_hist.png")


p1 = np.array([0, 0, 1, 1])
p2 = np.array([1, 1, 0, 0])
patterns = np.vstack((p1, p2))
cues = np.array([[1, 0, 1, 1], [1, 0, 0, 0]])

""" parameters """

M = 100  # number of patterns
n = 100  # size of patterns
sparsity = 0.01
patterns = generate_random_sparse_patterns(M, n, sparsity)


verbose = False


wn1 = train(patterns.shape[0], patterns.shape[1], patterns, verbose)
ret1, s = retreive(patterns, wn1, verbose)

plot_dendritic_potential_hist(s)
plot_dendritic_potential(s)


""" wn2, _ = WN.incremental_train(csr_matrix(patterns), None)
ret2, _, _ = WN.retreive(csr_matrix(patterns), patterns.shape[0], wn2)
wn2 = wn2.toarray()
ret2 = ret2.toarray()

diff1 = patterns - ret1
diff2 = patterns - ret2

print(np.array_equal(wn1, wn2))
print(np.array_equal(ret1, ret2))

print(
    f"C-R:\n0: {diff1.size - np.count_nonzero(diff1)}\n1: {np.count_nonzero(diff1 == 1)}\n-1: {np.count_nonzero(diff1 == -1)}\n"
)

size = patterns.shape[1]
weight = torch.from_numpy(wn1)
bias = torch.full((1, size), -1)

model = pseudoWN(size, weight, bias)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

tensor = torch.from_numpy(patterns)
optimizer.zero_grad()  # Zero the gradients
outputs = model(tensor)
out_np = outputs.detach().numpy()
print("out_np=", out_np)
diff2 = patterns - out_np

print(
    f"C-R:\n0: {diff2.size - np.count_nonzero(diff2)}\n1: {np.count_nonzero(diff2 == 1)}\n-1: {np.count_nonzero(diff2 == -1)}\n"
) """
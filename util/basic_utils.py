import numpy as np
import math


def binary_sparsity(X):
    return np.count_nonzero(X) / X.size


def best_layout(N):
    best = int(math.sqrt(N))
    while N % best != 0:
        best -= 1
    return best, int(N / best)


def mse(pred, truth):
    difference_array = np.subtract(pred, truth)
    squared_array = np.square(difference_array)
    return squared_array.mean()

#%%
import numpy as np
import math
import torch
from torch import nn
import time


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


def mse_detailed(pred, truth):
    """
    Measures the negative and positive squared error.
    In order to convert the result of this function to MSE do:
    (n+p)/(28*28) (input: 28 by 28 images)

    @param pred: flattened np array with the prediction
    @param truth: flattened np array with the ground truth

    @return neg: squared error due to extra info
    @return pos: squared error due to lost info
    """
    num_patterns = truth.shape[0]
    pred = pred.reshape(num_patterns, -1)
    truth = truth.reshape(num_patterns, -1)

    pattern_size = truth.shape[1]

    diff = np.subtract(truth, pred)

    neg = np.square(diff[diff < 0])
    pos = np.square(diff[diff >= 0])

    neg = neg.sum() / num_patterns if neg.size != 0 else 0
    pos = pos.sum() / num_patterns if pos.size != 0 else 0

    mse = (neg + pos) / pattern_size

    return (neg, pos, mse)


def mse_torch(pred, truth):
    """
    Measures the Mean squared error between two arrays

    @param pred: flattened np array with the prediction
    @param truth: flattened np array with the ground truth

    @return mse: mean swared error between pred and truth
    """
    pred = torch.from_numpy(pred)
    truth = torch.from_numpy(truth)
    mse = nn.MSELoss()

    loss = mse(pred.float(), truth.float())
    # TODO:
    return loss.item()


# %%

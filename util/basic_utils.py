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


def mse_torch(pred, truth):
    """
    Measures the Mean squared error between two arrays

    @param pred: np array with the prediction
    @param truth: np array with the ground truth

    @return mse: mean swared error between pred and truth
    """
    pred = torch.from_numpy(pred)
    truth = torch.from_numpy(truth)
    mse = nn.MSELoss()

    loss = mse(pred.float(), truth.float())

    return loss.item()


if __name__ == "__main__":

    x = np.arange(10)
    y = np.arange(10) + 1

    print(mse(x, y))
    print(mse_torch(x,y))

    a = np.random.rand(600000, 28, 28)
    b = np.random.rand(600000, 28, 28)

    start = time.time()
    print(mse(a, b))
    end = time.time()
    print(f"Regualr MSE done in {end-start:.2f}")

    start = time.time()
    print(mse_torch(a, b))
    end = time.time()
    print(f"Torch MSE done in {end-start:.2f}")

import numpy as np
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm, trange
import random


def add_zero_noise(data, prob=0.1):

    noisy = np.copy(data.toarray())

    for i in trange(noisy.shape[0], desc="adding zero-noise", unit="data-sample"):
        ones = np.argwhere(noisy[i] != 0)
        for j in range(len(ones)):
            if prob > random.random():
                noisy[i][ones[j]] = 0

    return csr_matrix(noisy)


def add_one_noise(data, prob=0.1):

    noisy = data.toarray()

    for i in trange(noisy.shape[0], desc="adding one-noise", unit="data-sample"):
        ones = np.argwhere(noisy[i] == 0)
        for j in range(len(ones)):
            if prob >= random.random():
                noisy[i][ones[j]] = 1

    return csr_matrix(noisy)

import numpy as np
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm, trange
import random


def add_zero_noise(data, prob=0.1):
    """
    Adds noise to a binary dataset.
    Each 1 in the dataset will become a 0 with the given probability.

    @param data: 2-Dim csr matrix
    @param prob: probability of turning a 1 into a 0

    @return noisy: noisy version of the data
    """
    noisy = np.copy(data.toarray())

    for i in trange(noisy.shape[0], desc="adding noise", unit="data-sample"):
        ones = np.argwhere(data[i] != 0)
        for j in range(len(ones)):
            if prob > random.random():
                noisy[i][ones[j]] = 0

    return csr_matrix(noisy)


def add_one_noise(data, prob=0.1):
    """
    Adds noise to a binary dataset.
    Each 0 in the dataset will become a 1 with the given probability.

    @param data: 2-Dim csr matrix
    @param prob: probability of turning a 0 into a 1

    @return noisy: noisy version of the data
    """
    noisy = np.copy(data.toarray())

    for i in trange(noisy.shape[0], desc="adding noise", unit="data-sample"):
        zeros = np.argwhere(data[i] == 0)
        for j in range(len(zeros)):
            if prob > random.random():
                noisy[i][zeros[j]] = 1

    return csr_matrix(noisy)


def add_one_noise_relative(data, Prepl=0.1):
    """
    Adds noise to a binary dataset.
    Each 1 in the dataset will have a given probability to "replicate"
    and originate another 1 in a random position.

    @param data: 2-Dim csr matrix
    @param Prepl: probability that a 1 replicates

    @return noisy: noisy version of the data
    """
    noisy = np.copy(data.toarray())

    for i in trange(noisy.shape[0], desc="adding noise", unit="data-sample"):
        ones = np.argwhere(noisy[i] != 0)
        for _ in range(len(ones)):
            if Prepl > random.random():
                zeros = np.argwhere(noisy[i] == 0)
                noisy[i][np.random.permutation(zeros)[0]] = 1

    return csr_matrix(noisy)


def add_noise(codes, noise_type="none", prob=0.1):
    if noise_type == "zero":
        codes_noisy = add_zero_noise(codes, prob)
    elif noise_type == "one":
        codes_noisy = add_one_noise_relative(codes, prob)
    elif noise_type == "none":
        codes_noisy = codes
    return codes_noisy

import numpy as np
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm, trange
import random


def add_zero_noise(codes, prob=0.1):

    codes = codes.toarray()

    for i in trange(codes.shape[0], desc="adding zero-noise", unit="data-sample"):
        ones = np.argwhere(codes[i] != 0)
        for j in range(len(ones)):
            if prob > random.random():
                codes[i][ones[j]] = 0

    return csr_matrix(codes)


def add_one_noise(codes, prob=0.1):

    codes = codes.toarray()

    for i in trange(codes.shape[0], desc="adding one-noise", unit="data-sample"):
        ones = np.argwhere(codes[i] == 0)
        for j in range(len(ones)):
            if prob >= random.random():
                codes[i][ones[j]] = 1

    return csr_matrix(codes)

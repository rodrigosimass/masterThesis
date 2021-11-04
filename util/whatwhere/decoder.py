import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.util import view_as_windows
import pickle
from scipy.sparse import csr_matrix
from .plot import *
import sys
import torchvision as torchv
import torch
from tqdm import trange
from tqdm import tqdm


def reconstruct_set(codes, features, Q, K):
    recon = np.zeros((codes.shape[0], Q, Q))
    codes = codes.reshape((-1, Q, Q, K))
    for i in range(codes.shape[0]):
        recon[i] = reconstruct(codes[i], features)
    return recon


def reconstruct(code, features):
    reconstruction_kernels = np.flip(np.flip(features, axis=2), axis=1)
    reconstruction_kernels = np.moveaxis(reconstruction_kernels, (0, 1, 2), (2, 0, 1))
    Q, _, k = code.shape
    f, _, _ = reconstruction_kernels.shape
    p = f // 2
    x_windows = np.squeeze(
        view_as_windows(np.pad(code, ((p, p), (p, p), (0, 0))), (f, f, k), (1, 1, k))
    )
    x_contributions = x_windows * reconstruction_kernels
    weight = x_windows.reshape(Q, Q, -1).sum(axis=-1)
    weight[weight == 0] = 1
    vals = x_contributions.reshape(Q, Q, -1).sum(axis=-1)
    r = vals / weight
    r[r < 0] = 0
    return r


def reconstruct_slow(code, features, Q):
    k, f, _ = features.shape
    p = f // 2
    x_padded = np.pad(code, ((p, p), (p, p), (0, 0)))
    x_padded.shape
    r = np.zeros((Q, Q))
    for i in tqdm(range(Q)):
        for j in range(Q):
            i_pad = i + p
            j_pad = j + p
            count = 0
            pixel = 0
            for m in np.arange(-p, p + 1):
                for n in np.arange(-p, p + 1):
                    pos_i = i_pad + m
                    pos_j = j_pad + n
                    k_i = f - (m + p) - 1
                    k_j = f - (n + p) - 1
                    for l in range(k):
                        # if this kernel contributes added the corresponding pixel
                        if x_padded[pos_i, pos_j, l] > 0:
                            count += 1
                            pixel += features[l, k_i, k_j]
            if count > 0:
                r[i, j] = pixel / count
    r[r < 0] = 0
    return r


def recon_grid(recons, size, num_examples=10):
    recons = recons[:num_examples]
    recons = recons.reshape((num_examples, Q, Q, K))
    recons = np.sum(recons, axis=3)
    recons.reshape(num_examples, Q, Q)

    tensor = torch.from_numpy(recons)
    tensor = torch.unsqueeze(tensor, dim=1)
    grid = torchv.utils.make_grid(tensor, normalize=True, nrow=10, pad_value=1)
    return grid

import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.util import view_as_windows
import pickle
from scipy.sparse import csr_matrix
import sys
import torchvision as torchv
import torch
from tqdm import trange
from tqdm import tqdm
from .encoder import translation, scale


def unpack_polar_params(params):
    C, rad = params
    cx, cy = C
    return (cx, cy, rad)


# codes (N,Q,Q,K) -> recons(N,I,J) (image space)
def recon_img_space(codes, polar_params, features, Q, K, I, J):
    recon = np.zeros((codes.shape[0], I, J))
    codes = codes.reshape((-1, Q, Q, K))
    for i in trange(
        codes.shape[0], desc="reconstructing", unit="datasamples", leave=False
    ):
        params = polar_params[i]
        pol = ungrid(codes[i])
        ret = unpolar(pol, params)
        h = unenum(ret, I, J, K)
        recon[i] = reconstruct(h, features)
    return recon


# codes (N,Q,Q,K) -> recons(N,Q,Q) (memory space)
def recon_mem_space(codes, features, Q, K):
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


def ungrid(a):
    Q, Q, n_K = a.shape
    k = np.linspace(1, n_K, n_K)
    h = np.flip(np.linspace(-1, 1, Q))
    w = np.linspace(-1, 1, Q)
    W, H, K = np.meshgrid(w, h, k)
    k = K[a != 0]
    h = H[a != 0]
    w = W[a != 0]
    as_set = np.zeros((k.shape[0], 3))
    as_set[:, 0] = w
    as_set[:, 1] = h
    as_set[:, 2] = k
    return as_set


def unpolar(pol, params):
    C, rad = params
    cx, cy = C
    ret = translation(scale(pol, 1.0 / rad), cx, cy)
    # TODO if rad=0 then return empty array
    return ret


def unenum(ret, n_H, n_W, k):
    ret_set = np.zeros((ret.shape[0], 3))
    ret_set[:, 0] = ((ret[:, 1] + 1) / 2) * (n_W - 1)
    ret_set[:, 1] = ((ret[:, 0] + 1) / 2) * (n_H - 1)
    ret_set[:, 2] = ret[:, 2]
    h = np.histogramdd(
        ret_set, bins=(n_H, n_W, k), range=[(0, n_H - 1), (0, n_W - 1), (1, k)]
    )[0]
    h = np.flip(h, axis=0)  # for visualization purposes
    h[h > 0] = 1
    return h

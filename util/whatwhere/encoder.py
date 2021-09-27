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

# same padding
def to_windows(img, window_shape, step_shape, dims=None):
    pad = []
    if dims is None:
        dims = img.ndim
    # for all RBG channels (multi dim support)
    for i in range(img.ndim):
        if i < dims:
            n = img.shape[i]
            s = step_shape[i]
            f = window_shape[i]
            p = int((f + (n - 1) * s - n) / 2)
        else:
            p = 0
        pad.append((p, p))
    # pad has necessary padding for each axis at this point
    # apply paddings
    img_padded = np.pad(img, tuple(pad), "constant", constant_values=(0))
    # slide windows across image and save contents (padded image)
    output = view_as_windows(img_padded, window_shape, step_shape)
    output = output.reshape(-1, np.prod(window_shape))  # 28*28*5*5 -> 784*25
    return output


# heviside of matrix
def H(x):
    y = np.zeros(x.shape)
    y[x > 0] = 1
    return y


# outro paper... nao usar (maybe)
def regular_convolution(x, K):
    kernel_shape = K.shape
    step_shape = tuple(np.ones(len(kernel_shape)).astype(int))
    data = to_windows(x, kernel_shape, step_shape)
    W = K.reshape(1, -1)
    logits = data @ W.T
    return logits.reshape(x.shape)


# esta foi usada no paper (quase de certeza)
def cosine_convolution(x, K, wta=True):  # TODO: falta o W como argumnto
    k = K.shape[0]
    # print("k:",k)
    kernel_shape = K.shape[1:]
    # print("kernel_shape:",kernel_shape)
    step_shape = tuple(np.ones(len(kernel_shape)).astype(int))
    # print("step_shape:",step_shape)
    data = to_windows(x, kernel_shape, step_shape)
    # print("data shape:", data.shape)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    out = (norms <= 0)[:, 0]
    norms[out] = 1
    data = data / norms
    W = K.reshape(k, -1)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / norms
    d = ((data @ W.T) + 1) / 2  # @ is matrix mul TODO: ver o que é este +1 /2
    if wta:
        t = d.max(
            axis=1, keepdims=True
        )  # para cada window da imagem selecionar a mais similar
        t[np.linalg.norm(data, axis=1) <= 0] = np.inf
        o = np.zeros(d.shape)
        o[(t - d) <= 0] = d[(t - d) <= 0]
        # print("hey", o.shape)
        # print("hey", x.shape)
        o = o.reshape(x.shape + (k,))  # imagem 28*28*K(num features)
        return o
    return d.reshape(x.shape + (k,))


# tb posso usar esta
def euclid_convolution(
    x, km, wta=True
):  # km== objecto que contem o resultado do k-means
    k = km.cluster_centers_.shape[0]  # resultado dos centroids
    kernel_shape = int(np.sqrt(km.cluster_centers_.shape[1]))
    kernel_shape = (kernel_shape, kernel_shape)
    step_shape = tuple(np.ones(len(kernel_shape)).astype(int))
    data = to_windows(x, kernel_shape, step_shape)
    d = 1 / km.transform(data)  # transform calcula distancia aos centroids
    if wta:
        t = d.max(axis=1, keepdims=True)
        t[np.linalg.norm(data, axis=1) <= 0] = np.inf
        o = np.zeros(d.shape)
        o[(t - d) <= 0] = d[(t - d) <= 0]
        o = o.reshape(x.shape + (k,))
        return o
    return d.reshape(x.shape + (k,))


# recebe imagem, K(n features) e threshold; calcula a convolution
def mu_ret(img, K, T_ret, wta=True):
    z = cosine_convolution(img, K, wta=wta)
    # z = euclid_convolution(img,K,wta=wta)
    a = H(z - np.quantile(z[z > 0], q=T_ret))
    return a


def enum_set(a):
    n_H, n_W, n_K = a.shape
    k = np.linspace(1, n_K, n_K)
    h = np.flip(np.linspace(0, n_H - 1, n_H))
    w = np.linspace(0, n_W - 1, n_W)
    W, H, K = np.meshgrid(w, h, k)
    k = K[a != 0]
    h = H[a != 0]
    w = W[a != 0]
    ret_set = np.zeros((k.shape[0], 3))
    ret_set[:, 0] = ((w / (n_W - 1)) * 2) - 1
    ret_set[:, 1] = ((h / (n_H - 1)) * 2) - 1
    ret_set[:, 2] = k
    return ret_set  # 3 colunas: pos em x, pos em y, feature que ganhou


def translation(features, C_x, C_y):
    M = np.eye(3)
    M[0:-1, -1] = (C_x, C_y)
    pos = np.copy(features)
    pos[:, -1] = 1
    pos = np.dot(M, pos.T).T
    pos[:, -1] = features[:, -1]
    # print(pos)
    return pos


def scale(features, rad):
    if rad == 0:
        # TODO: VER COM O lUIS SE ISTO E NA BOA
        rad = 1
    pos = np.copy(features)
    M = np.diag([1.0 / rad, 1.0 / rad])
    pos[:, 0:2] = np.dot(M, pos[:, 0:2].T).T
    return pos


def rotation(features, theta):  # nao usado no paper
    pos = np.copy(features)
    M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pos[:, 0:2] = np.dot(M, pos[:, 0:2].T).T
    return pos


def polar_transform(x):
    cx, cy = np.mean(x[:, 0:2], axis=0)  # center
    w = x[:, 0]
    h = x[:, 1]
    rad = np.max(np.sqrt((w - cx) ** 2 + (h - cy) ** 2))  # radius
    pol = scale(translation(x, -cx, -cy), rad)
    return pol


def grid_encoding(x, Q, k):
    pol = np.zeros(x.shape)
    pol[:, 0] = x[:, 1]
    pol[:, 1] = x[:, 0]
    pol[:, 2] = x[:, 2]
    h = np.histogramdd(pol, bins=(Q, Q, k), range=[(-1, 1), (-1, 1), (1, k)])[
        0
    ]  # conta para cada posi da grelha qnts pontos estao la dentro
    h = np.flip(h, axis=0)  # for visualization purposes y axist pointing down
    h[h > 0] = 1  # transofrma histograma em binary [0,1,2] -> [0,1,1]
    return h


#### Dictionary learning
"""
D conjunto de treino
K num features
patch_size: tamnaho da window 
background :  minimum windows norm
 """


def learn_features(
    trn_imgs, k, Fs, rng, n_epochs, background=0.8, kmeans=None, verbose=False
):
    # background discards values of patterns that are small in norm
    print(f"Learning a new dictionary...")

    patch_size = (
        1 + 2 * Fs,
        1 + 2 * Fs,
    )

    if kmeans is None:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=rng, verbose=True)
    buffer = []
    t0 = time.time()
    index = 0
    for _ in range(n_epochs):
        for img in trn_imgs:
            data = to_windows(
                img, patch_size, tuple(np.ones(len(patch_size)).astype(int))
            )
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            keep = norms > background
            keep = keep[:, 0]
            data = data[keep, :]
            # norms = norms[keep,:]
            # data = data / norms
            data = np.reshape(data, (len(data), -1))
            buffer.append(data)
            index += 1
            if index % 10 == 0:
                data = np.concatenate(buffer, axis=0)
                # data -= np.mean(data, axis=0)
                # data /= np.std(data, axis=0)
                kmeans.partial_fit(data)
                buffer = []
            if index % 10000 == 0 and verbose:
                print(
                    "Partial fit of %4i out of %i"
                    % (index / 1000, n_epochs * len(trn_imgs) / 1000)
                )
    dt = time.time() - t0
    print("done in %.2fs." % dt)
    return kmeans.cluster_centers_.reshape((-1,) + patch_size), kmeans


def compute_features(trn_imgs, k, Fs, rng, n_epochs, b=0.8, plot=False, verbose=False):
    run_name = "k" + str(k) + "_Fs" + str(Fs) + "_ep" + str(n_epochs) + "_b" + str(b)
    try:
        features = pickle.load(open(f"pickles/{run_name}__features.p", "rb"))
        if verbose:
            print(f"loaded features from pickles/{run_name}__features.p")
    except (OSError, IOError) as _:
        features, _ = learn_features(
            trn_imgs, k, Fs, rng, n_epochs, background=b, verbose=verbose
        )
        pickle.dump(
            features,
            open(f"pickles/{run_name}__features.p", "wb"),
        )

    """ if plot:
        plot_features2(features, features_shape, run_name) """

    return features


def learn_codes(trn_imgs, k, Q, verbose, features, T_what, wta):
    if verbose:
        print("Generation codes...")
    codes = np.zeros((trn_imgs.shape[0], k * Q ** 2))
    for i in range(trn_imgs.shape[0]):  # for all images
        if i % 1000 == 0 and verbose:
            print(i / 1000, " out of ", trn_imgs.shape[0] / 1000)
        img = trn_imgs[i]
        a = mu_ret(img, features, T_what, wta=wta)
        s = enum_set(a)
        if s.size != 0:
            p = polar_transform(s)
            e = grid_encoding(p, Q, features.shape[0])
            codes[i] = e.flatten()
        else:
            codes[i] = np.zeros(k * Q * Q)

    return csr_matrix(codes)


def get_codes_examples(codes, k, Q, num_examples=10):
    examples = np.array(codes[:num_examples])
    examples = examples.reshape(num_examples, k, Q, Q)
    examples = np.sum(examples, axis=1)
    examples.reshape(num_examples, Q, Q)

    tensor = torch.from_numpy(examples)
    tensor = torch.unsqueeze(tensor, dim=1)
    grid = torchv.utils.make_grid(tensor, normalize=True, nrow=10, pad_value=1)
    return grid

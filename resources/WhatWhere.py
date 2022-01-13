"""
Created on Wed Nov 24 11:32:56 2021

@author: luissacouto
"""
import numpy as np
from skimage.util import view_as_windows
from sklearn.cluster import KMeans


class RetLayer:
    def __init__(
        self,
        K,
        f,  # size of window is (f*f)
        Tw,
        T_pixel=1.2,  # background
        COSINE=True,  # mode of similatiry
        RNG=42,
        VERBOSE=1,
        MAX_ITER=300,
        N_INIT=10,
    ):
        self.K = K
        self.f = f
        self.p = f // 2
        self.RNG = RNG
        self.VERBOSE = VERBOSE
        self.MAX_ITER = MAX_ITER
        self.N_INIT = N_INIT
        self.T_pixel = T_pixel
        self.Tw = Tw
        self.COSINE = COSINE

    def fit(self, x):
        K = self.K
        p = self.p
        f = self.f
        T_pixel = self.T_pixel
        v = view_as_windows(np.pad(x, ((0, 0), (p, p), (p, p))), (1, f, f), (1, 1, 1))
        v = v.reshape(-1, f * f)
        print(v[v > 0])
        nor = np.sqrt((v ** 2).sum(axis=-1))  # mas nao é tudo positivo?
        v = v[nor >= T_pixel]
        if self.COSINE:
            nor = nor[nor >= T_pixel]
            v = v / nor.reshape(-1, 1)
        v = v[np.random.permutation(v.shape[0])]
        self.km = KMeans(
            n_clusters=K,
            init="k-means++",
            n_init=self.N_INIT,
            max_iter=self.MAX_ITER,
            verbose=self.VERBOSE,
            random_state=self.RNG,
        )
        self.km.fit(v)

    def similarity(self, x):
        f = self.f
        p = self.p
        n, n_H, n_W = x.shape
        v = view_as_windows(np.pad(x, ((0, 0), (p, p), (p, p))), (1, f, f), (1, 1, 1))
        v = v.reshape(n, -1, f * f)
        v = v.reshape(-1, f * f)
        if self.COSINE:
            nor = np.sqrt((v ** 2).sum(axis=-1))
            nor[nor <= 0] = 1
            v = v / nor.reshape(-1, 1)
            W = self.km.cluster_centers_
            W = W / np.sqrt((W ** 2).sum(axis=-1, keepdims=True))
            s = v @ W.T
        else:
            d = self.km.transform(v)
            s = 1 / d
        return s

    def encode(self, x):
        s = self.similarity(x)
        y = s.argmax(axis=-1) + 1
        s = s.max(axis=-1)
        y[s <= self.Tw] = 0
        y = y.reshape(x.shape)
        y_blow = np.eye(self.K + 1)[y][:, :, :, 1:]
        return y_blow

    def decode(self, x):
        K = self.K
        f = self.f
        p = self.p
        W = self.km.cluster_centers_
        W = W.reshape(K, f, f)
        W_flip = np.flip(np.flip(W, axis=2), axis=1)
        W_flip = np.moveaxis(W_flip, (0, 1, 2), (2, 0, 1))
        n, n_H, n_W, K = x.shape
        x_win = np.squeeze(
            view_as_windows(
                np.pad(x, ((0, 0), (p, p), (p, p), (0, 0))), (1, f, f, K), (1, 1, 1, K)
            )
        )
        x_contrib = x_win * W_flip
        # esta parte faz o mean para cada pixel (se o valor do pixel)
        weight = x_win.reshape(n, n_H, n_W, -1).sum(axis=-1)
        weight[weight == 0] = 1
        # tentar outras normalizations
        vals = x_contrib.reshape(n, n_H, n_W, -1).sum(axis=-1)
        r = vals / weight
        r[r < 0] = 0
        return r


class PolLayer:
    def __init__(self, Q, K):
        self.Q = Q
        self.K = K

    def map_to_set(self, x):
        n, n_H, n_W, k = x.shape
        ys = np.linspace(1, -1, n_H)
        xs = np.linspace(-1, 1, n_W)
        ks = np.arange(1, k + 1)
        xx, yy, kk = np.meshgrid(xs, ys, ks)
        S = []
        for i in range(n):
            S.append(np.stack([xx[x[i] > 0], yy[x[i] > 0], kk[x[i] > 0]], axis=-1))
        return S

    def set_transform(self, x):
        S = self.map_to_set(x)
        P = []
        params = []
        for s in S:
            ks = s[:, -1]
            xy = s[:, 0:2]
            C = xy.mean(axis=0)
            xy_centered = xy - C
            R = np.sqrt((xy_centered ** 2).sum(axis=-1)).max()
            xy_pol = xy_centered / R
            p = np.zeros((xy_pol.shape[0], 3))
            p[:, 0:2] = xy_pol
            p[:, -1] = ks
            P.append(p)
            params.append((C, R))
        return P, params

    def grid_encoding(self, x, Qh, Qw, K):
        xs = np.linspace(-1, 1, Qw)
        ys = np.linspace(1, -1, Qh)
        n = len(x)
        x_grid = np.zeros((n, Qh, Qw, K))
        for i in range(n):
            p = x[i]
            dy = (p[:, 1].reshape(-1, 1) - ys.reshape(1, -1)) ** 2
            iis = dy.argmin(axis=-1)
            dx = (p[:, 0].reshape(-1, 1) - xs.reshape(1, -1)) ** 2
            jjs = dx.argmin(axis=-1)
            kks = p[:, -1].astype(int) - 1
            x_grid[i, iis, jjs, kks] = 1
        return x_grid

    def encode(self, x):
        pol_set, params = self.set_transform(x)
        self.params = params
        if self.Q == 0:
            return pol_set
        return self.grid_encoding(pol_set, self.Q, self.Q, self.K)

    def decode(self, x, shape=None, params=None):
        if shape is None:
            shape = (self.Q, self.Q)
        if params is None:
            params = self.params
        if self.Q > 0:
            Q = self.Q
            K = self.K
            P = []
            for i in range(x.shape[0]):
                xs = np.linspace(-1, 1, Q)
                ys = np.linspace(1, -1, Q)
                ks = np.arange(1, K + 1)
                xx, yy, kk = np.meshgrid(xs, ys, ks)
                g = x[i]
                p = np.stack([xx[g > 0], yy[g > 0], kk[g > 0]], axis=-1)
                P.append(p)
            x = P
        for i in range(len(x)):
            x[i][:, 0:2] = (x[i][:, 0:2] * params[i][-1]) + params[i][0]
        x_recon = self.grid_encoding(x, shape[0], shape[1], self.K)
        return x_recon


class sparseWWencoder:
    def __init__(
        self,
        K,
        f,
        Tw,
        Q,
        T_pixel=1.2,
        COSINE=True,
        RNG=42,
        VERBOSE=1,
        MAX_ITER=300,
        N_INIT=10,
    ):
        self.K = K
        self.f = f
        self.Tw = Tw
        self.Q = Q
        self.T_pixel = T_pixel
        self.COSINE = COSINE
        self.RNG = RNG
        self.VERBOSE = VERBOSE
        self.MAX_ITER = MAX_ITER
        self.N_INIT = N_INIT
        self.retlayer = RetLayer(
            K,
            f,
            Tw,
            T_pixel=T_pixel,
            COSINE=COSINE,
            RNG=RNG,
            VERBOSE=VERBOSE,
            MAX_ITER=MAX_ITER,
            N_INIT=N_INIT,
        )
        self.pollayer = PolLayer(Q, K)

    def fit(self, x):
        self.retlayer.fit(x)

    def encode(self, x):
        ret = self.retlayer.encode(x)
        pol = self.pollayer.encode(ret)
        return pol

    def decode(self, x, shape=None, params=None):
        ret = self.pollayer.decode(x, shape=shape, params=params)
        pix = self.retlayer.decode(ret)
        return pix


""" Esta class e para classificar """


class AttentionMatcher:
    def __init__(
        self,
        K,
        f,
        Tw,
        T_where,
        Q,
        T_pixel=1.2,
        COSINE=True,
        RNG=42,
        VERBOSE=1,
        MAX_ITER=300,
        N_INIT=10,
    ):
        self.K = K
        self.T_where = T_where
        self.encoder = sparseWWencoder(
            K,
            f,
            Tw,
            Q,
            T_pixel=T_pixel,
            COSINE=COSINE,
            RNG=RNG,
            VERBOSE=VERBOSE,
            MAX_ITER=MAX_ITER,
            N_INIT=N_INIT,
        )
        self.VERBOSE = VERBOSE

    def fit(self, x_train, y_train):

        if self.VERBOSE > 0:
            print("--------------")
            print("Learning the encoder now")
            print("--------------")
        self.encoder.fit(x_train)
        x_enc = self.encoder.encode(x_train)

        if self.VERBOSE > 0:
            print("--------------")
            print("Learning the matcher now")
            print("--------------")
        T_where = self.T_where
        K = self.K
        pw = T_where // 2
        x_win = np.squeeze(
            view_as_windows(
                np.pad(x_enc, ((0, 0), (pw, pw), (pw, pw), (0, 0))),
                (1, T_where, T_where, K),
            )
        )
        x_win = x_win.max(axis=-2).max(axis=-2)
        self.W = x_win.reshape(x_win.shape[0], -1)
        self.labels = y_train

    def predict(self, x_test):
        x_enc = self.encoder.encode(x_test).reshape(x_test.shape[0], -1)
        z = self.W @ x_enc.T
        return self.labels[z.argmax(axis=0)]

    def score(self, x_test, y_test):
        return (self.predict(x_test) == y_test).mean()

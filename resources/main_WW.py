"""
Created on Wed Nov 24 19:11:33 2021

@author: luissacouto
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from WhatWhere import sparseWWencoder
from WhatWhere import AttentionMatcher

#%%

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
plt.imshow(X_train[np.random.permutation(X_train.shape[0])[0]], cmap="gray")
plt.show()


x_test = X_test[0:500]
y_test = Y_test[0:500]
x_used = X_train[0:100]
y_used = Y_train[0:100]

K = 12
f = 5
Tw = 0.85
Q = 50

swwe = sparseWWencoder(K, f, Tw, Q, COSINE=False)

swwe.fit(x_used)

x_enc = swwe.encode(x_used)
x_dec = swwe.decode(x_enc, shape=(28, 28), params=None)

l = 98
plt.imshow(x_used[l])
plt.imshow(x_dec[l])

T_where = 5
att = AttentionMatcher(
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
)
att.fit(x_used, y_used)
att.predict(x_test)
att.score(x_test, y_test)

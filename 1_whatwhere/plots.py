import matplotlib.pyplot as plt
from Utils import binary_sparsity
from Utils import best_layout
import random
import numpy as np


def plot_features(W, patch_size):
    plt.figure(figsize=(4.2, 4))
    plt.title("Features learnt with k-means")
    for i, patch in enumerate(W):
        plt.subplot(9, 9, i + 1)
        plt.imshow(
            patch.reshape(patch_size),
            cmap=plt.cm.gray,
            vmax=1,
            vmin=0,
            interpolation="nearest",
        )
        plt.xticks(())
        plt.yticks(())

    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()


def plot_examples(D, X_trn, K, k, Q, num_examples=3):

    Tr = X_trn.toarray()
    size = Tr.shape[0]
    Tr = Tr.reshape(size, k, Q * Q)

    fig, axs = plt.subplots(num_examples + 1, 2, constrained_layout=True)
    fig.suptitle(f"%B = {binary_sparsity(Tr)}")

    rand = np.random.permutation(D.shape[0])[0]
    axs[0][0].imshow(
        D[rand],
        vmax=1,
        vmin=0,
        cmap=plt.cm.gray,
        interpolation="nearest",
    )
    axs[0][0].set_title("Original pattern")

    axs[0][1].imshow(
        Tr[rand].reshape(best_layout(Tr[rand].size)),
        vmax=1,
        vmin=0,
        cmap=plt.cm.gray,
        interpolation="nearest",
    )
    axs[0][1].set_title("Coded pattern")

    for i in range(1, num_examples + 1):
        rand_k = random.randint(0, k - 1)
        axs[i][0].imshow(
            K[rand_k], vmax=1, vmin=0, cmap=plt.cm.gray, interpolation="nearest"
        )
        axs[i][0].set_title(f"filter {i} out of {k}")

        axs[i][1].imshow(
            Tr[rand][rand_k].reshape(Q, Q),
            vmax=1,
            vmin=0,
            cmap=plt.cm.gray,
            interpolation="nearest",
        )
        axs[i][1].set_title("filter map")

    # fig.set_size_inches(15, 8)
    plt.show()

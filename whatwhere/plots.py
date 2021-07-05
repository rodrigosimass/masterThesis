import matplotlib.pyplot as plt
from util import *
import random
import numpy as np


def plot_features(W, patch_size, run_name):
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

    plt.savefig(f"whatwhere/img/{run_name}__features.png")


def plot_examples(D, X_trn, K, k, Q, run_name, num_examples=3):

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
    plt.savefig(f"whatwhere/img/{run_name}__codes.png")


def plot_sparsity_distribution(codes, k, Q, run_name):
    plt.close()

    Tr = codes.toarray()
    size = Tr.shape[0]
    Tr = Tr.reshape(size, k * Q * Q)

    avg = np.average(Tr, axis=1)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    fig.suptitle("Sparsity distribution of the coded set")

    ax = axs[1]
    n, _, _ = ax.hist(
        range=(0, 1), x=avg, bins=1000, color="#0504aa", alpha=0.7, rwidth=0.85
    )
    maxfreq = n.max()
    ax.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    ax = axs[0]
    n, _, _ = ax.hist(x=avg, bins=100, color="#0504aa", alpha=0.7, rwidth=0.85)
    maxfreq = n.max()
    ax.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    plt.savefig(f"whatwhere/img/{run_name}__sparsityDistribution.png")


def plot_sparse_dense_examples(D, X_trn, K, k, Q, run_name):

    plt.close()

    Tr = X_trn.toarray()
    size = Tr.shape[0]
    Tr = Tr.reshape(size, k * Q * Q)

    sums = Tr.sum(axis=1)
    dense_idx = sums.argmax()
    sparse_idx = sums.argmin()

    fig, axs = plt.subplots(2, 2, constrained_layout=True)

    avg = np.average(Tr)
    str = "{:.4f}".format(avg)
    fig.suptitle(f"Avergae activity = {str}", fontsize=16)

    Tr = Tr.reshape(size, k, Q * Q)

    bl = best_layout(k * Q * Q)

    print(Tr[0].shape)

    axs[0][0].imshow(
        D[dense_idx],
        cmap=plt.cm.gray,
    )
    axs[0][0].set_title(f"densest (pat. {dense_idx})")
    axs[0][1].imshow(
        Tr[dense_idx].reshape(bl),
        cmap=plt.cm.gray,
    )
    avg = np.average(Tr[dense_idx])
    str = "{:.4f}".format(avg)
    axs[0][1].set_title(f"sparsity = {str}")

    axs[1][0].imshow(
        D[sparse_idx],
        cmap=plt.cm.gray,
    )
    axs[1][1].imshow(
        Tr[sparse_idx].reshape(bl),
        cmap=plt.cm.gray,
    )
    axs[1][0].set_title(f"sparsest (pat. {sparse_idx})")
    avg = np.average(Tr[sparse_idx])
    str = "{:.4f}".format(avg)
    axs[1][1].set_title(f"sparsity = {str}")

    plt.savefig(f"whatwhere/img/{run_name}__sparsedense.png")


def plot_mnist_codes_activity(mnist, s_codes, k, Q, run_name):

    plt.close()

    fig, axs = plt.subplots(2, 1)
    fig.suptitle("Activity in the MNIST dataset vs Whatwhere codes")

    codes = s_codes.toarray().reshape(s_codes.shape[0], k * Q * Q)

    sum_codes = (np.sum(codes, axis=0) / codes.shape[0]).flatten()
    sum_mnist = (np.sum(mnist, axis=0) / mnist.shape[0]).flatten()

    axs[0].set_title(f"Mnist activity (1D)")
    axs[0].bar(
        np.arange(sum_mnist.shape[0]),
        sum_mnist.flatten(),
        align="center",
        alpha=0.7,
        color="Black",
    )
    axs[0].set_ylabel("frequency")
    axs[0].set_xlim([1, sum_mnist.shape[0]])
    axs[0].set_ylim([0.0, 0.5])

    axs[1].set_title(f"Whatwhere activity (1D)")
    axs[1].bar(
        np.arange(sum_codes.shape[0]),
        sum_codes,
        align="center",
        alpha=0.7,
        color="Black",
    )
    axs[1].set_ylabel("frequency")
    axs[1].set_xlim([1, sum_codes.shape[0]])
    axs[1].set_ylim([0.0, 0.1])

    fig.tight_layout()

    plt.savefig(f"whatwhere/img/{run_name}__comparisson.png")
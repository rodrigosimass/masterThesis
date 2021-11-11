from util import *
import matplotlib.pyplot as plt
from util import *
import random
import numpy as np
from ..basic_utils import best_layout, binary_sparsity
from util.kldiv import *

rand = 2


def plot_features(W, Fs, run_name):

    plt.close()

    patch_size = (2 * Fs + 1, 2 * Fs + 1)

    bl = best_layout(W.shape[0])

    _, axs = plt.subplots(bl[0], bl[1], constrained_layout=True)

    aux = 0
    for i in range(bl[0]):
        for j in range(bl[1]):
            axs[i][j].imshow(
                W[aux].reshape(patch_size),
                vmax=1,
                vmin=0,
                cmap=plt.cm.gray,
                interpolation="nearest",
            )
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            aux += 1

    plt.savefig(f"img/whatwhere/{run_name}__features.png")


def plot_feature_maps(codes, k, Q, run_name, set="C"):
    plt.close()

    codes = codes.toarray().reshape((-1, Q, Q, k))
    example = codes[rand]

    bl = best_layout(k)

    fig, axs = plt.subplots(bl[0], bl[1], constrained_layout=True)

    fig.suptitle(f"Feature maps for pattern {rand} ({set})", fontsize=16)

    aux = 0
    for i in range(bl[0]):
        for j in range(bl[1]):
            axs[i][j].imshow(
                example[:, :, aux],
                vmax=1,
                vmin=0,
                cmap=plt.cm.gray,
                interpolation="nearest",
            )
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            aux += 1

    plt.savefig(f"img/whatwhere/{run_name}__Fmaps_{set}.png")


def plot_feature_maps_overlaped(data, X_trn, k, Q, run_name, set="C"):

    plt.close()

    Tr = X_trn.toarray()
    size = Tr.shape[0]
    Tr = Tr.reshape(size, Q, Q, k)

    example = Tr[rand]

    combined = np.sum(example, axis=2)

    _, axs = plt.subplots(1, 2, constrained_layout=True)

    axs[0].imshow(
        data[rand].reshape((28, 28)),
        cmap=plt.cm.gray,
        interpolation="nearest",
    )

    axs[1].imshow(
        combined,
        cmap=plt.cm.gray,
        interpolation="nearest",
    )

    plt.savefig(f"img/whatwhere/{run_name}__FmapsCombined_{set}.png")


def plot_examples(D, X_trn, K, k, Q, run_name, set, num_examples=3):

    Tr = X_trn.toarray()
    size = Tr.shape[0]
    Tr = Tr.reshape(size, k, Q * Q)

    fig, axs = plt.subplots(num_examples + 1, 2, constrained_layout=True)
    str = "{:.5f}".format(binary_sparsity(Tr))
    fig.suptitle(f"Examples of feature maps ({set} set) %B = {str}")

    axs[0][0].imshow(
        D[rand],
        vmax=1,
        vmin=0,
        cmap=plt.cm.gray,
        interpolation="nearest",
    )
    axs[0][0].set_title("Original pattern")

    axs[0][1].imshow(
        Tr[rand],
        vmax=1,
        vmin=0,
        cmap=plt.cm.gray,
        interpolation="nearest",
    )
    axs[0][1].set_title("Coded pattern")

    random.seed(0)
    for i in range(1, num_examples + 1):
        rand_k = random.randint(0, k - 1)
        axs[i][0].imshow(
            K[rand_k], vmax=1, vmin=0, cmap=plt.cm.gray, interpolation="nearest"
        )
        axs[i][0].set_title(f"filter {rand_k} out of {k}")

        axs[i][1].imshow(
            Tr[rand][rand_k].reshape(Q, Q),
            vmax=1,
            vmin=0,
            cmap=plt.cm.gray,
            interpolation="nearest",
        )
        axs[i][1].set_title("filter map")

    # fig.set_size_inches(15, 8)
    plt.savefig(f"img/whatwhere/{run_name}__codes_{set}.png")


def plot_sparsity_distribution(codes, k, Q, run_name, set="C"):
    plt.close()

    Tr = codes.toarray()
    size = Tr.shape[0]
    Tr = Tr.reshape(size, k * Q * Q)

    avg = np.average(Tr, axis=1)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    fig.suptitle("Sparsity distribution of the coded set")

    ax = axs[1]
    n, _, _ = ax.hist(
        range=(0, 1), x=avg, bins=100, color="#0504aa", alpha=0.7, rwidth=0.85
    )
    maxfreq = n.max()
    ax.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    ax = axs[0]
    n, _, _ = ax.hist(x=avg, bins=100, color="#0504aa", alpha=0.7, rwidth=0.85)
    maxfreq = n.max()
    ax.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    plt.savefig(f"img/whatwhere/{run_name}__sparsityDistribution_{set}.png")


def plot_mnist_codes_activity(mnist, codes, k, Q, run_name, set="C"):

    plt.close()

    fig, axs = plt.subplots(2, 1)
    if set == "C":
        fig.suptitle("Activity in the MNIST dataset vs Coded Set")
    else:
        fig.suptitle("Activity in the MNIST dataset vs Retrieved Set")

    codes = codes.toarray().reshape(-1, k * Q * Q)
    mnist = mnist.reshape(-1, 28 * 28)

    _, _, entropy_codes = measure_data_distribution_set(codes)
    _, _, entropy_mnist = measure_data_distribution_set(mnist)

    avg_codes = np.sort(np.average(codes, axis=0))
    avg_mnist = np.sort(np.average(mnist, axis=0))

    axs[0].set_title(f"Mnist activity (1D) e={entropy_mnist:.5f}")
    axs[0].bar(
        np.arange(avg_mnist.shape[0]),
        avg_mnist.flatten(),
        align="center",
        alpha=0.7,
        color="Black",
    )
    axs[0].set_ylabel("frequency")
    axs[0].set_xlim([1, avg_mnist.shape[0]])
    axs[0].set_ylim([0.0, np.max(avg_mnist)])

    axs[1].set_title(f"Whatwhere activity (1D) e={entropy_codes:.5f}")
    axs[1].bar(
        np.arange(avg_codes.shape[0]),
        avg_codes,
        align="center",
        alpha=0.7,
        color="Black",
    )
    axs[1].set_ylabel("frequency")
    axs[1].set_xlim([1, avg_codes.shape[0]])
    axs[1].set_ylim([0.0, np.max(avg_codes)])

    fig.tight_layout()

    plt.savefig(f"img/whatwhere/{run_name}__comparisson_{set}.png")


def plot_class_activity_1D(codes, labels, k, Q, run_name, set="C"):

    plt.close()

    codes = codes.toarray()
    size = codes.shape[0]
    labels = labels[:size]
    codes = codes.reshape(size, Q * Q, k)

    fig, axs = plt.subplots(10, 1)

    fig.suptitle(f"Usage of kernels per class", fontsize=16)

    for i in range(10):
        imgData = np.average(codes[labels == i], 0)  # (N,Q*Q,K) -> (Q*Q,K)
        imgData = np.average(imgData, axis=0)  # (Q*Q,K) -> (K)

        ax = axs[i]
        ax.bar(
            np.arange(k),
            imgData.flatten(),
            align="center",
            alpha=0.7,
            color="Black",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(i, rotation=0)
        ax.set_xlabel("k")

    plt.savefig(f"img/whatwhere/{run_name}__WW_classes_1D_{set}.png")


def plot_class_activity_1D_stacked(codes, labels, k, Q, run_name, set="C"):

    plt.close()

    codes = codes.toarray()
    size = codes.shape[0]
    labels = labels[:size]
    codes = codes.reshape(size, Q * Q, k)

    fig, ax = plt.subplots()

    fig.suptitle(f"Usage of kernels", fontsize=16)

    prev = np.zeros(k)
    for i in range(10):
        curr = np.average(codes[labels == i], 0)  # (N,Q*Q,K) -> (Q*Q,K)
        curr = np.average(curr, axis=0)  # (Q*Q,K) -> (K)

        ax.bar(x=np.arange(k), height=curr, bottom=prev, label=str(i))
        prev = prev + curr

    ax.set_ylabel("frequency")
    ax.legend()

    plt.savefig(f"img/whatwhere/{run_name}__WW_classes_1D_stacked_{set}.png")


def plot_class_activity_2D(codes, labels, k, Q, run_name, set="C"):

    plt.close()

    Tr = codes.toarray()
    size = Tr.shape[0]
    labels = labels[:size]
    Tr = Tr.reshape(size, Q, Q, k)
    overlaped = np.sum(Tr, axis=3)  # make k dim disapear

    fig, axs = plt.subplots(2, 5, constrained_layout=True)
    fig.suptitle(f"Average pixel activity per class ({set} set)", fontsize=16)
    for i in range(10):
        avgImg = np.average(overlaped[labels == i], 0)  # average across all samples
        ax = axs[i // 5][i % 5]
        ax.imshow(avgImg.reshape(Q, Q), cmap="Greys")
        ax.set_xticks([])
        ax.set_yticks([])
        activity = np.sum(avgImg) / (k * Q * Q)
        ax.set_title(f"({i})s={activity:.4f}")

    plt.savefig(f"img/whatwhere/{run_name}__WW_classes_2D_{set}.png")

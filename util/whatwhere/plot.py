from util import *
import matplotlib.pyplot as plt
from util import *
import random
import numpy as np

rand = 10


def plot_features(W, patch_size, run_name):
    plt.figure(figsize=(4.2, 4))
    plt.suptitle("Features learnt with k-means", fontsize=16)
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


def plot_features2(W, patch_size, run_name):

    plt.close()

    bl = best_layout(W.shape[0])

    fig, axs = plt.subplots(bl[0], bl[1], constrained_layout=True)

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

    plt.savefig(f"whatwhere/img/{run_name}__features.png")


def plot_feature_maps(X_trn, k, Q, run_name, set):

    plt.close()

    Tr = X_trn.toarray()
    size = Tr.shape[0]
    Tr = Tr.reshape(size, k, Q, Q)

    example = Tr[rand]

    bl = best_layout(k)

    fig, axs = plt.subplots(bl[0], bl[1], constrained_layout=True)

    fig.suptitle(f"Feature maps for pattern {rand} ({set})", fontsize=16)

    aux = 0
    for i in range(bl[0]):
        for j in range(bl[1]):
            axs[i][j].imshow(
                example[aux].reshape(Q, Q),
                vmax=1,
                vmin=0,
                cmap=plt.cm.gray,
                interpolation="nearest",
            )
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            aux += 1

    plt.savefig(f"whatwhere/img/{run_name}__Fmaps_{set}.png")


def plot_feature_maps_overlaped(data, X_trn, k, Q, run_name, set):

    plt.close()

    Tr = X_trn.toarray()
    size = Tr.shape[0]
    Tr = Tr.reshape(size, k, Q, Q)

    example = Tr[rand]

    combined = np.average(example, axis=0)

    fig, axs = plt.subplots(1, 2, constrained_layout=True)

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

    plt.savefig(f"whatwhere/img/{run_name}__FmapsCombined_{set}.png")


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
    plt.savefig(f"whatwhere/img/{run_name}__codes_{set}.png")


def plot_sparsity_distribution(codes, k, Q, run_name, set):
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

    plt.savefig(f"whatwhere/img/{run_name}__sparsityDistribution_{set}.png")


def plot_sparse_dense_examples(D, X_trn, K, k, Q, run_name, set):

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

    # print(Tr[0].shape)

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

    plt.savefig(f"whatwhere/img/{run_name}__sparsedense_{set}.png")


def plot_mnist_codes_activity(mnist, s_codes, k, Q, run_name, set):

    plt.close()

    fig, axs = plt.subplots(2, 1)
    if set == "C":
        fig.suptitle("Activity in the MNIST dataset vs Coded Set")
    else:
        fig.suptitle("Activity in the MNIST dataset vs Retrieved Set")

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
    axs[0].set_ylim([0.0, np.max(sum_mnist)])

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
    axs[1].set_ylim([0.0, np.max(sum_codes)])

    fig.tight_layout()

    plt.savefig(f"whatwhere/img/{run_name}__comparisson_{set}.png")


def plot_class_activity_1D(codes, labels, k, Q, run_name, set):

    plt.close()

    Tr = codes.toarray()
    size = Tr.shape[0]
    labels = labels[:size]
    Tr = Tr.reshape(size, k * Q * Q)

    fig, axs = plt.subplots(10, 1, sharex=True)

    fig.suptitle(f"Average pixel activity per class ({set} set)", fontsize=16)

    for i in range(10):
        avgImg = np.average(Tr[labels == i], 0)
        ax = axs[i]
        ax.bar(
            np.arange(k * Q * Q),
            avgImg.flatten(),
            align="center",
            alpha=0.7,
            color="Black",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(i, rotation=0)

    plt.savefig(f"whatwhere/img/{run_name}__WW_classes_1D_{set}.png")


def plot_class_activity_2D(codes, labels, k, Q, run_name, set):

    plt.close()

    Tr = codes.toarray()
    size = Tr.shape[0]
    labels = labels[:size]
    Tr = Tr.reshape(size, k, Q * Q)
    sums = np.sum(Tr, axis=1)

    fig, axs = plt.subplots(2, 5, constrained_layout=True)
    fig.suptitle(f"Average pixel activity per class ({set} set)", fontsize=16)
    for i in range(10):
        avgImg = np.average(sums[labels == i], 0)
        ax = axs[i // 5][i % 5]
        ax.imshow(avgImg.reshape(Q, Q), cmap="Greys")
        ax.set_xticks([])
        ax.set_yticks([])
        activity = np.sum(avgImg) / (k * Q * Q)
        str = "{:.4f}".format(activity)
        ax.set_title(f"({i})s={str}")

    plt.savefig(f"whatwhere/img/{run_name}__WW_classes_2D_{set}.png")
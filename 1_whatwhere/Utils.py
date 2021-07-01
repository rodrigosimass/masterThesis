import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from parse import parse
import math


def parse_par(parfile, parname):
    with open(parfile) as search:
        for line in search:
            tag = parname + " {v}"
            v = parse(tag, line)
            if v:
                return int(v.named["v"])

    return None


def load_imgs_lbls(imgfile, lblfile, num_imgs, img_size):
    imgs = np.fromfile(imgfile, dtype=np.float32)
    imgs = imgs.reshape(num_imgs, img_size, img_size, 1)

    lbls = np.fromfile(lblfile, dtype=np.int64)

    return imgs, lbls


def show_pat_example(title, imgs, lbls, hidden, num_imgs, num_examples=3):
    rand_seed = 7623
    m = np.min(hidden)
    M = np.max(hidden)

    random.seed(rand_seed)

    hidmax = abs(hidden).max()

    fig, axs = plt.subplots(num_examples, 2)
    fig.suptitle(
        f"Examples of {title} patterns and their hidden activity, random_seed={rand_seed}"
    )
    for i in range(num_examples):
        rand = random.randint(0, num_imgs - 1)

        image = np.asarray(imgs[rand])
        hid = hidden[rand]

        axs[i][0].imshow(
            image,
            cmap="Greys",
            interpolation=None,
        )
        # axs[i][0].set_title(f"Pattern {rand}/{num_imgs}, lbl:{lbls[rand]}")
        axs[i][0].set_title("inpats2")

        im = axs[i][1].imshow(
            hid,
            cmap="bwr",
            vmin=-hidmax,
            vmax=hidmax,
            interpolation=None,
        )
        # sparsity = np.count_nonzero(hid) / (hid.size)
        # axs[i][1].set_title(f"Hidden activity, \n min={np.min(hid)}, max={np.max(hid)}")
        axs[i][1].set_title("hid")
        axs[i][1].set_ylabel("H")
        axs[i][1].set_xlabel("M")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # fig.set_size_inches(15, 8)
    # fig.tight_layout(pad=0.5)
    plt.show()


def load_W(filename, i, j):
    W = np.fromfile(filename, dtype=np.float32)
    W = W.reshape(i, j)
    return W


def show_W(title, W, i, j):

    s, _ = sparsity(W)
    form_s = "{:.2f}".format(s)

    Wmax = abs(W).max()

    fig, axs = plt.subplots(2)

    fig.suptitle(f"{title} ; {i}x{j} ; s={form_s}")

    axs[0].set_ylabel("input")
    axs[0].set_xlabel("hidden")
    im = axs[0].imshow(
        W,
        cmap="bwr",
        vmin=-Wmax,
        vmax=Wmax,
        interpolation=None,
    )

    # TODO: fix the histogram for the weight matrices
    axs[1].set_ylabel("bias")
    axs[1].set_xlabel("j")
    print(W.shape)
    axs[1].hist(W.flatten(), bins=50)

    plt.colorbar(im, ax=axs[0])
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.show()


def best_layout(N):
    best = int(math.sqrt(N))
    while N % best != 0:
        best -= 1
    return best, int(N / best)


def show_W_HC_examples(title, W, M, H, num_examples=9):

    row, col = best_layout(num_examples)

    Wmax = abs(W).max()

    fig, axs = plt.subplots(row, col)
    fig.suptitle("HC's from " + title + " examples")

    for i in range(row):
        for j in range(col):
            rand = random.randint(0, H - 1)
            W_copy = W.copy()
            W_slice = W_copy[:, rand * M : (rand + 1) * M]
            # print(W_slice.shape)
            combined = np.sum(W_slice, axis=1) / M
            # print(combined.shape)
            combined = combined.reshape(28, 28)

            mi = np.min(combined)
            ma = np.max(combined)
            # print(mi, "----", ma)

            axs[i][j].set_title(title + " HC_" + str(rand))
            im = axs[i][j].imshow(
                combined,
                cmap="bwr",
                vmin=-Wmax,
                vmax=Wmax,
                interpolation=None,
            )
            # axs[i].colorbar()
            # axs[i].show()
    # fig.tight_layout(pad=0.5)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.set_size_inches(15, 8)
    plt.show()


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def load_bias(file, N):
    b = np.fromfile(file, dtype=np.float32)
    b = b.reshape(N)
    return b


def show_bias(title, b, N):
    # plt.title(f"{title}\n min={np.min(b)}, max={np.max(b)}")
    plt.title(title)
    plt.ylabel("bias")
    plt.xlabel("j")
    plt.hist(b, bins=100)
    # plt.bar(np.arange(N), b)
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.show()


def load_hidden_rep(file, num_patts, H, M):
    hid = np.fromfile(file, dtype=np.float32)
    hid = hid.reshape(num_patts, H, M)

    return hid


def binary_sparsity(X):
    return np.count_nonzero(X) / X.size


# counts the total ammount of non-zeros elements
def sparsity(matrix, tolerance=0.01):
    m = np.min(matrix)
    M = np.max(matrix)
    ran = M - m
    tol = ran * tolerance

    # print(f"tolerance for a zero is {0.0 - tol} to {0.0 + tol}")

    cnt = 0
    for x in np.nditer(matrix):
        if math.isclose(x, 0.0, abs_tol=tol):
            cnt += 1
    # print(cnt)
    percent_zeros = cnt / matrix.size
    percent_ones = 1 - percent_zeros
    return percent_ones, tol
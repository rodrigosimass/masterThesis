import numpy as np
import random
import matplotlib.pyplot as plt

from .basic_utils import best_layout


def plot_dists(l_dists, l_d, l_kl, l_e):

    plt.close()
    fig, _ = plt.subplots(len(l_dists), 1)

    for i in range(len(l_dists)):
        d = l_dists[i]
        ax = plt.subplot(4, 5, i + 1)
        ax.bar(np.arange(d.size), d)
        ax.set_xticks([])
        ax.set_title(f"Dice {i+1}")
        ax.set_ylim([0, 1])
        ax.set_ylabel("P")
        ax.set_xlabel("i")

    ax = plt.subplot(4, 5, (6, 10))
    ax.plot(np.arange(1, len(l_d) + 1), np.array(l_d), "ro")
    ax.set_xticks(np.arange(1, len(l_dists) + 1))
    ax.set_xlabel("Distribution #")
    ax.set_ylabel("|P-U|")

    ax = plt.subplot(4, 5, (11, 15))
    ax.plot(np.arange(1, len(l_kl) + 1), np.array(l_kl), "go")
    ax.set_xticks(np.arange(1, len(l_dists) + 1))
    ax.set_xlabel("Distribution #")
    ax.set_ylabel("KL-div")

    ax = plt.subplot(4, 5, (16, 20))
    ax.plot(np.arange(1, len(l_e) + 1), np.array(l_e), "bo")
    ax.set_xticks(np.arange(1, len(l_dists) + 1))
    ax.set_xlabel("Distribution #")
    ax.set_ylabel("Entropy")

    fig.set_size_inches(15, 10)
    plt.tight_layout()
    plt.savefig(f"img/kldiv/dists.png")


def multiple_imshow(imgs, layout=None, title=None, m=0, M=1):
    n_imgs = imgs.shape[0]
    if layout == None:
        layout = best_layout(n_imgs)
    fig, _ = plt.subplots(layout[0], layout[1])
    if title:
        fig.suptitle(title)

    for i, ax in enumerate(fig.axes):
        ax.imshow(imgs[i], vmin=m, vmax=M, cmap=plt.cm.gray)
        ax.set_xticks([])
        ax.set_yticks([])

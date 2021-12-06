import numpy as np
import random
import matplotlib.pyplot as plt

from util.basic_utils import best_layout


def plot_dists(l_dists, l_d, l_kl, l_e):

    plt.close()
    fig, _ = plt.subplots(len(l_dists), 1)

    for i in range(len(l_dists)):
        d = l_dists[i]
        ax = plt.subplot(4, 5, i + 1)
        ax.bar(np.arange(d.size), d)
        ax.set_xticks([])
        ax.set_title(f"Distribution {i+1}")

    ax = plt.subplot(4, 5, (6, 10))
    ax.plot(np.arange(1, len(l_d) + 1), np.array(l_d))
    ax.set_xticks(np.arange(1, len(l_dists) + 1))
    ax.set_xlabel("Distribution")
    ax.set_ylabel("d")

    ax = plt.subplot(4, 5, (11, 15))
    ax.plot(np.arange(1, len(l_kl) + 1), np.array(l_kl))
    ax.set_xticks(np.arange(1, len(l_dists) + 1))
    ax.set_xlabel("Distribution")
    ax.set_ylabel("KL-divergence")

    ax = plt.subplot(4, 5, (16, 20))
    ax.plot(np.arange(1, len(l_e) + 1), np.array(l_e))
    ax.set_xticks(np.arange(1, len(l_dists) + 1))
    ax.set_xlabel("Distribution")
    ax.set_ylabel("entropy")

    fig.set_size_inches(30, 10)
    plt.tight_layout()
    plt.savefig(f"img/kldiv/dists.png")


def multiple_imshow(imgs, title=None, m=0, M=1):
    num_imgs = imgs.shape[0]
    fig, axs = plt.subplots(1, num_imgs)
    if title:
        fig.suptitle(title)

    for i in range(num_imgs):
        axs[i].imshow(imgs[i], vmin=m, vmax=M, cmap=plt.cm.gray)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

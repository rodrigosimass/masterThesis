import matplotlib.pyplot as plt
import numpy as np

path = "mnist/data/"


def viz_class_activity_2D(data, labels):

    fig, axs = plt.subplots(2, 5, constrained_layout=True)

    fig.suptitle("Average pixel activity per class", fontsize=16)

    for i in range(10):
        avgImg = np.average(data[labels == i], 0)
        ax = axs[i // 5][i % 5]
        ax.imshow(avgImg.reshape(28, 28), cmap="Greys")
        ax.set_xticks([])
        ax.set_yticks([])
        activity = np.sum(avgImg) / 784
        str = "{:.2f}".format(activity)
        ax.set_title(f"s={str}")

    plt.savefig(f"img/mnist/class_activity_2D.png", bbox_inches="tight")


def viz_class_activity_1D(data, labels, sorted=False):

    fig, axs = plt.subplots(10, 1, sharex=True, sharey=True)

    if not sorted:
        fig.suptitle("MNIST Average pixel activity per class", fontsize=16)
    else:
        fig.suptitle("MNIST Average pixel activity per class (sorted)", fontsize=16)

    for i in range(10):
        print(i)
        avgImg = np.average(data[labels == i], 0)

        if sorted:
            avgImg = np.flip(np.sort(avgImg))  # descending sort

        ax = axs[i]
        ax.bar(
            np.arange(28 * 28),
            avgImg.flatten(),
            align="center",
            alpha=0.7,
            color="Black",
        )
        if i == 9:
            ax.set_xticks([0, 783])
        ax.set_yticks(())
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(i, rotation=0)
        ax.set_ylim((0, 1))
        ax.set_xlim((0, 783))

    if sorted:
        plt.savefig(f"img/mnist/class_activity_1D_sorted.png", bbox_inches="tight")
    else:
        plt.savefig(f"img/mnist/class_activity_1D.png", bbox_inches="tight")


def viz_activity(data, labels):
    sum_X = np.sum(data, axis=0)
    # print(np.max(sum_X))
    sum_X = sum_X / data.shape[0]
    # print(np.max(sum_X))
    # print(sum_X.shape)

    fig, axs = plt.subplots(1, 3)

    fig.suptitle("Activity in the MNIST dataset")

    ax0 = plt.subplot(221)
    ax0.set_title("Example: digit 2")
    ax0.imshow(data[5].reshape((28, 28)), cmap="Greys")
    ax0.set_xticks([1, 14, 28])
    ax0.set_yticks([1, 14, 28])

    ax1 = plt.subplot(222)
    ax1.set_title("Pixel usage frequency (2D)")
    plt1 = ax1.imshow(sum_X.reshape(28, 28), cmap="Greys", vmax=1, vmin=0)
    ax1.set_xticks([1, 14, 28])
    ax1.set_yticks([1, 14, 28])

    plt.colorbar(plt1, ticks=[0, 0.5, 1])

    avg_act = np.average(sum_X)
    str = "{:.2f}".format(avg_act)

    ax2 = plt.subplot(212)
    ax2.set_title(f"Pixel usage frequency (1D), avg={str}")
    ax2.bar(
        np.arange(28 * 28), sum_X.flatten(), align="center", alpha=0.7, color="Black"
    )
    ax2.set_xlabel("Pixel")
    ax2.set_ylabel("frequency")
    ax2.set_xlim([1, 784])
    ax2.set_xticks([1, 98, 196, 294, 392, 490, 588, 686, 784])
    ax2.set_yticks([0, 0.5, 1])

    fig.tight_layout()

    plt.savefig(f"img/mnist/activity.png", bbox_inches="tight")

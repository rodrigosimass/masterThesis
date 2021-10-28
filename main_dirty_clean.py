import torch
from util.pickleInterface import load_codes, load_ret
from util.mnist.tools import *
from models.MLP import *
import matplotlib.pyplot as plt
import numpy as np


def plot_error_evolution(clean, dirty):

    diff = dirty - clean

    noise = np.copy(diff)
    noise[noise < 0] = 0
    noise = noise.reshape(noise.shape[0], -1)
    noise_total = np.sum(noise, axis=1)

    loss = np.copy(diff)
    loss[loss > 0] = 0
    loss = loss.reshape(loss.shape[0], -1)
    loss_total = np.sum(loss, axis=1)
    loss_total = loss_total * -1

    absMax = max(np.amax(noise_total), np.amax(loss_total))

    fig, _ = plt.subplots(2, 1, sharey=True)

    ax1 = plt.subplot(2, 1, 1)
    ax1.bar(
        np.arange(noise_total.shape[0]),
        noise_total,
        color=(132 / 255, 9 / 255, 35 / 255),
    )
    ax1.title.set_text(f"noise")
    ax1.set_xlabel(f"MNIST index")
    ax1.set_ylim((0, absMax))

    ax2 = plt.subplot(2, 1, 2)
    ax2.bar(
        np.arange(loss_total.shape[0]),
        loss_total,
        color=(23 / 255, 84 / 255, 147 / 255),
    )
    ax2.title.set_text(f"loss")
    ax2.set_xlabel(f"MNIST index")
    ax2.set_ylim((0, absMax))

    fig.set_size_inches(10, 5)
    plt.tight_layout()
    plt.savefig(f"img/decoder/error_evolution_shareY.png")


def plot_examples(imgs, lbls, clean, dirty, name="dirtyClean"):
    n_rows = 4
    n_cols = 10
    fig_width = 15

    fig, _ = plt.subplots(n_rows, n_cols)
    fig.suptitle(name, fontsize=16)

    for i in range(10):
        ax1 = plt.subplot(n_rows, n_cols, i + 1)
        ax2 = plt.subplot(n_rows, n_cols, i + n_cols + 1)
        ax3 = plt.subplot(n_rows, n_cols, i + 2 * n_cols + 1)
        ax4 = plt.subplot(n_rows, n_cols, i + 3 * n_cols + 1)

        ax1.imshow(imgs[i], cmap="Greys", vmin=0.0, vmax=1.0)
        ax1.title.set_text(f"{lbls[i]}")
        ax2.imshow(clean[i], cmap="Greys", vmin=0.0, vmax=1.0)
        ax3.imshow(dirty[i], cmap="Greys", vmin=0.0, vmax=1.0)
        ax4.imshow(dirty[i] - clean[i], cmap="RdBu", vmin=-1.0, vmax=1.0)

        if i == 0:
            ax1.set_ylabel("MNIST", fontsize=16)
            ax2.set_ylabel("Coded", fontsize=16)
            ax3.set_ylabel("Retr", fontsize=16)
            ax4.set_ylabel("R - C", fontsize=16)

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.set_xticks([])
        ax4.set_yticks([])

    fig.set_size_inches(fig_width, fig_width * (n_rows / n_cols))
    plt.tight_layout()
    plt.savefig(f"img/decoder/{name}.png")


param_id = "k20_Fs2_ep5_b0.8_Q21_Tw0.95"

# MLP
dim_hid = []
dim_in = 20 * 21 * 21
dim_out = 28 * 28
max_epochs = 200
lr = 0.01
batch_size = 1000
patience = 20
delta = 0.0001
shuffle = False

loss = "MSE"

trn_n = 10000
val_n = (
    2000  # if (trn_n=60k;val_n=10k), then we will train with 50k and use 10k for valid
)

model_name = "MLP_" + loss + "_" + str(dim_hid) + "_n_" + str(trn_n)
model_PATH = f"pickles/{model_name}.pt"

imgs, lbls, _, _ = read_mnist()
codes, _ = load_codes(param_id)
ret = load_ret(param_id)

imgs = imgs[:trn_n].reshape((trn_n, 28, 28))
lbls = lbls[:trn_n]
codes = codes[:trn_n].toarray()
ret = ret[:trn_n].toarray()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim=dim_in, output_dim=dim_out, hidden_dim=dim_hid).to(device)
model.load_state_dict(torch.load(model_PATH))

clean = (model(torch.Tensor(codes))).detach().numpy().reshape((trn_n, 28, 28))

dirty = (model(torch.Tensor(ret))).detach().numpy().reshape((trn_n, 28, 28))

plot_error_evolution(clean, dirty)

""" # First from each class
idxs = idxs_first_per_class(lbls)
plot_examples(imgs[idxs], lbls[idxs], clean[idxs], dirty[idxs], "first10")

# last from each class
idxs = idxs_last_per_class(lbls)
plot_examples(imgs[idxs], lbls[idxs], clean[idxs], dirty[idxs], "last10")

for i in range(10):
    idxs = idxs_class_evolution(lbls, i, 10)
    plot_examples(
        imgs[idxs], lbls[idxs], clean[idxs], dirty[idxs], f"class_{i}_evolution"
    ) """

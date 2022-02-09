import numpy as np
import random
import matplotlib.pyplot as plt
from util.distribution import *
from util.pickleInterface import load_codes, load_ret
from util.mnist.tools import *


def generate_bin_data_from_dist(dist, M, n):
    data = np.zeros((M, n))
    for i in range(M):
        for j in range(n):
            roll = random.uniform(0, 1)
            if roll <= dist[j]:
                data[i][j] = 1
    return data


def plot_dists_kldivs(l_dists, l_names, filename):

    plt.close()
    fig, axs = plt.subplots(len(l_dists), 3)

    fig.suptitle("KL_div", fontsize=16)

    for i in range(len(l_dists)):
        d = l_dists[i]
        sorted = np.sort(d)
        u = 1 / d.size
        unif = np.full_like(sorted, u)

        axs[i][0].set_title("Distribution")
        axs[i][0].bar(np.arange(d.size), d)

        axs[i][1].set_title("Sorted Distribution (log scale)")
        axs[i][1].bar(np.arange(sorted.size), sorted)
        axs[i][1].set_yscale("log")
        axs[i][1].axhline(y=u, color="r", linestyle="-")
        axs[i][1].legend(["Uniform", "Distribution"])

        axs[i][2].set_xticks([])
        axs[i][2].set_yticks([])
        axs[i][2].text(
            x=0,
            y=0,
            s=f"""
            
            {l_names[i]}  


            kl_div = {kl_divergence(d,unif):.7f}
            entropy = {shannon_entropy(d):.7f}
            dist to unif = {dist_difference(d,unif):.7f}
            """,
            fontsize=20,
        )

    fig.set_size_inches(20, 15)
    plt.tight_layout()
    plt.savefig(f"img/kldiv/{filename}.png")


""" ------------------------------------------- """
# MNIST
mnist, _, _, _ = read_mnist(n_train=60000)
mnist = mnist.reshape((mnist.shape[0], -1))
s_mnist = mnist[:30000]
mnist_dist, _ = distributions(mnist)
s_mnist_dist, _ = distributions(s_mnist)

# WHATWHERE
param_id = "k20_Fs2_ep5_b0.8_Q21_Tw0.95"
codes = load_codes(param_id)[0].toarray()
s_codes = codes[:30000]
codes_dist, _ = distributions(codes)
s_codes_dist, _ = distributions(s_codes)

l_dists = [s_mnist_dist, mnist_dist, s_codes_dist, codes_dist]
l_names = [
    "MNIST n=784 M =30k",
    "MNIST n=784 M =60k",
    "WW n=8820 M=30k",
    "WW n=8820 M=60k",
]

plot_dists_kldivs(l_dists, l_names, "mnist")
print("Done with mnist/codes")
""" ------------------------------------------- """

l_n = [100, 500]
l_M = [5000, 10000]


l_names = []
l_dists = []

for n in l_n:
    for M in l_M:
        q = np.full((n), 1 / n)
        data = generate_bin_data_from_dist(q, M, n)
        p, _ = distributions(data)
        l_dists.append(p)
        l_names.append(f"Uniform: n={n}, M={M}")


plot_dists_kldivs(l_dists, l_names, "unif")
print("Done with unif")

""" ------------------------------------------- """
l_dists = []
l_names = []

for n in l_n:
    for M in l_M:
        q = np.random.normal(1, 0.5, size=M)
        q = np.histogram(q, bins=n, density=True)[0]

        data = generate_bin_data_from_dist(q, M, n)
        p, _ = distributions(data)

        l_dists.append(p)
        l_names.append(f"Normal: n={n}, M={M}")


plot_dists_kldivs(l_dists, l_names, "normal")
print("Done with normal")

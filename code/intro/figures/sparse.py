import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

rng = np.random.RandomState(10)  # deterministic random data

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

axs[0][0].set_xlim([0, 50])
axs[0][0].set_ylim([0, 15])

axs[0][0].set_title("Non-sparse, non-distributed")
axs[0][1].set_title("Sparse, non-distributed")
axs[1][0].set_title("Non-sparse, distributed")
axs[1][1].set_title("Sparse, distributed")

axs[1][0].set_xlabel(r"$x_{i}$")
axs[1][1].set_xlabel(r"$x_{i}$")

axs[0][0].set_ylabel(r"$P(x_i)=1$")
axs[1][0].set_ylabel(r"$P(x_i)=1$")

s_d = np.hstack((rng.uniform(size=30)) * 50)
_ = axs[1][1].hist(s_d, bins=50, color="green", alpha=0.8)

ns_d = np.hstack((rng.uniform(size=350)) * 50)
_ = axs[1][0].hist(ns_d, bins=50, color="green", alpha=0.8)

s_nd = np.hstack(rng.normal(0.5, 0.1, size=50) * 50)
_ = axs[0][1].hist(s_nd, bins=50, color="green", alpha=0.8)

ns_nd = np.hstack(rng.normal(0.5, 0.2, size=200) * 50)
_ = axs[0][0].hist(ns_nd, bins=50, color="green", alpha=0.8)


fig.tight_layout()
fig.savefig("sparse.png")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

rng = np.random.RandomState(10)  # deterministic random data

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

# axs[0][0].set_xlim([0, 50])
axs[0][0].set_ylim([0, 1])
plt.setp(axs, xticks=np.arange(1, 11), yticks=[0, 0.5, 1])

axs[0][0].set_title("Non-sparse, non-distributed")
axs[0][1].set_title("Sparse, non-distributed")
axs[1][0].set_title("Non-sparse, distributed")
axs[1][1].set_title("Sparse, distributed")

axs[1][0].set_xlabel(r"$i$")
axs[1][1].set_xlabel(r"$i$")

axs[0][0].set_ylabel(r"$P(i)$")
axs[1][0].set_ylabel(r"$P(i)$")

objects = (
    r"$x_1$",
    r"$x_2$",
    r"$x_3$",
    r"$x_4$",
    r"$x_5$",
    r"$x_6$",
    r"$x_7$",
    r"$x_8$",
    r"$x_9$",
    r"$x_10$",
)
y_pos = np.arange(1, len(objects) + 1)

s_d = np.array([0.08, 0.11, 0.12, 0.11, 0.08, 0.09, 0.13, 0.1, 0.09, 0.1])
_ = axs[1][1].bar(y_pos, s_d, align="center", alpha=0.5)

ns_d = np.array([0.42, 0.48, 0.51, 0.52, 0.5, 0.51, 0.52, 0.57, 0.5, 0.51])
_ = axs[1][0].bar(y_pos, ns_d, align="center", alpha=0.5)

s_nd = np.array([0.1, 0.14, 0.18, 0.16, 0.13, 0.08, 0.0, 0.0, 0.0, 0.0])
_ = axs[0][1].bar(y_pos, s_nd, align="center", alpha=0.5)

ns_nd = np.array([0.41, 0.52, 0.61, 0.52, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0])
_ = axs[0][0].bar(y_pos, ns_nd, align="center", alpha=0.5)


fig.tight_layout()
fig.savefig("sparse.png")
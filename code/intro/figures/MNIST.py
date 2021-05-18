import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print(train_X.shape)
sum_X = np.sum(train_X, axis=0)
print(train_X[0])
print(np.max(sum_X))
sum_X = sum_X / (60000 * 256)
print(np.max(sum_X))
print(sum_X.shape)

fig, axs = plt.subplots(1, 3)

ax0 = plt.subplot(221)
ax0.set_title("Example: digit 2")
ax0.imshow(train_X[5], cmap="Greys")
ax0.set_xticks([1, 14, 28])
ax0.set_yticks([1, 14, 28])

ax1 = plt.subplot(222)
ax1.set_title("Pixel usage frequency (2D)")
plt1 = ax1.imshow(sum_X.reshape(28, 28), cmap="Greys", vmax=1, vmin=0)
ax1.set_xticks([1, 14, 28])
ax1.set_yticks([1, 14, 28])

plt.colorbar(plt1, ticks=[0, 0.5, 1])

ax2 = plt.subplot(212)
ax2.set_title("Pixel usage frequency (1D)")
ax2.bar(np.arange(28 * 28), sum_X.flatten(), align="center", alpha=0.7, color="Black")
ax2.set_xlabel("Pixel")
ax2.set_ylabel("frequency")
ax2.set_xlim([1, 784])
ax2.set_xticks([1, 98, 196, 294, 392, 490, 588, 686, 784])
ax2.set_yticks([0, 0.5, 1])

fig.tight_layout()

plt.savefig("MNIST.png")

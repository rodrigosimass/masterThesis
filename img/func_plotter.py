import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 70000, 10)
y1 = (x * 6272) / 8e6
y2 = (1120 * x) / 8e6
y3 = [(8820 * 8820 / 2) / 8e6] * 10
plt.plot(x, y1, "-r", label="Raw")
plt.plot(x, y2, "-b", label="Sparse Codes")
plt.plot(x, y3, "-g", label="Memory matrix")
plt.title("Space required to store the MNIST dataset")
plt.xlabel("N", color="#1C2833")
plt.ylabel("memory (MB)", color="#1C2833")
plt.legend(loc="upper left")
plt.grid()
plt.show()
plt.savefig("img/writing/compression.png")

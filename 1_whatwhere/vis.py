import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import glob, os
import random
import gzip

from Utils import *

verbose = 2

""" con = np.fromfile("con.bin", dtype=np.int32)
con = con.reshape(28, 28, 30)
plt.imshow(con[:, :, 0]) """

H = parse_par("mnistL4.par", "H")
M = parse_par("mnistL4.par", "M")
ntrpat = parse_par("mnistL4.par", "ntrpat")
ntetrpat = parse_par("mnistL4.par", "ntetrpat")
ntepat = parse_par("mnistL4.par", "ntepat")

if verbose > 0:
    print(f"H:{H},  M:{M},  ntrpat:{ntrpat},  ntetrpat:{ntetrpat},  ntepat:{ntepat}")

N = H * M
""" load images and labels """
trimgs, trlbls = load_imgs_lbls("trinpats2.dat", "trlbl.dat", ntrpat, 28)
teimgs, telbls = load_imgs_lbls("teinpats2.dat", "telbl.dat", ntepat, 28)
tetrimgs, tetrlbls = load_imgs_lbls("tetrinpats2.dat", "trlbl.dat", ntetrpat, 28)

""" hidden activities of patterns """
trhid = load_hidden_rep("trhid.dat", ntrpat, H, M)
tehid = load_hidden_rep("tehid.dat", ntepat, H, M)
tetrhid = load_hidden_rep("tetrhid.dat", ntetrpat, H, M)


""" load weight matrixes """
WL4 = load_W("wijl4.dat", 784, N)
WL23 = load_W("wijl23.bin", N, N)

""" load biases """
bL4 = load_bias("bjl4.dat", N)
bL23 = load_bias("bjl23.bin", N)

""" my_hid = np.dot(trimgs.reshape(600, 784), WL4).reshape((600, 30, 100))
print("my_hid max:", np.max(my_hid), " min:", np.min(my_hid))
print("trhid max:", np.max(trhid), " min:", np.min(trhid))
print(trhid.max)
print()
fig, axs = plt.subplots(2)
axs[0].imshow(trhid[0])
axs[1].imshow(my_hid[0])
plt.show()
 """
if verbose > 1:
    show_pat_example("TRAIN", trimgs, trlbls, trhid, ntrpat)
    # show_pat_example("TEST", teimgs, telbls, ntepat, 28, num_examples=1)
    # show_pat_example("TRAINTEST", tetrimgs, tetrlbls, ntetrpat, 28, num_examples=1)

    show_W("WL4", WL4, 784, N)
    show_W_HC_examples("WL4", WL4, M, H, num_examples=9)

    show_W("WL23", WL23, N, N)
    print("Is WL23 symmetric: " + str(check_symmetric(WL23)))

    show_bias("bL4", bL4, N)
    show_bias("bL23", bL23, N)

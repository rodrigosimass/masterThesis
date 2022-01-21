import numpy as np
from scipy.sparse import csr_matrix
from tqdm import trange


def naive_encode1(imgs, B):
    """
    Naive Coding Strategy 1 from: Sa Couto and Whichert 2020.
    The B most active bits of each image will be set to 1, the rest to 0.
    @param imgs: (3d np array) of images
    @param B: (int) number of active pits per code
    @returns imgs_copy: (2d csr_matrix) with binary codes.
    """
    imgs_copy = np.zeros_like(imgs)
    imgs_copy = imgs_copy.reshape(imgs.shape[0], -1)

    for i in trange(imgs.shape[0], desc="Naive encoding"):
        img = imgs[i].flatten()
        copy = imgs_copy[i]

        most_active = np.argsort(img)[-B:]
        copy[most_active] = 1

    return csr_matrix(imgs_copy)


def naive_encode2(imgs, T):
    """
    Naive Coding Strategy 2 from: Sa Couto and Whichert 2020.
    Pixel values below T are set to 0, above T are set to 1.
    @param imgs: (3d np array) of images
    @param T: (float) threshold [0,1]
    @returns imgs_copy: (2d csr_matrix) with binary codes.
    """
    imgs_copy = np.copy(imgs).reshape(imgs.shape[0], -1)

    imgs_copy[imgs_copy >= T] = 1
    imgs_copy[imgs_copy < T] = 0

    return csr_matrix(imgs_copy)

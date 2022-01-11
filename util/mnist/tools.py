import numpy as np
from random import randrange


def idxs_first_per_class(lbls):
    l_idxs = []
    for i in range(10):
        class_idxs = np.array(np.where(lbls == i)).flatten()
        l_idxs.append(class_idxs[0])
    return l_idxs


def idxs_last_per_class(lbls):
    l_idxs = []
    for i in range(10):
        class_idxs = np.array(np.where(lbls == i)).flatten()
        l_idxs.append(class_idxs[-1])
    return l_idxs


def idxs_x_random_per_class(lbls, x=1, seed=False):
    idxs = np.zeros((x, 10))

    if seed:
        np.random.seed(0)

    for i in range(10):  # class
        class_idxs = np.array(np.where(lbls == i)).flatten()
        rand_idxs = np.random.permutation(class_idxs)[:x]
        idxs[:, i] = rand_idxs
    return idxs.flatten().astype(int)


def idxs_class_evolution(lbls, lbl, n):
    l_idxs = []
    class_idxs = np.array(np.where(lbls == lbl)).flatten()
    step = int(class_idxs.shape[0] / n)
    for i in range(n):
        idx = class_idxs[i * step]
        l_idxs.append(idx)
    return l_idxs


def load_idxfile(filename):
    import struct

    with open(filename, "rb") as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0:
            raise Exception("Invalid idx file: unexpected magic number!")
        dtype, ndim = ord(_file.read(1)), ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0] for _ in range(ndim)]
        data = np.fromfile(_file, dtype=np.dtype(np.uint8).newbyteorder(">")).reshape(
            shape
        )
    return data


def read_mnist(dim=[28, 28], n_train=60000, n_test=10000, one_hot=False):

    """
    Read mnist train and test data.
    Images are normalized to be in range [0,1].
    Labels are one-hot coded if flag is True.
    """

    path = "datasets/mnist/"

    train_imgs = load_idxfile(path + "train-images-idx3-ubyte")
    train_imgs = train_imgs / 255.0
    train_imgs = train_imgs.reshape(-1, dim[0], dim[1])

    train_lbls = load_idxfile(path + "train-labels-idx1-ubyte")
    if one_hot:
        train_lbls_1hot = np.zeros((len(train_lbls), 10), dtype=np.float32)
        train_lbls_1hot[range(len(train_lbls)), train_lbls] = 1.0

    test_imgs = load_idxfile(path + "t10k-images-idx3-ubyte")
    test_imgs = test_imgs / 255.0
    test_imgs = test_imgs.reshape(-1, dim[0], dim[1])

    test_lbls = load_idxfile(path + "t10k-labels-idx1-ubyte")

    if one_hot:
        test_lbls_1hot = np.zeros((len(test_lbls), 10), dtype=np.float32)
        test_lbls_1hot[range(len(test_lbls)), test_lbls] = 1.0
    if one_hot:
        return (
            train_imgs[:n_train],
            train_lbls_1hot[:n_train],
            test_imgs[:n_test],
            test_lbls_1hot[:n_test],
        )
    else:
        return (
            train_imgs[:n_train],
            train_lbls[:n_train],
            test_imgs[:n_test],
            test_lbls[:n_test],
        )

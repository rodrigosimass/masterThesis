import numpy as np
from tqdm import trange
from scipy.sparse import csr_matrix
import math
import random
import matplotlib.pyplot as plt


""" ------------------------------------------------------------ """
""" ------------------- Description creators ------------------- """
""" ------------------------------------------------------------ """


def noisy_x_hot_encoding(lbls, x=50, Pc=0.5, Pr=0.0, n_classes=10):
    """
    "Stochastic" vervion of x_hot: the x units correscoping to the class will
    activate with Pclass (high value), the rest of the array will activate with Prest (low value)
    @returns descs: 2d np array with descriptions
    """
    descs = np.zeros(shape=(lbls.shape[0], n_classes * x))
    for i in trange(lbls.shape[0], desc="Noisy-x-hot", leave=False, unit="lbls"):
        lbl = lbls[i]
        start = lbl * x
        end = start + x
        for j in range(descs.shape[1]):
            if start <= j < end:
                if Pc > random.random():
                    descs[i][j] = 1
            else:
                if Pr > random.random():
                    descs[i][j] = 1
    return descs


def cochlea_encodeing(lbls, class_total=11, class_on=3, n_classes=10):
    """
    Sparser version of x-hot encoding.
    Each class has its own space (class_total) but will only have the central bits of its sapce on.
    Sparsity = class_total * n_classes / class_on
    Size = n_classes * class_total
    @param lbls : 1D np array (n_patterns * 1) with int labels like {0,1,2,...,n_class}
    @param class_total : total class space (should be odd number)
    @param class_on : on bits per class
    @param n_class : number of different classes (MNIST is 10)
    @returns descs: 2D np binary array with encoding
    """
    descs = np.zeros(shape=(lbls.shape[0], n_classes * class_total))
    for i in range(lbls.shape[0]):
        lbl = lbls[i]
        start = lbl * class_total + int((class_total - class_on) / 2)
        end = start + 3
        descs[i][start:end] = 1
    return descs


def bin_array(num, m):
    """Convert a positive integer num into an m-sized binary vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def bin_encoding(lbls, n_classes=10):
    """
    Transforms integer lables into binary labels.
    @param lbls : 1D np array (n_patterns * 1) with int labels like {0,1,2,...,n_class}
    @param n_class : number of different classes (MNIST is 10)
    @returns bin_enc: 2D np binary array with encoding
    """
    bits_per_lbl = math.ceil(math.log2(n_classes))
    n_patterns = lbls.shape[0]

    bin_enc = np.zeros(shape=(n_patterns, bits_per_lbl), dtype=int)

    for i in trange(n_patterns, desc="Encoding labels", leave=False):
        bin_enc[i] = bin_array(lbls[i], bits_per_lbl)

    return bin_enc


def x_hot_encoding(lbls, x=1, n_classes=10, reordered=False, order=None):
    """
    Transforms an array of integer labels into binary x-hot codes.
    Sparsity of encoding will always be 1/n_class.
    @param lbls : 1D np array (n_patterns * 1) with int labels like {0,1,2,...,n_class}
    @param x : desired number of 1s in the encoding (default is one-hot encoding)
    @param n_class : number of different classes (MNIST is 10)
    @returns: 2D np binary array with encoding
    """
    n_patterns = lbls.shape[0]

    x_hot = np.zeros(shape=(n_patterns, n_classes * x), dtype=int)

    for i in trange(n_patterns, desc="Encoding labels", leave=False):
        lbl = lbls[i]
        start = lbl * x
        end = start + x
        x_hot[i][start:end] = 1

    return x_hot


def join(descs, codes):
    """
    Concatenates description (Desc|Codes).
    @param codes: 2D csr matrix (n_codes * code_size) with binary code.
    @param descs: 2D np array (n_codes * desc_size) with binary coded description.
    @returns desCodes: 2D csr binary matrix (n_codes * (desc_size + code_size)) (Desc|Codes)
    """
    if codes.shape[0] != descs.shape[0]:
        print(
            f"WARNING: codes shape[0] ({codes.shape[0]}) differs from descs.shape[0] ({descs.shape[0]})"
        )
        exit(0)
    n = codes.shape[0]
    code_size = codes.shape[1]
    desc_size = descs.shape[1]

    desCodes = np.empty((n, desc_size + code_size))
    desCodes[:, 0:desc_size] = descs
    desCodes[:, desc_size : desc_size + code_size] = codes.toarray()

    return csr_matrix(desCodes)


def separate(desCodes, desc_size):
    """
    UNDO of join() function
    @param desCodes: 2D csr binary matrix (n_patterns * (desc_size + code_size)) descs (left) + codes (right)
    @param desc_size: index for column-wise slice
    @returns descs: 2D np array (n_codes * desc_size)
    @returns codes: 2D csr matrix (n_codes * desc_size)
    """
    descs = desCodes[:, 0:desc_size].toarray()
    codes = csr_matrix(desCodes[:, desc_size:])
    return (descs, codes)


def get_descs(desCodes, desc_size):
    return desCodes[:, 0:desc_size]


def get_codes(desCodes, desc_size):
    return desCodes[:, desc_size:]


def delete_descs(desCodes, desc_size):
    desCodes = desCodes.toarray()
    desCodes[:, 0:desc_size] = 0
    return csr_matrix(desCodes)


def plot_class_act(descs, lbls, c):
    s = np.mean(descs[lbls == c], axis=0)
    plt.bar(np.arange(s.shape[0]), s)


""" ------------------------------------------------------------ """
""" ------------------- Evaluation functions ------------------- """
""" ------------------------------------------------------------ """


def interval_classifier(prediction, truth, n=11, n_classes=10, verbose=False):
    correct = 0
    for i in trange(prediction.shape[0], leave=False, desc="Interval classifier"):
        max = 0
        pred = 0
        for c in range(n_classes):
            class_sum = np.sum(prediction[i][c * n : (c + 1) * n])
            if class_sum > max:
                max = class_sum
                pred = c
        if pred == truth[i]:
            correct += 1
    if verbose:
        print(f"Interval accuracy: {correct/prediction.shape[0]*100:.2f}")
    return correct / prediction.shape[0]


def generate(gen_desCodes, full_memory, desc_size):
    ret = full_memory.retrieve(gen_desCodes)
    gen_codes = separate(ret, desc_size)[1]
    return gen_codes


# %%
if __name__ == "__main__":

    lbls = np.arange(10)
    print("Labels before encoding:")
    print(lbls)

    one_hot = x_hot_encoding(lbls)
    print("Description (One-hot):")
    print(one_hot)

    three_hot = x_hot_encoding(lbls, x=3, n_classes=10)
    print("Description (Three-hot):")
    print(three_hot)

    print("\n Now with some binary codes:\n")

    print("Codes (10 random binary patterns of size 5):")
    codes = csr_matrix(np.random.randint(low=0, high=2, size=(10, 5)))
    print(codes.toarray())

    desc_codes = join(one_hot, codes)
    print("Desc_Codes (Desc|Codes)")
    print("||L L L L L L L L L L C C C C C|")
    print(desc_codes.toarray())
    desc, codes = separate(desc_codes, desc_size=10)

    print("\nseparate(Desc + Codes) yields:")
    print("desc:")
    print(desc)
    print("and codes:")
    print(codes.toarray())

    cochlea = cochlea_encodeing(lbls)
    print("Cochlea encoding")
    print(cochlea.shape)
    cochlea[2:4] = 0
    interval_classifier(cochlea, lbls)

# %%

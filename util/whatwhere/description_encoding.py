import numpy as np
from tqdm import trange
from scipy.sparse import csr_matrix
import math
import random
import matplotlib.pyplot as plt


""" ------------------------------------------------------------ """
""" ------------------- Description creators ------------------- """
""" ------------------------------------------------------------ """


def noisy_x_hot_encoding(lbls, x=50, p_c=0.5, p_r=0.0, num_classes=10):
    """
    "Stochastic" vervion of x_hot: the x units correscoping to the class will
    activate with p_class (high value), the rest of the array will activate with p_rest (low value)
    """
    descs = np.zeros(shape=(lbls.shape[0], num_classes * x))
    for i in range(lbls.shape[0]):
        lbl = lbls[i]
        start = lbl * x
        end = start + x
        for j in range(descs.shape[1]):
            if start <= j < end:
                if p_c > random.random():
                    descs[i][j] = 1
            else:
                if p_r > random.random():
                    descs[i][j] = 1
    return descs


def cochlea_encodeing(lbls, class_total=11, class_on=3, num_classes=10):
    """
    Sparser version of x-hot encoding.
    Each class has its own space (class_total) but will only have the central bits of its sapce on.
    Sparsity = class_total * num_classes / class_on
    Size = num_classes * class_total
    @param lbls : 1D np array (num_patterns * 1) with int labels like {0,1,2,...,num_class}
    @param class_total : total class space (should be odd number)
    @param class_on : on bits per class
    @param num_class : number of different classes (MNIST is 10)
    @returns descs: 2D np binary array with encoding
    """
    descs = np.zeros(shape=(lbls.shape[0], num_classes * class_total))
    for i in range(lbls.shape[0]):
        lbl = lbls[i]
        start = lbl * class_total + int((class_total - class_on) / 2)
        end = start + 3
        descs[i][start:end] = 1
    return descs


def bin_array(num, m):
    """Convert a positive integer num into an m-sized binary vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def bin_encoding(lbls, num_classes=10):
    """
    Transforms integer lables into binary labels.
    @param lbls : 1D np array (num_patterns * 1) with int labels like {0,1,2,...,num_class}
    @param num_class : number of different classes (MNIST is 10)
    @returns bin_enc: 2D np binary array with encoding
    """
    bits_per_lbl = math.ceil(math.log2(num_classes))
    num_patterns = lbls.shape[0]

    bin_enc = np.zeros(shape=(num_patterns, bits_per_lbl), dtype=int)

    for i in trange(num_patterns, desc="Encoding labels", leave=False):
        bin_enc[i] = bin_array(lbls[i], bits_per_lbl)

    return bin_enc


def x_hot_encoding(lbls, x=1, num_classes=10, reordered=False, order=None):
    """
    Transforms an array of integer labels into binary x-hot codes.
    Sparsity of encoding will always be 1/num_class.
    @param lbls : 1D np array (num_patterns * 1) with int labels like {0,1,2,...,num_class}
    @param x : desired number of 1s in the encoding (default is one-hot encoding)
    @param num_class : number of different classes (MNIST is 10)
    @returns: 2D np binary array with encoding
    """
    num_patterns = lbls.shape[0]

    x_hot = np.zeros(shape=(num_patterns, num_classes * x), dtype=int)

    for i in trange(num_patterns, desc="Encoding labels", leave=False):
        lbl = lbls[i]
        start = lbl * x
        end = start + x
        x_hot[i][start:end] = 1

    return x_hot


def concatenate(descs, codes):
    """
    Concatenates description (Desc|Codes).
    @param codes: 2D csr matrix (num_codes * code_size) with binary code.
    @param descs: 2D np array (num_codes * desc_size) with binary coded description.
    @returns desc_code: 2D csr binary matrix (num_codes * (desc_size + code_size)) (Desc|Codes)
    """
    if codes.shape[0] != descs.shape[0]:
        print(
            f"WARNING: codes shape[0] ({codes.shape[0]}) differs from descs.shape[0] ({descs.shape[0]})"
        )
        exit(0)

    desc_code = np.hstack((descs, codes.toarray()))

    return csr_matrix(desc_code)


def deconcatenate(concatenated, desc_size=10):
    """
    UNDO of concatenate() function
    @param concatenated: 2D csr binary matrix (num_patterns * (desc_size + code_size)) descs (left) + codes (right)
    @param desc_size: index for column-wise slice
    @returns descs: 2D np array (num_codes * desc_size)
    @returns codes: 2D csr matrix (num_codes * desc_size)
    """

    split = np.split(concatenated.toarray(), indices_or_sections=[desc_size], axis=1)
    descs = split[0]
    codes = csr_matrix(split[1])

    return descs, codes


def detach(arr, idx):
    l = arr[:, 0:idx]
    r = arr[:, idx:]
    return (l, r)


def plot_class_act(descs, lbls, c):
    s = np.mean(descs[lbls == c], axis=0)
    plt.bar(np.arange(s.shape[0]), s)


""" ------------------------------------------------------------ """
""" ------------------- Evaluation functions ------------------- """
""" ------------------------------------------------------------ """


def interval_classifier(prediction, truth, n=11, num_classes=10, verbose=False):
    correct = 0
    for i in trange(prediction.shape[0], leave=False, desc="Interval classifier"):
        max = 0
        pred = 0
        for c in range(num_classes):
            class_sum = np.sum(prediction[i][c * n : (c + 1) * n])
            if class_sum > max:
                max = class_sum
                pred = c
        if pred == truth[i]:
            correct += 1
    if verbose:
        print(f"Interval accuracy: {correct/prediction.shape[0]*100:.2f}")
    return correct / prediction.shape[0]


def autoassociation(descs_codes, desc_size, full_memory):
    """
    Measures how good the memory is at maintaining the correct label in
    auto-association.
    @param descs_codes: 2D binary csr_matrix (desc|code)
    @param desc_size: size of description
    @param full_memory: AAWN memory already storing descs_codes
    @returns descs: descs before memory
    @returns descs_ret: descs after memory
    """
    ret = full_memory.retrieve(descs_codes)
    descs_ret = deconcatenate(ret, desc_size)[0]

    return descs_ret


def completion(descs_codes, desc_size, full_memory):
    """
    Measures how good the memory is at completing the missing
    desc of stored patterns
    @param descs_codes: 2D binary csr_matrix (desc|code)
    @param desc_size: size of description
    @param full_memory: AAWN memory already storing descs_codes
    """
    codes = deconcatenate(descs_codes, desc_size)[1]
    empty_descs = np.zeros((codes.shape[0], desc_size))
    just_codes = concatenate(empty_descs, codes)
    ret = full_memory.retrieve(just_codes)
    descs_ret = deconcatenate(ret, desc_size)[0]

    return descs_ret


def classification(tst_descs_codes, desc_size, full_memory):
    """
    Measures how good the memory is classifying unseen codes.
    @param descs_codes: 2D binary csr_matrix (desc|code) with tst set
    @param desc_size: size of description
    @param full_memory: AAWN memory already storing descs_codes
    """
    tst_codes = deconcatenate(tst_descs_codes, desc_size)[1]
    empty_descs = np.zeros((tst_codes.shape[0], desc_size))
    just_codes = concatenate(empty_descs, tst_codes)
    ret = full_memory.retrieve(just_codes)
    descs_ret = deconcatenate(ret, desc_size)[0]

    return descs_ret


def generate(gen_descs_codes, full_memory, desc_size):
    ret = full_memory.retrieve(gen_descs_codes)
    gen_codes = deconcatenate(ret, desc_size)[1]
    return gen_codes


# %%
if __name__ == "__main__":

    lbls = np.arange(10)
    print("Labels before encoding:")
    print(lbls)

    one_hot = x_hot_encoding(lbls)
    print("Description (One-hot):")
    print(one_hot)

    three_hot = x_hot_encoding(lbls, x=3, num_classes=10)
    print("Description (Three-hot):")
    print(three_hot)

    print("\n Now with some binary codes:\n")

    print("Codes (10 random binary patterns of size 5):")
    codes = csr_matrix(np.random.randint(low=0, high=2, size=(10, 5)))
    print(codes.toarray())

    desc_codes = concatenate(one_hot, codes)
    print("Desc_Codes (Desc|Codes)")
    print("||L L L L L L L L L L C C C C C|")
    print(desc_codes.toarray())
    desc, codes = deconcatenate(desc_codes, desc_size=10)

    print("\ndeconcatenate(Desc + Codes) yields:")
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

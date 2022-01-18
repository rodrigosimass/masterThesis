#%%
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import trange


def H(vec):
    """
    Applies the heaviside function to a vector.
    @param vec: np array
    @return vec: np array (binary)
    """
    vec[vec >= 0] = 1
    vec[vec < 0] = 0
    return vec


class AAWN:
    """
    Binary Willshaw Network for Auto-Association of binary patterns
    Atributes:
        n          : number of neurons
        W          : weight matrix (n*n csr_matrix)
        n_stored : number of stored pattenrs
    """

    def __init__(self, n):
        self.n = n  # number of neurons
        self.W = None  # csr_matrix weight matrix
        self.n_stored = 0  # num of stored pattens

    def sparsity(self):
        """
        Computes the sparsity of the weight matrix
        @returns s: sparsity
        """
        if self.W != None:
            return self.W.nnz / (self.n * self.n)
        else:
            print("EMPTY memory!")
            return 0.0

    def store(self, patterns):
        """
        Stores patterns in the memory
        @param pattenrs: 2D-np array (num patterns * size of patterns)
        """
        if self.W == None:
            weights = np.zeros((self.n, self.n))
        else:
            weights = self.W.toarray()

        for i in trange(patterns.shape[0], desc="Training WN", leave=False):
            pattern = patterns[i]
            n_nz = len(pattern.indices)
            for i in range(n_nz):
                for j in range(i, n_nz):
                    idx_i = pattern.indices[i]
                    idx_j = pattern.indices[j]
                    weights[idx_i, idx_j] = 1
                    weights[idx_j, idx_i] = 1

        self.W = csr_matrix(weights)
        self.n_stored += patterns.shape[0]

    def retrieve(self, cues):
        """
        Retrives cues with soft-thresholding (Palm et al.)
        @param cues: 2D csr_matrix, pattern size must match self.n
        @return ret: 2D csr_matrix with the same shape as cues
        """
        ret = np.zeros(cues.shape)

        s = csr_matrix.dot(cues, self.W)  # dentritic potential

        for i in trange(cues.shape[0], desc="Retrieving", leave=False):
            aux = s[i].toarray()
            m = np.max(aux)
            if m != 0:
                aux = aux - m
                ret[i] = H(aux)
            else:
                ret[i] = 0

        return csr_matrix(ret)

    def forget(self):
        """
        Reset the memory
        """
        self.W = None
        self.nnz = 0
        self.s = 0.0
        self.n_stored = 0


""" ------------------------------------------------------------ """
""" ------------------- Evaluation functions ------------------- """
""" ------------------------------------------------------------ """


def perfect_retrieval_error(codes, ret):
    """
    Computes the Perfect retrieval error
    @param codes: 2D csr_matrix with original codes
    @param ret: 2D csr_matrix with memory output
    @return err: perfect retrievals / num codes
    """
    codes = csr_matrix(codes)
    ret = csr_matrix(ret)

    miss = 0
    for i in range(ret.shape[0]):
        if (codes[i] != ret[i]).nnz != 0:
            miss += 1

    err = miss / ret.shape[0]

    return err


def hamming_distance_detailed(codes, ret):
    """
    Negative&Positive version of hamming distance.
    Computes the hamming distance from extra bits (0s that become 1s),
    and from lost bits (1s that become 0s)
    @param codes: 2D csr_matrix with original codes
    @param ret: 2D csr_matrix with memory output
    @return err_extra: hamm dist from extra bits
    @return err_lost: hamm dist from lost bits
    @return hamm_dist: total hamming distance
    """
    diff = (codes - ret).toarray()

    extra_bits = np.count_nonzero(diff == -1)
    lost_bits = np.count_nonzero(diff == 1)

    total_bits = ret.shape[0] * ret.shape[1]

    hamm_dist = (extra_bits + lost_bits) / total_bits

    return (extra_bits / total_bits, lost_bits / total_bits, hamm_dist)


def err_1NNclassifier(trn, trn_lbls, tst, tst_lbls):
    """
    Computes the classification error of a simple 1NN dot product classifier (Sa-Couto 2020).
    @param trn: 2D csr_matrix with trn (ret set)
    @param trn_lbls: np array with true labels
    @param tst: 2D csr_matrix with tst (coded set)
    @param tst_lbls: np array with labels of tst
    @tsturn err: classification error
    """
    # (n_tst*size).(n_trn*size).T
    sim = csr_matrix.dot(tst, trn.T)  # (n_tst*n_trn)
    # for each tst example, which is the nearest trn example
    nn = csr_matrix.argmax(sim, axis=1)  # (n_tst*1)

    truth = tst_lbls.flatten()  # (n_tst*1)
    prediction = trn_lbls[nn].flatten()  # label of the most similar neighbour

    diff = prediction - truth
    err = np.count_nonzero(diff) / diff.shape[0]

    return err


def eval(codes, codes_lbls, ret, ret_lbls):
    """
    Calls all the evaluation functions at once
    @param codes: 2D csr_matrix with coded set (classifier tst set)
    @param codes_lbls: np array with labels of the coded set
    @param ret: 2D csr_matrix with ret set (classifier trn set)
    @param ret_lbls: np array with true labels
    @return pre: perfect retrieval error
    @return hd_extra: hamming distance from extra bits
    @return hd_lost: hamming distance from lost bits
    @return hd: total hamming distance
    @return err_1nn: classification error of 1NN classifier
    """
    pre = perfect_retrieval_error(codes, ret)
    hd_extra, hd_lost, hd = hamming_distance_detailed(codes, ret)
    err_1nn = err_1NNclassifier(ret, ret_lbls, codes, codes_lbls)

    return pre, hd_extra, hd_lost, hd, err_1nn


if __name__ == "__main__":

    right = np.array([0, 0, 1, 1])
    left = np.array([1, 1, 0, 0])
    dataset = csr_matrix(np.vstack((right, left)))

    print("Dataset:\n", dataset.toarray())
    n_patterns = dataset.shape[0]
    pattern_size = dataset.shape[1]

    wn = AAWN(pattern_size)
    wn.store(dataset)

    print(f"Memory has {wn.n_stored} stored patterns.")
    print("Weight matrix:\n", wn.W.toarray())
    print(f"Sparsity of W = {wn.sparsity()}")

    right_noisy = np.array([1, 0, 1, 1])
    left_noisy = np.array([1, 0, 0, 0])
    cues = csr_matrix(np.vstack((right_noisy, left_noisy)))

    print("Cues (before memory):\n", cues.toarray())
    ret = wn.retrieve(cues)
    print("Retrieved (after memory):\n", ret.toarray())

    prr = perfect_retrieval_error(dataset, ret)
    print(f"Perfect retrieval error: {prr}")

    hd_extra, hd_lost, hd = hamming_distance_detailed(dataset, ret)

    print(
        f"Hamming Distance (Extra): {hd_extra} + Hamming Distance (Lost): {hd_lost} = Hamming Distance: {hd}"
    )

    print("Forgeting...")
    wn.forget()
    print(f"Memory has {wn.n_stored} stored patterns.")

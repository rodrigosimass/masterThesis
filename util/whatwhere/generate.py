import numpy as np
from scipy.sparse import csr_matrix

from util.whatwhere.description_encoding import separate


def compute_dist(codes):
    d = np.mean(codes, axis=0)
    return np.array(d).reshape(
        -1,
    )


def compute_dists(codes, lbls, n_classes=10):
    """
    Computes de probability of activity for eapch dimention of the codes, classwise.
    @returns dists: (2d np array) with the distribution of each class
    """
    dists = np.empty((n_classes, codes.shape[1]))

    for c in range(n_classes):
        dists[c] = compute_dist(codes[lbls == c])
    return dists


def avg_bits_per_code(codes):
    nnz = codes.nnz
    n_codes = codes.shape[0]
    return nnz / n_codes

def std_bits_per_code(codes):
    return np.std(np.sum(codes,axis=-1))

def sample_from_dist(dist, n=1):
    samples = np.random.rand(n, dist.shape[0])
    samples = samples - dist  # subtracts the dist from each row
    samples[samples >= 0] = 0
    samples[samples < 0] = 1
    return samples


def sample_from_dists(dists, gen_lbls):
    """
    @param dists: distribution for each class
    """
    n_samples = gen_lbls.shape[0]
    code_size = dists.shape[1]
    samples = np.empty((n_samples, code_size))

    for i in range(n_samples):
        lbl = gen_lbls[i]
        dist = dists[lbl]
        s = sample_from_dist(dist).flatten()
        samples[i] = s

    return csr_matrix(samples)


def create_gen_lbls(n_classes=10, n_exs=10, transpose=False):
    if transpose:
        gen_lbls = np.repeat(np.arange(n_classes), n_exs)
    else:
        gen_lbls = np.tile(np.arange(n_classes), n_exs).reshape(n_exs, n_classes)

    return gen_lbls.flatten()

def iterative_generate(first_descCode, desc_size, code_sparsify = 20, increment=2, code_m=200, code_M=300, desc_sparsify=50, desc_m=90, desc_M=100, num_tries=3,  plot=True):

    l_code_s = []
    l_desc_s = []

    gen_lbl = np.array([c])
    first_desc, forst_code  = separate(first_descCode, desc_size)
    
    in_desc = first_desc
    in_code = first_code


    for _ in range(num_tries):
        in_desCode = join(in_desc, in_code)

        out_desCode = wn_desCodes.retrieve(in_desCode)
        out_desc, out_code = separate(out_desCode, desc_size)

        in_code_bits = np.sum(in_code)
        in_desc_bits = np.sum(in_desc)
        l_code_s.append(in_code_bits)
        l_desc_s.append(in_desc_bits)

        out_code_bits = np.sum(out_code)
        out_desc_bits = np.sum(out_desc)
        l_code_s.append(out_code_bits)
        l_desc_s.append(out_desc_bits)


        if plot:
            plot_gen_code_ret(in_code, out_code, in_desc.flatten(), out_desc.flatten(), c, just_class_interval=True)

        if (out_code_bits >= code_m and out_code_bits <= code_M) and (out_desc_bits>= desc_m and out_desc_bits <= desc_M):
            print("Success!")
            break
        
        else:
            Pdel_code = 1 - code_sparsify / out_code_bits
            Pdel_desc = 1 - desc_sparsify / out_desc_bits
            if Pdel_code < 1.0:
                print("Pdel code = ", Pdel_code)
                out_code = add_zero_noise(out_code, Pdel_code)
            if Pdel_desc < 1.0:
                print("Pdel desc = ", Pdel_desc)
                #out_desc = add_zero_noise(csr_matrix(out_desc), Pdel_desc)
                #out_desc = out_desc.toarray()
            
            in_code = out_code
            in_desc = first_desc
            code_sparsify += increment
    
    return l_code_s, l_desc_s

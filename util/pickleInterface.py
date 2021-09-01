import pickle
from .whatwhere.encoder import *
from .willshaw.memory import *


def store_ret(ret, k, Q, num_stored, Fs, n_epochs, b, T_what):
    run_name = "k" + str(k) + "_Fs" + str(Fs) + "_ep" + str(n_epochs) + "_b" + str(b)
    run_name += "_Q" + str(Q) + "_Tw" + str(T_what)
    pickle.dump(
        ret,
        open(f"pickles/{run_name}_n{num_stored}__ret.p", "wb"),
    )


def load_or_compute_features(
    trn_imgs, k, Fs, rng, n_epochs, b, plot=False, verbose=False
):
    run_name = "k" + str(k) + "_Fs" + str(Fs) + "_ep" + str(n_epochs) + "_b" + str(b)
    try:
        features = pickle.load(open(f"pickles/{run_name}__features.p", "rb"))
        if verbose:
            print(f"loaded features from pickles/{run_name}__features.p")
    except (OSError, IOError) as _:
        features, _ = learn_features(
            trn_imgs, k, Fs, rng, n_epochs, background=b, verbose=verbose
        )
        pickle.dump(
            features,
            open(f"pickles/{run_name}__features.p", "wb"),
        )

    return features


def load_or_compute_codes(
    trn_imgs,
    tst_imgs,
    k,
    Q,
    features,
    T_what,
    wta,
    n_epochs,
    b,
    Fs,
    verbose=False,
    test=False,
):
    run_name = "k" + str(k) + "_Fs" + str(Fs) + "_ep" + str(n_epochs) + "_b" + str(b)
    run_name += "_Q" + str(Q) + "_Tw" + str(T_what)

    try:
        codes = pickle.load(open(f"pickles/{run_name}__ww_trn.p", "rb"))
        if verbose:
            print(f"loaded codes from pickle: pickles/{run_name}__ww_trn.p")
    except (OSError, IOError) as _:
        codes = learn_codes(trn_imgs, k, Q, verbose, features, T_what, wta)
        if verbose:
            print(f"saving codes to pickles/{run_name}__ww_trn.p")
        pickle.dump(
            codes,
            open(f"pickles/{run_name}__ww_trn.p", "wb"),
        )

    AS = codes.nnz / (codes.shape[0] * codes.shape[1])
    densest = np.max(csr_matrix.sum(codes, axis=1)) / codes.shape[1]

    codes_tst = None
    if test:
        try:
            codes_tst = pickle.load(open(f"pickles/{run_name}__ww_tst.p", "rb"))
            if verbose:
                print(f"loaded codes from pickle: pickles/{run_name}__ww_tst.p")
        except (OSError, IOError) as _:
            codes_tst = learn_codes(tst_imgs, k, Q, verbose, features, T_what, wta)
            if verbose:
                print(f"saving codes to pickles/{run_name}__ww_tst.p")
            pickle.dump(
                codes_tst,
                open(f"pickles/{run_name}__ww_tst.p", "wb"),
            )

    if verbose:
        print(
            f"""Coded training set:
                avg sparsity = {AS}
                densest (%B) = {densest}
        """
        )

    return (codes, codes_tst, AS, densest)


def load_or_compute_will(run_name, codes_csr, factor_of_stored, verbose=False):
    num_stored = 784 * factor_of_stored
    try:
        will = pickle.load(
            open(f"pickles/{run_name}_fac{factor_of_stored}__will.p", "rb")
        )
        if verbose:
            print(
                f"loaded will from pickle: pickles/{run_name}_fac{factor_of_stored}__will.p"
            )
    except (OSError, IOError) as _:
        will = train(codes_csr, num_stored)
        pickle.dump(
            will,
            open(f"pickles/{run_name}_fac{factor_of_stored}__will.p", "wb"),
        )
        if verbose:
            print(
                f"saving trained willshaw to pickles/{run_name}_fac{factor_of_stored}__will.p"
            )

    sparsity = will.nnz / (will.shape[0] * will.shape[1])

    if verbose:
        if np.array_equal(will.toarray(), (will.toarray()).T):
            print("[OK] Willshaw matrix is symmetric")

        print(
            f"""W martix sparsity = {sparsity}
        """
        )

    return (will, sparsity)


def load_or_compute_ret(
    run_name,
    codes,
    will,
    factor_of_stored,
    verbose=False,
):

    try:
        ret = pickle.load(
            open(f"pickles/{run_name}_fac{factor_of_stored}__ret.p", "rb")
        )
        if verbose:
            print(
                f"loaded ret from pickle: pickles/{run_name}_fac{factor_of_stored}__ret.p"
            )
    except (OSError, IOError) as _:
        ret = retreive(codes, factor_of_stored, will)
        pickle.dump(
            ret,
            open(f"pickles/{run_name}_fac{factor_of_stored}__ret.p", "wb"),
        )
        if verbose:
            print(f"saving ret to pickles/{run_name}_fac{factor_of_stored}__ret.p")

    AS = ret.nnz / (ret.shape[0] * ret.shape[1])
    densest = np.max(csr_matrix.sum(ret, axis=1)) / ret.shape[1]
    if verbose:
        print(
            f"""Coded set:
                avg sparsity = {AS}
                densest (%B) = {densest}
        """
        )

    return (ret, AS, densest)
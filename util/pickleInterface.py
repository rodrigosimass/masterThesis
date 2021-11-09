import pickle
from .whatwhere.encoder import *
from .willshaw.memory import *

""" RUN-NAME GETTERS """


def get_features_run_name(k, Fs, n_epochs, b, classwise):
    return (
        "k"
        + str(k)
        + "_Fs"
        + str(Fs)
        + "_ep"
        + str(n_epochs)
        + "_b"
        + str(b)
        + "_cw"
        + str(classwise)
    )


def get_codes_run_name(k, Fs, n_epochs, b, Q, T_what, wta):
    return (
        "k"
        + str(k)
        + "_Fs"
        + str(Fs)
        + "_ep"
        + str(n_epochs)
        + "_b"
        + str(b)
        + "_Q"
        + str(Q)
        + "_Tw"
        + str(T_what)
        + "_wta"
        + str(wta)
    )


""" LOADERS - load file, exit() on error"""


def load_ret(run_name, set="trn"):
    try:
        ret = pickle.load(open(f"pickles/{run_name}__ret_{set}.p", "rb"))
    except (OSError, IOError) as _:
        print(f"ERROR: file <<pickles/{run_name}__ret_{set}.p>> not found...")
        exit(1)
    return ret


def load_codes(run_name, set="trn"):
    try:
        codes = pickle.load(open(f"pickles/{run_name}__codes_{set}.p", "rb"))
    except (OSError, IOError) as _:
        print(f"ERROR: file <<pickles/{run_name}__codes_{set}.p>> not found...")
        exit(1)

    return codes


def load_features(run_name):
    try:
        features = pickle.load(open(f"pickles/{run_name}__features.p", "rb"))
    except (OSError, IOError) as _:
        print(f"ERROR: file <<pickles/{run_name}__features.p>> not found...")
        exit(1)
    return features


def compute_features(
    trn_imgs,
    trn_lbls,
    k,
    Fs,
    rng,
    n_epochs,
    b,
    verbose=False,
    classwise=False,
):
    run_name = get_features_run_name(k, Fs, n_epochs, b, classwise)
    try:
        features = pickle.load(open(f"pickles/{run_name}__features.p", "rb"))
        if verbose:
            print(f"loaded features from pickles/{run_name}__features.p")
    except (OSError, IOError) as _:
        if classwise:
            features = learn_classwise_features(
                trn_imgs, trn_lbls, k, Fs, rng, n_epochs, background=b
            )
        else:
            features = learn_features(trn_imgs, k, Fs, rng, n_epochs, background=b)
        pickle.dump(
            features,
            open(f"pickles/{run_name}__features.p", "wb"),
        )

    return features


def compute_codes(
    imgs, k, Q, features, T_what, wta, n_epochs, b, Fs, verbose=False, set="trn"
):
    run_name = get_codes_run_name(k, Fs, n_epochs, b, Q, T_what, wta)

    try:
        codes = pickle.load(open(f"pickles/{run_name}__codes_{set}.p", "rb"))
        polar_params = pickle.load(open(f"pickles/{run_name}__polar_{set}.p", "rb"))
        if verbose:
            print(f"loaded codes from : pickles/{run_name}__codes_{set}.p")
            print(f"loaded polar params from : pickles/{run_name}__polar_{set}.p")
    except (OSError, IOError) as _:
        codes, polar_params = learn_codes(imgs, k, Q, features, T_what, wta)
        if verbose:
            print(f"saving codes to pickles/{run_name}__codes_{set}.p")
            print(f"saving polar params to pickles/{run_name}__polar_{set}.p")
        pickle.dump(
            codes,
            open(f"pickles/{run_name}__codes_{set}.p", "wb"),
        )
        pickle.dump(
            polar_params,
            open(f"pickles/{run_name}__polar_{set}.p", "wb"),
        )

    return (codes, polar_params)

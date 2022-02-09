import numpy as np
import wandb
from tqdm import trange
from util.mnist.tools import *
from util.pickleInterface import *
from util.whatwhere.encoder import *
from util.whatwhere.decoder import *
from util.whatwhere.generate import *
from util.whatwhere.noise import *
from util.willshaw.memory import *
from util.willshaw.plot import *
from util.pytorch.tools import np_to_grid
from util.distribution import *
from util.basic_utils import mse_detailed
from util.whatwhere.description_encoding import *


if len(sys.argv) < 2:
    print("ERROR! \nusage: python3 MLP.py <<0/1>> for wandb on or off")
    exit(1)
USE_WANDB = bool(int(sys.argv[1]))


""" Code generation params """
rng = np.random.RandomState(0)  # reproducible
K = 20
Q = 21
n_epochs = 5
b = 0.8
wta = True

Fs = 2
Tw = 0.75

"""Noisy-x-hot Description params"""
nxh_x = 5
nxh_Pc = 0.5
nxh_Pr = 0.0

""" Noise params """
l_prob = [1.0]  # each item in this list is a different wandb run
noise_type = "zero"  # none, zero, one TODO: add "both" noise option

TRIAL_RUN = True

""" load mnist """
imgs, lbls, _, _ = read_mnist(n_train=60000)
I = imgs.shape[1]
J = imgs.shape[2]


""" generate codes """
features = compute_features(imgs, lbls, K, Fs, rng, n_epochs, b, verbose=False)
codes, polar_params = compute_codes(
    imgs,
    K,
    Q,
    features,
    Tw,
    wta,
    n_epochs,
    b,
    Fs,
    verbose=False,
)

code_size = codes.shape[1]

if TRIAL_RUN:
    # Reduce the size of the datasets for debugging (2*fos)
    imgs = imgs[: code_size * 2]
    lbls = lbls[: code_size * 2]
    codes = codes[: code_size * 2]
    polar_params = polar_params[: code_size * 2]

""" Create clean desCodes (patterns to store in the memory) """
descs = noisy_x_hot_encoding(lbls, nxh_x, nxh_Pc, nxh_Pr)
desCodes = join(descs, codes)
desc_size = descs.shape[1]

""" Create artificial codes and descs """
gen_lbls = create_gen_lbls(n_classes=10, n_exs=3)
gen_descs = noisy_x_hot_encoding(gen_lbls, nxh_x, nxh_Pc, nxh_Pr)

prob_dists = compute_dists(codes, lbls)
gen_codes = sample_from_dists(prob_dists, gen_lbls)


for prob in tqdm(l_prob, desc="main", unit="run"):

    """Create the noisy desCodes"""
    codes_noisy = add_noise(codes, noise_type, prob)
    desCodes_noisy = join(descs, codes_noisy)

    gen_codes_noisy = add_noise(gen_codes, noise_type, prob)
    gen_desCodes_noisy = join(gen_descs, gen_codes_noisy)

    if USE_WANDB:
        wandb.init(
            project="generator_WN",
            entity="rodrigosimass",
            config={
                "km_K": K,
                "km_epochs": n_epochs,
                "km_wta": wta,
                "km_b": b,
                "km_Fs": Fs,
                "ww_Q": Q,
                "ww_Tw": Tw,
                "noise_type": noise_type,
                "noise_prob": prob,
                "NXH_x": nxh_x,
                "NXH_Pc": nxh_Pc,
                "NXH_Pr": nxh_Pr,
            },
        )

        name = "TRIAL_" if TRIAL_RUN else ""
        name += f"desCodes_{nxh_x}_"
        if noise_type == "none" or prob == 0.0:
            name += "none"
        else:
            name += noise_type
            name += "_p" + str(prob)
        wandb.run.name = name

        """ log the initial state (no memory) """
        log_dict = {}

        # reconstructions
        """ recons = recon_with_polar(codes_noisy, features, polar_params, Q, K)
        extra, lost, mse = mse_detailed(recons, imgs)

        log_dict["mse_extra"] = extra
        log_dict["mse_lost"] = lost
        log_dict["mse"] = mse """

        """ examples for visualization purposes """
        """ ex_idxs = idxs_x_random_per_class(lbls[:code_size], x=3, seed=True)
        ex_img = imgs[ex_idxs]
        ex_codes_noisy = codes_noisy[ex_idxs]
        ex_polar_params = polar_params[ex_idxs]
        ex_recon = recon_with_polar(ex_codes_noisy, features, ex_polar_params, Q, K)

        log_dict["MNIST"] = wandb.Image(np_to_grid(ex_img))
        log_dict["Ret"] = wandb.Image(code_grid(ex_codes_noisy, K, Q))
        log_dict["Recon_codes"] = wandb.Image(np_to_grid(ex_recon)) """

        wandb.log(log_dict, step=0)

    """ Initialize empty memories """
    """ wn_codes = AAWN(code_size) """
    wn_desCodes = AAWN(desc_size + code_size)

    max_fos = int(codes.shape[0] / codes.shape[1])

    for fos in trange(max_fos, desc="Storing", unit="factor of stored (fos)"):

        n_stored = code_size * (fos + 1)

        """ train the memory with clean data """
        """ new_data_codes = codes[n_stored - code_size : n_stored]
        wn_codes.store(new_data_codes) """

        new_data_desCodes = desCodes[n_stored - code_size : n_stored]
        wn_desCodes.store(new_data_desCodes)

        """ present the memory with cues """
        """ ret_codes = wn_codes.retrieve(codes_noisy[:n_stored])

        ret_desCodes = wn_desCodes.retrieve(desCodes_noisy[:n_stored])
        ret_desCodes = separate(ret_desCodes, desc_size)[1] """

        ret_gen = wn_desCodes.retrieve(gen_desCodes_noisy)
        ret_gen = separate(ret_gen, desc_size)[1]

        # TODO: maybe measure AA_score for the generated desCodes, to see if the memory keeps the correct label

        """ create Reconstructions """
        """ recons_codes = recon_with_polar(
            ret_codes[ex_idxs], features, polar_params[ex_idxs], Q, K
        )
        recons_desCodes = recon_with_polar(
            ret_desCodes[ex_idxs], features, polar_params[ex_idxs], Q, K
        ) """

        """ create generations """
        generations = recon_no_polar(ret_gen, features, Q, K)

        if USE_WANDB:
            log_dict = {}

            """ log images """
            """ log_dict["MNIST"] = wandb.Image(np_to_grid(ex_img))
            log_dict["Recon_codes"] = wandb.Image(np_to_grid(recons_codes))
            log_dict["Recon_desCodes"] = wandb.Image(np_to_grid(recons_desCodes)) """
            log_dict["Generations"] = wandb.Image(np_to_grid(generations))

            wandb.log(log_dict, step=fos + 1)

    if USE_WANDB:
        wandb.finish()

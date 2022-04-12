#%%
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


""" Noise params """
l_prob = [0.0, 0.25, 0.5, 0.75]  # each item in this list is a different wandb run
noise_type = "zero"  # none, zero, one TODO: add "both" noise option

TRIAL_RUN = False
class_dist = True

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

""" Create artificial codes and descs """
if class_dist:
    gen_lbls = create_gen_lbls(n_classes=10, n_exs=3, transpose=False)
    prob_dists = compute_dists(codes, lbls)
    gen_codes = sample_from_dists(prob_dists, gen_lbls)
else:
    dist = compute_dist(codes)
    gen_codes = csr_matrix(sample_from_dist(dist, n=30))


for prob in l_prob:

    gen_codes_noisy = add_noise(gen_codes, noise_type, prob)

    if USE_WANDB:
        wandb.init(
            project="naive_generator_WN",
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
            },
        )

        name = "TRIAL_" if TRIAL_RUN else ""
        name += f"withDecodings_"
        if noise_type == "none" or prob == 0.0:
            name += "none"
        else:
            name += str(noise_type)
            name += "_p" + str(prob)
        wandb.run.name = name

        """ log the initial state (no memory) """
        log_dict = {}
        decodings = recon_no_polar(gen_codes, features, Q, K)
        decodings_noisy = recon_no_polar(gen_codes_noisy, features, Q, K)
        log_dict["Decodings"] = wandb.Image(np_to_grid(decodings))
        log_dict["Decodings_noisy"] = wandb.Image(np_to_grid(decodings_noisy))

        wandb.log(log_dict, step=0)

    wn = AAWN(code_size)

    max_fos = int(codes.shape[0] / codes.shape[1])

    for fos in trange(max_fos, desc="Storing", unit="factor of stored (fos)"):

        n_stored = code_size * (fos + 1)

        """ train the memory with clean data """
        """ new_data_codes = codes[n_stored - code_size : n_stored]
        wn_codes.store(new_data_codes) """

        new_data = codes[n_stored - code_size : n_stored]
        wn.store(new_data)

        ret_gen = wn.retrieve(gen_codes_noisy)

        """ create generations """
        generations = recon_no_polar(ret_gen, features, Q, K)

        if USE_WANDB:
            log_dict = {}

            log_dict["Generations"] = wandb.Image(np_to_grid(generations))

            wandb.log(log_dict, step=fos + 1)

    if USE_WANDB:
        wandb.finish()

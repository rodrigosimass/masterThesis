import torch
import torchvision as torchv
import numpy as np


def np_to_grid(array, norm=True, nr=10, pad=1):
    tensor = torch.from_numpy(array)
    tensor = torch.unsqueeze(tensor, dim=1)
    return torchv.utils.make_grid(tensor, normalize=norm, nrow=nr, pad_value=pad)


def swap_codes_axes(codes, k, Q):
    codes = codes.toarray()
    codes = codes.reshape(-1, Q * Q, k)
    codes = np.swapaxes(codes, 1, 2)
    codes = codes.reshape(-1, k, Q, Q)

    return codes

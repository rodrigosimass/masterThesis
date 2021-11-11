import torch
import torchvision as torchv


def np_to_grid(array, norm=True, nr=10, pad=1):
    tensor = torch.from_numpy(array)
    tensor = torch.unsqueeze(tensor, dim=1)
    return torchv.utils.make_grid(tensor, normalize=norm, nrow=nr, pad_value=pad)

import torch
from torch import nn

class deCNN_MLP(nn.Module):
    def __init__(
        self,
        l1_in_dim,
        l1_out_dim,
        convT_in_ch,
        convT_out_ch,
        convT_k_size,
        convT_pad,
    ):
        self.convT_in_ch = convT_in_ch

        super().__init__()

        self.fc1 = nn.Linear(l1_in_dim, l1_out_dim)
        self.convT = nn.ConvTranspose2d(
            in_channels=convT_in_ch,
            out_channels=convT_out_ch,
            kernel_size=convT_k_size,
            padding=convT_pad,
        )
        self.norm = nn.BatchNorm2d(convT_out_ch)


    def forward(self, x0):
        x1 = torch.sigmoid(self.fc1(x0))
        x1 = torch.reshape(x1, (-1, self.convT_in_ch, 28, 28))

        x2 = self.convT(x1)
        x2 = self.norm(x2)
        x2 = torch.flatten(x2, start_dim=1)

        return x2   
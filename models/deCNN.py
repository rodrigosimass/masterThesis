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

        self.convT1 = nn.ConvTranspose2d(
            in_channels=20,
            out_channels=1,
            kernel_size=5,
            padding=2,
        )
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(1)

        """ self.convT2 = nn.ConvTranspose2d(
            in_channels=5,
            out_channels=1,
            kernel_size=5,
            padding=2,
        )
        self.act2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(1) """

        self.fc3 = nn.Linear(21 * 21, 28 * 28)
        self.act3 = nn.Sigmoid()
        self.norm3 = nn.BatchNorm1d(28*28)
        


    def forward(self, x0):

        x1 = self.convT1(x0)

        x1 = self.act1(x1)
        x1 = self.norm1(x1)

        #x2 = self.convT2(x1)
        #x2 = self.act2(x2)
        #x2 = self.norm2(x2)

        x1 = torch.flatten(x1, start_dim=1)


        x2 = self.fc3(x1)
        x2 = self.act3(x2)
        #print("x3:",x3.size())

        return x2
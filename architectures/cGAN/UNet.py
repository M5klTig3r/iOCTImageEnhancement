import torch.nn as nn

from architectures.cGAN.Encoder import Encoder
from architectures.cGAN.Decoder import Decoder


class UNet(nn.Module):
    def __init__(self, retain_dim=False):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.head = nn.Sequential(nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                  nn.Tanh())
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][0:])
        out = self.head(out)
        return out

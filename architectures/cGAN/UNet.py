import torch.nn as nn
import GeneratorEncoder
import GeneratorDecoder


class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 1024, 1024, 1024, 512, 256, 128, 3),
                 num_class=1, retain_dim=False):
        super().__init__()
        self.encoder = GeneratorEncoder(enc_chs)
        self.decoder = GeneratorDecoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        decFeatures = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        map = self.head(decFeatures)
        return map

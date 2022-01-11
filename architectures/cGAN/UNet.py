import torch
import torch.nn as nn
from torch import Tensor

from architectures.cGAN.GeneratorEncoder import GeneratorEncoder
from architectures.cGAN.GeneratorDecoder import GeneratorDecoder


class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 64, 128, 256, 512, 512, 512, 512, 512),
                 dec_chs=(512, 1024, 1024, 1024, 1024, 512, 256, 128, 64, 1),
                 #(512, 512), (1024, 512), (1024, 512), (1024, 512), (1024, 256), (512, 128), (256, 64), (128, 1)),
                 num_class=1, retain_dim=False):
        super().__init__()
        self.encoder = GeneratorEncoder(enc_chs)
        self.decoder = GeneratorDecoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, (1, 1))
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        decFeatures = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        map = self.head(decFeatures)
        return map


encoder = GeneratorEncoder((1, 64, 128, 256, 512, 512, 512, 512, 512))
# input image
x: Tensor = torch.randn(1, 1, 512, 512)
ftrs = encoder(x)
for ftr in ftrs:
    print(ftr.shape)

decoder = GeneratorDecoder((512, 1024, 1024, 1024, 1024, 512, 256, 128, 64, 1))
x = torch.randn(1, 1024, 28, 28)
decoder(x, ftrs[::-1]).shape()


#unet = UNet()
#img = torch.randn(1, 1, 512, 512)
#unet(img).shape()

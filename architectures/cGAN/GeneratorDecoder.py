import torch
import torchvision
import torch.nn as nn
from architectures.cGAN.ConvolutionBlock import ConvolutionBlock


class GeneratorDecoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.Sequential(nn.ConvTranspose2d(
                in_channels=chs[i], out_channels=chs[i+1], kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)),
                nn.BatchNorm2d(chs[i+1]),
                nn.ReLU())

                for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList(
            [ConvolutionBlock(chs[i], chs[i+1])
             for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            print(f"Input {x.shape}")
            print(f"Features from encoder: {enc_ftrs.shape}")
            x = torch.cat([x, enc_ftrs], dim=1)
            # x.resize(self.chs[i])
            print(f"Concat {x.shape}")
            x = self.dec_blocks[i](x)
        x = nn.Tanh()
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


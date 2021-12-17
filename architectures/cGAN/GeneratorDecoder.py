import torch
import torchvision
import torch.nn as nn
import ConvolutionBlock


class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels=chs[i], out_channels=chs[i + 1], kernel_size=(4,4), stride=(2,2), padding=(1,1))
             for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList(
            [ConvolutionBlock[chs[i], chs[i + 1]]
             for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
import torch
import torch.nn as nn

from architectures.cGAN.ConvolutionBlock import ConvolutionBlock


class GeneratorEncoder(nn.Module):

    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([ConvolutionBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs
            x = block(x)
            blockOutputs.append(x)

        # return the list containing the intermediate outputs
        return blockOutputs


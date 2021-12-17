import torch.nn as nn


class ConvolutionBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(4, 4), stride=(2, 2),
                              padding=1)
        self.batch = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.ReLU()

    def forward(self, image):
        return self.relu(self.batch(self.conv(image)))

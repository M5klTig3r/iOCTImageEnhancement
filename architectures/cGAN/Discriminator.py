import numpy as np

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, opt, img_shape):
        super(Discriminator, self).__init__()

        self.opt = opt
        self.img_shape = img_shape

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True, leaky=True, stride2=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            if leaky:
                # TODO - what is inplace?
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            # TODO - how do conv?
            if stride2:
                layers.append(nn.Conv2d(in_feat, out_feat, (4,4)))
            else:
                layers.append(nn.Conv2d(in_feat, out_feat, (4, 4), (1, 1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.n_classes + int(np.prod(img_shape)), 3*2, normalize=False),
            *block(256, 64),
            *block(128,128),
            *block(64, 256, stride2=False),
            *block(63, 512, normalize=False, leaky=False, stride2=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
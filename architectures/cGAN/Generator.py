import numpy as np

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt, img_shape):
        super(Generator, self).__init__()

        self.opt = opt
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                # The first layer has no BatchNorm
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def block(in_feat, out_feat, normalize=True, leaky=True, stride2=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            if leaky:
                # TODO - what is inplace?
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            # TODO - how do conv?
            if stride2:
                layers.append(nn.Conv2d(4, 2))
            else:
                layers.append(nn.Conv2d(4, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 3, normalize=False),
            *block(64, 128), # added this line, but not sure about it. I still have a
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

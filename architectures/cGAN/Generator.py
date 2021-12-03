import numpy as np

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt, img_shape):
        super(Generator, self).__init__()

        self.opt = opt
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True, leaky=True, stride2=True):
            layers = [nn.Linear(in_feat, out_feat)]
            # TODO - how do conv?
            if stride2:
                layers.append(nn.Conv2d(in_feat, out_feat, (4, 4)))
            else:
                layers.append(nn.Conv2d(in_feat, out_feat, (4, 4)))

            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))

            if leaky:
                # TODO - what is inplace?
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.ReLU())
            return layers

        def encode():
            self.model = nn.Sequential(
                *block(opt.latent_dim + opt.n_classes, 3, normalize=False),
                *block(256, 64),  # added this line, but not sure about it. I still have a
                *block(128, 128),
                *block(64, 256),
                *block(32, 512),
                *block(16, 512),
                *block(8, 512),
                *block(4, 512)
            )

        def decode():
            self.model = nn.Sequential(
                nn.ConvTranspose2d(2, 512, (4,4)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(4, 1024, (4, 4)),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 1024, (4, 4)),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1024, (4, 4)),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1024, (4, 4)),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 512, (4, 4)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 256, (4, 4)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, (4, 4)),
                nn.Tanh()
            )

        encode()
        decode()

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

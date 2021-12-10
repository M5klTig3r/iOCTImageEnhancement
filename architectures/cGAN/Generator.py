import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt, img_shape):
        super(Generator, self).__init__()

        self.opt = opt
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True, leaky=True, stride2=True):
            layers = [nn.Linear(in_feat, out_feat)]
            # TODO - how do conv?
            if stride2:
                layers.append(nn.Conv2d(in_feat, out_feat, (4, 4)))
            else:
                layers.append(nn.Conv2d(in_feat, out_feat, (4, 4)))

            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))

            if leaky:
                # TODO - what is inplace?
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.ReLU())
            return layers

        def encode():
            print("Generator - encode() - started.")
            self.model = nn.Sequential(
                *block(opt.img_size, 256, normalize=False),
                *block(256, 128),  # added this line, but not sure about it. I still have a
                *block(128, 64),
                *block(64, 32),
                *block(32, 16),
                *block(16, 8),
                *block(8, 4),
                *block(4, opt.latent_dim)
            )
            print("Generator - encode() - finished.")

        def decode():
            print("Generator - decode() - started.")
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels=opt.latent_dim + 1, out_channels=4, stride=(2, 2), kernel_size=(4, 4), padding=opt.padding, dilation=opt.dilation, output_padding=opt.output_padding, groups=opt.groups),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=4, out_channels=8, stride=(2, 2), kernel_size=(4, 4)),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 16, (4, 4),(2, 2),  padding=opt.padding),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 32, (4, 4), (2, 2), padding=opt.padding),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 64,  (4, 4),(2, 2),  padding=opt.padding),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 128,  (4, 4),(2, 2),  padding=opt.padding),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 256,  (4, 4), (2, 2), padding=opt.padding),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 512,  (4, 4), (2, 2), padding=opt.padding),
                nn.Tanh()
            )
            print("Generator - decode() - finished.")

        encode()
        decode()

    def forward(self, images):
        # Concatenate label embedding and image to produce input
        # gen_input = torch.cat((labels, noise), -1)
        img = self.model(images)
        img = img.view(img.size(0), *self.img_shape)
        return img

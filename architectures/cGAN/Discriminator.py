import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.channel = img_shape[0]
        self.img_size = img_shape[1]

        def block(img_channels, features_d, normalize=True, leaky=True, stride2=True):
            layers = [nn.Linear(in_features=img_channels, out_features=features_d)]
            if normalize:
                layers.append(nn.BatchNorm2d(num_features=features_d, momentum=0.8))
            if leaky:
                # TODO - what is inplace?
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            # TODO - how do conv?
            if stride2:
                layers.append(nn.Conv2d(in_channels=img_channels, out_channels=features_d, kernel_size=(4, 4)))
            else:
                layers.append(
                    nn.Conv2d(in_channels=img_channels, out_channels=features_d, kernel_size=(4, 4), stride=(1, 1)))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *block(img_channels=(3 * 2), features_d=self.img_size, normalize=False),
            *block(img_channels=64, features_d=256),
            *block(img_channels=128, features_d=128),
            *block(img_channels=64, features_d=512, stride2=False),
            *block(img_channels=63, features_d=1, normalize=False, leaky=False, stride2=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), labels), -1)
        validity = self.model(d_in)
        return validity

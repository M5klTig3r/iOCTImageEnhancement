import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.channel = img_shape[0]
        self.img_size = img_shape[1]

        def block(img_channels, features_d):
            return nn.Sequential(
                nn.Conv2d(in_channels=img_channels, out_channels=features_d, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(num_features=features_d, momentum=0.8),
                nn.LeakyReLU(0.2)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=(1 * 2), out_channels=64, kernel_size=(4,4), stride=(2, 2), padding=(1,1)),
            nn.LeakyReLU(0.2),
            *block(img_channels=64, features_d=128),
            *block(img_channels=128, features_d=256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        #        print(img.shape)
        #        print(labels.shape)
        #        print("Image size(0)")
        #        print(img.size(0))
        # first round we need d_in to be: [1, 6, 512, 512]
        # second round we need d_in to be:
        # third round we need d_in to be:
        d_in = torch.cat((img, labels), 1)
        #        print(d_in.shape)
        #        print("d_in size(0)")
        #        print(d_in.size(0))
        validity = self.model(d_in)
        #        print(validity.shape)
        #        print("validity size(0)")
        #        print(validity.size(0))
        return validity

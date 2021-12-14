import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, img_shape):
        super(Generator, self).__init__()
        self.img_size = img_shape[1]
        self.channels = img_shape[0]
        self.label_emb = nn.Embedding(self.img_size, self.img_size * self.img_size)

        def block(in_feat, out_feat):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=(4, 4)),
                nn.BatchNorm2d(num_features=out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # encode
        self.model = nn.Sequential(
            # N x channels x 512 x 512
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4)),  # 256x256
            nn.LeakyReLU(0.2, inplace=True),
            *block(in_feat=64, out_feat=128),  # 128x128
            *block(in_feat=128, out_feat=256),  # 64x64
            *block(in_feat=256, out_feat=512),  # 32x32
            #*block(in_feat=512, out_feat=512),  # 16x16
            #*block(in_feat=512, out_feat=512),  # 8x8
            #*block(in_feat=512, out_feat=512),  # 4x4
            #*block(in_feat=512, out_feat=512),  # 2x2
            #nn.ConvTranspose2d(in_channels=512, out_channels=1024, stride=(2, 2), kernel_size=(4, 4)),  # 4x4
            #nn.BatchNorm2d(num_features=1024),
            #nn.ReLU(),
            #nn.ConvTranspose2d(in_channels=1024, out_channels=1024, stride=(2, 2), kernel_size=(4, 4)),  # 8x8
            #nn.BatchNorm2d(num_features=1024),
            #nn.ReLU(),
            #nn.ConvTranspose2d(in_channels=1024, out_channels=1024,kernel_size=(4, 4), stride=(2, 2)),  # 16x16
            #nn.BatchNorm2d(num_features=1024),
            #nn.ReLU(),
            #nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=(4, 4), stride=(2, 2)),  # 32x32
            #nn.BatchNorm2d(num_features=1024),
            #nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=512,  kernel_size=(4, 4), stride=(2, 2)),  # 64x64
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # N x channels x 64 x 64
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(2, 2)),  # 128x128
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(2, 2)),  # 256x256
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=(4, 4), stride=(2, 2)),  # 512x512
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((labels, noise), dim=0)
        print("Generated input for generator")
        print(gen_input.shape)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        print("Generator output")
        print(img.shape)
        return img

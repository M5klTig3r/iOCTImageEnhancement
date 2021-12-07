#
# Inspired by https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
#

import argparse
import os
import numpy as np
import torchvision.datasets

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch

from architectures.cGAN.Discriminator import Discriminator
from architectures.cGAN.Generator import Generator

os.makedirs("images", exist_ok=True)

# TODO - i might not need all of this
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=127, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # done
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")  # done
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")  # done
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=2,
                    help="dimensionality of the latent space")  # would be the bottleneck - the lowest size
# use 512 - rescale to squared
parser.add_argument("--img_size", type=int, default=512,
                    help="size of each image dimension")  # TODO - 512 x 1024; size was 32 for images. of size 28x28
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # done
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss functions
# L1 and edge loss
adversarial_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(opt, img_shape)
discriminator = Discriminator(opt, img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(
        "../../ImageDenoising(Averaging)Cubes/sorted/cut_eye_no_needle/86271bd2-31fb-436f-9e31-9ec5a3a4f7648203"
        "/bigVol_9mm",
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),

    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_images = generator.forward(z, labels)
    save_image(gen_images.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
# Debug purpose.
#    print("Dataset Size: \n")
#    print(dataloader.__sizeof__())
#    print("Data 0: \n")
#    print(dataloader.dataset.__getitem__(0))
#    print("\nData Size: \n")
#    print(dataloader.dataset.__getitem__(0).__sizeof__())
    for i, (images, labels) in enumerate(dataloader):

        batch_size = images.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_images = Variable(images.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, batch_size)))

        # Generate a batch of images
        gen_images = generator.forward(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator.forward(gen_images, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator.forward(real_images, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator.forward(gen_images.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
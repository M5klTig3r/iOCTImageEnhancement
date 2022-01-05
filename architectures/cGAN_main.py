#
# Inspired by https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
#

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch

from architectures.cGAN.Discriminator import Discriminator
from architectures.cGAN.Generator import Generator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("images", exist_ok=True)

# TODO - i might not need all of this
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # done
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")  # done
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")  # done
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")  # would be the bottleneck - the lowest size
# use 512 - rescale to squared
parser.add_argument("--img_size", type=int, default=512,
                    help="size of each image dimension")  # TODO - 512 x 1024; size was 32 for images. of size 28x28
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # done
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--padding", type=int, default=(1, 1), help="Padding for image convolution.")
parser.add_argument("--dilation", type=int, default=(1, 1), help="Dilation for image convolution.")
parser.add_argument("--output_padding", type=int, default=(0, 0), help="output_padding for image convolution.")
parser.add_argument("--groups", type=int, default=1, help="groups for image convolution.")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
print(img_shape)

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache()

# Loss functions
# L1 and edge loss
adversarial_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(img_shape)
discriminator = Discriminator(img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
# os.makedirs("../data/mnist", exist_ok=True)

dataloader = torch.utils.data.DataLoader(datasets.ImageFolder(
    "../../iOCT/bigVol_9mm",
    # "../../ImageDenoising(Averaging)Cubes/sorted/cut_eye_no_needle/86271bd2-31fb-436f-9e31-9ec5a3a4f7648203/bigVol_9mm",
    transform=transforms.Compose(
        [  # transforms.Grayscale(num_output_channels=1),
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]
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


def sample_image(n_row, batches_done, current_epoch, real_images, labels):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    gen_images = generator.forward(real_images)
    # TODO - create folder conditionally
    save_image(gen_images.data, f"images/{current_epoch}_{batches_done}.png", nrow=n_row, normalize=True)


# ----------
#  Training
# ----------
def my_plot(epochs, g_loss, d_loss):
    plt.plot(epochs, g_loss)
    plt.plot(epochs, d_loss)


groundTruth = 0
for j, (images, labels) in enumerate(dataloader):
    if j == 1:
        groundTruth = images
loss_vals = []
for epoch in range(opt.n_epochs):
    # Debug purpose.
    #    print("Dataset Size: \n")
    #    print(dataloader.__sizeof__())
    #    print("Data 0: \n")
    #    print(dataloader.dataset.__getitem__(0))
    #    print("\nData Size: \n")
    #    print(dataloader.dataset.__getitem__(0).__sizeof__())
    epoch_g_loss = []
    epoch_d_loss = []
    for i, (images, labels) in enumerate(dataloader):

        if i == 0:
            continue
        batch_size = images.shape[0]
        # print(images.shape)
        # print(labels.shape)
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        # real_images are the slices
        # labels = averaged images
        # print("Real imges before making them variables")
        # print(images.shape)
        real_images = Variable(images.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        # labels - ground truth
        # images - real images
        gen_labels = Variable(groundTruth.type(FloatTensor))  # labels are the input images
        # print("Real images")
        # print(real_images.shape)
        # print("Labels aka ground truth")
        # print(gen_labels.shape)

        # Generate a batch of images
        gen_images = generator.forward(real_images)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator.forward(gen_images, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        epoch_g_loss.append(g_loss.item())
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator.forward(real_images, gen_labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator.forward(gen_images.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        epoch_d_loss.append(d_loss.item())
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done, current_epoch=epoch, real_images=real_images,
                         labels=labels)

        # my_plot(np.linspace(1, opt.n_epochs, opt.n_epochs).astype(int), g_loss.detach().numpy(),
        #        d_loss.detach().numpy())

plt.show()

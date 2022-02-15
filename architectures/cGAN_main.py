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

from pytorch_msssim import SSIM

from architectures.cGAN.EdgeSensitiveLoss import EdgeSensitiveLoss
from architectures.cGAN.Discriminator import Discriminator
from architectures.cGAN.Generator import Generator
from architectures.cGAN.UNet import UNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("images", exist_ok=True)

# TODO - i might not need all of this
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")  # more than 8 is trouble for VRam
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")  # done
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")  # done
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")  # done
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")  # would be the bottleneck - the lowest size
# use 512 - rescale to squared
parser.add_argument("--img_size", type=int, default=512,
                    help="size of each image dimension")  # TODO - 512 x 1024; size was 32 for images. of size 28x28
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # done
parser.add_argument("--sample_interval", type=int, default=8, help="interval between image sampling")
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
# L1
adversarial_loss = torch.nn.L1Loss()
generator_loss = EdgeSensitiveLoss()
# L1 and edge loss
# MSE
# adversarial_loss = torch.nn.MSELoss()
# SSIM adversarial_loss = SSIM(win_size=11, win_sigma=1.5, data_range=opt.batch_size, size_average=True, channel=1)
# might be enough to set win_size
# channel is needed

# Initialize generator and discriminator
generator = Generator(img_shape)
discriminator = Discriminator(img_shape)

# TEST
testGenerator = UNet()

# Optimizers

optimizer_G = torch.optim.Adam(testGenerator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

if cuda:
    print("Cuda TRUE")
    testGenerator.cuda()
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
else:
    print("Cuda FALSE")

# Configure data loader
# os.makedirs("../data/mnist", exist_ok=True)

dataloader = torch.utils.data.DataLoader(datasets.ImageFolder(
    "../data/input",
    transform=transforms.Compose(
        [transforms.Grayscale(num_output_channels=opt.channels),
         transforms.Resize((opt.img_size, opt.img_size)),
         transforms.ToTensor(), # automatically normalizes to [0, 1]
         transforms.Normalize([0.5], [0.5])
        ]
    ),
),
    batch_size=opt.batch_size,
    shuffle=True,
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def normalize(image):
    normalized_image = torch.empty(image.size())
    for i, channels in enumerate(image):
        for j, rows in enumerate(channels):
            div = torch.div(rows, 2.0)
            normalized_image[i, j] = torch.mul(torch.add(div, 0.5), 255)
    return normalized_image


def sample_image(n_row, batches_done, current_epoch, real_images):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    input = Variable(real_images.type(FloatTensor))
    gen_images = testGenerator.forward(input)
    # TODO - create folder conditionally
    path = f"../../output_images/L1_b{opt.batch_size}_e{opt.n_epochs}_lr0004"
    if not os.path.exists(path):
        os.makedirs(path)
    save_image(input.data,
               f"{path}/{current_epoch}_{batches_done}_input.png",
               nrow=n_row, normalize=True)
    save_image(normalize(gen_images).data,
               f"{path}/{current_epoch}_{batches_done}_output.png",
               nrow=n_row)  # do not normalize


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

generator_loss_set = []
discriminator_loss_set = []

fig = plt.figure()
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
        # print(images.shape)
        # print(labels.shape)
        # Adversarial ground truths
        valid = Variable(FloatTensor(opt.batch_size, 1, 62, 62).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(opt.batch_size, 1, 62, 62).fill_(0.0), requires_grad=False)

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
        gen_images = testGenerator.forward(real_images)
       # trans = transforms.Compose([transforms.ToTensor()])
      #  test = trans(gen_images)
       # gen_images_norm = normalize(gen_images)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator.forward(gen_images, gen_labels)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator.forward(real_images, gen_labels)
        # L1 loss & MSE
        d_real_loss = adversarial_loss(validity_real, valid)
        # L1 and edge loss
        # SSIM
        # d_real_loss = 1 - adversarial_loss(validity_real, valid)

        # Loss for fake images
        # Is it input vs truth or input vs generated?
        validity_fake = discriminator.forward(gen_images.detach(), gen_labels)
        validity_fake_edge_loss = discriminator.forward(real_images, gen_labels)
        # L1 loss & MSE & Edge loss
        d_fake_loss = adversarial_loss(validity_fake, fake)
        # SSIM
        # d_fake_loss = 1 - adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  Calculate loss Generator
        # ---------------------

        # L1 loss and MSE
        # print(f"Validity is size: {validity.size()}")
        # print(f"Valid is size: {valid.size()}")
        g_loss = adversarial_loss(validity, valid)
        # L1 and edge loss
        # g_loss = generator_loss.optimization(generated_images=gen_images, d_fake=validity_fake_edge_loss, d_real=validity_real)
        # SSIM
        # g_loss = 1 - adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        generator_loss_set.append(g_loss.item())
        discriminator_loss_set.append(d_loss.item())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss, g_loss)
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=4, batches_done=batches_done, current_epoch=epoch, real_images=images)

        # my_plot(np.linspace(1, opt.n_epochs, opt.n_epochs).astype(int), g_loss.detach().numpy(),
        #        d_loss.detach().numpy())

# plot the results
plt.xlabel('batches')
plt.ylabel('loss')
plt.title(f'cGAN on iOCT data')
epochs_array = np.arange(0, len(discriminator_loss_set))
plt.plot(epochs_array, discriminator_loss_set, label="Discriminator loss")
plt.plot(epochs_array, generator_loss_set, label="Generator loss")
plt.legend()
# plt.savefig(f"../../L1_b{opt.batch_size}_e{opt.n_epochs}_lr0004.png")
plt.savefig(f"../../output_images/L1_b{opt.batch_size}_e{opt.n_epochs}_lr0004.png")
plt.show()

# for automatic shutdown after finishing
# os.system("shutdown /s /t 60")

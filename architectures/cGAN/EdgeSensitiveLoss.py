import numpy as np
import torch
import tensorflow as tf
from tensorflow import math  # .math as tf

class EdgeSensitiveLoss:

    def __init__(self):
        super().__init__()

    def loss_cgan_per_image(self, d_real_img, d_fake_img):

        # TODO probabilities
        # TODO expectation (?)
        log_real_images = self.map_fn(d_real_img, torch.log, d_real_img.size())
        mean_real_images = self.map_fn(log_real_images, torch.mean, log_real_images.size()[0:-1])

        subtract = torch.neg(torch.subtract(d_fake_img, 1))
        log_fake_images = self.map_fn(subtract, torch.log, subtract.size())
        mean_fake_images = self.map_fn(log_fake_images, torch.mean, log_fake_images.size()[0:-1])

        add_mean = self.add_tensors(mean_real_images, mean_fake_images) # 2D

        # maximize mean
        maximize = torch.empty(add_mean.size(0))
        for i, element in enumerate(add_mean):
            maximize[i] = torch.max(element)

        return torch.argmin(maximize)

    def add_tensors(self, t1, t2):
        output = torch.empty(t1.size())

        for i, (img_ch1, img_ch2) in enumerate(zip(t1, t2)):
            for j, (img_row1, img_row2) in enumerate(zip(img_ch1, img_ch2)):
                output[i, j] = torch.add(img_row1, img_row2)
        return output

    def map_fn(self, img, fkt, res_size):
        new_img = torch.empty(res_size)
        for i, img_channel in enumerate(img):
            for j, img_row in enumerate(img_channel):
                new_img[i, j] = fkt(img_row)
        return new_img

    def loss_cgan(self, d_real_batch, d_fake_batch):
        total_loss = 0
        for d_real_img, d_fake_img in zip(d_real_batch, d_fake_batch):
            total_loss += self.loss_cgan_per_image(d_real_img, d_fake_img)
        return total_loss / d_real_batch.size()

    def loss_l1(self, generated_images):
        return torch.nn.L1Loss(generated_images)

    def loss_edge(self):
        # TODO
        return

    def optimization(self, generated_images, d_real, d_fake, alpha=1, beta=0.5):
        # TODO
        # Add edge loss
        return self.loss_cgan(d_real_batch=d_real, d_fake_batch=d_fake) + alpha * self.loss_l1(
            generated_images=generated_images).item()

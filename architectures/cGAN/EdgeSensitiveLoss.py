import numpy as np
import torch
import tensorflow.math as tf

class EdgeSensitiveLoss:

    def __init__(self, generated_images, d_real, d_fake, alpha=0.5, beta=0.5):
        super().__init__()
        self.d_fake = d_fake
        self.alpha = alpha
        self.beta = beta
        self.generated_images = generated_images
        self.d_real = d_real
        return self.optimization()

    def loss_cgan(self):
        log_fkt = lambda img: tf.map_fn(lambda x: tf.log(x), img)
        # TODO probabilities
        # TODO expectation (?)
        log_real_images = log_fkt(self.d_real)
        mean_real_images = tf.mean(log_real_images)

        subtract = tf.subtract(1, self.d_fake)
        log_fake_images = log_fkt(subtract)
        mean_fake_images = tf.mean(log_fake_images)
        return np.argmin(np.max(tf.sum(mean_real_images, mean_fake_images)))

    def loss_l1(self):
        return torch.nn.L1Loss(self.generated_images)

    def loss_edge(self):
        # TODO
        return

    def optimization(self):
        # TODO
        # Add edge loss
        return self.loss_cgan() + self.alpha*self.loss_l1().item()

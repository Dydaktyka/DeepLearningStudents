import torch as t
import torch.nn as tnn
import torch.nn.functional as F
import torchvision
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time

import sys

sys.path.append("..")
import utils


class Embedding(tnn.Module):
    def __init__(self, in_features, out_features, s=0.01):
        super(Embedding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter("weights", self.weights)

    def forward(self,x):
        return self.weights[x]


class Generator(tnn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.embed = tnn.Linear(in_features=10, out_features=z_dim)
        self.project = tnn.Linear(in_features=z_dim, out_features=256 * 7 * 7)

        self.generate = tnn.Sequential(
            tnn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            tnn.BatchNorm2d(128),
            tnn.LeakyReLU(0.01),
            tnn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            tnn.BatchNorm2d(64),
            tnn.LeakyReLU(0.01),
            tnn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            tnn.Tanh()
        )

    def forward(self, x, y):
        y = self.embed(y)

        x = t.reshape(self.project(x * y), (-1, 256, 7, 7))
        x = self.generate(x)
        return x


class Discriminator(tnn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embed = tnn.Linear(in_features=10, out_features=28 * 28)
        self.discriminator = tnn.Sequential(
            tnn.Conv2d(2, 32, 3, 2, 1),
            tnn.LeakyReLU(0.01),

            tnn.Conv2d(32, 64, 3, 2, 1),
            tnn.BatchNorm2d(64),
            tnn.LeakyReLU(0.01),

            tnn.Conv2d(64, 128, 3, 2),
            tnn.BatchNorm2d(128),
            tnn.LeakyReLU(0.01),

            tnn.Flatten(),
            tnn.Linear(in_features=1152, out_features=1),
            tnn.Sigmoid()
        )

    def forward(self, x, y):
        y = self.embed(y).reshape(-1, 1, 28, 28)
        x = t.cat([x, y], 1)
        return self.discriminator(x)


def discriminator_loss(discriminator, generator, imgs, oh_labels, z_dim, device):
    data_size = len(imgs)

    real_labels = t.ones(data_size, 1, device=device)
    d_real_loss = F.binary_cross_entropy(discriminator(imgs, oh_labels), 0.9 * real_labels)  # Using one-sided label smoothing
    fake_labels = t.zeros(data_size, 1, device=device)
    z = t.empty(data_size, z_dim, device=device).uniform_(-1, 1)
    g_out = generator(z, oh_labels).detach()
    d_fake_loss = F.binary_cross_entropy(discriminator(g_out, oh_labels), fake_labels)

    d_loss = d_fake_loss + d_real_loss
    return d_loss


def generator_loss(discriminator, generator, data_size, oh_labels,  z_dim, device):
    z = t.empty(data_size, z_dim, device=device).uniform_(-1, 1)
    g_out = generator(z, oh_labels)
    real_labels = t.ones(data_size, 1, device=device)
    g_loss = F.binary_cross_entropy(discriminator(g_out, oh_labels), real_labels)
    return g_loss


def train_discriminator(discriminator, generator, opt, d, oh_labels, z_dim, k_diskriminator, device):
    for di in range(k_diskriminator):
        opt.zero_grad()
        loss = discriminator_loss(discriminator, generator, d, oh_labels, z_dim, device)
        loss.backward()
        opt.step()
    return loss


def train_generator(disciminator, generator, opt, l,oh_labels, z_dim, k_generator, device):
    for gi in range(k_generator):
        opt.zero_grad()
        loss = generator_loss(disciminator, generator, l, oh_labels, z_dim, device)
        loss.backward()
        opt.step()
    return loss


def gen_and_save(rows, cols, generator,  noise, oh_labels, filename=None):
    with t.no_grad():
        generator.eval()
        out_t = generator(noise, oh_labels);
        # Plot the samples on rows x cols grid.
    utils.display_img_grid(rows, cols, out_t.cpu().data.numpy().reshape(-1, 28, 28), filename)

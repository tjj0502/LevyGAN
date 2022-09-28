import torchdata.datapipes as dp
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import ot
import matplotlib.pyplot as plt
import timeit
import copy
from math import sqrt

from torch.autograd import grad as torch_grad

from aux_functions import *
from Generator import Generator
from Discriminator import Discriminator


config = {
    'device': torch.device('cpu'),
    'noise size': 62,
    'num epochs': 20,
    'num Chen iters': 5000,
    'optimizer': 'Adam',
    'lrG': 0.0001,
    'lrD': 0.0005,
    'beta1': 0,
    'beta2': 0.99,
    'ngpu': 0,
    'weight clipping limit': 0.01,
    'gp weight': 10.0,
    'batch size': 1024,
    'test batch size': 65536,
    'w dim': 4,
    'a dim': 6,
    'which generator': 1,
    'which discriminator': 1,
    'generator symmetry mode': 'Hsym',
    'generator last width': 6,
    's dim': 16,
    'leakyReLU slope': 0.2,
    'num tests for 2d': 8,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7]
}

init_config(config)


class LevyGAN:

    def __init__(self, cf: dict):
        self.device = cf['device']

        self.noise_size = cf['noise size']
        self.w_dim = cf['w dim']
        self.a_dim = cf['a dim']
        self.s_dim = cf['s dim']

        # Number of training epochs using classical training
        self.num_epochs = cf['num epochs']

        # Number of iterations of Chen training
        self.num_Chen_iters = cf['num Chen iters']

        # 'Adam' of 'RMSProp'
        self.which_optimizer = cf['optimizer']

        # Learning rate for optimizers
        self.lrG = cf['lrG']
        self.lrD = cf['lrD']

        # Beta1 hyperparam for Adam optimizers
        self.beta1 = cf['beta1']

        self.beta2 = cf['beta2']

        self.ngpu = cf['ngpu']

        # To keep the criterion Lipschitz
        self.weight_cliping_limit = cf['weight clipping limit']

        # for gradient penalty
        self.gp_weight = cf['gp weight']

        self.batch_size = cf['batch size']

        self.test_batch_size = cf['test batch size']

        # if 1 use GAN1, if 2 use GAN2, etc.
        self.which_discriminator = cf['which discriminator']
        self.which_generator = cf['which generator']
        self.generator_symmetry_mode = cf['generator symmetry mode']
        self.generator_last_width = cf['generator last width']

        # slope for LeakyReLU
        self.leakyReLU_slope = cf['leakyReLU slope']

        # this gives the option to rum the training process multiple times with differently initialised GANs
        self.num_trials = cf['num trials']

        self.num_tests_for2d = cf['num tests for 2d']

        self.W_fixed_whole = cf['W fixed whole']

        self.T, self.M, self.S = generate_TMS(self.w_dim)

        self.netG = Generator(cf)
        self.netD = Discriminator(cf)

    def compute_wth(self, w_in: torch.Tensor, h_in: torch.Tensor):
        return aux_compute_wth(w_in, h_in, self.S, self.T, self.w_dim)

    def compute_wthmb(self, wth_in: torch.Tensor, b_in: torch.Tensor):
        return aux_compute_wthmb(wth_in, b_in, self.M, self.w_dim)
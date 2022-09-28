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

        if self.which_optimizer == 'Adam':
            self.optG = torch.optim.Adam(self.netG.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
            self.optD = torch.optim.Adam(self.netD.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))
        elif self.which_optimizer == 'RMSProp':
            self.optG = torch.optim.RMSprop(self.netG.parameters(), lr=self.lrG)
            self.optD = torch.optim.RMSprop(self.netD.parameters(), lr=self.lrD)

        self.W_fixed = torch.tensor(self.W_fixed_whole)[:self.w_dim].unsqueeze(1).transpose(1, 0)
        self.W_fixed = self.W_fixed.expand((self.test_batch_size, self.w_dim))

        self.st_dev_W_fixed = np.diag(true_st_devs(self.W_fixed_whole[:self.w_dim]))

        # Load "true" samples generated from this fixed W increment
        fixed_test_data_filename = f"samples/fixed_samples_{self.w_dim}-dim.csv"
        self.A_fixed_true = np.genfromtxt(fixed_test_data_filename, dtype=float, delimiter=',')[:self.test_batch_size,
                            self.w_dim:(self.w_dim + self.a_dim)]

        unfixed_test_data_filename = f"samples/non-fixed_samples_{self.w_dim}-dim.csv"
        self.unfixed_test_data = np.genfromtxt(unfixed_test_data_filename, dtype = float, delimiter=',')[:self.test_batch_size]

        self.fixed_data_for_2d = []
        if self.w_dim == 2:
            self.fixed_data_for_2d = [np.genfromtxt(f"samples/fixed_samples_2-dim{i+1}.csv",dtype=float,delimiter=',') for i in range(self.num_tests_for2d)]

    def _gradient_penalty(self, real_data, generated_data):
        b_size_gp = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(b_size_gp, 1)
        alpha = alpha.expand_as(real_data)
        interpolated = (alpha * real_data.data + (1 - alpha) * generated_data.data).requires_grad_(True)

        if self.ngpu > 0:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.netD(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.ngpu > 0 else torch.ones(
                                   prob_interpolated.size(), device=self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (b_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(b_size_gp, -1)
        # grad_norm = gradients.norm(2, dim=1).mean().item()

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        avg_grad_norm = gradients_norm.mean().item()

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean(), avg_grad_norm

    def all_2dim_errors(self):
        assert self.w_dim == 2
        errs = []
        for i in range(self.num_tests_for2d):
            # Test Wasserstein error for fixed W
            data_fixed_true = self.fixed_data_for_2d[i]
            a_fixed_true = data_fixed_true[:, 2]
            w_combo = torch.tensor(data_fixed_true[:, :2], dtype=torch.float)
            noise = torch.randn((self.test_batch_size, self.noise_size), dtype=torch.float, device=self.device)
            g_in = torch.cat((noise, w_combo), 1)
            a_fixed_gen = self.netG(g_in)[:, 3].detach().numpy().squeeze()
            errs.append(sqrt(ot.wasserstein_1d(a_fixed_true, a_fixed_gen, p=2)))
        return errs

    # def multi_dim_wasserstein_errors(self):
    #     noise = torch.randn((self.test_batch_size, self.noise_size), dtype=torch.float, device=self.device)
    #     g_in = torch.cat((noise, self.W_fixed), 1)
    #     A_fixed_gen = self.netG(g_in)[:, self.w_dim:self.w_dim + self.a_dim].detach().numpy()
    #     errors = [sqrt(ot.wasserstein_1d(self.A_fixed_true[:, i], A_fixed_gen[:, i], p=2)) for i in range(self.a_dim)]

    def chen_errors(self):
        W = torch.randn((self.test_batch_size, self.w_dim), dtype=torch.float, device=self.device)
        noise = torch.randn((self.test_batch_size, self.noise_size), dtype=torch.float, device=self.device)
        gen_in = torch.cat((noise, W), 1)
        generated_data = self.netG(gen_in).detach()
        return chen_error_3step(generated_data, self.w_dim)

    def avg_st_dev_error(self, _a_generated):
        difference = np.abs(self.st_dev_W_fixed - np.sqrt(np.abs(empirical_second_moments(_a_generated))))
        return difference.mean()

    def make_report(self, epoch: int = None, iter: int = None):
        report = ""
        if not (epoch is None or iter is None):
            report = f"epoch: {epoch}/{self.num_epochs}, iter: {iter}, "

        pretty_chen_errors = make_pretty(self.chen_errors())

        data = self.unfixed_test_data
        b_size = data.size(0)
        noise = torch.randn((b_size, self.noise_size), dtype=torch.float, device=self.device)
        w = data[:, :self.w_dim]
        z = torch.cat((noise, w), dim=1)
        fake_data = self.netG(z)
        fake_data = fake_data.detach()
        pruning_indices = torch.randperm(b_size * self.s_dim)[:b_size]
        pruned_fake_data = fake_data[pruning_indices]
        gradient_penalty, gradient_norm = self._gradient_penalty(data, pruned_fake_data)
        prob_real = self.netD(data)
        prob_fake = self.netD(fake_data)
        lossD_fake = prob_fake.mean(0).view(1)
        lossD_real = prob_real.mean(0).view(1)
        lossD = lossD_fake - self.s_dim * lossD_real

        pretty_chen_errors = make_pretty(chen_error_3step(fake_data, self.w_dim))
        report += f"gradient norm: {gradient_norm:.4f}, discriminator dist: {lossD.item():.7f}"

        # Test Wasserstein error for fixed W
        if self.w_dim > 2:
            noise = torch.randn((self.test_batch_size, self.noise_size), dtype=torch.float, device=self.device)
            g_in = torch.cat((noise, self.W_fixed), 1)
            A_fixed_gen = self.netG(g_in)[:, self.w_dim:self.w_dim + self.a_dim].detach().numpy()
            errors = [sqrt(ot.wasserstein_1d(self.A_fixed_true[:, i], A_fixed_gen[:, i], p=2)) for i in range(self.a_dim)]
            pretty_errors = make_pretty(errors)
            st_dev_err = self.avg_st_dev_error(A_fixed_gen)
            joint_err = joint_wass_dist(self.A_fixed_true[:1000],A_fixed_gen[:1000])
            pretty_chen_errors = make_pretty(self.chen_errors())
            report += f", st_dev error: {st_dev_err: .4f}, joint_wass_dist: {joint_err: .5f}\nerrs: {pretty_errors}, ch_err: {pretty_chen_errors} "
        else:
            pretty_errors = make_pretty(self.all_2dim_errors())
            report += f"\nerrs: {pretty_errors}, ch_err: {pretty_chen_errors[0]}"
        return report

    def load_dicts(self, descriptor: str = ""):
        self.netG.load_state_dict(torch.load(f'model_saves/generator{self.which_generator}_{self.generator_symmetry_mode}_{self.w_dim}d_{self.noise_size}noise_{descriptor}.pt'))
        self.netD.load_state_dict(torch.load(f'model_saves/discriminator{self.which_discriminator}_{self.generator_symmetry_mode}_{self.w_dim}d_{self.noise_size}noise_{descriptor}.pt'))

    def save_dicts(self, descriptor: str = ""):
        paramsD = copy.deepcopy(self.netD.state_dict())
        paramsG = copy.deepcopy(self.netG.state_dict())
        torch.save(paramsD, f'model_saves/discriminator{self.which_discriminator}_{self.generator_symmetry_mode}_{self.w_dim}d_{self.noise_size}noise_{descriptor}.pt')
        torch.save(paramsG, f'model_saves/generator{self.which_generator}_{self.generator_symmetry_mode}_{self.w_dim}d_{self.noise_size}noise_{descriptor}.pt')

    def compute_wth(self, w_in: torch.Tensor, h_in: torch.Tensor):
        return aux_compute_wth(w_in, h_in, self.S, self.T, self.w_dim)

    def compute_wthmb(self, wth_in: torch.Tensor, b_in: torch.Tensor):
        return aux_compute_wthmb(wth_in, b_in, self.M, self.w_dim)
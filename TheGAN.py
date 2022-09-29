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
    'ngpu': 0,
    'w dim': 4,
    'a dim': 6,
    'noise size': 62,
    'which generator': 1,
    'which discriminator': 1,
    'generator symmetry mode': 'Hsym',
    'generator last width': 6,
    's dim': 16,
    'leakyReLU slope': 0.2,
    'num epochs': 20,
    'num Chen iters': 5000,
    'optimizer': 'Adam',
    'lrG': 0.0001,
    'lrD': 0.0005,
    'beta1': 0,
    'beta2': 0.99,
    'Lipschitz mode': 'gp',
    'weight clipping limit': 0.01,
    'gp weight': 10.0,
    'batch size': 1024,
    'test batch size': 65536,
    'num tests for 2d': 8,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7]
}

training_config = {
    'num epochs': 20,
    'num Chen iters': 5000,
    'optimizer': 'Adam',
    'lrG': 0.0001,
    'lrD': 0.0005,
    'beta1': 0,
    'beta2': 0.99,
    'Lipschitz mode': 'gradient penalty',
    'weight clipping limit': 0.01,
    'gp weight': 10.0,
    'batch size': 1024,
}


class LevyGAN:

    def __init__(self, cf: dict):
        init_config(cf)

        # ============ Model config ===============
        self.device = cf['device']
        self.ngpu = cf['ngpu']
        self.noise_size = cf['noise size']
        self.w_dim = cf['w dim']
        self.a_dim = cf['a dim']
        self.s_dim = cf['s dim']

        # if 1 use GAN1, if 2 use GAN2, etc.
        self.which_discriminator = cf['which discriminator']
        self.which_generator = cf['which generator']

        self.generator_symmetry_mode = cf['generator symmetry mode']
        self.generator_last_width = cf['generator last width']

        # Which method of keeping the critic Lipshcitz to use. 'gp' for gradient penalty, 'wc' for weight clipping
        self.Lipschitz_mode = cf['Lipschitz mode']

        # slope for LeakyReLU
        self.leakyReLU_slope = cf['leakyReLU slope']

        self.T, self.M, self.S = generate_tms(self.w_dim)

        # create the nets
        self.netG = Generator(cf)
        self.netD = Discriminator(cf)

        self.dict_saves_folder = f'model_G{self.which_generator}_D{self.which_discriminator}_{self.Lipschitz_mode}_{self.generator_symmetry_mode}_{self.w_dim}d_{self.noise_size}noise'

        self.serial_number = read_serial_number(self.dict_saves_folder)

        # ============== Training config ===============
        # Number of training epochs using classical training
        self.num_epochs = cf['num epochs']

        # Number of iterations of Chen training
        self.num_Chen_iters = cf['num Chen iters']

        # 'Adam' of 'RMSProp'
        # self.which_optimizer = cf['optimizer']

        # Learning rate for optimizers
        # self.lrG = cf['lrG']
        # self.lrD = cf['lrD']

        # Beta hyperparam for Adam optimizers
        # self.beta1 = cf['beta1']
        # self.beta2 = cf['beta2']
        #
        # if self.which_optimizer == 'Adam':
        #     self.optG = torch.optim.Adam(self.netG.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        #     self.optD = torch.optim.Adam(self.netD.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))
        # elif self.which_optimizer == 'RMSProp':
        #     self.optG = torch.optim.RMSprop(self.netG.parameters(), lr=self.lrG)
        #     self.optD = torch.optim.RMSprop(self.netD.parameters(), lr=self.lrD)

        # this gives the option to rum the training process multiple times with differently initialised GANs
        # self.num_trials = cf['num trials']

        # for weight clipping
        # self.weight_clipping_limit = cf['weight clipping limit']

        # for gradient penalty
        # self.gp_weight = cf['gp weight']

        # self.batch_size = cf['batch size']

        # ============ Testing config ============
        self.num_tests_for2d = cf['num tests for 2d']

        self.test_batch_size = cf['test batch size']

        self.W_fixed_whole = cf['W fixed whole']
        self.W_fixed = torch.tensor(self.W_fixed_whole)[:self.w_dim].unsqueeze(1).transpose(1, 0)
        self.W_fixed = self.W_fixed.expand((self.test_batch_size, self.w_dim))

        self.st_dev_W_fixed = np.diag(true_st_devs(self.W_fixed_whole[:self.w_dim]))

        # Load "true" samples generated from this fixed W increment
        fixed_test_data_filename = f"samples/fixed_samples_{self.w_dim}-dim.csv"
        self.A_fixed_true = np.genfromtxt(fixed_test_data_filename, dtype=float, delimiter=',')[:self.test_batch_size,
                            self.w_dim:(self.w_dim + self.a_dim)]

        unfixed_test_data_filename = f"samples/non-fixed_test_samples_{self.w_dim}-dim.csv"
        self.unfixed_test_data = np.genfromtxt(unfixed_test_data_filename, dtype=float, delimiter=',')[
                                 :self.test_batch_size]

        self.fixed_data_for_2d = []
        if self.w_dim == 2:
            self.fixed_data_for_2d = [
                np.genfromtxt(f"samples/fixed_samples_2-dim{i + 1}.csv", dtype=float, delimiter=',') for i in
                range(self.num_tests_for2d)]

    def _gradient_penalty(self, real_data, generated_data, gp_weight):
        b_size_gp = real_data.shape[0]

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
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda() if self.ngpu > 0 else torch.ones(
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
        return gp_weight * ((gradients_norm - 1) ** 2).mean(), avg_grad_norm

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

    def make_report(self, epoch: int = None, iters: int = None, chen_iters: int = None, add_line_break=True):
        report = ""
        if add_line_break:
            line_break = "\n"
        else:
            line_break = " "
        if not (epoch is None or iters is None):
            report = f"epoch: {epoch}/{self.num_epochs}, iter: {iters}, "

        if not (chen_iters is None):
            report = f"chen_iters: {chen_iters}/{self.num_Chen_iters}"

        data = torch.tensor(self.unfixed_test_data, dtype=torch.float)
        b_size = data.shape[0]

        noise = torch.randn((b_size, self.noise_size), dtype=torch.float, device=self.device)
        w = data[:, :self.w_dim]
        z = torch.cat((noise, w), dim=1)
        fake_data = self.netG(z)
        fake_data = fake_data.detach()
        pruning_indices = torch.randperm(b_size * self.s_dim)[:b_size]
        pruned_fake_data = fake_data[pruning_indices]

        gradient_penalty, gradient_norm = self._gradient_penalty(data, pruned_fake_data, gp_weight= 0)

        prob_real = self.netD(data)

        prob_fake = self.netD(fake_data)

        loss_d_fake = prob_fake.mean(0).view(1)
        loss_d_real = prob_real.mean(0).view(1)
        loss_d = loss_d_fake - self.s_dim * loss_d_real

        ch_errors = chen_error_3step(fake_data, self.w_dim)
        pretty_chen_errors = make_pretty(ch_errors)
        report += f"gradient norm: {gradient_norm:.5f}, discriminator dist: {loss_d.item():.5f}"

        # Test Wasserstein error for fixed W
        if self.w_dim > 2:
            noise = torch.randn((self.test_batch_size, self.noise_size), dtype=torch.float, device=self.device)
            g_in = torch.cat((noise, self.W_fixed), 1)
            a_fixed_gen = self.netG(g_in)[:, self.w_dim:self.w_dim + self.a_dim].detach().numpy()
            errors = [sqrt(ot.wasserstein_1d(self.A_fixed_true[:, i], a_fixed_gen[:, i], p=2)) for i in
                      range(self.a_dim)]
            pretty_errors = make_pretty(errors)
            st_dev_err = self.avg_st_dev_error(a_fixed_gen)
            joint_err = joint_wass_dist(self.A_fixed_true[:1000], a_fixed_gen[:1000])
            pretty_chen_errors = make_pretty(self.chen_errors())
            report += f", st_dev error: {st_dev_err: .5f}, joint_wass_dist: {joint_err: .5f}{line_break}errs: {pretty_errors}, ch_err: {pretty_chen_errors} "
        else:
            errors = self.all_2dim_errors()
            pretty_errors = make_pretty(errors)
            report += f"{line_break}errs: {pretty_errors}, ch_err: {pretty_chen_errors[0]}"

        return report, errors, ch_errors

    def draw_error_graphs(self,wass_errors_through_training, chen_errors_through_training):
        labels = list_pairs(self.w_dim)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))
        ax1.set_title("Individual 2-Wasserstein errors")
        ax1.plot(wass_errors_through_training, label=labels)
        ax1.set_xlabel("iterations")
        ax1.legend(prop={'size': 15})
        ax2.set_title("Chen errors")
        ax2.plot(chen_errors_through_training, label=labels)
        ax2.set_xlabel("iterations")
        ax2.legend(prop={'size': 15})
        fig.show()
        graph_filename = f"model_saves/{self.dict_saves_folder}/graph_num{self.serial_number}.png"
        fig.savefig(graph_filename)

    def load_dicts(self, descriptor: str = ""):
        folder_name = f'model_saves/{self.dict_saves_folder}/'
        self.netG.load_state_dict(torch.load(folder_name + f'generator_{descriptor}.pt'))
        self.netD.load_state_dict(torch.load(folder_name + f'discriminator_{descriptor}.pt'))

    def load_dicts_unstructured(self, gen_filename, discr_filename):
        self.netG.load_state_dict(torch.load(gen_filename))
        self.netD.load_state_dict(torch.load(discr_filename))

    def save_current_dicts(self, report: str, descriptor: str = ""):
        params_g = copy.deepcopy(self.netG.state_dict())
        params_d = copy.deepcopy(self.netD.state_dict())
        self.save_dicts(params_g, params_d, report, descriptor)

    def save_dicts(self, params_g, params_d, report: str, descriptor: str = ""):
        filename = f'model_saves/{self.dict_saves_folder}/summary_file.txt'

        with open(filename, 'a+') as summary_file:
            summary = f"{self.serial_number} {descriptor}: {report} \n"
            summary_file.write(summary)

        folder_name = f'model_saves/{self.dict_saves_folder}/'
        torch.save(params_g, folder_name + f'generator_num{self.serial_number}_{descriptor}.pt')
        torch.save(params_d, folder_name + f'discriminator_num{self.serial_number}_{descriptor}.pt')

    def increase_serial(self):
        self.serial_number += 1

    def compute_wth(self, w_in: torch.Tensor, h_in: torch.Tensor):
        return aux_compute_wth(w_in, h_in, self.S, self.T, self.w_dim)

    def compute_wthmb(self, wth_in: torch.Tensor, b_in: torch.Tensor):
        return aux_compute_wthmb(wth_in, b_in, self.M, self.w_dim)

    def classic_train(self, tr_conf: dict):
        print("blub")
        # Number of training epochs using classical training
        self.num_epochs = tr_conf['num epochs']

        # 'Adam' of 'RMSProp'
        which_optimizer = tr_conf['optimizer']

        # Learning rate for optimizers
        lrG = tr_conf['lrG']
        lrD = tr_conf['lrD']

        # Beta hyperparam for Adam optimizers
        beta1 = tr_conf['beta1']
        beta2 = tr_conf['beta2']

        if which_optimizer == 'Adam':
            opt_g = torch.optim.Adam(self.netG.parameters(), lr=lrG, betas=(beta1, beta2))
            opt_d = torch.optim.Adam(self.netD.parameters(), lr=lrD, betas=(beta1, beta2))
        elif which_optimizer == 'RMSProp':
            opt_g = torch.optim.RMSprop(self.netG.parameters(), lr=lrG)
            opt_d = torch.optim.RMSprop(self.netD.parameters(), lr=lrD)
        else:
            opt_g = torch.optim.RMSprop(self.netG.parameters(), lr=lrG)
            opt_d = torch.optim.RMSprop(self.netD.parameters(), lr=lrD)

        # Which method of keeping the critic Lipshcitz to use. 'gp' for gradient penalty, 'wc' for weight clipping
        self.Lipschitz_mode = tr_conf['Lipschitz mode']

        # for weight clipping
        weight_clipping_limit = tr_conf['weight clipping limit']

        # for gradient penalty
        gp_weight = tr_conf['gp weight']

        batch_size = tr_conf['batch size']

        # create dataloader for samples
        def row_processer(row):
            return torch.tensor(np.array(row, dtype=np.float32), dtype=torch.float, device=self.device)

        filename = f"samples/samples_{self.w_dim}-dim.csv"
        datapipe = dp.iter.FileOpener([filename], mode='t')
        datapipe = datapipe.parse_csv(delimiter=',')
        datapipe = datapipe.map(row_processer)
        dataloader = DataLoader(dataset=datapipe, batch_size=batch_size, num_workers=2)

        # Check if the dimensions match
        d = next(iter(dataloader))
        if d.size(1) != self.a_dim + self.w_dim:
            print("!!!!!!!!!!!!!!!!!!!!!!!!! WRONG DATA DIMENSIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Early stopping setup
        min_sum = float('inf')
        min_chen_err_sum = float('inf')

        # For graphing
        wass_errors_through_training = []
        chen_errors_through_training = []

        iters = 0

        for epoch in range(self.num_epochs):

            for i, data in enumerate(dataloader):
                self.netD.zero_grad()
                self.netG.zero_grad()

                # weight clipping so critic is lipschitz
                if self.Lipschitz_mode == 'wc':
                    for p in self.netD.parameters():
                        p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)

                # check actual batch size (last batch could be shorter)
                b_size = data.shape[0]

                noise = torch.randn((b_size, self.noise_size), dtype=torch.float, device=self.device)
                w = data[:, :self.w_dim]
                z = torch.cat((noise, w), dim=1)
                fake_data = self.netG(z)
                fake_data = fake_data.detach()
                pruning_indices = torch.randperm(b_size * self.s_dim)[:b_size]
                pruned_fake_data = fake_data[pruning_indices]

                gradient_penalty, gradient_norm = self._gradient_penalty(data, pruned_fake_data, gp_weight= gp_weight)

                prob_real = self.netD(data)

                prob_fake = self.netD(fake_data)

                loss_d_fake = prob_fake.mean(0).view(1)
                loss_d_real = prob_real.mean(0).view(1)
                loss_d = loss_d_fake - self.s_dim * loss_d_real
                if self.Lipschitz_mode == 'gp':
                    loss_d += gradient_penalty
                loss_d.backward()
                opt_d.step()

                # train Generator with probability 1/5
                if iters % 5 == 0:
                    self.netG.zero_grad()
                    noise = torch.randn((b_size, self.noise_size), dtype=torch.float, device=self.device)
                    w = data[:, :self.w_dim]
                    z = torch.cat((noise, w), dim=1)
                    fake_data = self.netG(z)
                    lossG = self.netD(fake_data)
                    lossG = - lossG.mean(0).view(1)
                    lossG.backward()
                    opt_g.step()

                if iters % 100 == 0:
                    report, errors, chen_errors = self.make_report(epoch=epoch, iters=iters)
                    print(report)
                    wass_errors_through_training.append(errors)
                    chen_errors_through_training.append(chen_errors)
                    # Early stopping checkpoint
                    error_sum = sum(errors)
                    if error_sum <= min_sum:
                        min_sum = error_sum
                        self.save_current_dicts(report=report, descriptor="min_sum")
                        print("Saved parameters (fixed error)")

                    chen_err_sum = sum(chen_errors)
                    if chen_err_sum < min_chen_err_sum:
                        min_chen_err_sum = chen_err_sum
                        self.save_current_dicts(report=report, descriptor="min_chen")
                        print("Saved parameters (chen errors)")

                iters +=1

        self.draw_error_graphs(wass_errors_through_training,chen_errors_through_training)


    def chen_train(self, tr_conf: dict):
        print("blub")
        # Number of iterations of Chen training
        num_Chen_iters = tr_conf['num Chen iters']

        # 'Adam' of 'RMSProp'
        which_optimizer = tr_conf['optimizer']

        # Learning rate for optimizers
        lrG = tr_conf['lrG']
        lrD = tr_conf['lrD']

        # Beta hyperparam for Adam optimizers
        beta1 = tr_conf['beta1']
        beta2 = tr_conf['beta2']

        if which_optimizer == 'Adam':
            optG = torch.optim.Adam(self.netG.parameters(), lr=lrG, betas=(beta1, beta2))
            optD = torch.optim.Adam(self.netD.parameters(), lr=lrD, betas=(beta1, beta2))
        elif which_optimizer == 'RMSProp':
            optG = torch.optim.RMSprop(self.netG.parameters(), lr=lrG)
            optD = torch.optim.RMSprop(self.netD.parameters(), lr=lrD)

        # To keep the criterion Lipschitz
        weight_clipping_limit = tr_conf['weight clipping limit']

        # for gradient penalty
        gp_weight = tr_conf['gp weight']

        batch_size = tr_conf['batch size']

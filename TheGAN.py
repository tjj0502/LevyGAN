from pathlib import Path
from statistics import mean
import timeit
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
from hyperopt import fmin, tpe, hp, STATUS_OK

from aux_functions import *
from Generator import Generator
from Discriminator import Discriminator
import configs_folder.configs as configs
import importlib

from torch.autograd import grad as torch_grad

config = configs.config

training_config = configs.training_config


class LevyGAN:

    def __init__(self, config_in: dict = None, serial_num_in: int = -1, do_load_samples=True):
        if config_in is None:
            importlib.reload(configs)
            cf = configs.config
        else:
            cf = config_in
        init_config(cf)

        # ============ Model config ===============
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.ngpu = 1
        else:
            self.device = torch.device('cpu')
            self.ngpu = 0
        self.noise_size = cf['noise size']
        self.w_dim = cf['w dim']
        self.a_dim = cf['a dim']
        self.s_dim = cf['s dim']
        # if 1 use GAN1, if 2 use GAN2, etc.
        self.which_discriminator = cf['which discriminator']
        self.which_generator = cf['which generator']

        self.generator_symmetry_mode = cf['generator symmetry mode']
        self.generator_last_width = cf['generator last width']

        # slope for LeakyReLU
        self.leakyReLU_slope = cf['leakyReLU slope']

        self.T, self.M, self.S = generate_tms(self.w_dim, self.device)

        # create the nets
        self.netG = Generator(cf)
        self.netD = Discriminator(cf)

        self.dict_saves_folder = f'model_G{self.which_generator}_D{self.which_discriminator}_' + \
                                 f'{self.generator_symmetry_mode}_{self.w_dim}d_{self.noise_size}noise'
        Path(f"model_saves/{self.dict_saves_folder}/").mkdir(parents=True, exist_ok=True)

        if serial_num_in < 0:
            self.serial_number = read_serial_number(self.dict_saves_folder)
        else:
            self.serial_number = serial_num_in

        # ============== Training config ===============
        # Number of training epochs using classical training
        self.num_epochs = 0

        # Number of iterations of Chen training
        self.num_Chen_iters = 0

        self.testing_frequency = 100

        self.descriptor = ""

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

        # self.bsz = cf['bsz']

        # ============ Testing config ============
        self.num_tests_for_lowdim = cf['num tests for 2d']
        if self.w_dim == 3 and self.num_tests_for_lowdim > 7:
            self.num_tests_for_lowdim = 5

        self.test_bsz = cf['test bsz']
        self.unfixed_test_bsz = cf['unfixed test bsz']
        self.joint_wass_dist_bsz = cf['joint wass dist bsz']

        self.W_fixed_whole = cf['W fixed whole']
        self.W_fixed = torch.tensor(self.W_fixed_whole, device=self.device)[:self.w_dim].unsqueeze(1).transpose(1, 0)
        self.W_fixed = self.W_fixed.expand((self.test_bsz, self.w_dim))

        self.joint_labels = ["joint error"]

        self.st_dev_W_fixed = np.diag(true_st_devs(self.W_fixed_whole[:self.w_dim]))

        # Load "true" samples generated from this fixed W increment
        fixed_test_data_filename = f"samples/fixed_samples_{self.w_dim}-dim.csv"
        self.fixed_data = np.genfromtxt(fixed_test_data_filename, dtype=float, delimiter=',')[:self.test_bsz]
        self.A_fixed_true = self.fixed_data[:self.test_bsz, self.w_dim:(self.w_dim + self.a_dim)]

        samples_filename = f"samples/samples_{self.w_dim}-dim.csv"
        self.samples = np.array([], dtype=float)
        self.samples_torch = torch.tensor([], dtype=torch.float, device=self.device)
        if do_load_samples:
            self.samples = np.genfromtxt(samples_filename, dtype=float, delimiter=',')
            self.samples_torch = torch.tensor(self.samples, dtype=torch.float, device=self.device)

        self.single_coord_labels = []
        self.fixed_data_for_lowdim = []
        if self.w_dim <= 3:
            self.joint_labels = []
            for i in range(self.num_tests_for_lowdim):
                data = np.genfromtxt(f"samples/fixed_samples_{self.w_dim}-dim{i + 3}.csv", dtype=float, delimiter=',')
                self.fixed_data_for_lowdim.append(data)
                w = list(data[0, :self.w_dim])
                self.joint_labels.append(w)
                self.single_coord_labels += list_pairs(self.w_dim, w)
        else:
            self.single_coord_labels = list_pairs(self.w_dim)

        self.test_results = {
            'errors': [],
            'chen errors': [],
            'joint wass errors': [float('inf')],
            'best joint errors': [float('inf')],
            'st dev error': -1.0,
            'loss d': 0.0,
            'gradient norm': 0.0,
            'min sum': float('inf'),
            'best fixed errors': [],
            'min chen sum': float('inf'),
            'best chen errors': [],
            'best score': float('inf'),
            'best score report': ''
        }

        self.do_timeing = cf['do timeing']
        self.start_time = timeit.default_timer()
        self.print_reports = True
        if 'print reports' in cf:
            self.print_reports = cf['print reports']
        self.should_draw_graphs = True
        if 'should draw graphs' in cf:
            self.should_draw_graphs = cf['should draw graphs']

    def config(self):
        cf = {
            'device': self.device,
            'ngpu': self.ngpu,
            'w dim': self.w_dim,
            'a dim': self.a_dim,
            'noise size': self.noise_size,
            'which generator': self.which_generator,
            'which discriminator': self.which_discriminator,
            'generator symmetry mode': self.generator_symmetry_mode,
            'generator last width': self.generator_last_width,
            's dim': self.s_dim,
            'leakyReLU slope': self.leakyReLU_slope,
            'test bsz': self.test_bsz,
            'unfixed test bsz': self.unfixed_test_bsz,
            'joint wass dist bsz': self.joint_wass_dist_bsz,
            'num tests for 2d': self.num_tests_for_lowdim,
            'W fixed whole': self.W_fixed_whole,
            'do timeing': self.do_timeing
        }
        return cf

    def load_samples(self):
        samples_filename = f"samples/samples_{self.w_dim}-dim.csv"
        self.samples = np.genfromtxt(samples_filename, dtype=float, delimiter=',')
        self.samples_torch = torch.tensor(self.samples, dtype=torch.float, device=self.device)

    def reload_testing_config(self, cf: dict):
        self.num_tests_for_lowdim = cf['num tests for 2d']

        self.test_bsz = cf['test bsz']
        self.unfixed_test_bsz = cf['unfixed test bsz']
        self.joint_wass_dist_bsz = cf['joint wass dist bsz']

        self.W_fixed_whole = cf['W fixed whole']
        self.W_fixed = torch.tensor(self.W_fixed_whole, device=self.device)[:self.w_dim].unsqueeze(1).transpose(1, 0)
        self.W_fixed = self.W_fixed.expand((self.test_bsz, self.w_dim))

        self.st_dev_W_fixed = np.diag(true_st_devs(self.W_fixed_whole[:self.w_dim]))

        # Load "true" samples generated from this fixed W increment
        fixed_test_data_filename = f"samples/fixed_samples_{self.w_dim}-dim.csv"
        self.fixed_data = np.genfromtxt(fixed_test_data_filename, dtype=float, delimiter=',')[:self.test_bsz]
        self.A_fixed_true = self.fixed_data[:self.test_bsz, self.w_dim:(self.w_dim + self.a_dim)]

        samples_filename = f"samples/samples_{self.w_dim}-dim.csv"
        self.samples = np.genfromtxt(samples_filename, dtype=float, delimiter=',')
        self.samples_torch = torch.tensor(self.samples, dtype=torch.float, device=self.device)

        self.fixed_data_for_lowdim = []
        if self.w_dim == 2:
            self.fixed_data_for_lowdim = [
                np.genfromtxt(f"samples/fixed_samples_2-dim{i + 1}.csv", dtype=float, delimiter=',') for i in
                range(self.num_tests_for_lowdim)]

        self.reset_test_results()

        self.do_timeing = cf['do timeing']
        self.start_time = timeit.default_timer()

    def reset_test_results(self):
        self.test_results = {
            'errors': [],
            'chen errors': [],
            'joint wass errors': [float('inf')],
            'best joint errors': [float('inf')],
            'st dev error': -1.0,
            'loss d': 0.0,
            'gradient norm': 0.0,
            'min sum': float('inf'),
            'best fixed errors': [],
            'min chen sum': float('inf'),
            'best chen errors': [],
            'best score': float('inf'),
            'best score report': ''
        }

    def _gradient_penalty(self, real_data, generated_data, gp_weight):
        b_size_gp = real_data.shape[0]
        # Calculate interpolation
        alpha = torch.rand(b_size_gp, 1, device=self.device)
        alpha = alpha.expand_as(real_data)
        interpolated = (alpha * real_data + (1 - alpha) * generated_data).requires_grad_(True)

        if self.ngpu > 0:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.netD(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
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

    def all_2dim_errors(self, comp_loss_d = False):
        assert self.w_dim == 2
        all_errors = []
        losses = []
        for i in range(self.num_tests_for_lowdim):
            # Test Wasserstein error for fixed W
            data_fixed_true = self.fixed_data_for_lowdim[i]
            data_fixed_true = data_fixed_true[:self.test_bsz]
            errors, joint_err, loss_d = self.compute_fixed_errors(data_fixed_true, comp_loss_d=comp_loss_d)
            all_errors += errors
            if comp_loss_d:
                losses.append(loss_d)
        return all_errors, losses

    def compute_fixed_errors(self, input_true_data, comp_joint_error=False, comp_loss_d = False):
        noise = torch.randn((self.test_bsz, self.noise_size), dtype=torch.float, device=self.device)
        true_data = input_true_data[:self.test_bsz]
        true_data_torch = torch.tensor(true_data, dtype=torch.float, device=self.device)
        w_torch = true_data_torch[:, :self.w_dim]
        a_true = true_data[:, self.w_dim : (self.w_dim + self.a_dim)]
        g_in = torch.cat((noise, w_torch), 1)
        fake_data = self.netG(g_in)
        pruning_indices = self.unfixed_test_bsz * \
                          torch.randint(high=self.s_dim, size=(self.unfixed_test_bsz,), device=self.device) + \
                          torch.arange(self.unfixed_test_bsz, dtype=torch.int, device=self.device)
        fake_data = (fake_data[pruning_indices]).detach()
        a_fake = fake_data[:, self.w_dim:self.w_dim + self.a_dim].cpu().numpy()
        errors = [sqrt(ot.wasserstein_1d(a_true[:, i], a_fake[:, i], p=2)) for i in
                  range(self.a_dim)]
        self.print_time("FIXED ERRORS")
        if self.w_dim >= 3:
            st_dev_err = self.avg_st_dev_error(a_fake)
            self.test_results['st dev error'] = st_dev_err
            self.print_time("ST DEV ERRORS")
        else:
            st_dev_err = float('inf')

        if comp_loss_d:
            prob_real = self.netD(true_data_torch)
            prob_fake = self.netD(fake_data)
            self.print_time("netD FOR REPORT")
            loss_d_fake = prob_fake.mean(0).view(1)
            loss_d_real = prob_real.mean(0).view(1)
            loss_d = loss_d_fake - loss_d_real
            loss_d = loss_d.detach().item()
        else:
            loss_d = float('inf')

        if comp_joint_error and (self.w_dim >= 3):
            joint_err = joint_wass_dist(a_true[:self.joint_wass_dist_bsz],
                                        a_fake[:self.joint_wass_dist_bsz])
        else:
            joint_err = float('inf')

        return errors, joint_err, loss_d

    def all_3dim_errors(self, comp_joint_error=False, comp_loss_d = False):
        assert self.w_dim == 3
        all_errors = []
        joint_errors = []
        losses = []
        for i in range(self.num_tests_for_lowdim):
            data_fixed_true = self.fixed_data_for_lowdim[i]
            data_fixed_true = data_fixed_true[:self.test_bsz]
            errors, joint_err, loss_d = self.compute_fixed_errors(data_fixed_true, comp_joint_error, comp_loss_d)
            all_errors += errors
            if comp_joint_error:
                joint_errors.append(joint_err)
            if comp_loss_d:
                losses.append(loss_d)

        assert (len(all_errors) == len(self.single_coord_labels))
        if not comp_joint_error:
            joint_errors = [float('inf')]
        return all_errors, joint_errors, losses

    # def multi_dim_wasserstein_errors(self):
    #     noise = torch.randn((self.test_bsz, self.noise_size), dtype=torch.float, device=self.device)
    #     g_in = torch.cat((noise, self.W_fixed), 1)
    #     A_fixed_gen = self.netG(g_in)[:, self.w_dim:self.w_dim + self.a_dim].detach().numpy()
    #     errors = [sqrt(ot.wasserstein_1d(self.A_fixed_true[:, i], A_fixed_gen[:, i], p=2)) for i in range(self.a_dim)]

    def chen_errors(self):
        _W = torch.randn((self.test_bsz, self.w_dim), dtype=torch.float, device=self.device)
        noise = torch.randn((self.test_bsz, self.noise_size), dtype=torch.float, device=self.device)
        gen_in = torch.cat((noise, _W), 1)
        generated_data = self.netG(gen_in).detach()
        return chen_error_3step(generated_data, self.w_dim)

    def avg_st_dev_error(self, _a_generated):
        difference = np.abs(self.st_dev_W_fixed - np.sqrt(np.abs(empirical_second_moments(_a_generated))))
        return difference.mean()

    def do_tests(self, comp_joint_err=False, comp_grad_norm=False, comp_loss_d=False, comp_chen_error=False, save_models=False, save_best_results=True):
        self.netG.eval()
        self.netD.eval()

        chen_errors = []
        if comp_chen_error or comp_grad_norm:
            unfixed_data = self.samples_torch[:self.unfixed_test_bsz]
            actual_bsz = unfixed_data.shape[0]

            noise = torch.randn((actual_bsz, self.noise_size), dtype=torch.float, device=self.device)
            w = unfixed_data[:, :self.w_dim]
            z = torch.cat((noise, w), dim=1)
            self.print_time("Z FOR REPORT")
            fake_data = self.netG(z)
            pruning_indices = self.unfixed_test_bsz * \
                              torch.randint(high=self.s_dim, size=(self.unfixed_test_bsz,), device=self.device) + \
                              torch.arange(self.unfixed_test_bsz, dtype=torch.int, device=self.device)
            fake_data = (fake_data[pruning_indices]).detach()
            self.print_time("RUNNING netG FOR REPORT")

            if comp_grad_norm:
                gradient_penalty, gradient_norm = self._gradient_penalty(unfixed_data, fake_data, gp_weight=0)
                self.test_results['gradient norm'] = gradient_norm

            self.print_time("UNFIXED PART OF REPORT")
            chen_errors = []
            if comp_chen_error:
                chen_errors = chen_error_3step(fake_data, self.w_dim)
                self.test_results['chen errors'] = chen_errors
                self.print_time("CHEN ERRORS")

        joint_wass_errors = [float('inf')]
        losses = [float('inf')]
        # Test Wasserstein error for fixed W
        errors = []
        if self.w_dim > 3:
            errors, joint_wass_error, loss_d = self.compute_fixed_errors(self.fixed_data, comp_joint_err, comp_loss_d)
            joint_wass_errors = [joint_wass_error]
            losses = [loss_d]
            self.test_results['joint wass errors'] = joint_wass_errors
        elif self.w_dim == 2:
            errors, losses = self.all_2dim_errors()
        elif self.w_dim == 3:
            errors, joint_wass_errors, losses = self.all_3dim_errors(comp_joint_err, comp_loss_d)
            self.test_results['joint wass errors'] = joint_wass_errors

        self.test_results['errors'] = errors
        self.test_results['loss d'] = losses
        flag_for_joint_err = True  # just to avoid computing joint error twice
        if sum(errors) < self.test_results['min sum']:
            self.test_results['min sum'] = sum(errors)
            if save_best_results:
                self.test_results['best fixed errors'] = make_pretty(errors)

        if comp_chen_error and (sum(chen_errors) < self.test_results['min chen sum']) and save_best_results:
            self.test_results['min chen sum'] = sum(chen_errors)
            self.test_results['best chen errors'] = make_pretty(chen_errors)

        report = self.make_report(add_line_break=False)
        if sum(joint_wass_errors) < sum(self.test_results['best joint errors']):
            if save_best_results:
                self.test_results['best joint errors'] = make_pretty(joint_wass_errors)
            # if save_models:
                # self.save_current_dicts(report=report,
                #                         descriptor=f"{self.descriptor}_min_sum")
                # print("Warning: not saving on min sum")


        if comp_joint_err:
            score = self.model_score()
        else:
            score = self.model_score(c=0.0)

        if score < self.test_results['best score']:
            report = self.make_report(add_line_break=False)
            if save_best_results:
                self.test_results['best score'] = make_pretty(score)
                self.test_results['best score report'] = report
            if save_models:
                self.save_current_dicts(report=report,
                                        descriptor=f"{self.descriptor}_max_scr")
                print(f"Saved model with best score: {make_pretty(score)}")

        self.netG.train()
        self.netD.train()

        return

    def eval(self, w_in):
        actual_bsz = w_in.shape[0]
        self.netG.eval()
        #self.start_time = timeit.default_timer()
        noise = torch.randn((actual_bsz, self.noise_size), dtype=torch.float, device=self.device)
        z = torch.cat((noise, w_in), dim=1)
        fake_data = self.netG(z)
        #self.print_time("EVAL")

        pruning_indices = actual_bsz * \
                          torch.randint(high=self.s_dim, size=(actual_bsz,), device=self.device) + \
                          torch.arange(actual_bsz, dtype=torch.int, device=self.device)
        fake_data = (fake_data[pruning_indices]).detach()
        #elapsed = timeit.default_timer() - self.start_time
        self.netG.train()
        return fake_data

    def model_score(self, a: float = 20.0, b: float = 0.0, c: float = 0.6):
        res = 0.0
        res += a * mean(self.test_results['errors'])
        if b > 0.0:
            res += b * mean(self.test_results['chen errors'])
        if c > 0.0:
            res += c * self.a_dim * mean(self.test_results['joint wass errors'])
        return res

    def compute_objective(self, optimizer, lrG, lrD, num_discr_iters, beta1, beta2, gp_weight, leaky_slope,
                          tr_conf_in: dict = None, trials: int = 5):
        self.leakyReLU_slope = leaky_slope
        cf = self.config()

        if tr_conf_in is None:
            importlib.reload(configs)
            tr_conf = configs.training_config
        else:
            tr_conf = tr_conf_in
        tr_conf['optimizer'] = optimizer
        tr_conf['lrG'] = lrG
        tr_conf['lrD'] = lrD
        tr_conf['num discr iters'] = num_discr_iters
        tr_conf['beta1'] = beta1
        tr_conf['beta2'] = beta2
        tr_conf['gp_weight'] = gp_weight

        scores = []
        attachments = {}
        for i in range(trials):
            if optimizer == 'Adam':
                descr = f"COMP_OBJ_Adam_b1_{beta1:.3f}_b2_{beta2:.4f}_lrG{lrG:.6f}_lrD{lrD:.6f}_numDitr{num_discr_iters}_gp{gp_weight:.0f}_lkslp{leaky_slope:.3f}_trial{i}"
            else:
                descr = f"COMP_OBJ_RMSProp_lrG{lrG:.6f}_lrD{lrD:.6f}_numDitr{num_discr_iters}_gp{gp_weight:.0f}_lkslp{leaky_slope:.3f}_trial{i}"
            tr_conf['descriptor'] = descr
            print(descr, end="")
            self.netG = Generator(cf)
            self.netD = Discriminator(cf)
            self.reset_test_results()
            self.classic_train(tr_conf, save_models=False)
            scores.append(self.test_results['best score'])
            attachments[f'trial {i} best joint errors'] = self.test_results['best joint errors']
            attachments[f'trial {i} best fixed errors'] = self.test_results['best fixed errors']
            attachments[f'trial {i} best chen errors'] = self.test_results['best chen errors']
            attachments[f'trial {i} best score'] = self.test_results['best score']
            attachments[f'trial {i} best score report'] = self.test_results['best score report']

        variance = 2 * np.var(scores)
        if len(scores) == 1:
            variance = 0.3
        mean = np.mean(scores)
        result_dict = {
            'status': STATUS_OK,
            'loss': mean,
            'loss_variance': variance,
            'other stuff': attachments
        }

        return result_dict

    def make_report(self, epoch: int = None, iters: int = None, chen_iters: int = None, add_line_break=True, short = False):
        report = ""
        if add_line_break:
            line_break = "\n"
        else:
            line_break = ", "
        if not (epoch is None):
            report += f"ep: {epoch}/{self.num_epochs}, "
        if not (iters is None):
            report += f"itr: {iters}, "
        if not (chen_iters is None):
            report += f"chen_iters: {chen_iters}/{self.num_Chen_iters}, "

        score = self.test_results['best score']
        report += f"scr: {score:.5f}, "
        grad_norm = self.test_results['gradient norm']
        report += f"discr grad norm: {grad_norm:.5f}, "
        if len(self.test_results['loss d']) > 0:
            if short:
                avg_loss_d = sum(self.test_results['loss d'])/len(self.test_results['loss d'])
                report += f"discr loss(es): {make_pretty(avg_loss_d)}"
            else:
                report += f"discr loss(es): {make_pretty(self.test_results['loss d'])}"
        joint_wass_errors = self.test_results['joint wass errors']
        if len(joint_wass_errors) > 0:
            if short:
                avg_joint = sum(joint_wass_errors)/len(joint_wass_errors)
                report += f", avg joint: {make_pretty(avg_joint)}"
            else:
                report += f", joint errs: {make_pretty(joint_wass_errors)}"
        st_dev_error = self.test_results['st dev error']
        if st_dev_error < 100 and not short:
            report += f", st dev err: {st_dev_error:.5f}"
        # pretty_errors = make_pretty(self.test_results['errors'])
        # report += f"{line_break}errs: {pretty_errors}"
        if len(self.test_results['errors']) > 0:
            if short:
                avg_err = sum(self.test_results['errors'])/len(self.test_results['errors'])
                report += f" avg err: {make_pretty(avg_err)}"
            else:
                report +=f"{line_break}errs: {make_pretty(self.test_results['errors'])}"

        if len(self.test_results['chen errors']) > 0:
            pretty_chen_errors = make_pretty(self.test_results['chen errors'])
            if len(pretty_chen_errors) == 1:
                pretty_chen_errors = pretty_chen_errors[0]
            report += f", chen errs: {pretty_chen_errors}"

        return report

    def draw_error_graphs(self, wass_errors_through_training, losses_through_training,
                          joint_errors_through_training=None, descriptor: str = ''):
        if not self.should_draw_graphs:
            return
        labels = self.single_coord_labels
        if not (joint_errors_through_training is None):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 35))
            ax3.set_title("Joint 2-Wasserstein errors")
            ax3.plot(joint_errors_through_training, label=self.joint_labels)
            ax3.set_ylim([-0.01, 0.4])
            ax3.set_xlabel("iterations")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(20, 15))
            ax2.set_xlabel("iterations")
        ax1.set_title("Individual 2-Wasserstein errors")
        ax1.plot(wass_errors_through_training, label=labels)
        ax1.set_ylim([-0.005, 0.3])
        ax1.legend(prop={'size': 15})
        ax2.set_title("Discriminator losses")
        ax2.plot(losses_through_training, label=self.joint_labels)
        ax2.set_ylim([-0.01, 0.06])
        ax2.legend(prop={'size': 15})

        fig.show()
        graph_filename = f"model_saves/{self.dict_saves_folder}/graph_{self.dict_saves_folder}_num{self.serial_number}_{descriptor}.png"
        fig.savefig(graph_filename)

    def load_dicts(self, serial_num_to_load: int = -1, descriptor: str = ""):
        if serial_num_to_load < 0:
            sn = self.serial_number
        else:
            sn = serial_num_to_load
        folder_name = f'model_saves/{self.dict_saves_folder}/'
        self.netG.load_state_dict(
            torch.load(folder_name + f'generator_num{sn}_{descriptor}.pt', map_location=self.device))
        self.netD.load_state_dict(torch.load(folder_name + f'discriminator_num{sn}_{descriptor}.pt',
                                             map_location=self.device))

    def load_dicts_unstructured(self, gen_filename, discr_filename):
        self.netG.load_state_dict(torch.load(gen_filename, map_location=self.device))
        self.netD.load_state_dict(torch.load(discr_filename, map_location=self.device))

    def save_current_dicts(self, report: str, descriptor: str = ""):
        params_g = copy.deepcopy(self.netG.state_dict())
        params_d = copy.deepcopy(self.netD.state_dict())
        self.save_dicts(params_g, params_d, report, descriptor)

    def save_dicts(self, params_g, params_d, report: str, descriptor: str = ""):
        filename = f'model_saves/{self.dict_saves_folder}/summary_file.txt'
        line_header = f"{self.serial_number} {descriptor}"
        summary = f"{line_header}: {report} \n"

        with open(filename, 'r+') as summary_file:
            lines = summary_file.readlines()
            summary_file.seek(0)

            flag = False
            for i in range(len(lines)):
                line_header_from_file = lines[i].split(':')[0]
                if line_header == line_header_from_file:
                    lines[i] = summary
                    flag = True
                    break

            if not flag:
                lines.append(summary)

            summary_file.writelines(lines)
            summary_file.truncate()

        folder_name = f'model_saves/{self.dict_saves_folder}/'
        torch.save(params_g, folder_name + f'generator_num{self.serial_number}_{descriptor}.pt')
        torch.save(params_d, folder_name + f'discriminator_num{self.serial_number}_{descriptor}.pt')

    def print_time(self, description: str = ""):
        if self.do_timeing:
            elapsed = timeit.default_timer() - self.start_time
            print(f"{description} TIME: {elapsed}")
            self.start_time = timeit.default_timer()

    def classic_train(self, tr_conf_in: dict = None, save_models=True):
        self.print_time("START TRAIN")
        if tr_conf_in is None:
            importlib.reload(configs)
            tr_conf = configs.training_config
        else:
            tr_conf = tr_conf_in

        # Number of training epochs using classical training
        self.num_epochs = tr_conf['num epochs']
        max_iters = float('inf')
        if 'max iters' in tr_conf and isinstance(tr_conf['max iters'], int):
            max_iters = tr_conf['max iters']

        # 'Adam' of 'RMSProp'
        which_optimizer = tr_conf['optimizer']

        # Learning rate for optimizers
        lr_g = tr_conf['lrG']
        lr_d = tr_conf['lrD']
        num_discr_iters = tr_conf['num discr iters']

        # Beta hyperparam for Adam optimizers
        beta1 = tr_conf['beta1']
        beta2 = tr_conf['beta2']

        if which_optimizer == 'Adam':
            opt_g = torch.optim.Adam(self.netG.parameters(), lr=lr_g, betas=(beta1, beta2))
            opt_d = torch.optim.Adam(self.netD.parameters(), lr=lr_d, betas=(beta1, beta2))
        elif which_optimizer == 'RMSProp':
            opt_g = torch.optim.RMSprop(self.netG.parameters(), lr=lr_g)
            opt_d = torch.optim.RMSprop(self.netD.parameters(), lr=lr_d)
        else:
            opt_g = torch.optim.RMSprop(self.netG.parameters(), lr=lr_g)
            opt_d = torch.optim.RMSprop(self.netD.parameters(), lr=lr_d)

        # Which method of keeping the critic Lipshcitz to use. 'gp' for gradient penalty, 'wc' for weight clipping
        self.Lipschitz_mode = tr_conf['Lipschitz mode']

        # for weight clipping
        weight_clipping_limit = tr_conf['weight clipping limit']

        # for gradient penalty
        gp_weight = tr_conf['gp weight']

        bsz = tr_conf['bsz']

        compute_joint_error = tr_conf['compute joint error']

        self.descriptor = tr_conf['descriptor']

        if 'print reports' in tr_conf:
            self.print_reports = tr_conf['print reports']

        if 'should draw graphs' in tr_conf:
            self.should_draw_graphs = tr_conf['should draw graphs']

        filename = f"samples/samples_{self.w_dim}-dim.csv"
        whole_training_data = self.samples_torch.split(bsz)

        # Early stopping setup
        self.test_results['min sum'] = float('inf')
        self.test_results['min chen sum'] = float('inf')

        # For graphing
        wass_errors_through_training = []
        chen_errors_through_training = []
        joint_errors_through_training = []
        losses_through_training = []

        iters = 0
        for epoch in range(self.num_epochs):
            if not self.print_reports:
                print(f"{epoch}", end="  ")

            if 'custom lrs' in tr_conf:
                lrs = tr_conf['custom lrs']
                if epoch in lrs:
                    lr_g_cust, lr_d_cust = lrs[epoch]
                    opt_g = torch.optim.Adam(self.netG.parameters(), lr=lr_g_cust, betas=(beta1, beta2))
                    opt_d = torch.optim.Adam(self.netD.parameters(), lr=lr_d_cust, betas=(beta1, beta2))
                    print(f"changed lrs to G: {lr_g_cust}, D: {lr_d_cust}")

            for i, data in enumerate(whole_training_data):
                if iters >= max_iters:
                    break
                self.print_time("TOP")
                self.netD.zero_grad()
                self.netG.zero_grad()

                # weight clipping so critic is lipschitz
                if self.Lipschitz_mode == 'wc':
                    for p in self.netD.parameters():
                        p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)

                # check actual bsz (last batch could be shorter)
                actual_bsz = data.shape[0]

                noise = torch.randn((actual_bsz, self.noise_size), dtype=torch.float, device=self.device)
                w = data[:, :self.w_dim]
                z = torch.cat((noise, w), dim=1)
                self.print_time(description="MAKE Z 1")
                fake_data = self.netG(z)
                self.print_time(description="netG 1")
                fake_data_detached = fake_data.detach()
                prob_real = self.netD(data)
                prob_fake = self.netD(fake_data_detached)
                self.print_time(description="netD 1 twice")

                loss_d_fake = prob_fake.mean(0).view(1)
                loss_d_real = prob_real.mean(0).view(1)
                loss_d = loss_d_fake - loss_d_real

                if self.Lipschitz_mode == 'gp':
                    pruning_indices = bsz * torch.randint(low=0, high=self.s_dim, size=(bsz,), device=self.device) + \
                                      torch.arange(bsz, dtype=torch.int, device=self.device)
                    pruned_fake_data = (fake_data_detached[pruning_indices]).detach()
                    gradient_penalty, gradient_norm = self._gradient_penalty(data, pruned_fake_data,
                                                                             gp_weight=gp_weight)
                    self.test_results['gradient norm'] = gradient_norm
                    self.print_time(description="GRAD PENALTY")
                    loss_d += gradient_penalty

                self.print_time(description="BEFORE BACKPROP")
                loss_d.backward()
                self.print_time(description="netD BACKPROP")
                opt_d.step()
                self.print_time(description="OPT D")

                # train Generator with probability 1/5
                if (iters % num_discr_iters == 0) and (iters % self.testing_frequency != 0):
                    self.netG.zero_grad()
                    loss_g = self.netD(fake_data)
                    self.print_time(description="netD 2")
                    loss_g = - loss_g.mean(0).view(1)
                    loss_g.backward()
                    self.print_time(description="netG BACKPROP")
                    opt_g.step()
                    self.print_time(description="OPT G")

                if iters % self.testing_frequency == 0:
                    self.print_time(description="BEFORE TESTS")
                    self.do_tests(comp_joint_err=compute_joint_error, comp_loss_d=True, save_models=save_models)
                    self.print_time(description="AFTER TESTS")
                    if self.print_reports:
                        report = self.make_report(epoch=epoch, iters=iters, short=True)
                        print(report)
                    self.print_time(description="AFTER REPORT")
                    errors = self.test_results['errors']
                    wass_errors_through_training.append(errors)
                    losses = self.test_results['loss d']
                    losses_through_training.append(losses)
                    # chen_errors = self.test_results['chen errors']
                    # chen_errors_through_training.append(chen_errors)
                    if compute_joint_error:
                        joint_errors_through_training.append(self.test_results['joint wass errors'])

                    # chen_err_sum = sum(chen_errors)
                    # if chen_err_sum <= self.test_results['min chen sum']:
                    #     self.test_results['min chen sum'] = chen_err_sum
                    #     if save_models:
                    #         self.save_current_dicts(report=report_for_saving_dicts, descriptor=f"{descriptor}min_chen")
                    #     if self.print_reports:
                    #         print("Min Chen sum")

                    self.print_time(description="SAVING DICTS")
                    self.do_timeing = False
                iters += 1

        best_score = self.test_results['best score']
        self.draw_error_graphs(wass_errors_through_training,
                               joint_errors_through_training=joint_errors_through_training,
                               losses_through_training=losses_through_training,
                               descriptor=f"{self.descriptor}_score_{best_score}")
        if save_models:
            report = self.make_report(add_line_break=False)
            self.save_current_dicts(report=report, descriptor=f"{self.descriptor}_end_trn")

    def chen_train(self, tr_conf_in: dict = None, save_models=True):
        self.print_time("START CHEN TRAIN")
        if tr_conf_in is None:
            importlib.reload(configs)
            tr_conf = configs.training_config
        else:
            tr_conf = tr_conf_in

        # Number of training epochs using classical training
        self.num_Chen_iters = tr_conf['num Chen iters']

        # 'Adam' of 'RMSProp'
        which_optimizer = tr_conf['optimizer']

        # Learning rate for optimizers
        lr_g = tr_conf['lrG']
        lr_d = tr_conf['lrD']
        num_discr_iters = tr_conf['num discr iters']

        # Beta hyperparam for Adam optimizers
        beta1 = tr_conf['beta1']
        beta2 = tr_conf['beta2']

        if which_optimizer == 'Adam':
            opt_g = torch.optim.Adam(self.netG.parameters(), lr=lr_g, betas=(beta1, beta2))
            opt_d = torch.optim.Adam(self.netD.parameters(), lr=lr_d, betas=(beta1, beta2))
        elif which_optimizer == 'RMSProp':
            opt_g = torch.optim.RMSprop(self.netG.parameters(), lr=lr_g)
            opt_d = torch.optim.RMSprop(self.netD.parameters(), lr=lr_d)
        else:
            opt_g = torch.optim.RMSprop(self.netG.parameters(), lr=lr_g)
            opt_d = torch.optim.RMSprop(self.netD.parameters(), lr=lr_d)

        # Which method of keeping the critic Lipshcitz to use. 'gp' for gradient penalty, 'wc' for weight clipping
        self.Lipschitz_mode = tr_conf['Lipschitz mode']

        # for weight clipping
        weight_clipping_limit = tr_conf['weight clipping limit']

        # for gradient penalty
        gp_weight = tr_conf['gp weight']

        bsz = tr_conf['bsz']

        compute_joint_error = tr_conf['compute joint error']

        self.descriptor = tr_conf['descriptor']

        if 'print reports' in tr_conf:
            self.print_reports = tr_conf['print reports']

        if 'should draw graphs' in tr_conf:
            self.should_draw_graphs = tr_conf['should draw graphs']

        # Early stopping setup
        self.test_results['min sum'] = float('inf')
        self.test_results['min chen sum'] = float('inf')

        # For graphing
        wass_errors_through_training = []
        chen_errors_through_training = []
        joint_errors_through_training = []
        losses_through_training = []

        iters = 0
        for i in range(self.num_Chen_iters):

            if 'custom Chen lrs' in tr_conf:
                lrs = tr_conf['custom Chen lrs']
                if iters in lrs:
                    lr_g_cust, lr_d_cust = lrs[iters]
                    opt_g = torch.optim.Adam(self.netG.parameters(), lr=lr_g_cust, betas=(beta1, beta2))
                    opt_d = torch.optim.Adam(self.netD.parameters(), lr=lr_d_cust, betas=(beta1, beta2))
                    print(f"its: {iters} changed lrs to G: {lr_g_cust}, D: {lr_d_cust}")

            self.print_time("TOP")
            self.netD.zero_grad()
            self.netG.zero_grad()

            # weight clipping so critic is lipschitz
            if self.Lipschitz_mode == 'wc':
                for p in self.netD.parameters():
                    p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)

            z = torch.randn((bsz, self.noise_size + self.w_dim), dtype=torch.float, device=self.device)
            w = z[:, self.noise_size:]
            self.print_time(description="MAKE Z 1")
            fake_data = self.netG(z)
            self.print_time(description="netG 1")
            fake_data_detached = fake_data.detach()

            # Key step: generate true data using chen combine
            true_data = chen_combine(chen_combine(fake_data_detached, self.w_dim), self.w_dim)

            prob_real = self.netD(true_data)
            prob_fake = self.netD(fake_data_detached)
            self.print_time(description="netD 1 twice")

            loss_d_fake = prob_fake.mean(0).view(1)
            loss_d_real = prob_real.mean(0).view(1)
            loss_d = loss_d_fake - loss_d_real

            true_data_bsz = (self.s_dim * bsz)//4

            if self.Lipschitz_mode == 'gp':
                pruning_indices = true_data_bsz * torch.randint(low=0, high=4, size=(true_data_bsz,), device=self.device) + \
                                  torch.arange(true_data_bsz, dtype=torch.int, device=self.device)
                pruned_fake_data = (fake_data_detached[pruning_indices]).detach()
                gradient_penalty, gradient_norm = self._gradient_penalty(true_data, pruned_fake_data,
                                                                         gp_weight=gp_weight)
                self.test_results['gradient norm'] = gradient_norm
                self.print_time(description="GRAD PENALTY")
                loss_d += gradient_penalty

            self.print_time(description="BEFORE BACKPROP")
            loss_d.backward()
            self.print_time(description="netD BACKPROP")
            opt_d.step()
            self.print_time(description="OPT D")

            # train Generator every num_discr_iters iterations
            if (iters % num_discr_iters == 0) and (iters % self.testing_frequency != 0):
                self.netG.zero_grad()
                loss_g = self.netD(fake_data)
                self.print_time(description="netD 2")
                loss_g = - loss_g.mean(0).view(1)
                loss_g.backward()
                self.print_time(description="netG BACKPROP")
                opt_g.step()
                self.print_time(description="OPT G")

            if iters % self.testing_frequency == 0:
                self.print_time(description="BEFORE TESTS")
                self.do_tests(comp_joint_err=compute_joint_error, comp_loss_d=True, save_models=save_models)
                self.print_time(description="AFTER TESTS")
                if self.print_reports:
                    report = self.make_report(chen_iters=iters, short=True)
                    print(report)
                self.print_time(description="AFTER REPORT")
                errors = self.test_results['errors']
                wass_errors_through_training.append(errors)
                losses = self.test_results['loss d']
                losses_through_training.append(losses)
                # chen_errors = self.test_results['chen errors']
                # chen_errors_through_training.append(chen_errors)
                if compute_joint_error:
                    joint_errors_through_training.append(self.test_results['joint wass errors'])


                # chen_err_sum = sum(chen_errors)
                # if chen_err_sum <= self.test_results['min chen sum']:
                #     self.test_results['min chen sum'] = chen_err_sum
                #     if save_models:
                #         self.save_current_dicts(report=report_for_saving_dicts, descriptor=f"{descriptor}min_chen")
                #     if self.print_reports:
                #         print("Min Chen sum")

                self.print_time(description="SAVING DICTS")
                self.do_timeing = False
            iters += 1

        best_score = self.test_results['best score']
        self.draw_error_graphs(wass_errors_through_training,
                               joint_errors_through_training=joint_errors_through_training,
                               losses_through_training=losses_through_training,
                               descriptor=f"{self.descriptor}_score_{best_score}")
        if save_models:
            report = self.make_report(add_line_break=False)
            self.save_current_dicts(report=report, descriptor=f"{self.descriptor}_end_trn")
from pathlib import Path
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

    def __init__(self, config_in: dict = None, serial_num_in: int = -1):
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
        self.num_tests_for2d = cf['num tests for 2d']

        self.test_bsz = cf['test bsz']
        self.unfixed_test_bsz = cf['unfixed test bsz']
        self.joint_wass_dist_bsz = cf['joint wass dist bsz']

        self.W_fixed_whole = cf['W fixed whole']
        self.W_fixed = torch.tensor(self.W_fixed_whole, device=self.device)[:self.w_dim].unsqueeze(1).transpose(1, 0)
        self.W_fixed = self.W_fixed.expand((self.test_bsz, self.w_dim))

        self.st_dev_W_fixed = np.diag(true_st_devs(self.W_fixed_whole[:self.w_dim]))

        # Load "true" samples generated from this fixed W increment
        fixed_test_data_filename = f"samples/fixed_samples_{self.w_dim}-dim.csv"
        self.A_fixed_true = np.genfromtxt(fixed_test_data_filename, dtype=float, delimiter=',')
        self.A_fixed_true = self.A_fixed_true[:self.test_bsz, self.w_dim:(self.w_dim + self.a_dim)]

        samples_filename = f"samples/samples_{self.w_dim}-dim.csv"
        self.samples = np.genfromtxt(samples_filename, dtype=float, delimiter=',')
        self.samples_torch = torch.tensor(self.samples, dtype=torch.float, device=self.device)

        self.fixed_data_for_2d = []
        if self.w_dim == 2:
            self.fixed_data_for_2d = [
                np.genfromtxt(f"samples/fixed_samples_2-dim{i + 1}.csv", dtype=float, delimiter=',') for i in
                range(self.num_tests_for2d)]

        self.test_results = {
            'errors': [],
            'chen errors': [],
            'joint wass error': -1.0,
            'st dev error': -1.0,
            'loss d': 0.0,
            'gradient norm': 0.0,
            'min sum': float('inf'),
            'min chen sum': float('inf'),
            'best score': float('inf')
        }

        self.do_timeing = cf['do timeing']
        self.start_time = timeit.default_timer()

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
            'num tests for 2d': self.num_tests_for2d,
            'W fixed whole': self.W_fixed_whole,
            'do timeing': self.do_timeing
        }
        return cf

    def reload_testing_config(self, cf: dict):
        self.num_tests_for2d = cf['num tests for 2d']

        self.test_bsz = cf['test bsz']
        self.unfixed_test_bsz = cf['unfixed test bsz']
        self.joint_wass_dist_bsz = cf['joint wass dist bsz']

        self.W_fixed_whole = cf['W fixed whole']
        self.W_fixed = torch.tensor(self.W_fixed_whole, device=self.device)[:self.w_dim].unsqueeze(1).transpose(1, 0)
        self.W_fixed = self.W_fixed.expand((self.test_bsz, self.w_dim))

        self.st_dev_W_fixed = np.diag(true_st_devs(self.W_fixed_whole[:self.w_dim]))

        # Load "true" samples generated from this fixed W increment
        fixed_test_data_filename = f"samples/fixed_samples_{self.w_dim}-dim.csv"
        self.A_fixed_true = np.genfromtxt(fixed_test_data_filename, dtype=float, delimiter=',')
        self.A_fixed_true = self.A_fixed_true[:self.test_bsz, self.w_dim:(self.w_dim + self.a_dim)]

        samples_filename = f"samples/samples_{self.w_dim}-dim.csv"
        self.samples = np.genfromtxt(samples_filename, dtype=float, delimiter=',')
        self.samples_torch = torch.tensor(self.samples, dtype=torch.float, device=self.device)

        self.fixed_data_for_2d = []
        if self.w_dim == 2:
            self.fixed_data_for_2d = [
                np.genfromtxt(f"samples/fixed_samples_2-dim{i + 1}.csv", dtype=float, delimiter=',') for i in
                range(self.num_tests_for2d)]

        self.reset_test_results()

        self.do_timeing = cf['do timeing']
        self.start_time = timeit.default_timer()

    def reset_test_results(self):
        self.test_results = {
            'errors': [],
            'chen errors': [],
            'joint wass error': -1.0,
            'st dev error': -1.0,
            'loss d': 0.0,
            'gradient norm': 0.0,
            'min sum': float('inf'),
            'min chen sum': float('inf'),
            'best score': float('inf')
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

    def all_2dim_errors(self):
        assert self.w_dim == 2
        errs = []
        for i in range(self.num_tests_for2d):
            # Test Wasserstein error for fixed W
            data_fixed_true = self.fixed_data_for_2d[i]
            data_fixed_true = data_fixed_true[:self.test_bsz]
            a_fixed_true = data_fixed_true[:, 2]
            w_combo = torch.tensor(data_fixed_true[:, :2], dtype=torch.float, device=self.device)
            noise = torch.randn((self.test_bsz, self.noise_size), dtype=torch.float, device=self.device)
            g_in = torch.cat((noise, w_combo), 1)
            a_fixed_gen = self.netG(g_in)[:, 2].detach().cpu().numpy().squeeze()
            errs.append(sqrt(ot.wasserstein_1d(a_fixed_true, a_fixed_gen, p=2)))
        return errs

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

    def do_tests(self, comp_joint_err=False, comp_grad_norm=False, comp_loss_d=False):
        unfixed_data = self.samples_torch[:self.unfixed_test_bsz]
        actual_bsz = unfixed_data.shape[0]

        noise = torch.randn((actual_bsz, self.noise_size), dtype=torch.float, device=self.device)
        w = unfixed_data[:, :self.w_dim]
        z = torch.cat((noise, w), dim=1)
        self.print_time("Z FOR REPORT")
        fake_data = self.netG(z)
        fake_data = fake_data.detach()
        self.print_time("RUNNING netG FOR REPORT")

        if comp_grad_norm:
            pruned_fake_data = fake_data[:actual_bsz]
            gradient_penalty, gradient_norm = self._gradient_penalty(unfixed_data, pruned_fake_data, gp_weight=0)
            self.test_results['gradient norm'] = gradient_norm

        if comp_loss_d:
            prob_real = self.netD(unfixed_data)
            prob_fake = self.netD(fake_data)
            self.print_time("netD FOR REPORT")
            loss_d_fake = prob_fake.mean(0).view(1)
            loss_d_real = prob_real.mean(0).view(1)
            loss_d = loss_d_fake - self.s_dim * loss_d_real
            self.test_results['loss d'] = loss_d.item()
        self.print_time("UNFIXED PART OF REPORT")

        chen_errors = chen_error_3step(fake_data, self.w_dim)
        self.print_time("CHEN ERRORS")
        joint_wass_error = -1.0
        st_dev_err = -1.0
        # Test Wasserstein error for fixed W
        if self.w_dim > 2:
            noise = torch.randn((self.test_bsz, self.noise_size), dtype=torch.float, device=self.device)
            g_in = torch.cat((noise, self.W_fixed), 1)
            a_fixed_gen = self.netG(g_in)[:, self.w_dim:self.w_dim + self.a_dim].detach().cpu().numpy()
            errors = [sqrt(ot.wasserstein_1d(self.A_fixed_true[:, i], a_fixed_gen[:, i], p=2)) for i in
                      range(self.a_dim)]
            self.print_time("FIXED ERRORS")
            st_dev_err = self.avg_st_dev_error(a_fixed_gen)
            self.print_time("ST DEV ERRORS")

            if comp_joint_err:
                joint_wass_error = joint_wass_dist(self.A_fixed_true[:self.joint_wass_dist_bsz],
                                                   a_fixed_gen[:self.joint_wass_dist_bsz])
            self.print_time("JOINT WASS ERRORS")

        else:
            errors = self.all_2dim_errors()

        self.test_results['errors'] = errors
        self.test_results['chen errors'] = chen_errors
        self.test_results['joint wass error'] = joint_wass_error
        self.test_results['st dev error'] = st_dev_err
        score = self.model_score()

        if score < self.test_results['best score']:
            self.test_results['best score'] = score

    def model_score(self, a: float = 1.0, b: float = 0.2, c: float = 1.0):
        res = 0.0
        res += a * sum(self.test_results['errors'])
        res += b * sum(self.test_results['chen errors'])
        res += c * self.a_dim * self.test_results['joint wass error']
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
        tr_conf['compute joint error'] = True

        scores = []
        for i in range(trials):
            self.netG = Generator(cf)
            self.netD = Discriminator(cf)
            self.reset_test_results()
            self.classic_train(tr_conf)
            scores.append(self.test_results['best score'])

        variance = np.var(scores)
        mean = np.mean(scores)
        result_dict = {
            'status': STATUS_OK,
            'loss': mean,
            'loss_variance': variance
        }

        return self.model_score()

    def make_report(self, epoch: int = None, iters: int = None, chen_iters: int = None, add_line_break=True):
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

        grad_norm = self.test_results['gradient norm']
        report += f"discr grad norm: {grad_norm:.5f}, "
        report += f"discr loss: {self.test_results['loss d']:.5f}"
        joint_wass_error = self.test_results['joint wass error']
        if joint_wass_error >= 0:
            report += f", joint err: {joint_wass_error:.5f}"
        st_dev_error = self.test_results['st dev error']
        if st_dev_error >= 0:
            report += f", st dev err: {st_dev_error:.5f}"
        pretty_errors = make_pretty(self.test_results['errors'])
        pretty_chen_errors = make_pretty(self.test_results['chen errors'])
        if len(pretty_chen_errors) == 1:
            pretty_chen_errors = pretty_chen_errors[0]
        report += f"{line_break}errs: {pretty_errors}, chen errs: {pretty_chen_errors}"

        return report

    def draw_error_graphs(self, wass_errors_through_training, chen_errors_through_training,
                          joint_errors_through_training=None, descriptor: str = ''):
        if not (joint_errors_through_training is None):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 15))
            ax3.set_title("Joint 2-Wasserstein errors")
            ax3.plot(joint_errors_through_training)
            ax3.set_ylim([-0.01, 0.8])
            ax3.set_xlabel("iterations")
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))
        labels = list_pairs(self.w_dim)
        ax1.set_title("Individual 2-Wasserstein errors")
        ax1.plot(wass_errors_through_training, label=labels)
        ax1.set_ylim([-0.005, 0.2])
        ax1.set_xlabel("iterations")
        ax1.legend(prop={'size': 15})
        ax2.set_title("Chen errors")
        ax2.plot(chen_errors_through_training, label=labels)
        ax2.set_ylim([-0.01, 0.8])
        ax2.set_xlabel("iterations")
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
            torch.load(folder_name + f'generator_num{self.serial_number}_{descriptor}.pt', map_location=self.device))
        self.netD.load_state_dict(torch.load(folder_name + f'discriminator_num{self.serial_number}_{descriptor}.pt',
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

    def classic_train(self, tr_conf_in: dict = None, save_models = True):
        if tr_conf_in is None:
            importlib.reload(configs)
            tr_conf = configs.training_config
        else:
            tr_conf = tr_conf_in

        # Number of training epochs using classical training
        self.num_epochs = tr_conf['num epochs']

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

        descriptor = tr_conf['descriptor']

        filename = f"samples/samples_{self.w_dim}-dim.csv"
        whole_training_data = self.samples_torch.split(bsz)
        print(len(whole_training_data))

        # Early stopping setup
        self.test_results['min sum'] = float('inf')
        self.test_results['min chen sum'] = float('inf')

        # For graphing
        wass_errors_through_training = []
        chen_errors_through_training = []
        joint_errors_through_training = []

        iters = 0
        for epoch in range(self.num_epochs):

            for i, data in enumerate(whole_training_data):
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
                loss_d = loss_d_fake - self.s_dim * loss_d_real
                self.test_results['loss d'] = loss_d.item()

                if self.Lipschitz_mode == 'gp':
                    pruned_fake_data = fake_data_detached[:actual_bsz]
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
                if iters % num_discr_iters == 0:
                    self.netG.zero_grad()
                    loss_g = self.netD(fake_data)
                    self.print_time(description="netD 2")
                    loss_g = - loss_g.mean(0).view(1)
                    loss_g.backward()
                    self.print_time(description="netG BACKPROP")
                    opt_g.step()
                    self.print_time(description="OPT G")

                if iters % 100 == 0:
                    self.print_time(description="BEFORE TESTS")
                    self.do_tests(comp_joint_err=compute_joint_error)
                    self.print_time(description="AFTER TESTS")
                    report = self.make_report(epoch=epoch, iters=iters)
                    print(report)
                    self.print_time(description="AFTER REPORT")
                    errors = self.test_results['errors']
                    wass_errors_through_training.append(errors)
                    chen_errors = self.test_results['chen errors']
                    chen_errors_through_training.append(chen_errors)
                    if compute_joint_error:
                        joint_errors_through_training.append(self.test_results['joint wass error'])
                    report_for_saving_dicts = self.make_report(add_line_break=False)
                    # Early stopping checkpoint
                    error_sum = sum(errors)
                    if error_sum <= self.test_results['min sum']:
                        self.test_results['min sum'] = error_sum
                        if save_models:
                            self.save_current_dicts(report=report_for_saving_dicts, descriptor=f"{descriptor}_min_sum")
                        print("Min fixed sum")

                    chen_err_sum = sum(chen_errors)
                    if chen_err_sum < self.test_results['min chen sum']:
                        self.test_results['min chen sum'] = chen_err_sum
                        if save_models:
                            self.save_current_dicts(report=report_for_saving_dicts, descriptor=f"{descriptor}min_chen")
                        print("Min Chen sum")

                    self.print_time(description="SAVING DICTS")
                    self.do_timeing = False
                iters += 1

        self.draw_error_graphs(wass_errors_through_training, chen_errors_through_training,
                               joint_errors_through_training=joint_errors_through_training, descriptor=descriptor)

    # def chen_train(self, tr_conf: dict):
    #     print("blub")
    #     # Number of iterations of Chen training
    #     num_Chen_iters = tr_conf['num Chen iters']
    #
    #     # 'Adam' of 'RMSProp'
    #     which_optimizer = tr_conf['optimizer']
    #
    #     # Learning rate for optimizers
    #     lrG = tr_conf['lrG']
    #     lrD = tr_conf['lrD']
    #
    #     # Beta hyperparam for Adam optimizers
    #     beta1 = tr_conf['beta1']
    #     beta2 = tr_conf['beta2']
    #
    #     if which_optimizer == 'Adam':
    #         optG = torch.optim.Adam(self.netG.parameters(), lr=lrG, betas=(beta1, beta2))
    #         optD = torch.optim.Adam(self.netD.parameters(), lr=lrD, betas=(beta1, beta2))
    #     elif which_optimizer == 'RMSProp':
    #         optG = torch.optim.RMSprop(self.netG.parameters(), lr=lrG)
    #         optD = torch.optim.RMSprop(self.netD.parameters(), lr=lrD)
    #
    #     # To keep the criterion Lipschitz
    #     weight_clipping_limit = tr_conf['weight clipping limit']
    #
    #     # for gradient penalty
    #     gp_weight = tr_conf['gp weight']
    #
    #     bsz = tr_conf['bsz']

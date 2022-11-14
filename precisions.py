from TheGAN import LevyGAN
from aux_functions import *
import timeit

config = {
    'w dim': 3,
    'noise size': 16,
    'which generator': 7,
    'which discriminator': 8,
    'generator symmetry mode': 'Hsym',
    'leakyReLU slope': 0.2,
    'test bsz': 16384,
    'unfixed test bsz': 16384,
    'joint wass dist bsz': 8192,
    'num tests for 2d': 4,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    'should draw graphs': True,
    'do timeing': False
}

w_dim = 3
a_dim = int((w_dim * (w_dim - 1)) // 2)
bsz = 262144
data = np.genfromtxt(f'samples/high_prec_samples_altfixed_3-dim.csv', dtype=float, delimiter=',', max_rows=bsz)
a_true = data[:, w_dim:(w_dim + a_dim)]
W = data[:, :w_dim]
W_torch = torch.tensor(W, dtype=torch.float, device=torch.device('cuda'))
print(data.shape)

def check_precision(_samples, _elapsed, name):
    a_samples = _samples[:, w_dim:(w_dim + a_dim)]
    err = [sqrt(ot.wasserstein_1d(a_true[:, i], a_samples[:, i], p=2)) for i in range(a_dim)]
    joint_err = joint_wass_dist(a_true[:20000], a_samples[:20000])
    print(f"{name} time:{elapsed}, individual errs:{make_pretty(err)}, joint err: {joint_err}")

def check_precision_a(_a_samples, _elapsed, name):
    err = [sqrt(ot.wasserstein_1d(a_true[:, i], _a_samples[:, i], p=2)) for i in range(a_dim)]
    joint_err = joint_wass_dist(a_true[:20000], _a_samples[:20000])
    print(f"{name} time:{elapsed}, individual errs:{make_pretty(err)}, joint err: {joint_err}")

samples = np.genfromtxt(f'samples/mid_prec_fixed_samples_{w_dim}-dim.csv', dtype=float, delimiter=',', max_rows=bsz) # 0.68s
elapsed = 0.111565
check_precision(samples, elapsed, "julia p3")

samples = np.genfromtxt(f'samples/p4_samples_3-dim.csv', dtype=float, delimiter=',', max_rows=bsz) # 0.68s
elapsed = 0.162031
check_precision(samples, elapsed, "julia p4")


start_time = timeit.default_timer()
samples = gen_2mom_approx(w_dim, bsz, _W = W)
elapsed = timeit.default_timer() - start_time
check_precision(samples, elapsed, "2mom")

start_time = timeit.default_timer()
samples = gen_4mom_approx(w_dim, bsz, _W=W)
elapsed = timeit.default_timer() - start_time
check_precision(samples, elapsed, "4mom")

T, M, S = generate_tms(w_dim, torch.device('cpu'))
start_time = timeit.default_timer()
h = sqrt(1 / 12) * torch.randn((bsz, w_dim), dtype=torch.float, device=torch.device('cuda'))
wth = aux_compute_wth(W_torch, h, S, T, w_dim).detach()
b = sqrt(1 / 12) * torch.randn((bsz, w_dim), dtype=torch.float, device=torch.device('cuda'))
a_wthmb = aux_compute_wthmb(wth, b, M, w_dim)
elapsed = timeit.default_timer() - start_time
a_wthmb_np = a_wthmb.cpu().numpy()
check_precision_a(a_wthmb_np, elapsed, "F&L")

levG = LevyGAN(config_in=config, do_load_samples=False)
levG.do_timeing = True
levG.load_dicts(serial_num_to_load=3, descriptor="CHEN_max_scr")
a_gan_np = (levG.eval(W_torch).detach()[:, w_dim:(w_dim + a_dim)]).numpy()
check_precision_a(a_gan_np, elapsed, "GAN")
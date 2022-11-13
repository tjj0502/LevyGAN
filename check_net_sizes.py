import pickle
import time
from TheGAN import LevyGAN
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

config = {
    'w dim': 3,
    'noise size': 16,
    'which generator': 4,
    'which discriminator': 4,
    'generator symmetry mode': 'Hsym',
    'leakyReLU slope': 0.2,
    'test bsz': 65536,
    'unfixed test bsz': 65536,
    'joint wass dist bsz': 4096,
    'num tests for 2d': 3,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    'should draw graphs': True,
    'do timeing': False
}

training_config = {
    'num epochs': 3,
    'max iters': None,
    'num Chen iters': 4000,
    'optimizer': 'Adam',
    'lrG': 0.000001,
    'lrD': 0.000002,
    'num discr iters': 3,
    'beta1': 0.2,
    'beta2': 0.98,
    'Lipschitz mode': 'gp',
    'weight clipping limit': 0.01,
    'gp weight': 10.0,
    'bsz': 1024,
    'compute joint error': True,
    'print reports': False,
    'descriptor': '',
    'custom Chen lrs': {
            0: (0.00001, 0.0001),
            610: (0.00001, 0.00003),
            2000: (0.000004, 0.00001),
            3000: (0.000001, 0.000002),
        },
    'custom lrs': {
            0: (0.000001, 0.000005),
            1: (0.0000005, 0.000001),
            2: (0.00000005, 0.0000001)
        },
}

short_tr_conf = training_config.copy()
short_tr_conf['num Chen iters'] = 30

with open("net_test_report.txt", "a+") as f:
    for gen in range(7, 8):
        for discr in range(8, 9):
            for trials in range(1):
                config['which generator'] = gen
                config['which discriminator'] = discr
                # training_config['descriptor'] = "CLASSIC"
                # levG = LevyGAN(config)
                # levG.classic_train(training_config)
                # report = levG.test_results['best score report']
                # line = f"CLASSIC gen: {gen}, discr: {discr}, {report}\n"
                # f.write(line)

                training_config['descriptor'] = "CHEN"
                levG = LevyGAN(config)
                levG.should_draw_graphs = False
                levG.chen_train(short_tr_conf, save_models=False)
                levG.should_draw_graphs = True
                levG.chen_train(training_config)
                report = levG.test_results['best score report']
                levG.joint_wass_dist_bsz = 20000
                levG.load_dicts(descriptor="CHEN_max_scr")
                levG.do_tests(comp_joint_err=True)
                joint_errs = levG.test_results['joint wass errors']
                line = f"CHEN gen: {gen}, discr: {discr}, num{levG.serial_number}, best joint: {joint_errs} {report}\n"

                f.write(line)

                # a bit more classical training
                training_config['descriptor'] = "CLAS"
                levG.joint_wass_dist_bsz = 4096
                levG.classic_train(training_config)
                report = levG.test_results['best score report']
                levG.joint_wass_dist_bsz = 20000
                levG.load_dicts(descriptor="CHEN_max_scr")
                levG.do_tests(comp_joint_err=True)
                joint_errs = levG.test_results['joint wass errors']
                line = f"CLS+ gen: {gen}, discr: {discr}, num{levG.serial_number}, best joint: {joint_errs} {report}\n"

                f.write(line)



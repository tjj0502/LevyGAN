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
    'num tests for 2d': 4,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    'should draw graphs': True,
    'do timeing': False
}

training_config = {
    'num epochs': 7,
    'max iters': None,
    'num Chen iters': 3000,
    'optimizer': 'Adam',
    'lrG': 0.000001,
    'lrD': 0.000002,
    'num discr iters': 3,
    'beta1': 0.2,
    'beta2': 0.98,
    'Lipschitz mode': 'gp',
    'weight clipping limit': 0.01,
    'gp weight': 5.0,
    'bsz': 1024,
    'compute joint error': True,
    'print reports': False,
    'descriptor': '',
    'custom Chen lrs': {
            0: (0.000003, 0.00003),
            410: (0.000002, 0.00001),
            2410: (0.0000001, 0.0000003),
        },
    'custom lrs': {
            0: (0.000002, 0.00002),
            1: (0.000001, 0.000005),
            3: (0.0000001, 0.0000003),
            5: (0.000005, 0.00001)
        },
}

short_tr_conf = training_config.copy()
short_tr_conf['num Chen iters'] = 20


for gen in range(9, 10):
    for discr in range(9, 10):
        for trials in range(5):
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
            levG.do_tests(comp_joint_err=True, save_best_results=False)
            joint_errs = levG.test_results['joint wass errors']
            line = f"CHEN gen: {gen}, discr: {discr}, num{levG.serial_number}, best joint: {joint_errs} {report}\n"
            with open("net_test_report.txt", "a+") as f:
                f.write(line)

            # a bit more classical training
            training_config['descriptor'] = "CLAS"
            levG.joint_wass_dist_bsz = 4096
            levG.reset_test_results()
            levG.classic_train(training_config)
            report = levG.test_results['best score report']
            levG.joint_wass_dist_bsz = 20000
            levG.load_dicts(descriptor="CLAS_max_scr")
            levG.do_tests(comp_joint_err=True, save_best_results=False)
            joint_errs = levG.test_results['joint wass errors']
            line = f"CLS+ gen: {gen}, discr: {discr}, num{levG.serial_number}, best joint: {joint_errs} {report}\n"
            with open("net_test_report.txt", "a+") as f:
                f.write(line)



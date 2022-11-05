import pickle
import time
from TheGAN import LevyGAN
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

config = {
    'w dim': 3,
    'noise size': 62,
    'which generator': 4,
    'which discriminator': 4,
    'generator symmetry mode': 'Hsym',
    'leakyReLU slope': 0.2,
    'test bsz': 16384,
    'unfixed test bsz': 16384,
    'joint wass dist bsz': 4096,
    'num tests for 2d': 4,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    'should draw graphs': True,
    'do timeing': False
}

training_config = {
    'num epochs': 8,
    'max iters': None,
    'num Chen iters': 8000,
    'optimizer': 'Adam',
    'lrG': 0.000008,
    'lrD': 0.0001,
    'num discr iters': 3,
    'beta1': 0.2,
    'beta2': 0.97,
    'Lipschitz mode': 'gp',
    'weight clipping limit': 0.01,
    'gp weight': 20.0,
    'bsz': 1024,
    'compute joint error': True,
    'print reports': False,
    'descriptor': ''
}

with open("net_test_report.txt", "a+") as f:
    for gen in range(1, 5):
        for discr in range(1, 5):
            config['which generator'] = gen
            config['which discriminator'] = discr
            training_config['descriptor'] = "CLASSIC"
            levG = LevyGAN(config)
            levG.classic_train(training_config)
            report = levG.test_results['best score report']
            line = f"CLASSIC gen: {gen}, discr: {discr}, {report}\n"
            f.write(line)

            training_config['descriptor'] = "CHEN"
            levG = LevyGAN(config)
            levG.chen_train(training_config)
            report = levG.test_results['best score report']
            line = f"CHEN gen: {gen}, discr: {discr}, {report}\n"
            f.write(line)



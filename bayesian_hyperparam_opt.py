import pickle
import time
from TheGAN import LevyGAN
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

config = {
    'w dim': 4,
    'noise size': 62,
    'which generator': 4,
    'which discriminator': 4,
    'generator symmetry mode': 'Hsym',
    'leakyReLU slope': 0.2,
    'test bsz': 16384,
    'unfixed test bsz': 16384,
    'joint wass dist bsz': 8192,
    'num tests for 2d': 8,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    'should draw graphs': False,
    'do timeing': False
}

training_config = {
    'num epochs': 10,
    'max iters': None,
    'optimizer': 'Adam',
    'lrG': 0.00005,
    'lrD': 0.0001,
    'num discr iters': 3,
    'beta1': 0,
    'beta2': 0.99,
    'Lipschitz mode': 'gp',
    'weight clipping limit': 0.01,
    'gp weight': 20.0,
    'bsz': 1024,
    'compute joint error': True,
    'print reports': False,
    'descriptor': ''
}

levG = LevyGAN(config)


def objective(x):
    opt = x[0]
    lrG = x[1]
    lrD = x[2]
    numDiters = x[3]
    gpw = x[4]
    leaky_slp = x[5]
    if opt[0] == 'Adam':
        b1 = opt[1]
        b2 = opt[2]
        # print(f'\n\n!!!!!!!!!!!!!!\nAdam, {b1}, {b2}, {lrG}, {lrD}, {numDiters}, {gpw}, {leaky_slp}\n!!!!!!!!\n\n')
        return levG.compute_objective('Adam', lrG, lrD, numDiters, b1, b2, gpw, leaky_slp, training_config, trials=5)
    else:
        # print(f'\n\n!!!!!!!!!!!!\nRMSProp, {lrG}, {lrD}, {numDiters}, {gpw}, {leaky_slp}\n!!!!!!!!\n\n')
        return levG.compute_objective('RMSProp', lrG, lrD, numDiters, 0, 0, gpw, leaky_slp, training_config, trials=5)


trials = Trials()
space = [
    hp.choice('opt', [('Adam', hp.uniform('b1', 0.0, 1.0), 1 - hp.loguniform('b2', -9, -3)), ('RMSProp')]),
    hp.loguniform('lrG', -14, -9),
    hp.loguniform('lrD', -14, -9),
    1 + hp.quniform('numDiters', 2, 10, 2),
    hp. uniform('gpw', 3, 50,),
    hp.loguniform('leaky_slp', -4, -1)
]

best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=300,
            trials=trials)

print(best)

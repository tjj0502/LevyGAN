import pickle
import time
from TheGAN import LevyGAN
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

config = {
    'w dim': 3,
    'noise size': 62,
    'which generator': 3,
    'which discriminator': 3,
    'generator symmetry mode': 'Hsym',
    'leakyReLU slope': 0.2,
    'test bsz': 4096,
    'unfixed test bsz': 4096,
    'joint wass dist bsz': 4096,
    'num tests for 2d': 8,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    'do timeing': True
}

training_config = {
    'num epochs': 1,
    'max iters': 120,
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
    'descriptor': ''
}

levG = LevyGAN(config)


def objective(opt, b1, b2, lrG, lrD, numDiters, gpw, leaky_slp):
    if opt == 'Adam':
        print(f'\n\n!!!!!!!!!!!!!!!!!!\nAdam, {b1}, {b2}, {lrG}, {lrD}, {numDiters}, {gpw}, {leaky_slp}\n!!!!!!!!\n\n')
        return levG.compute_objective('Adam', lrG, lrD, numDiters, b1, b2, gpw, leaky_slp, tr_conf_in=training_config)
    else:
        print(f'\n\n!!!!!!!!!!!!!!!\nRMSProp, {b1}, {b2}, {lrG}, {lrD}, {numDiters}, {gpw}, {leaky_slp}\n!!!!!!!!\n\n')
        return levG.compute_objective('RMSProp', lrG, lrD, numDiters, 0, 0, gpw, leaky_slp, tr_conf_in=training_config)


trials = Trials()
space = [
    hp.choice('opt', [('Adam', hp.uniform('b1', 0.0, 1.0), 1 - hp.loguniform('b2', -9, -3)), ('RMSProp')]),
    hp.loguniform('lrG', -9, -14),
    hp.loguniform('lrD', -9, -14),
    1 + hp.quniform('numDiters', 2, 10, 2),
    hp. uniform('gpw', 3, 50,),
    hp.loguniform('leaky_slp', -1, -4)
]

best = fmin(objective,
            space=hp.uniform('x', -10, 10),
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)

print(best)

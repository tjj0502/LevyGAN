import torch

long_config = {
    'device': torch.device('cpu'),
    'ngpu': 0,
    'w dim': 4,
    'a dim': 6,
    'noise size': 62,
    'which generator': 2,
    'which discriminator': 2,
    'generator symmetry mode': 'Hsym',
    'generator last width': 6,
    's dim': 16,
    'leakyReLU slope': 0.2,
    'num epochs': 30,
    'num Chen iters': 5000,
    'optimizer': 'Adam',
    'lrG': 0.00005,
    'lrD': 0.0001,
    'beta1': 0,
    'beta2': 0.99,
    'Lipschitz mode': 'gp',
    'weight clipping limit': 0.01,
    'gp weight': 20.0,
    'bsz': 1024,
    'test bsz': 16384,
    'unfixed test bsz': 16384,
    'joint wass dist bsz': 8192,
    'num tests for 2d': 8,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    'do timeing': True
}

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


training_config = {
    'num epochs': 10,
    'max iters': None,
    'num Chen iters': 10000,
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
    'descriptor': '',
    'custom lrs': {
        0: (0.00001, 0.0001),
        2: (0.00001, 0.00004),
        4: (0.000004, 0.00002),
        8: (0.000001, 0.000005)
    },
    'custom Chen lrs': {
        0: (0.00001, 0.0001),
        2000: (0.00001, 0.00004),
        4000: (0.000004, 0.00002),
        8000: (0.000001, 0.000005)
    },
}


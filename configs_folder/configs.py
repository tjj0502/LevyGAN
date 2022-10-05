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
    'w dim': 4,
    'noise size': 62,
    'which generator': 2,
    'which discriminator': 2,
    'generator symmetry mode': 'Hsym',
    'leakyReLU slope': 0.2,
    'test bsz': 16384,
    'unfixed test bsz': 16384,
    'joint wass dist bsz': 8192,
    'num tests for 2d': 8,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    'do timeing': True
}


training_config = {
    'num epochs': 2,
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
    'compute joint error': False,
    'descriptor': ''
}


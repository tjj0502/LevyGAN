import torch.nn as nn


def generator_main(cf: dict):
    which_gen = cf['which generator']
    w_dim = cf['w dim']
    device = cf['device']
    a_dim = int((w_dim * (w_dim - 1)) // 2)
    generator_last_width = cf['generator last width']
    noise_size = cf['noise size']
    if which_gen == 1:
        layers = nn.Sequential(
            nn.Linear(w_dim + noise_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, generator_last_width)
        ).to(device=device)
    if which_gen == 2:
        layers = nn.Sequential(
            nn.Linear(w_dim + noise_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            # nn.Linear(1024,1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, generator_last_width)
        ).to(device=device)
    if which_gen == 3:
        layers = nn.Sequential(
            nn.Linear(w_dim+noise_size,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128,generator_last_width)
        ).to(device=device)
    if which_gen == 4:
        layers = nn.Sequential(
            nn.Linear(w_dim + noise_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, generator_last_width)
        ).to(device=device)

    return layers


def discriminator_main(cf: dict):
    which_disc = cf['which discriminator']
    device = cf['device']
    w_dim = cf['w dim']
    a_dim = int((w_dim * (w_dim - 1)) // 2)
    generator_last_width = cf['generator last width']
    noise_size = cf['noise size']
    leakyReLU_slope = cf['leakyReLU slope']
    if which_disc == 1:
        layers = nn.Sequential(
            nn.Linear(w_dim + a_dim,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128,1),
            nn.Sigmoid()
        ).to(device=device)
    if which_disc == 2:
        layers = nn.Sequential(
            nn.Linear(w_dim + a_dim,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(leakyReLU_slope),

            # nn.Linear(1024,1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(1024,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128,1),
            nn.Sigmoid()
        ).to(device=device)
    if which_disc == 3:
        layers = nn.Sequential(
            nn.Linear(w_dim + a_dim,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128,1),
            nn.Sigmoid()
        ).to(device=device)
    if which_disc == 4:
        layers = nn.Sequential(
            nn.Linear(w_dim + a_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(2048,2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device=device)

    return layers




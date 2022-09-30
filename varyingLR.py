import configs
from TheGAN import LevyGAN



def check_learning_rates():
    lrGs = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000002]
    lrs = []
    tr_conf = configs.training_config
    for mult in [20, 10, 5, 2, 1]:
        lr_pairs = [(x, mult * x) for x in lrGs]
        lrs += lr_pairs
    for lr_g, lr_d in lrs:
        gan = LevyGAN()
        tr_conf['lrG'] = lr_g
        tr_conf['lrD'] = lr_d
        gan.classic_train(tr_conf)
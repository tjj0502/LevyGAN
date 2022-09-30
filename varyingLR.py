import configs
from TheGAN import LevyGAN


def check_learning_rates():
    lr_gs = [0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000002]
    lrs = []
    tr_conf = configs.training_config
    gan = LevyGAN()
    gan.do_tests()
    report = gan.make_report(add_line_break=False)
    gan.save_current_dicts(report=report, descriptor="INITIAL")
    for mult in [20, 10, 5, 2, 1]:
        lr_pairs = [(x, mult * x) for x in lr_gs]
        lrs += lr_pairs
    for lr_g, lr_d in lrs:
        gan = LevyGAN()
        tr_conf['lrG'] = lr_g
        tr_conf['lrD'] = lr_d
        tr_conf['descriptor'] = f"lrG_{lr_g}_lrD_{lr_d}"
        gan.load_dicts(descriptor="INITIAL")
        gan.classic_train(tr_conf)


def check_betas():
    tr_conf = configs.training_config
    gan = LevyGAN()
    gan.do_tests()
    report = gan.make_report(add_line_break=False)
    gan.save_current_dicts(report=report, descriptor="INITIAL")
    for beta2 in [0.99, 0.999]:
        for beta1 in [0.0, 0.05, 0.1, 0.5, 0.8, 0.95]:
            tr_conf['beta1'] = beta1
            tr_conf['beta2'] = beta2
            tr_conf['descriptor'] = f"beta1_{beta1}_beta2_{beta2}"
            gan.load_dicts(descriptor="INITIAL")
            gan.classic_train(tr_conf)

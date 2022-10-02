import configs_folder.configs as configs
from TheGAN import LevyGAN
import pandas

config = configs.config
tr_conf = configs.training_config


def check_learning_rates():
    lr_gs = [0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000002]
    lrs = []
    gan = LevyGAN(config)
    gan.do_tests()
    report = gan.make_report(add_line_break=False)
    gan.save_current_dicts(report=report, descriptor="INITIAL")
    for mult in [20, 10, 5, 2, 1, 0.5]:
        lr_pairs = [(x, mult * x) for x in lr_gs]
        lrs += lr_pairs
    for lr_g, lr_d in lrs:
        tr_conf['lrG'] = lr_g
        tr_conf['lrD'] = lr_d
        tr_conf['descriptor'] = f"lrG_{lr_g: .6f}_lrD_{lr_d: .6f}"
        gan.load_dicts(descriptor="INITIAL")
        gan.classic_train(tr_conf)


def check_betas():
    result_grid = []
    gan = LevyGAN()
    gan.do_tests()
    report = gan.make_report(add_line_break=False)
    gan.save_current_dicts(report=report, descriptor="INITIAL")
    for beta2 in [0.95, 0.98, 0.999]:
        result_row = []
        for beta1 in [0.0, 0.05, 0.1, 0.5, 0.8, 0.95]:
            tr_conf['beta1'] = beta1
            tr_conf['beta2'] = beta2
            tr_conf['descriptor'] = f"beta1_{beta1: .2f}_beta2_{beta2: .3f}"
            gan.load_dicts(descriptor="INITIAL")
            gan.classic_train(tr_conf)
            entry = (gan.test_results['min sum'], gan.test_results['min chen sum'], gan.test_results['joint wass error'])
            result_row.append(entry)
        result_grid.append(result_row)

    nice_results = pandas.DataFrame(result_grid, columns=[0.0, 0.05, 0.1, 0.5, 0.8, 0.95], index=[0.95, 0.98, 0.999])
    filename = f"model_saves/{gan.dict_saves_folder}/betas_result_grid_{gan.dict_saves_folder}.txt"
    with open(filename, 'a+') as file:
        file.write("\n")
        file.write(nice_results.to_string())

import configs_folder.configs as configs
from TheGAN import LevyGAN
import pandas
import torch


def check_learning_rates():
    config = {
        'device': torch.device('cuda'),
        'ngpu': 1,
        'w dim': 4,
        'a dim': 6,
        'noise size': 62,
        'which generator': 4,
        'which discriminator': 4,
        'generator symmetry mode': 'Hsym',
        'generator last width': 6,
        's dim': 16,
        'leakyReLU slope': 0.2,
        'num epochs': 30,
        'num Chen iters': 5000,
        'optimizer': 'Adam',
        'lrG': 0.00005,
        'lrD': 0.0001,
        'beta1': 0.1,
        'beta2': 0.99,
        'Lipschitz mode': 'gp',
        'weight clipping limit': 0.01,
        'gp weight': 20.0,
        'batch size': 1024,
        'test batch size': 16384,
        'unfixed test batch size': 65536,
        'joint wass dist batch size': 8192,
        'num tests for 2d': 8,
        'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
        'do timeing': True
    }

    tr_conf = {
        'num epochs': 30,
        'num Chen iters': 5000,
        'optimizer': 'Adam',
        'lrG': 0.00005,
        'lrD': 0.0001,
        'beta1': 0.1,
        'beta2': 0.99,
        'Lipschitz mode': 'gp',
        'weight clipping limit': 0.01,
        'gp weight': 30.0,
        'batch size': 1024,
        'compute joint error': False,
        'descriptor': ''
    }

    result_grid = []
    lr_gs = [0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002]
    gan = LevyGAN(config)
    gan.do_tests()
    report = gan.make_report(add_line_break=False)
    gan.save_current_dicts(report=report, descriptor="INITIAL")
    mults = [20, 10, 5, 2, 1, 0.5]
    for mult in mults:
        result_row = []
        for lr_g in lr_gs:
            lr_d = mult * lr_g
            tr_conf['lrG'] = lr_g
            tr_conf['lrD'] = lr_d
            tr_conf['descriptor'] = f"lrG_{lr_g:.6f}_lrD_{lr_d:.6f}"
            gan.load_dicts(descriptor="INITIAL")
            gan.classic_train(tr_conf)
            min_sum = gan.test_results['min sum']
            min_chen_sum = gan.test_results['min chen sum']
            gan.load_dicts(descriptor=f'lrG_{lr_g:.6f}_lrD_{lr_d:.6f}_min_sum')
            gan.do_tests(comp_joint_err=True)
            joint_err_min_sum = gan.test_results['joint wass error']
            gan.load_dicts(descriptor=f'lrG_{lr_g:.6f}_lrD_{lr_d:.6f}min_chen')
            gan.do_tests(comp_joint_err=True)
            joint_err_min_chen = gan.test_results['joint wass error']
            entry = f" {min_sum:.5f}, {min_chen_sum:.5f}, {joint_err_min_sum:.5f}, {joint_err_min_chen:.5f} "
            result_row.append(entry)
        result_grid.append(result_row)

    nice_results = pandas.DataFrame(result_grid, columns=lr_gs, index=mults)
    filename = f"model_saves/{gan.dict_saves_folder}/LRs_result_grid_num{gan.serial_number}_{gan.dict_saves_folder}.txt"
    with open(filename, 'a+') as file:
        file.write("\n")
        file.write(nice_results.to_string())


def check_gp_and_leaky():
    print("a")


def check_betas():
    config = {
        'device': torch.device('cuda'),
        'ngpu': 1,
        'w dim': 4,
        'a dim': 6,
        'noise size': 62,
        'which generator': 4,
        'which discriminator': 4,
        'generator symmetry mode': 'Hsym',
        'generator last width': 6,
        's dim': 16,
        'leakyReLU slope': 0.2,
        'num epochs': 30,
        'num Chen iters': 5000,
        'optimizer': 'Adam',
        'lrG': 0.00005,
        'lrD': 0.0001,
        'beta1': 0.1,
        'beta2': 0.99,
        'Lipschitz mode': 'gp',
        'weight clipping limit': 0.01,
        'gp weight': 20.0,
        'batch size': 1024,
        'test batch size': 16384,
        'unfixed test batch size': 65536,
        'joint wass dist batch size': 8192,
        'num tests for 2d': 8,
        'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
        'do timeing': True
    }

    tr_conf = {
        'num epochs': 30,
        'num Chen iters': 5000,
        'optimizer': 'Adam',
        'lrG': 0.00005,
        'lrD': 0.0001,
        'beta1': 0.1,
        'beta2': 0.99,
        'Lipschitz mode': 'gp',
        'weight clipping limit': 0.01,
        'gp weight': 30.0,
        'batch size': 1024,
        'compute joint error': False,
        'descriptor': ''
    }

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
            tr_conf['descriptor'] = f"beta1_{beta1:.2f}_beta2_{beta2:.3f}"
            gan.load_dicts(descriptor="INITIAL")
            gan.classic_train(tr_conf)
            entry = (gan.test_results['min sum'], gan.test_results['min chen sum'])
            result_row.append(entry)
        result_grid.append(result_row)

    nice_results = pandas.DataFrame(result_grid, columns=[0.0, 0.05, 0.1, 0.5, 0.8, 0.95], index=[0.95, 0.98, 0.999])
    filename = f"model_saves/{gan.dict_saves_folder}/LRs_result_grid_num{gan.serial_number}_{gan.dict_saves_folder}.txt"
    with open(filename, 'a+') as file:
        file.write("\n")
        file.write(nice_results.to_string())
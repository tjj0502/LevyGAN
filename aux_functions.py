import torch
import numpy as np
import ot
from math import sqrt

config = {
    'device': torch.device('cpu'),
    'ngpu': 0,
    'w dim': 4,
    'a dim': 6,
    'noise size': 62,
    'which generator': 1,
    'which discriminator': 1,
    'generator symmetry mode': 'Hsym',
    'generator last width': 6,
    's dim': 16,
    'leakyReLU slope': 0.2,
    'num epochs': 20,
    'num Chen iters': 5000,
    'optimizer': 'Adam',
    'lrG': 0.0001,
    'lrD': 0.0005,
    'beta1': 0,
    'beta2': 0.99,
    'weight clipping limit': 0.01,
    'gp weight': 10.0,
    'batch size': 1024,
    'test batch size': 65536,
    'num tests for 2d': 8,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7]
}


def init_config(cf: dict):
    _w_dim = cf['w dim']
    _a_dim = int((_w_dim * (_w_dim - 1)) // 2)
    cf['a dim'] = _a_dim
    if cf['generator symmetry mode'] == 'sym':
        cf['generator last width'] = _a_dim + _w_dim
    else:
        cf['generator last width'] = _a_dim
    if cf['generator symmetry mode'] in ["Hsym", "sym"]:
        s_dim = int(2 ** _w_dim)
    else:
        s_dim = 1
    cf['s dim'] = s_dim
    if torch.cuda.is_available():
        cf['ngpu'] = 1
        cf['device'] = torch.device('cuda')



def chen_combine(w_a_in: torch.Tensor, _w_dim: int):
    _a_dim = int((_w_dim * (_w_dim - 1)) // 2)
    # the batch dimension of the inputs will be quartered
    out_size = w_a_in.shape[0] // 2
    assert 2 * out_size == w_a_in.shape[0]
    assert w_a_in.shape[1] == _w_dim + _a_dim

    # w_0_s is from 0 to t/2 and w_s_t is from t/2 to t
    w_0_s, w_s_t = w_a_in.chunk(2)
    result = torch.clone(w_0_s + w_s_t)
    result[:, :_w_dim] = sqrt(0.5) * result[:, :_w_dim]
    result[:, _w_dim:(_w_dim + _a_dim)] = 0.5 * result[:, _w_dim:(_w_dim + _a_dim)]

    idx = _w_dim
    for k in range(_w_dim - 1):
        for l in range(k + 1, _w_dim):
            correction_term = 0.25 * (w_0_s[:, k] * w_s_t[:, l] - w_0_s[:, l] * w_s_t[:, k])
            result[:, idx] += correction_term
            idx += 1

    return result


def chen_error_3step(w_a_in: torch.Tensor, _w_dim: int):
    _a_dim = int((_w_dim * (_w_dim - 1)) // 2)
    combined_data = chen_combine(w_a_in, _w_dim)
    combined_data = chen_combine(combined_data, _w_dim)
    combined_data = chen_combine(combined_data, _w_dim)
    return [sqrt(ot.wasserstein_1d(combined_data[:, _w_dim + i], w_a_in[:, _w_dim + i], p=2)) for i in range(_a_dim)]


def true_st_devs(_w: np.ndarray):
    _w = np.array(_w)
    _w_dim = _w.shape[0]
    _w_squared = np.square(_w)
    _a_dim = int((_w_dim - 1) * _w_dim / 2)
    st_devs = []
    for k in range(_w_dim):
        for l in range(k + 1, _w_dim):
            st_devs.append(sqrt((1.0 / 12.0) * (1.0 + _w_squared[k] + _w_squared[l])))
    assert len(st_devs) == _a_dim
    return st_devs


def empirical_second_moments(_a_generated: np.ndarray):
    _batch_dim = _a_generated.shape[0]
    const = 1.0 / _batch_dim
    _a_dim = _a_generated.shape[1]
    result = np.zeros((_a_dim, _a_dim))
    for i in range(_a_dim):
        for j in range(i, _a_dim):
            result[i, j] = const * np.dot(_a_generated[:, i], _a_generated[:, j])
            if i != j:
                result[j, i] = result[i, j]

    return result


def empirical_variances(_a_generated):
    return np.diagonal(empirical_second_moments(_a_generated))


def joint_wass_dist(x1: torch.Tensor, x2: torch.Tensor):
    closest_pairing_matrix = ot.dist(x1, x2, metric='sqeuclidean')
    return sqrt(ot.emd2([], [], closest_pairing_matrix, numItermax=1000000))


def a_idx(i: int, j: int, _w_dim: int):
    if i == j:
        return None
    idx = 0
    for k in range(_w_dim):
        for l in range(k + 1, _w_dim):
            if (i == k and j == l) or (j == k and i == l):
                return idx
            else:
                idx += 1


def w_indices(a_i: int, _w_dim: int):
    if a_i >= int(_w_dim * (_w_dim - 1) // 2) or a_i < 0:
        return None
    idx = 0
    for k in range(_w_dim):
        for l in range(k + 1, _w_dim):
            if idx == a_i:
                return k, l
            else:
                idx += 1


def list_pairs(_w_dim: int):
    fixed_w_list = [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7]
    lst = []
    for k in range(_w_dim):
        for l in range(k + 1, _w_dim):
            lst.append((fixed_w_list[k], fixed_w_list[l]))
    return lst


def make_pretty(errs):
    return ["{0:0.4f}".format(i) for i in errs]


def read_serial_number(dict_saves_folder):
    filename = f'model_saves/{dict_saves_folder}/summary_file.txt'
    with open(filename, 'a+') as summary_file:
        summary_file.seek(0)
        lines = summary_file.read().splitlines()
        if not lines:
            serial_num = 1
        else:
            last_line = lines[-1]
            serial_num = int(last_line.split()[0]) + 1
    return serial_num


def generate_signs(_w_dim: int):
    lst = []
    for i in range(2 ** _w_dim):
        binary_exp = list(bin(i)[2:])
        lst.append((_w_dim - len(binary_exp)) * [0] + binary_exp)

    res = 2 * np.array(lst, dtype=float) - np.ones((2 ** _w_dim, _w_dim), dtype=float)
    return res


def generate_tms(_w_dim: int, device):
    _a_dim = int((_w_dim * (_w_dim - 1)) // 2)
    signs = torch.tensor(generate_signs(_w_dim), dtype=torch.float, device=device).view(1, 2**_w_dim, _w_dim).contiguous()
    m_list = []

    for s in range(signs.shape[1]):
        m_row = []
        for i in range(_w_dim):
            for j in range(i + 1, _w_dim):
                m_row.append(signs[0, s, j].item() * signs[0, s, i].item())
        m_list.append(m_row)

    _M = torch.tensor(m_list, dtype=torch.float, device=device).unsqueeze(1).contiguous().detach()
    first_dim = []
    second_dim = []
    third_dim = []
    values = []
    idx = 0
    for i in range(_w_dim):
        for j in range(i + 1, _w_dim):
            first_dim.append(i)
            second_dim.append(j)
            third_dim.append(idx)
            values.append(-1.0)
            first_dim.append(j)
            second_dim.append(i)
            third_dim.append(idx)
            values.append(1.0)
            idx += 1

    indices = torch.tensor([first_dim, second_dim, third_dim], device=device)
    _T = torch.sparse_coo_tensor(indices=indices, values=values, size=(_w_dim, _w_dim, _a_dim),
                                 dtype=torch.float, device=device).to_dense().contiguous().detach()
    return _T, _M, signs


def aux_compute_wth(w_in: torch.Tensor, h_in: torch.Tensor, _s: torch.Tensor, _t: torch.Tensor, _w_dim: int):
    _bsz = w_in.shape[0]
    assert w_in.shape == (_bsz, _w_dim)
    assert h_in.shape == (_bsz, _w_dim)
    _H = torch.mul(_s, h_in.view(_bsz, 1, _w_dim))
    _WT = torch.tensordot(w_in, _t, dims=1)
    _WTH = torch.flatten(torch.matmul(_H, _WT).permute(1, 0, 2), start_dim=0, end_dim=1)
    return _WTH


def aux_compute_wthmb(wth_in: torch.Tensor, b_in: torch.Tensor, _m: torch.Tensor, _w_dim):
    _a_dim = int((_w_dim * (_w_dim - 1)) // 2)
    _bsz = b_in.shape[0]
    assert wth_in.shape == (_bsz * (2 ** _w_dim), _a_dim)
    assert b_in.shape == (_bsz, _a_dim)
    _B = b_in.view(1, _bsz, _a_dim)
    _MB = torch.flatten(torch.mul(_m, _B), start_dim=0, end_dim=1)
    return wth_in + _MB

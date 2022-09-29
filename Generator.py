import torch.nn as nn
import torch
from math import sqrt

from aux_functions import generate_tms, aux_compute_wth, aux_compute_wthmb
from models import generator_main


class Generator(nn.Module):
    def __init__(self, cf: dict):
        super(Generator, self).__init__()
        self.device = cf['device']
        self.noise_size = cf['noise size']
        self.w_dim = cf['w dim']
        self.a_dim = cf['a dim']
        self.s_dim = cf['s dim']
        self.generator_symmetry_mode = cf['generator symmetry mode']
        self.main = generator_main(cf)
        self.T, self.M, self.S = generate_tms(self.w_dim)

    def compute_wth(self, w_in: torch.Tensor, h_in: torch.Tensor):
        _bsz = w_in.shape[0]
        assert w_in.shape == (_bsz, self.w_dim)
        assert h_in.shape == (_bsz, self.w_dim)
        _H = torch.mul(self.S, h_in.view(_bsz, 1, self.w_dim))
        _WT = torch.tensordot(w_in, self.T, dims=1)
        _WTH = torch.flatten(torch.matmul(_H, _WT).permute(1, 0, 2), start_dim=0, end_dim=1)
        return _WTH

    def compute_wthmb(self, wth_in: torch.Tensor, b_in: torch.Tensor):
        _bsz = b_in.shape[0]
        assert wth_in.shape == (_bsz * (2 ** self.w_dim), self.a_dim)
        assert b_in.shape == (_bsz, self.a_dim)
        _B = b_in.view(1, _bsz, self.a_dim)
        _MB = torch.flatten(torch.mul(self.M, _B), start_dim=0, end_dim=1)
        return wth_in + _MB

    def forward(self, input):
        noise, w = torch.split(input, [self.noise_size, self.w_dim], dim=1)
        if self.generator_symmetry_mode == "Hsym":
            bsz = input.shape[0]
            h = sqrt(1 / 12) * torch.randn((bsz, self.w_dim), dtype=torch.float, device=self.device)
            wth = self.compute_wth(w, h).detach()
            x = torch.cat((noise, h), dim=1).detach()
            b = self.main(x)
            a = self.compute_wthmb(wth, b)
            w = w.repeat((self.s_dim, 1))
        elif self.generator_symmetry_mode == "sym":
            x = self.main(input)
            h = x[:, :self.w_dim]
            b = x[:, self.w_dim:self.w_dim + self.a_dim]
            wth = self.compute_wth(w, h)
            a = self.compute_wthmb(wth, b)
            w = w.repeat((self.s_dim, 1))
        else:
            a = self.main(input)

        return torch.cat((w, a), dim=1)

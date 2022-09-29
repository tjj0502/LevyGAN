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
        return aux_compute_wth(w_in, h_in, self.S, self.T, self.w_dim)

    def compute_wthmb(self, wth_in: torch.Tensor, b_in: torch.Tensor):
        return aux_compute_wthmb(wth_in, b_in, self.M, self.w_dim)

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

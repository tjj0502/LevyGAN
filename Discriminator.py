import torch.nn as nn
from models import discriminator_main


class Discriminator(nn.Module):
    def __init__(self, cf: dict):
        super(Discriminator, self).__init__()
        self.main = discriminator_main(cf)

    def forward(self, input):
        return self.main(input)

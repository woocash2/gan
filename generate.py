import os

import torch
from models import Generator
from util import *

latent_size = 64
how_many=10
start_from=11
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
sample_dir = 'generated'


if __name__ == '__main__':

    device = get_default_device()
    generator = Generator(latent_size).to(device)

    checkpoint = torch.load(os.path.join(sample_dir, 'model_{0:0=4d}.pth'.format(start_from)))
    generator.load_state_dict(checkpoint["gen_sd"])

    for i in range(1,how_many+1):
        fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
        gen_save_samples(generator, sample_dir, i, fixed_latent, stats, show=True)
    print('done')

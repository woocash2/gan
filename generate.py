import os

import torch
from models import Generator, GeneratorSkip
from models64 import GeneratorSkip64
from util import *

latent_size = 64
how_many=20
start_from=501
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
sample_dir = 'Final64skip_res_16_40_0002'


if __name__ == '__main__':

    device = get_default_device(3)
    generator = GeneratorSkip64(latent_size,device).to(device)

    checkpoint = torch.load(os.path.join(sample_dir, 'model_{0:0=4d}.pth'.format(start_from)))
    generator.load_state_dict(checkpoint["gen_sd"])

    for i in range(1,how_many+1):
        fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
        gen_save_samples(generator, sample_dir, i, fixed_latent, stats, show=True,prefix="onDemand_")
    print('done')

import os
import random

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
import time

from tqdm import tqdm

import util
from models import *
from models64 import *
from train import Trainer
from util import *

DATA_DIR='organic'
epochs_per_sample=5
epochs_per_checkpoint=10
latent_size = 64
start_from=0
lr = 0.0002
epochs = 301
sample_dir = 'fresh-64-Gskip-Dstd'
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
image_size = 64
batch_size = 64
noise_std = 0.08
noise_fade = 1/3
small_train_set = True
small_set_size = 5000

on_miracle = True




def fit(epochs, lr,fixed_latent,generator,discriminator, start_idx=0,name="model",std=0.1,fade_noise=(True,1/2)):
        torch.cuda.empty_cache()
        gl_time=time.time()
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []

        # Create optimizers
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        if start_idx!=0: # load model
            checkpoint= torch.load(os.path.join(sample_dir,'{}_{0:0=4d}.pth'.format(name,start_idx)))
            start_idx=checkpoint["epoch"]
            generator.load_state_dict(checkpoint["gen_sd"])
            opt_g.load_state_dict(checkpoint["opt_g_sd"])
            losses_g=checkpoint["loss_g"]
            discriminator.load_state_dict(checkpoint["dis_sd"])
            opt_d.load_state_dict(checkpoint["opt_d_sd"])
            losses_d=checkpoint["loss_d"]
            fixed_latent=checkpoint["fixed_latent"]

        trainer = Trainer(discriminator, generator, batch_size, device, latent_size)

        train_std=std
        for epoch in range(start_idx,epochs):
            tim =time.time()
            if fade_noise[0]:
                train_std=std*(1-min(0.95,fade_noise[1]*epoch/(epochs)))
                # linearly fade to a fraction over first half of the training
            for real_images, _ in train_dl:
                # Train discriminator
                real_images=util.addGaussianNoise(real_images,device,std=train_std)
                loss_d, real_score, fake_score = trainer.train_discriminator(real_images, opt_d)
                # Train generator
                loss_g = trainer.train_generator(opt_g,std=train_std)

            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}, epoch_time:{:.4f}, time:{:.4f}".format(
                epoch + 1, epochs, loss_g, loss_d, real_score, fake_score,time.time()-tim,time.time()-gl_time))

            # Save generated images
            if (epoch!=start_idx and (epoch)%epochs_per_sample==0) or epoch==epochs-1:
                gen_save_samples(generator, sample_dir, epoch + 1, fixed_latent, stats, show=False)
            if not small_train_set:
                if (epoch!=start_idx and (epoch)%epochs_per_checkpoint==0) or epoch==epochs-1:
                    torch.save({
                        "epoch":epoch+1,"gen_sd":generator.state_dict(),"opt_g_sd":opt_g.state_dict(),"loss_g":losses_g,
                                      "dis_sd": discriminator.state_dict(), "opt_d_sd": opt_d.state_dict(),"loss_d": losses_d,
                        "fixed_latent":fixed_latent
                    },os.path.join(sample_dir,'{}_{:0=4d}.pth'.format(name,epoch+1) ))
                    print("saved checkpoint {}_{:0=4d}.pth".format(name,epoch+1))

        return losses_g, losses_d, real_scores, fake_scores


#8k images, 32*32, ls=64, eps=10 , bs=256 => 1782 s, results= okeish
if __name__ == '__main__':

    torch.manual_seed(42)

    train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats),
    ]))

    if small_train_set:
        train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=3, pin_memory=True, sampler=range(0, small_set_size))
    else:
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    

    if on_miracle:
        device = get_default_device(0)
    else:
        device = get_default_device(0)
    train_dl = DeviceDataLoader(train_dl, device)

    #discriminatorModel = Discriminator().to(device)
    #discriminatorModel = DiscriminatorResidual().to(device)
    #discriminatorModel = DiscriminatorSkip(device).to(device)
    discriminatorModel = Discriminator64().to(device)
    #discriminatorModel = DiscriminatorSkip64(device).to(device)

    #generatorModel = Generator(latent_size).to(device)
    #generatorModel = GeneratorResidual(latent_size,device).to(device)
    #generatorModel = GeneratorSkip(latent_size,device).to(device)
    #generatorModel = Generator64(latent_size).to(device)
    generatorModel = GeneratorSkip64(latent_size, device).to(device)

    os.makedirs(sample_dir, exist_ok=True)

    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    gen_save_samples(generatorModel, sample_dir, 0, fixed_latent, stats)
    history = fit(epochs, lr,fixed_latent,generatorModel,discriminatorModel,start_idx=start_from,std=noise_std,fade_noise=(True,noise_fade))
    print('done')


"""
Notes:
    - disable blur
    - add noise to discriminator input
    - disable spectral norm (kills some of variation, and bleaches vibrant colors)
    - initiate from orthogonal
    - make noise linearly fade to fractional (nonzero) value
    - penalize overconfidence
"""
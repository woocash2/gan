import torch
import torch.nn.functional as F

import util


class Trainer():
    
    def __init__(s, discriminator, generator, batch_size, device, latent_size) -> None:
        s.discriminator = discriminator
        s.generator = generator
        s.batch_size = batch_size
        s.device = device
        s.latent_size = latent_size

    def train_discriminator(s, real_images, opt_d):
        # Clear discriminator gradients
        opt_d.zero_grad()

        # Pass real images through discriminator
        real_preds = s.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=s.device) * 0.9 # ones for reals
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Generate fake images
        latent = torch.randn(s.batch_size, s.latent_size, 1, 1, device=s.device)
        fake_images = s.generator(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=s.device)+0.05 # zeros for fakes
        fake_preds = s.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        return loss.item(), real_score, fake_score


    def train_generator(s, opt_g,std=0):
        # Clear generator gradients
        opt_g.zero_grad()

        # Generate fake images
        latent = torch.randn(s.batch_size, s.latent_size, 1, 1, device=s.device)
        fake_images = s.generator(latent)

        # Try to fool the discriminator
        fake_images = util.addGaussianNoise(fake_images,s.device,std=std)
        preds = s.discriminator(fake_images)
        targets = torch.ones(s.batch_size, 1, device=s.device)
        loss = F.binary_cross_entropy(preds, targets)

        # Update generator weights
        loss.backward()
        opt_g.step()

        return loss.item()
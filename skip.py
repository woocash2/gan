import torch
import torch.nn as nn
from models import Layer, LayerT
import torch.nn.functional as F
import util

class G32(nn.Module):
    def __init__(self, latent) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            # in: latent_size  x 1 x 1
            LayerT(latent, 64, 1, 0),
            # out: 64 x 4 x 4

            LayerT(64, 64, 2, 1),
            # out: 64 x 8 x 8

            LayerT(64, 32, 2, 1),
            # out: 32 x 16 x 16
        ])
        self.finisher = nn.ModuleList([
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # out: 3 x 32 x 32
            nn.Tanh()
        ])


class G64(nn.Module):
    def __init__(self, latent) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            # in: latent_size  x 1 x 1
            LayerT(latent, 64, 1, 0),
            # out: 64 x 4 x 4

            LayerT(64, 64, 2, 1),
            # out: 64 x 8 x 8

            LayerT(64, 64, 2, 1),
            # out: 64 x 16 x 16

            LayerT(64, 32, 2, 1),
            # out: 32 x 32 x 32
        ])
        self.finisher = nn.ModuleList([
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # out: 3 x 64 x 64
            nn.Tanh()
        ])


class D32(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            # in: 3 x 32 x 32
            Layer(3, 32),
            # out: 32 x 16 x 16

            Layer(32, 64),
            # out: 64 x 8 x 8

            Layer(64, 64),
            # out: 64 x 4 x 4
        ])
        self.finisher = nn.ModuleList([
            nn.Conv2d(64,1,kernel_size=4,stride=1,padding=0),
            # out: 1 x 1 x 1
            nn.Flatten(),
            nn.Sigmoid(),
        ])


class D64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            # in: 3 x 64 x 64
            Layer(3, 8),
            # out: 16 x 32 x 32

            Layer(8, 16),
            # out: 16 x 16 x 16

            Layer(16, 32),
            # out: 32 x 8 x 8

            Layer(32, 64),
            # out: 64 x 4 x 4
        ])
        self.finisher = nn.ModuleList([
            nn.Conv2d(64,1,kernel_size=4,stride=1,padding=0),
            # out: 1 x 1 x 1
            nn.Flatten(),
            nn.Sigmoid(),
        ])


class GSkip(nn.Module):
    def __init__(self, latent, res) -> None:
        super().__init__()
        self.latent = latent
        self.res = str(res)
        self.skipped = []

        self.g_models = nn.ModuleDict({
            '32': G32(latent),
            '64': G64(latent),
        })

    def forward(self, x):
        self.skipped = []
        for layer in self.g_models[self.res].layers:
            x = layer(x)
            self.skipped.append(x)
        for fin in self.g_models[self.res].finisher:
            x = fin(x)
        self.skipped.reverse()
        return x
    

class DSkip(nn.Module):
    def __init__(self, res) -> None:
        super().__init__()
        self.res = str(res)
        self.skipped = []

        self.d_models = nn.ModuleDict({
            '32': D32(),
            '64': D64(),
        })

    def forward(self, x):
        for i, layer in enumerate(self.d_models[self.res].layers):
            x = layer(x)
            x = (x + self.skipped[i]) / 2.0
        for fin in self.d_models[self.res].finisher:
            x = fin(x)
        return x


class TrainerGDSkip:
    def __init__(s, d, g, b_size, dev, latent) -> None:
        s.gen = g
        s.dis = d
        s.bsize = b_size
        s.dev = dev
        s.latent = latent

    def train_discriminator(s, real_images, opt_d):
        # Generate fake images
        latent = torch.randn(s.bsize, s.latent, 1, 1, device=s.dev)
        fake_images = s.gen(latent)

        # Clear discriminator gradients
        opt_d.zero_grad()

        # Pass real images through discriminator
        s.dis.skipped = s.gen.skipped
        real_preds = s.dis(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=s.dev) * 0.9 # ones for reals
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=s.dev) # zeros for fakes
        fake_preds = s.dis(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        return loss.item(), real_score, fake_score


    def train_generator(s, opt_g, std=0):
        # Clear generator gradients
        opt_g.zero_grad()

        # Generate fake images
        latent = torch.randn(s.bsize, s.latent, 1, 1, device=s.dev)
        fake_images = s.gen(latent)

        # Try to fool the discriminator
        fake_images = util.addGaussianNoise(fake_images,s.dev,std=std)
        s.dis.skipped = s.gen.skipped
        preds = s.dis(fake_images)

        targets = torch.ones(s.bsize, 1, device=s.dev)
        loss = F.binary_cross_entropy(preds, targets)

        # Update generator weights
        loss.backward()
        opt_g.step()

        return loss.item()

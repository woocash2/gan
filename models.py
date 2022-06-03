import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        self.sublayers = nn.ModuleList([
            nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2, inplace=True),
        ])
    
    def forward(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x

class LayerT(nn.Module):
    def __init__(self, in_size, out_size, stride, padding) -> None:
        super().__init__()
        self.sublayers = nn.ModuleList([
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True),
        ])
    
    def forward(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            # in: 3 x 32 x 32
            Layer(3, 32),
            # out: 32 x 16 x 16

            Layer(32, 64),
            # out: 64 x 8 x 8

            Layer(64, 128),
            # out: 128 x 4 x 4
        ])

        self.finisher = nn.ModuleList([
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        for fin in self.finisher:
            x = fin(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_size) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            # in: latent_size x 1 x 1
            LayerT(latent_size, 128, 1, 0),
            # out: 128 x 4 x 4

            LayerT(128, 64, 2, 1),
            # out: 64 x 8 x 8

            LayerT(64, 32, 2, 1),
            # out: 32 x 16 x 16
        ])

        self.finisher = nn.ModuleList([
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 32 x 32
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        for fin in self.finisher:
            x = fin(x)
        return x


class GeneratorSkip(Generator):
    def __init__(self, latent_size, device) -> None:
        super().__init__(latent_size)
        self.device = device

    def forward(self, x):
        img = torch.zeros([x.shape[0], 64, 1, 1]).to(self.device)
        for layer in self.layers:
            x = layer(x)
            img = nn.Upsample([x.shape[2], x.shape[3]]).to(self.device)(img)
            img = nn.Conv2d(img.shape[1], x.shape[1], 1).to(self.device)(img)
            img = (img + x)
        for fin in self.finisher:
            img = fin(img)
        return img
from models import Layer, LayerT, ResidualLayer, ResidualLayerT
import torch.nn as nn
import torch.nn.functional as F
import torch

class Discriminator64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            # in: 3 x 64 x 64
            Layer(3, 16),
            # out: 16 x 32 x 32

            Layer(16, 36),
            # out: 36 x 16 x 16

            Layer(36, 84),
            # out: 84 x 8 x 8

            Layer(84, 84),
            # out: 84 x 4 x 4
        ])

        self.finisher = nn.ModuleList([
            nn.Conv2d(84, 1, kernel_size=4,stride=1,padding=0),
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


class DiscriminatorSkip64(Discriminator64):
    def __init__(self,device) -> None:
        super().__init__()
        self.device=device
        self.convs = [
            # in 3 x 64 x 64
            nn.Conv2d(3, 16, 1,bias=False).to(device),
            # out: 16 x 32 x 32
            nn.Conv2d(16, 36, 1,bias=False).to(device),
            # out: 36 x 16 x 16
            nn.Conv2d(36, 84, 1,bias=False).to(device),
            # out: 84 x 8 x 8
            nn.Conv2d(84, 84, 1,bias=False).to(device),
            # out: 84 x 4 x 4
        ]
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

    def forward(self, x):
        x_init=torch.clone(x).to(self.device)
        for conv,layer in zip(self.convs,self.layers):
            x_init = nn.AvgPool2d(kernel_size=2).to(self.device)(x_init)
            x_init = conv(x_init)
            x = layer(x)
            x=x+x_init
            x= nn.BatchNorm2d(x.shape[1]).to(self.device)(x)
        for fin in self.finisher:
            x = fin(x)
        return x


class Generator64(nn.Module):
    def __init__(self, latent_size) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            # in: latent_size  x 1 x 1
            LayerT(latent_size, 96, 1, 0),
            # out: 64 x 4 x 4

            LayerT(96, 96, 2, 1),
            # out: 64 x 8 x 8

            LayerT(96, 64, 2, 1),
            # out: 64 x 16 x 16

            LayerT(64, 32, 2, 1),
            # out: 32 x 32 x 32
        ])

        self.finisher = nn.ModuleList([
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # out: 3 x 64 x 64
            nn.Tanh()
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        for fin in self.finisher:
            x = fin(x)
        return x


class GeneratorSkip64(Generator64):
    def __init__(self, latent_size, device) -> None:
        super().__init__(latent_size)
        self.device = device
        self.latent_size=latent_size
        self.convs = [
            nn.Conv2d(latent_size, 96, 1,bias=False).to(device),
            nn.Conv2d(96, 96, 1,bias=False).to(device),
            nn.Conv2d(96, 64, 1,bias=False).to(device),
            nn.Conv2d(64, 32, 1,bias=False).to(device),
        ]
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False
        self.latent_size = latent_size

    def forward(self, x):
        img = torch.zeros([x.shape[0], self.latent_size, 1, 1]).to(self.device)
        for layer, conv in zip(self.layers, self.convs):
            x = layer(x)
            img = nn.Upsample([x.shape[2], x.shape[3]]).to(self.device)(img)
            img = conv(img)
            img = (img + x)
        for fin in self.finisher:
            img = fin(img)
        return img


class DiscriminatorResidual64(Discriminator64):
    def __init__(self) -> None:
        super().__init__()
        self.layers=nn.ModuleList([
            # in: 3 x 64 x 64
            ResidualLayer(3, 16),
            # out: 16 x 32 x 32

            ResidualLayer(16, 36),
            # out: 32 x 16 x 16

            ResidualLayer(36, 84),
            # out: 84 x 8 x 8

            ResidualLayer(84, 84),
            # out: 84 x 4 x 4
        ])


class GeneratorResidual64(Generator64):
    def __init__(self, latent_size, device) -> None:
        super().__init__(latent_size)
        self.device = device
        self.layers = nn.ModuleList([
            ResidualLayerT(latent_size, 96, 1, 0,self.device),
            # out: 64 x 4 x 4

            ResidualLayerT(96, 96, 2, 1,self.device),
            # out: 

            ResidualLayerT(96, 64, 2, 1,self.device),
            # out: 64 x 8 x 8

            ResidualLayerT(64, 32, 2, 1,self.device),
            # out: 32 x 16 x 16
        ])

class GeneratorIntermidiate64(Generator64):
    def __init__(self, latent_size, device) -> None:
        super().__init__(latent_size)
        self.device = device

        self.layers = nn.ModuleList([
            # in: latent_size  x 1 x 1
            LayerT(256, 192, 1, 0),
            # out: 64 x 4 x 4

            LayerT(192, 96, 2, 1),

            LayerT(96, 64, 2, 1),
            # out: 64 x 8 x 8

            LayerT(64, 32, 2, 1),
            # out: 32 x 16 x 16
        ])
        self.layers = nn.ModuleList([
            # in: latent_size  x 1 x 1
            LayerT(256, 192, 1, 0),
            # out: 64 x 4 x 4

            LayerT(192, 96, 2, 1),

            LayerT(96, 64, 2, 1),
            # out: 64 x 8 x 8

            LayerT(64, 32, 2, 1),
            # out: 32 x 16 x 16
        ])

    def forward(self, x):
        x = torch.view_as_real(torch.fft.rfftn(x,s=6,norm="ortho"))
        x = nn.Flatten()(x)
        x = x[:, :, None, None]
        for layer in self.layers:
            x = layer(x)
        for fin in self.finisher:
            x = fin(x)
        return x


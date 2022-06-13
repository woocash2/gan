import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        self.sublayers = nn.ModuleList([
            nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU( inplace=True),
        ])

    def forward(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x

class LayerT(nn.Module):
    def __init__(self, in_size, out_size, stride, padding) -> None:
        super().__init__()
        self.sublayers = nn.ModuleList([
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=stride, padding=padding)),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True),
        ])
        nn.init.orthogonal_(self.sublayers[0].weight)

    def forward(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x


class ResidualLayer(nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        self.sublayers = nn.ModuleList([
            nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_size+out_size,out_size,1,bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.2, inplace=True),
        ])

    def forward(self, x):
        for i, layer in enumerate(self.sublayers):
            if i==0:
                xConv=layer(x)
                x=nn.AvgPool2d(kernel_size=2)(x)
                x=torch.cat((x,xConv),1)
            else:
                x=layer(x)
        return x

class ResidualLayerT(nn.Module):
    def __init__(self, in_size, out_size, stride, padding,device) -> None:
        super().__init__()
        self.device=device
        self.sublayers = nn.ModuleList([
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_size,out_size,kernel_size=1,bias=False)),
            nn.utils.spectral_norm(nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=stride, padding=padding)),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True)
        ])
        nn.init.orthogonal_(self.sublayers[0].weight)
        nn.init.orthogonal_(self.sublayers[1].weight)

    def forward(self, x):
        for i, layer in enumerate(self.sublayers):
            if i==1:
                xConv=layer(x)
                x=nn.UpsamplingBilinear2d([xConv.shape[2],xConv.shape[3]]).to(self.device)(x)
                x=x+xConv
            else:
                x=layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            # in: 3 x 32 x 32
            Layer(3, 16),
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
            # in: latent_size  x 1 x 1
            LayerT(latent_size, 64, 1, 0),
            # out: 64 x 4 x 4

            LayerT(64, 64, 2, 1),
            # out: 64 x 8 x 8

            LayerT(64, 32, 2, 1),
            # out: 32 x 16 x 16
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


class GeneratorSkip(Generator):
    def __init__(self, latent_size, device) -> None:
        super().__init__(latent_size)
        self.device = device
        self.latent_size=latent_size
        self.convs = [
            nn.Conv2d(latent_size, 64, 1,bias=False).to(device),
            nn.Conv2d(64, 64, 1,bias=False).to(device),
            nn.Conv2d(64, 32, 1,bias=False).to(device),
        ]
        for layer in self.convs:
            layer.weight=nn.Parameter(torch.ones_like(layer.weight)/layer.weight.shape[0]).to(device)

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

class DiscriminatorSkip(Discriminator):
    def __init__(self,device) -> None:
        super().__init__()
        self.device=device
        self.convs = [
            # in 3 x 32 x 32
            nn.Conv2d(3, 16, 1,bias=False).to(device),
            # out: 16 x 16 x 16
            nn.Conv2d(16, 32, 1,bias=False).to(device),
            # out: 32 x 8 x 8
            nn.Conv2d(32, 64, 1,bias=False).to(device),
            # out: 64 x 4 x 4
        ]
        for layer in self.convs:
            layer.weight=nn.Parameter(torch.ones_like(layer.weight)/layer.weight.shape[0]).to(device)
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


class DiscriminatorResidual(Discriminator):
    def __init__(self) -> None:
        super().__init__()
        self.layers=nn.ModuleList([
            # in: 3 x 32 x 32
            ResidualLayer(3, 16),
            # out: 16 x 16 x 16

            ResidualLayer(16, 32),
            # out: 32 x 8 x 8

            ResidualLayer(32, 64),
            # out: 64 x 4 x 4
        ])
class GeneratorResidual(Generator):
    def __init__(self, latent_size, device) -> None:
        super().__init__(latent_size)
        self.device = device
        self.layers = nn.ModuleList([
            ResidualLayerT(latent_size, 64, 1, 0,self.device),
            # out: 64 x 4 x 4

            ResidualLayerT(64, 64, 2, 1,self.device),
            # out: 64 x 8 x 8

            ResidualLayerT(64, 32, 2, 1,self.device),
            # out: 32 x 16 x 16
        ])

class GeneratorIntermidiate(Generator):
    def __init__(self, latent_size, device) -> None:
        super().__init__(latent_size)
        self.device = device
        self.preproc = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(latent_size, latent_size//2),
            nn.ReLU().to(device),
            nn.Linear(latent_size//2, latent_size//2),
            nn.ReLU().to(device),
            nn.Linear(latent_size//2, latent_size),
            nn.ReLU().to(device),
        ])
    def forward(self, x):
        for layer in self.preproc:
            x = layer(x)
        x = x[:, :, None, None]
        for layer in self.layers:
            x = layer(x)
        for fin in self.finisher:
            x = fin(x)
        return x

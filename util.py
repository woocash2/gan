import os
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt


def denorm(img_tensors, stats):
        return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

def get_default_device(x=0):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{x}')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def gen_save_samples(generator, sample_dir, index, latent_tensors, stats, show=True):
        fake_images = generator(latent_tensors)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        save_image(denorm(fake_images, stats), os.path.join(sample_dir, fake_fname), nrow=8)
        print('Saving', fake_fname, "to", sample_dir)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]);
            ax.set_yticks([])
            ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

def addGaussianNoise(tensor,device,mean=0,std=1):
    return tensor + (torch.randn(tensor.size())*std +mean).to(device)
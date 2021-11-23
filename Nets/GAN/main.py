import os
from datetime import datetime
from pathlib import Path

import pytz
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FILE_DIR = Path(__file__).resolve().parent

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 24
num_epochs = 150

os.makedirs("images", exist_ok=True)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, image_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


disc = Discriminator().to(device)
gen = Generator().to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# trm = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# data = ImageFolder(root=f'{BASE_DIR}/Datasets/ScarletChoir', transform=trm)
# loader = DataLoader(data, batch_size=batch_size)
#
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        logger.info(f"Epoch [{epoch}/{num_epochs}]| "
                    f"Batch {batch_idx}/{len(loader)}| "
                    f"Loss Descriminator: {lossD:.4f}| "
                    f"Loss Generator: {lossG:.4f}")

        if batch_idx == 0:
            with torch.no_grad():
                now = datetime.now(tz=pytz.timezone('Asia/Vladivostok'))
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                # img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                # img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                # save_image(real, f"images/true_{str(uuid4())}.png", normalize=True)
                save_image(fake, f'images/fake_{now.strftime("%H:%M %d-%m-%Y")}.png', normalize=True)
                step += 1

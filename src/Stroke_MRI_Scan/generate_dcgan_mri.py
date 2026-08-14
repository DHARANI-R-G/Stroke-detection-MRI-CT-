import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = r"D:\Stroke-MRI"
CLASS_NAMES = ["Haemorrhagic", "Ischemic", "Normal"]
CLASS_DIRS = {c: os.path.join(DATASET_DIR, c) for c in CLASS_NAMES}
BALANCED_DIR = r"D:\Strokes_MRI_Balanced_data"
BALANCED_CLASS_DIRS = {c: os.path.join(BALANCED_DIR, c) for c in CLASS_NAMES}
for d in BALANCED_CLASS_DIRS.values():
    os.makedirs(d, exist_ok=True)
for c in CLASS_NAMES:
    for f in os.listdir(CLASS_DIRS[c]):
        src = os.path.join(CLASS_DIRS[c], f)
        dst = os.path.join(BALANCED_CLASS_DIRS[c], f)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

transform_gan = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class AllImagesDataset(Dataset):
    def __init__(self, class_dirs, transform=None):
        self.paths = []
        for d in class_dirs:
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    self.paths.append(p)
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

all_dirs_for_gan = [CLASS_DIRS["Haemorrhagic"], CLASS_DIRS["Ischemic"], CLASS_DIRS["Normal"]]
gan_dataset = AllImagesDataset(all_dirs_for_gan, transform=transform_gan)
batch_size = 64
gan_loader = DataLoader(gan_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

nz = 100
ngf = 64
ndf = 64
nc = 3

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 2500
for epoch in range(epochs):
    for real in gan_loader:
        real = real.to(device)
        optimizerD.zero_grad()
        output_real = netD(real)
        label_real = torch.ones(output_real.size(0), device=device)
        errD_real = criterion(output_real, label_real)
        noise = torch.randn(output_real.size(0), nz, 1, 1, device=device)
        fake = netG(noise)
        output_fake = netD(fake.detach())
        label_fake = torch.zeros(output_fake.size(0), device=device)
        errD_fake = criterion(output_fake, label_fake)
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        optimizerG.zero_grad()
        output_fake_forG = netD(fake)
        label_gen = torch.ones(output_fake_forG.size(0), device=device)
        errG = criterion(output_fake_forG, label_gen)
        errG.backward()
        optimizerG.step()

counts = {c: len(os.listdir(CLASS_DIRS[c])) for c in CLASS_NAMES}
target = max(counts.values())
needs = {c: max(0, target - len(os.listdir(BALANCED_CLASS_DIRS[c]))) for c in CLASS_NAMES}

netG.eval()
with torch.no_grad():
    total_needed = sum(needs.values())
    if total_needed > 0:
        noise = torch.randn(total_needed, nz, 1, 1, device=device)
        fake = netG(noise).cpu()
        fake = (fake * 0.5 + 0.5)
        idx = 0
        for c in CLASS_NAMES:
            n = needs[c]
            for i in range(n):
                img = fake[idx]
                img = transforms.ToPILImage()(img)
                img = img.resize((224, 224), Image.BILINEAR)
                base = len(os.listdir(BALANCED_CLASS_DIRS[c]))
                img.save(os.path.join(BALANCED_CLASS_DIRS[c], f"synthetic_{base + i}.png"))
                idx += 1

final_counts = {c: len(os.listdir(BALANCED_CLASS_DIRS[c])) for c in CLASS_NAMES}
print(f"Balanced Dataset Size -> Haemorrhagic: {final_counts['Haemorrhagic']}, Ischemic: {final_counts['Ischemic']}, Normal: {final_counts['Normal']}")

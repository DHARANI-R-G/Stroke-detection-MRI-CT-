import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = r"D:\Strokes_CT"
NORMAL_DIR = os.path.join(DATASET_DIR, "Normal")
STROKE_DIR = os.path.join(DATASET_DIR, "Stroke")


BALANCED_DIR = r"D:\Strokes_CT_Balanced_data_"
BALANCED_NORMAL = os.path.join(BALANCED_DIR, "Normal")
BALANCED_STROKE = os.path.join(BALANCED_DIR, "Stroke")
os.makedirs(BALANCED_NORMAL, exist_ok=True)
os.makedirs(BALANCED_STROKE, exist_ok=True)


for img_file in os.listdir(NORMAL_DIR):
    shutil.copy(os.path.join(NORMAL_DIR,img_file), BALANCED_NORMAL)
for img_file in os.listdir(STROKE_DIR):
    shutil.copy(os.path.join(STROKE_DIR,img_file), BALANCED_STROKE)

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


full_dataset = datasets.ImageFolder(DATASET_DIR)
stroke_class_idx = full_dataset.class_to_idx["Stroke"]

class StrokeDataset(Dataset):
    def __init__(self, dataset, class_idx, transform=None):
        self.transform = transform
        self.images = [img for img, label in dataset if label == class_idx]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img

stroke_dataset = StrokeDataset(full_dataset, stroke_class_idx, transform=transform)
stroke_loader = DataLoader(stroke_dataset, batch_size=64, shuffle=True, drop_last=True)

nz = 100
ngf = 64
ndf = 64
nc = 3

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
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
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1,1).squeeze(1)

netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5,0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5,0.999))


epochs = 8
for epoch in range(epochs):
    for real in stroke_loader:
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

    
    print(f"Epoch [{epoch+1}/{epochs}] | Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f}")


num_normal = len(os.listdir(BALANCED_NORMAL))
num_stroke = len(os.listdir(BALANCED_STROKE))
diff = num_normal - num_stroke

netG.eval()
with torch.no_grad():
    noise = torch.randn(diff, nz, 1, 1, device=device)
    fake = netG(noise).cpu()
    fake = (fake*0.5+0.5)
    for idx,img in enumerate(fake):
        img = transforms.ToPILImage()(img)
        img = img.resize((224,224), Image.BILINEAR)
        img.save(os.path.join(BALANCED_STROKE, f"synthetic_{idx}.png"))


final_normal = len(os.listdir(BALANCED_NORMAL))
final_stroke = len(os.listdir(BALANCED_STROKE))
print(f"Balanced Dataset Size -> Normal: {final_normal}, Stroke: {final_stroke}")

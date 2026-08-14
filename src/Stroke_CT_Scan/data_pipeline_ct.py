import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = r"D:\Strokes_CT"

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform_train)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size,val_size,test_size])

val_dataset.dataset.transform = transform_val
test_dataset.dataset.transform = transform_val

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=4,pin_memory=True)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)

print(f"Device: {device}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

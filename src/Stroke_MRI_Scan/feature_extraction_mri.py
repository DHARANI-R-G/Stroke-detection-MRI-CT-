import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BALANCED_DIR = r"D:\Strokes_MRI_Balanced_data"
FEATURE_SAVE = r"D:\Stroke_mri_features"
os.makedirs(FEATURE_SAVE, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(BALANCED_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")

def load_model_with_retry(model_fn, weights_attr):
    try:
        model = model_fn(weights=weights_attr)
    except RuntimeError as e:
        if "invalid hash value" in str(e):
            filename = str(weights_attr.url.split("/")[-1])
            filepath = os.path.join(cache_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            model = model_fn(weights=weights_attr)
        else:
            raise e
    return model

models_dict = {}
models_dict["efficientnet_b3"] = load_model_with_retry(models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1)
models_dict["densenet121"] = load_model_with_retry(models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1)
models_dict["resnet50"] = load_model_with_retry(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2)
models_dict["convnext_tiny"] = load_model_with_retry(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

inception_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
inception_model.eval()
inception_model.to(device)
inception_model.fc = nn.Identity()
inception_model.aux_logits = False
inception_model.AuxLogits = nn.Identity()
models_dict["inception_v3"] = inception_model

for name, model in models_dict.items():
    model.eval()
    model.to(device)
    if "efficientnet" in name:
        model.classifier = nn.Identity()
    elif "densenet" in name:
        model.classifier = nn.Identity()
    elif "resnet" in name:
        model.fc = nn.Identity()
    elif "convnext" in name:
        model.classifier = nn.Identity()

features_list = []
labels_list = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        batch_features = []
        for name, model in models_dict.items():
            feats = model(imgs)
            if feats.dim() == 4:
                feats = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feats, (1,1)), 1)
            batch_features.append(feats.cpu())
        batch_features = torch.cat(batch_features, dim=1)
        features_list.append(batch_features)
        labels_list.append(labels)

features = torch.cat(features_list, dim=0)
labels = torch.cat(labels_list, dim=0)

torch.save(features, os.path.join(FEATURE_SAVE, "features.pt"))
torch.save(labels, os.path.join(FEATURE_SAVE, "labels.pt"))

print("Feature extraction completed!")
print("Feature tensor shape:", features.shape)
print("Labels tensor shape:", labels.shape)
print(f"Saved features.pt and labels.pt in {FEATURE_SAVE}")


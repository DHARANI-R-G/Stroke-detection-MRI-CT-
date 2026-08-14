import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATASET_DIR = r"D:\Strokes_CT_Balanced_data"
NUM_CLASSES = 2
BATCH_SIZE = 4
EPOCHS = 8
LR = 3e-5
XAI_DIR = r"D:\XAI_vit_ctscan_3"
os.makedirs(XAI_DIR, exist_ok=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        return loss

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform_train)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

val_dataset.dataset.transform = transform_val
test_dataset.dataset.transform = transform_val

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)

print(f"Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

in_features = None
if isinstance(model.heads, nn.Linear):
    in_features = model.heads.in_features
elif isinstance(model.heads, nn.Sequential):
    for layer in model.heads:
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            break

if in_features is None:
    raise RuntimeError("Could not find final linear layer in model.heads to replace. Inspect model structure.")

model.heads = nn.Linear(in_features, NUM_CLASSES)
model.to(device)

criterion = FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
    
    if len(all_preds) > 0:
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())
    else:
        epoch_loss, epoch_acc = 0.0, 0.0

    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    scheduler.step()

class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
    
    def hook_fn(self, module, input, output):
        try:
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
                self.attention_weights.append(attn.detach().cpu())
            else:
                if hasattr(output, 'shape') and output.dim() >= 2:
                    self.attention_weights.append(output.detach().cpu())
        except Exception:
            pass
    
    def extract_attention(self, x):
        self.attention_weights = []
        hooks = []
        for name, module in self.model.named_modules():
            n = name.lower()
            if ('attn' in n or 'attention' in n) and 'drop' not in n:
                try:
                    hooks.append(module.register_forward_hook(self.hook_fn))
                except Exception:
                    pass
        with torch.no_grad():
            _ = self.model(x.to(next(self.model.parameters()).device))
        for hook in hooks:
            hook.remove()
        return self.attention_weights
    
    def compute_attention_rollout(self):
        if not self.attention_weights:
            return None
        last_attn = self.attention_weights[-1]
        att = last_attn[0]
        if att.dim() == 3:
            avg_attn = att.mean(dim=0)
        elif att.dim() == 2:
            avg_attn = att
        else:
            return None
        cls_to_patches = avg_attn[0, 1:]
        return cls_to_patches.cpu().numpy()

class GradientSaliency:
    def __init__(self, model):
        self.model = model
    
    def compute_saliency(self, x, target_class):
        x_copy = x.clone().detach().to(next(self.model.parameters()).device)
        x_copy.requires_grad = True
        
        output = self.model(x_copy)
        self.model.zero_grad()
        
        class_score = output[0, target_class]
        class_score.backward()
        
        saliency = x_copy.grad.data.abs().cpu().numpy()
        return saliency[0]

def analyze_saliency_regions(saliency_map):
    height, width = saliency_map.shape
    
    normalized_saliency = saliency_map.copy()
    if normalized_saliency.max() > 0:
        normalized_saliency = (normalized_saliency - normalized_saliency.min()) / (normalized_saliency.max() - normalized_saliency.min())
    
    zones = {
        'Left Cerebellum': (int(height*0.7), height, 0, int(width*0.4)),
        'Right Cerebellum': (int(height*0.7), height, int(width*0.6), width),
        'Brainstem': (int(height*0.6), int(height*0.8), int(width*0.4), int(width*0.6)),
        'Left Temporal': (int(height*0.4), int(height*0.7), 0, int(width*0.3)),
        'Right Temporal': (int(height*0.4), int(height*0.7), int(width*0.7), width),
        'Left Frontal': (0, int(height*0.4), 0, int(width*0.4)),
        'Right Frontal': (0, int(height*0.4), int(width*0.6), width),
        'Central/Basal Ganglia': (int(height*0.3), int(height*0.6), int(width*0.35), int(width*0.65))
    }
    
    region_scores = {}
    for region_name, (y1, y2, x1, x2) in zones.items():
        region_patch = normalized_saliency[y1:y2, x1:x2]
        region_scores[region_name] = {
            'mean_saliency': float(region_patch.mean()),
            'max_saliency': float(region_patch.max()),
            'percentage_active': float((region_patch > 0.5).sum() / region_patch.size * 100)
        }
    
    sorted_regions = sorted(region_scores.items(), 
                           key=lambda x: x[1]['mean_saliency'], 
                           reverse=True)
    
    return sorted_regions

def generate_region_description(sorted_regions, top_n=3):
    top_regions = sorted_regions[:top_n]
    
    region_names = [region[0] for region in top_regions]
    
    if len(region_names) == 1:
        text = f"The AI examined the {region_names[0]} closely"
    elif len(region_names) == 2:
        text = f"The AI examined the {region_names[0]} and {region_names[1]} closely"
    else:
        text = f"The AI examined the {', '.join(region_names[:-1])}, and {region_names[-1]} closely"
    
    return text + " and determined these structures appear normal, contributing to the overall 'Normal' classification."

def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlayed = (img * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return overlayed

def analyze_attention_statistics(attention_map):
    stats = {
        'mean': float(attention_map.mean()),
        'max': float(attention_map.max()),
        'min': float(attention_map.min()),
        'std': float(attention_map.std())
    }
    entropy = -np.sum(attention_map * np.log(attention_map + 1e-10))
    stats['entropy'] = float(entropy)
    top_k = 5
    flat_indices = np.argsort(attention_map)[-top_k:][::-1]
    grid_size = int(np.sqrt(len(attention_map)))
    top_patches = []
    for idx in flat_indices:
        row = idx // grid_size
        col = idx % grid_size
        value = attention_map[idx]
        top_patches.append({
            'index': int(idx),
            'position': (int(row), int(col)),
            'value': float(value)
        })
    stats['top_patches'] = top_patches
    attention_2d = attention_map.reshape(grid_size, grid_size)
    h, w = attention_2d.shape
    quadrants = {
        'Top-Left': attention_2d[:h//2, :w//2],
        'Top-Right': attention_2d[:h//2, w//2:],
        'Bottom-Left': attention_2d[h//2:, :w//2],
        'Bottom-Right': attention_2d[h//2:, w//2:]
    }
    stats['quadrants'] = {}
    for name, quad in quadrants.items():
        stats['quadrants'][name] = {
            'mean': float(quad.mean()),
            'max': float(quad.max())
        }
    return stats

def generate_xai_visualization(model, img_tensor, img_np, prediction, confidence, true_label, img_index):
    class_names = ['Normal', 'Stroke']
    attention_extractor = AttentionExtractor(model)
    try:
        attention_weights = attention_extractor.extract_attention(img_tensor)
        attention_map = attention_extractor.compute_attention_rollout()
    except Exception:
        attention_map = None
    gradient_saliency = GradientSaliency(model)
    saliency_map = gradient_saliency.compute_saliency(img_tensor, prediction)
    
    saliency_sum = np.abs(saliency_map).mean(axis=0)
    sorted_regions = analyze_saliency_regions(saliency_sum)
    region_description = generate_region_description(sorted_regions, top_n=3)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Image #{img_index} - Prediction: {class_names[prediction]} ({confidence*100:.2f}%)', 
                 fontsize=16, fontweight='bold')
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_np)
    ax1.set_title('Original CT Scan', fontweight='bold')
    ax1.axis('off')
    if attention_map is not None:
        grid_size = int(np.sqrt(len(attention_map)))
        attention_2d = attention_map.reshape(grid_size, grid_size)
        overlayed_attn = overlay_heatmap(img_np, attention_2d)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(overlayed_attn)
        ax2.set_title('Attention Heatmap', fontweight='bold')
        ax2.axis('off')
    
    overlayed_sal = overlay_heatmap(img_np, saliency_sum)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlayed_sal)
    ax3.set_title('Gradient Saliency', fontweight='bold')
    ax3.axis('off')
    ax_text = fig.add_subplot(gs[1:, :])
    ax_text.axis('off')
    
    text_content = []
    text_content.append("=" * 80)
    text_content.append(f"EXPLAINABLE AI ANALYSIS ")
    text_content.append("=" * 80)
    text_content.append("")
    text_content.append("PREDICTION RESULTS:")
    text_content.append(f"  Predicted Class: {class_names[prediction]}")
    text_content.append(f"  True Label: {class_names[true_label]}")
    text_content.append(f"  Confidence: {confidence*100:.2f}%")
    text_content.append(f"  Correct: {'YES' if prediction == true_label else 'NO'}")
    text_content.append("")
    if attention_map is not None:
        stats = analyze_attention_statistics(attention_map)
        text_content.append("ATTENTION MECHANISM ANALYSIS:")
        text_content.append(f"  Mean Attention: {stats['mean']:.6f}")
        text_content.append(f"  Max Attention: {stats['max']:.6f}")
        text_content.append(f"  Std Deviation: {stats['std']:.6f}")
        text_content.append(f"  Entropy: {stats['entropy']:.4f} (Lower = More Focused)")
        text_content.append("")
        text_content.append("  Top 5 Attended Patches:")
        for i, patch in enumerate(stats['top_patches'], 1):
            text_content.append(f"    #{i}. Grid Position ({patch['position'][0]}, {patch['position'][1]}) - Value: {patch['value']:.6f}")
        text_content.append("")
        text_content.append("  Spatial Distribution (Quadrants):")
        for quad_name, quad_stats in stats['quadrants'].items():
            text_content.append(f"    {quad_name:15s}: Mean={quad_stats['mean']:.6f}, Max={quad_stats['max']:.6f}")
        text_content.append("")
    
    saliency_stats = {
        'mean': float(saliency_map.mean()),
        'max': float(saliency_map.max()),
        'channel_means': {
            'R': float(saliency_map[0].mean()),
            'G': float(saliency_map[1].mean()),
            'B': float(saliency_map[2].mean())
        }
    }
    text_content.append("GRADIENT-BASED SALIENCY:")
    text_content.append(f"  Mean Saliency: {saliency_stats['mean']:.6f}")
    text_content.append(f"  Max Saliency: {saliency_stats['max']:.6f}")

    text_content.append("")
    
    text_content.append("GRADIENT SALIENCY INTERPRETATION:")
    text_content.append("=" * 80)
    text_content.append("")
    text_content.append("Regions analyzed by AI (ranked by attention):")
    for i, (region_name, scores) in enumerate(sorted_regions[:4], 1):
        display_score = scores['mean_saliency'] * 100
        text_content.append(f"  {i}. {region_name} (Saliency: {display_score:.2f}%)")
    text_content.append("")
    text_content.append("Clinical Interpretation:")
    if prediction == 0:
        modified_desc = region_description
    else:
        top_region = sorted_regions[0][0]
        modified_desc = f"The AI examined the {top_region} closely and detected potential abnormalities, contributing to the 'Stroke' classification."
    text_content.append(f"  {modified_desc}")

    
    
    text_content.append("CLINICAL INTERPRETATION:")
    if prediction == 1:
        if confidence > 0.9:
            certainty = "VERY HIGH confidence"
        elif confidence > 0.7:
            certainty = "HIGH confidence"
        else:
            certainty = "MODERATE confidence"
        text_content.append(f"  ALERT: STROKE DETECTED with {certainty}")
        if attention_map is not None:
            if stats['entropy'] < 3.5:
                text_content.append("  Model shows FOCUSED attention -> Localized abnormality")
            else:
                text_content.append("  Model shows DIFFUSE attention -> Widespread changes")
            max_quad = max(stats['quadrants'].items(), key=lambda x: x[1]['mean'])
            text_content.append(f"  Most suspicious region: {max_quad[0]}")
    else:
        text_content.append("  NO STROKE DETECTED")
        if confidence > 0.9:
            text_content.append("  High confidence in normal appearance")
        else:
            text_content.append("  Moderate confidence - Recommend manual review")
    text_content.append("")
    text_content.append("=" * 80)
    text_str = "\n".join(text_content)
    ax_text.text(0.01, 0.99, text_str, fontsize=8, verticalalignment='top', fontfamily='monospace', wrap=True)
    return fig, text_str

print("\nStarting XAI analysis on test dataset...")
model.eval()

test_preds_list, test_labels_list = [], []
test_loss = 0.0
class_names = ['Normal', 'Stroke']
summary_results = []

for idx, (imgs, labels) in enumerate(test_loader):
    imgs, labels = imgs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        test_preds_list.append(preds.detach().cpu())
        test_labels_list.append(labels.detach().cpu())
    for i in range(len(imgs)):
        img_tensor = imgs[i:i+1]
        prediction = int(preds[i].item())
        confidence = float(probs[i, prediction].item())
        true_label = int(labels[i].item())
        img_number = idx * BATCH_SIZE + i + 1
        img_np = imgs[i].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = (img_np * 255).astype(np.uint8)
        try:
            fig, text_explanation = generate_xai_visualization(
                model, img_tensor, img_np, prediction, confidence, true_label, img_number
            )
        except Exception as e:
            fig = plt.figure(figsize=(8,6))
            plt.imshow(img_np); plt.axis('off')
            text_explanation = f"Image #{img_number}\nError generating XAI viz: {e}"
        img_filename = f"xai_image_{img_number}.png"
        img_path = os.path.join(XAI_DIR, img_filename)
        try:
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
        except Exception:
            pass
        plt.close(fig)
        txt_filename = f"xai_text_{img_number}.txt"
        txt_path = os.path.join(XAI_DIR, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_explanation)
        summary_results.append({
            'image': img_number,
            'prediction': class_names[prediction],
            'true_label': class_names[true_label],
            'confidence': confidence,
            'correct': prediction == true_label
        })
        print(f"Processed image {img_number}/{len(test_dataset)}")

summary_path = os.path.join(XAI_DIR, "summary_all_images.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SUMMARY OF ALL TEST IMAGES\n")
    f.write("=" * 80 + "\n\n")
    for result in summary_results:
        f.write(f"Image #{result['image']}:\n")
        f.write(f"  Prediction: {result['prediction']}\n")
        f.write(f"  True Label: {result['true_label']}\n")
        f.write(f"  Confidence: {result['confidence']*100:.2f}%\n")
        f.write(f"  Correct: {'YES' if result['correct'] else 'NO'}\n")
        f.write("\n")

if len(test_preds_list) > 0:
    test_preds = torch.cat(test_preds_list).numpy()
    test_labels = torch.cat(test_labels_list).numpy()
else:
    test_preds = np.array([])
    test_labels = np.array([])

if len(test_labels) > 0:
    test_acc = accuracy_score(test_labels, test_preds)
    test_loss = test_loss / len(test_loader.dataset)
    cm = confusion_matrix(test_labels, test_preds)
else:
    test_acc = 0.0
    cm = np.array([[]])


print("FINAL TEST RESULTS")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print(f"\nXAI outputs saved in: {XAI_DIR}")
print(f"  - {len(summary_results)} image files")
print(f"  - {len(summary_results)} text files")
print(f"  - 1 summary file")

from datetime import datetime
os.makedirs("results", exist_ok=True)
results_file = os.path.join("results", "metrics_stroke_ct.txt")
with open(results_file, "w", encoding="utf-8") as f:
    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    f.write(f"Dataset: {DATASET_DIR}\n")
    f.write(f"Num classes: {NUM_CLASSES}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Learning rate: {LR}\n")
    f.write("\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write("\n")
    f.write("Confusion Matrix:\n")
    if cm.size == 0:
        f.write("No confusion matrix available\n")
    else:
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")
print(f"Saved metrics to: {results_file}")

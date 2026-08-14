import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve

import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_PATH = r"D:\Stroke_mri_features\features.pt"
LABEL_PATH = r"D:\Stroke_mri_features\labels.pt"

NUM_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 20
GRAD_CLIP = 1.0

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()

def build_model(input_dim, num_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(256, num_classes)
    ).to(device)

def stratified_split(features, labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_idx, temp_idx = next(sss.split(features, labels))

    val_size = 0.5
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
    val_idx, test_idx = next(sss2.split(features[temp_idx], labels[temp_idx]))

    return train_idx, temp_idx[val_idx], temp_idx[test_idx]


features = torch.load(FEATURE_PATH)
labels = torch.load(LABEL_PATH)

X = features.numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
y = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
y = y.reshape(-1).astype(int)


plt.figure()
sns.countplot(x=y)
plt.title("Class Distribution")
plt.show()


scaler = StandardScaler()
X = scaler.fit_transform(X)

train_idx, val_idx, test_idx = stratified_split(X, y)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]


X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)


model = build_model(X_train.shape[1], NUM_CLASSES)
criterion = FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    preds_all, labels_all = [], []
    total_loss = 0

    for x, yb in train_loader:
        x, yb = x.to(device), yb.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds_all.append(out.argmax(1).cpu())
        labels_all.append(yb.cpu())

    train_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(torch.cat(labels_all), torch.cat(preds_all))


    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0

    with torch.no_grad():
        for x, yb in val_loader:
            x, yb = x.to(device), yb.to(device)
            out = model(x)
            loss = criterion(out, yb)

            val_loss += loss.item()
            val_preds.append(out.argmax(1).cpu())
            val_labels.append(yb.cpu())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(torch.cat(val_labels), torch.cat(val_preds))

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()


plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.show()

# -------------------- TEST --------------------
model.eval()
test_preds, test_labels = [], []
probs_all = []

with torch.no_grad():
    for x, yb in test_loader:
        x = x.to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1)

        probs_all.append(probs.cpu())
        test_preds.append(out.argmax(1).cpu())
        test_labels.append(yb)

test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)
probs_all = torch.cat(probs_all).numpy()

# -------------------- CONFUSION MATRIX --------------------
cm = confusion_matrix(test_labels, test_preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

# -------------------- ROC CURVE --------------------
y_bin = label_binarize(test_labels, classes=[0,1,2])

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs_all[:, i])
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve Class {i}")
    plt.show()

# -------------------- PRECISION-RECALL --------------------
for i in range(NUM_CLASSES):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], probs_all[:, i])
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"PR Curve Class {i}")
    plt.show()

# -------------------- CONFIDENCE DISTRIBUTION --------------------
conf = np.max(probs_all, axis=1)
plt.figure()
plt.hist(conf, bins=20)
plt.title("Prediction Confidence")
plt.show()

# -------------------- MISCLASSIFICATION --------------------
wrong = test_preds != test_labels
plt.figure()
sns.countplot(x=test_labels[wrong].numpy())
plt.title("Misclassified Samples")
plt.show()

print("Final Test Accuracy:", accuracy_score(test_labels, test_preds))
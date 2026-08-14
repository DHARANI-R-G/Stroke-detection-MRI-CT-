import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (list, tuple, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float)
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                alpha = self.alpha.to(inputs.device)
            else:
                alpha = self.alpha
            at = alpha[targets]
            loss = at * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

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

def stratified_split(features, labels, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=SEED):
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    idx = np.arange(len(labels))
    train_idx, temp_idx = next(sss1.split(idx, labels))
    val_rel = val_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_rel, random_state=seed)
    val_idx_rel, test_idx_rel = next(sss2.split(temp_idx, labels[temp_idx]))
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]
    return train_idx, val_idx, test_idx

if __name__ == "__main__":
    features = torch.load(FEATURE_PATH)
    labels = torch.load(LABEL_PATH)
    if isinstance(features, torch.Tensor):
        X = features.numpy()
    else:
        X = np.asarray(features)
    y = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    if y.ndim > 1:
        y = y.reshape(-1)
    y = y.astype(int)
    unique = np.unique(y)
    print("Unique labels found:", unique.tolist())
    if unique.min() < 0 or unique.max() >= NUM_CLASSES:
        raise ValueError(f"Label values must be in [0, {NUM_CLASSES-1}]. Found min {int(unique.min())}, max {int(unique.max())}")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    train_idx, val_idx, test_idx = stratified_split(X, y, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type=="cuda"))
    class_counts = np.bincount(y_train)
    inv_freq = 1.0 / (class_counts + 1e-8)
    alpha = inv_freq / inv_freq.sum() * NUM_CLASSES
    alpha = torch.tensor(alpha, dtype=torch.float32)
    model = build_model(X_train.shape[1], NUM_CLASSES)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                outputs = model(x)
                loss = criterion(outputs, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            running_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(torch.cat(all_labels), torch.cat(all_preds))
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                val_preds.append(preds.cpu())
                val_labels.append(y.cpu())
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(torch.cat(val_labels), torch.cat(val_preds))
        print(f"Epoch {epoch}/{EPOCHS} Train Loss: {epoch_loss:.6f} Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.6f} Val Acc: {val_acc:.4f}")
        scheduler.step(val_loss)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= PATIENCE:
            print("Early stopping triggered.")
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            test_preds.append(preds.cpu())
            test_labels.append(y.cpu())
    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)
    test_acc = accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

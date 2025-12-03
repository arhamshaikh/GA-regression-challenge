# ======================================================
# 3D ResNet34 + Multi-Head Attention for GA Prediction
# GA in DAYS (Low RMSE Version)
# ======================================================

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# ---------------- Config ----------------
CSV_TRAIN = "final_train.csv"
CSV_VAL   = "final_valid.csv"
CSV_TEST  = "final_test.csv"

SWEEP_COL = "path_nifti4"
IMG_SIZE  = (64,64,64)

BATCH_SIZE = 2
EPOCHS     = 15
LR_BACKBONE = 1e-5
LR_HEAD     = 5e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "out_resnet34_attention"
os.makedirs(OUT_DIR, exist_ok=True)

print("Device:", DEVICE)

# ---------------- Utilities ----------------
def load_nifti_as_cdhw(path):
    img = nib.load(path).get_fdata()

    img = np.array(img, dtype=np.float32)

    # Case 1: 3D (H, W, D)
    if img.ndim == 3:
        # → convert to (C=1, D, H, W)
        return img[np.newaxis, ...].transpose(0, 3, 1, 2)

    # Case 2: 4D but channel last (H, W, D, C)
    if img.ndim == 4 and img.shape[3] <= 4:
        return img.transpose(3, 2, 0, 1)

    # Case 3: 4D but channel first (C, H, W, D)
    if img.ndim == 4 and img.shape[0] <= 4:
        return img.transpose(0, 3, 1, 2)

    # If anything else → auto-fix to single channel
    D = img.ndim
    shape = img.shape
    print("⚠ Warning: Unusual NIfTI shape:", shape, "Fixing automatically...")

    # flatten everything into 3D volume
    vol = img.squeeze()
    if vol.ndim != 3:
        # last fallback
        vol = vol.reshape(shape[0], shape[1], -1)

    return vol[np.newaxis, ...].transpose(0, 3, 1, 2)
def prepare_tensor_from_nifti(path, img_size=IMG_SIZE):
    arr = load_nifti_as_cdhw(path)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    tensor = torch.from_numpy(arr)

    with torch.no_grad():
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=img_size,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)
    
    return tensor


# ---------------- Dataset ----------------
class SweepDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[SWEEP_COL]
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        img = prepare_tensor_from_nifti(path)
        ga_days = float(row["ga"])

        return img, torch.tensor(ga_days, dtype=torch.float32)


# ---------------- 3D ResNet34 ----------------
# BasicBlock3D + simplified ResNet34
class BasicBlock3D(nn.Module):
    def __init__(self,in_c,out_c,stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c,out_c,3,stride=stride,padding=1)
        self.bn1   = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c,out_c,3,padding=1)
        self.bn2   = nn.BatchNorm3d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c,out_c,1,stride=stride),
                nn.BatchNorm3d(out_c)
            )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet34_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1,64,7,stride=2,padding=3)
        self.bn1   = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(3,stride=2,padding=1)

        # Layers
        self.layer1 = self._make_layer(64,64,3)
        self.layer2 = self._make_layer(64,128,4,stride=2)
        self.layer3 = self._make_layer(128,256,6,stride=2)
        self.layer4 = self._make_layer(256,512,3,stride=2)

    def _make_layer(self,in_c,out_c,blocks,stride=1):
        layers = [BasicBlock3D(in_c,out_c,stride)]
        for _ in range(1,blocks):
            layers.append(BasicBlock3D(out_c,out_c))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


# ---------------- Multi-Head Attention ----------------
class AttnHead(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self,x):
        B,C,D,H,W = x.shape
        x = x.view(B,C,-1).transpose(1,2)   # (B, N, C)
        x,_ = self.attn(x,x,x)
        x = x.mean(1)
        return x


# ---------------- Main GA Model ----------------
class GA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet34_3D()
        self.attn = AttnHead(512)
        self.fc = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,1)
        )

    def forward(self,x):
        x = self.backbone(x)
        x = self.attn(x)
        x = self.fc(x)
        return x.squeeze(1)


# ------------------------------------------------------
# LOAD DATA + NORMALIZE GA (IMPORTANT FOR LOW RMSE)
# ------------------------------------------------------
train_df = pd.read_csv(CSV_TRAIN)
val_df   = pd.read_csv(CSV_VAL)

train_df["ga"] = pd.to_numeric(train_df["ga"], errors="coerce")
val_df["ga"]   = pd.to_numeric(val_df["ga"], errors="coerce")

ga_mean = train_df["ga"].mean()
ga_std  = train_df["ga"].std()

train_df["ga_norm"] = (train_df["ga"] - ga_mean) / ga_std
val_df["ga_norm"]   = (val_df["ga"] - ga_mean) / ga_std

train_ds = SweepDS(train_df)
val_ds   = SweepDS(val_df)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)


# ------------------------------------------------------
# TRAINING
# ------------------------------------------------------
model = GA_Model().to(DEVICE)

optimizer = Adam([
    {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
    {"params": model.attn.parameters(),      "lr": LR_HEAD},
    {"params": model.fc.parameters(),        "lr": LR_HEAD},
])

# Hybrid Loss (lower RMSE)
l1 = nn.L1Loss()
sl1 = nn.SmoothL1Loss()

def hybrid_loss(pred,gt):
    return 0.5*l1(pred,gt) + 0.5*sl1(pred,gt)


# EMA
ema_model = GA_Model().to(DEVICE)
ema_model.load_state_dict(model.state_dict())

def update_ema():
    with torch.no_grad():
        for p, q in zip(model.parameters(), ema_model.parameters()):
            q.data.mul_(0.999).add_(0.001 * p.data)


train_losses = []
val_losses = []


for epoch in range(1, EPOCHS+1):
    model.train()
    run = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs = imgs.to(DEVICE)
        labels = ((labels - ga_mean)/ga_std).to(DEVICE)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = hybrid_loss(preds, labels)
        loss.backward()
        optimizer.step()
        update_ema()

        run += loss.item()

    train_losses.append(run/len(train_loader))

    # ---------------- Validate using EMA model ----------------
    ema_model.eval()
    all_p, all_l = [], []
    vloss = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            p = ema_model(imgs)

            # denormalize
            p = p*ga_std + ga_mean

            vloss += ((p-labels)**2).mean().item()

            all_p.append(float(p.cpu()))
            all_l.append(float(labels.cpu()))

    val_losses.append(vloss/len(val_loader))
    mae = mean_absolute_error(all_l, all_p)
    rmse = (mean_squared_error(all_l, all_p))**0.5

    print(f"Epoch {epoch}: MAE={mae:.3f}  RMSE={rmse:.3f}")

torch.save(ema_model.state_dict(), f"{OUT_DIR}/best_model.pth")
print("Model saved.")


# ------------------------------------------------------
# VALIDATION PRED CSV (STUDY_ID + PREDICTED GA)
# ------------------------------------------------------
print("\nGenerating validation predictions...")

pred_ids = []
pred_vals = []

for i,(imgs,_) in enumerate(val_loader):
    imgs = imgs.to(DEVICE)
    out = ema_model(imgs)
    out = out*ga_std + ga_mean

    pred_ids.append(val_df.iloc[i]["study_id"])
    pred_vals.append(float(out.cpu()))

res = pd.DataFrame({
    "study_id": pred_ids,
    "predicted_ga": pred_vals
})

res.to_csv(f"{OUT_DIR}/val_predictions.csv", index=False)
print("Saved:", f"{OUT_DIR}/val_predictions.csv")
print(res.head())

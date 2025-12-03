# train_final.py
"""
Final robust training script for Blindsweep GA prediction.

Features:
- Robust NIfTI loader handling common shapes (3D/4D, channels-first/last)
- Converts volumes to (C=1, D, H, W)
- Resize via torch.interpolate to IMG_SIZE
- Augmentations, balanced site sampler, EMA, cosine LR restarts
- Tries torchvision r3d_18 (3D) backbone, else fallback ResNet3D
- Saves best EMA model and predictions CSV

Edit constants below to match your files/paths.
"""
import os
import random
import warnings
from collections import defaultdict
from math import sqrt

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------- CONFIG -----------------
CSV_TRAIN = "final_train.csv"   # adjust
CSV_VAL   = "final_valid.csv"   # adjust

SWEEP_COL = "path_nifti4"       # column that contains NIfTI path
ID_COL    = "study_id"
GA_COL    = "ga"

IMG_SIZE  = (96, 96, 96)   # (D, H, W) target - adjust if GPU limited
BATCH_SIZE = 2
EPOCHS     = 70

LR_BACKBONE = 1e-5
LR_HEAD     = 5e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "out_final"
os.makedirs(OUT_DIR, exist_ok=True)

print("Device:", DEVICE)

# ------------- Utilities: robust NIfTI loader -------------
def load_nifti_as_cdhw(path):
    """
    Load NIfTI and return numpy array shaped (C, D, H, W)
    Heuristics:
      - 3D arrays: often (H, W, D) -> convert to (1, D, H, W) by transposing (2,0,1)
                 but if that seems wrong, we still proceed (this covers most datasets).
      - 4D with small last dim (<=4): assume channels last (H, W, D, C) -> transpose (3,2,0,1).
      - 4D with small first dim (<=4): assume channels first (C, H, W, D) -> transpose (0,3,1,2).
      - Fallback: squeeze then put channel dim.
    """
    img = nib.load(path).get_fdata()
    arr = np.asarray(img, dtype=np.float32)

    if arr.ndim == 3:
        # Common case: arr.shape could be (H,W,D) or (D,H,W)
        # Heuristic: if third dim is small (<=8), it might be channels or slices small -> else use (H,W,D) -> (D,H,W)
        h, w, d = arr.shape
        # assume (H, W, D) and convert to (1, D, H, W)
        cdhw = arr.transpose(2, 0, 1)[np.newaxis, ...]
        return cdhw

    if arr.ndim == 4:
        # Cases:
        # (H, W, D, C) -> channels last
        # (C, H, W, D) -> channels first
        s0, s1, s2, s3 = arr.shape
        if s3 <= 4:
            # channels last
            cdhw = arr.transpose(3, 2, 0, 1)
            return cdhw
        if s0 <= 4:
            # channels first
            cdhw = arr.transpose(0, 3, 1, 2)
            return cdhw
        # fallback: treat as (H,W,D,X) -> collapse last dim -> use mean
        collapsed = arr.mean(axis=3)
        cdhw = collapsed.transpose(2,0,1)[np.newaxis, ...]
        return cdhw

    # fallback: squeeze and add channel
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        return arr.transpose(2,0,1)[np.newaxis, ...]
    # unexpected shape
    raise ValueError(f"Unsupported NIfTI shape {arr.shape} for file {path}")


def prepare_tensor_from_nifti(path, img_size=IMG_SIZE):
    """
    Returns torch.Tensor shape (C, D, H, W) normalized to [0,1] and resized to img_size.
    """
    arr = load_nifti_as_cdhw(path)  # (C, D, H, W)
    # normalize per-volume
    if np.nanmax(arr) == np.nanmin(arr):
        # constant volume -> small noise to avoid divide by zero downstream
        arr = arr.astype(np.float32)
        arr = arr + np.random.randn(*arr.shape) * 1e-6
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    tensor = torch.from_numpy(arr).float()  # (C, D, H, W)
    # Resize using trilinear
    with torch.no_grad():
        tensor = F.interpolate(tensor.unsqueeze(0), size=img_size, mode="trilinear", align_corners=False).squeeze(0)
    return tensor  # (C, D, H, W)


# ---------------- Augmentations ----------------
def rand_bool(p=0.5):
    return random.random() < p

def center_crop_or_pad(x, shape):
    # x: (C, D, H, W)
    C, D, H, W = x.shape
    Td, Th, Tw = shape
    # crop/pad
    # crop
    ds = []
    for cur, tgt in zip((D,H,W), (Td,Th,Tw)):
        if cur > tgt:
            start = (cur - tgt) // 2
            ds.append(slice(start, start + tgt))
        else:
            ds.append(slice(0, cur))
    x_c = x[:, ds[0], ds[1], ds[2]]
    # pad if needed
    pd = [0, 0, 0, 0, 0, 0]  # pad W,H,D order for F.pad (reverse)
    # amount to pad per dim
    pads = []
    for cur, tgt in zip((D,H,W), (Td,Th,Tw)):
        if cur < tgt:
            total = tgt - cur
            left = total // 2
            right = total - left
            pads.append((left, right))
        else:
            pads.append((0,0))
    # F.pad expects (padWleft, padWright, padHleft, padHright, padDleft, padDright)
    pad_list = []
    for a,b in reversed(pads):
        pad_list.extend([a,b])
    if any([a+b>0 for a,b in pads]):
        x_c = F.pad(x_c, pad_list, mode="constant", value=0.0)
    return x_c

def augment_tensor(tensor, p_flip=0.6, p_noise=0.6, p_gamma=0.6, p_zoom=0.6):
    # tensor: torch.Tensor (C, D, H, W)
    x = tensor.clone()
    if rand_bool(p_flip):
        if rand_bool(0.5): x = x.flip(-1)  # W
        if rand_bool(0.5): x = x.flip(-2)  # H
        if rand_bool(0.5): x = x.flip(-3)  # D
    if rand_bool(p_zoom):
        z = random.uniform(0.9, 1.12)
        C, D, H, W = x.shape
        new_d = max(2, int(D*z))
        new_h = max(2, int(H*z))
        new_w = max(2, int(W*z))
        x = F.interpolate(x.unsqueeze(0), size=(new_d,new_h,new_w), mode="trilinear", align_corners=False).squeeze(0)
        x = center_crop_or_pad(x, (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    if rand_bool(p_gamma):
        gamma = random.uniform(0.85, 1.25)
        x = torch.clamp(x ** gamma, 0.0, 1.0)
    if rand_bool(p_noise):
        std = random.uniform(0.0, 0.05)
        x = x + torch.randn_like(x) * std
        x = x.clamp(0.0, 1.0)
    return x

# ---------------- Dataset ----------------
class SweepDS(Dataset):
    def __init__(self, df, transforms=None, train=True):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[SWEEP_COL]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        img = prepare_tensor_from_nifti(path)
        if self.train and self.transforms is not None:
            img = self.transforms(img)
        ga = float(row[GA_COL])
        sid = row[ID_COL]
        return img, torch.tensor(ga, dtype=torch.float32), sid

# ---------------- Balanced sampler ----------------
class BalancedSiteSampler(Sampler):
    def __init__(self, df, site_fn, epoch_len=None):
        self.df = df.reset_index(drop=True)
        self.site_fn = site_fn
        self.indices_by_site = defaultdict(list)
        for i, sid in enumerate(self.df[ID_COL]):
            s = site_fn(sid)
            self.indices_by_site[s].append(i)
        self.sites = list(self.indices_by_site.keys())
        self.epoch_len = epoch_len or len(self.df)

    def __iter__(self):
        n_sites = max(1, len(self.sites))
        per_site = max(1, self.epoch_len // n_sites)
        indices = []
        for s in self.sites:
            idxs = self.indices_by_site[s]
            if len(idxs) == 0:
                continue
            chosen = np.random.choice(idxs, per_site, replace=len(idxs) < per_site)
            indices.extend(chosen.tolist())
        while len(indices) < self.epoch_len:
            indices.append(random.randint(0, len(self.df)-1))
        random.shuffle(indices)
        return iter(indices[:self.epoch_len])

    def __len__(self):
        return self.epoch_len

# --------------- Model backbone (r3d_18 or fallback ResNet3D) ---------------
def get_backbone(out_features=512):
    try:
        from torchvision.models.video import r3d_18
        m = r3d_18(pretrained=False)
        # adjust first conv if needed (r3d_18 has .stem[0] conv)
        try:
            conv0 = m.stem[0]
            if conv0.in_channels != 1:
                m.stem[0] = nn.Conv3d(1, conv0.out_channels, kernel_size=conv0.kernel_size,
                                      stride=conv0.stride, padding=conv0.padding, bias=False)
        except Exception:
            # older torchvision versions use conv1
            if hasattr(m, "conv1") and m.conv1.in_channels != 1:
                orig = m.conv1
                m.conv1 = nn.Conv3d(1, orig.out_channels, kernel_size=orig.kernel_size,
                                    stride=orig.stride, padding=orig.padding, bias=False)
        num_ftrs = m.fc.in_features
        m.fc = nn.Identity()
        return m, num_ftrs
    except Exception:
        print("torchvision r3d_18 not available — using fallback ResNet3D")
        class BasicBlock3D(nn.Module):
            def __init__(self,in_c,out_c,stride=1):
                super().__init__()
                self.conv1 = nn.Conv3d(in_c,out_c,3,stride=stride,padding=1)
                self.bn1   = nn.BatchNorm3d(out_c)
                self.conv2 = nn.Conv3d(out_c,out_c,3,padding=1)
                self.bn2   = nn.BatchNorm3d(out_c)
                self.shortcut = nn.Sequential()
                if stride != 1 or in_c != out_c:
                    self.shortcut = nn.Sequential(nn.Conv3d(in_c,out_c,1,stride=stride), nn.BatchNorm3d(out_c))
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
                x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
                x = x.mean(dim=[2,3,4])
                return x
        m = ResNet34_3D()
        return m, 512

class GA_Model(nn.Module):
    def __init__(self, backbone, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = feats.view(feats.size(0), -1)
        out = self.fc(feats)
        return out.squeeze(1)

# ---------------- Load dataframes ----------------
train_df = pd.read_csv(CSV_TRAIN)
val_df   = pd.read_csv(CSV_VAL)

train_df[GA_COL] = pd.to_numeric(train_df[GA_COL], errors="coerce")
val_df[GA_COL] = pd.to_numeric(val_df[GA_COL], errors="coerce")

def site_fn(study_id):
    return str(study_id).split("-")[0]

train_df["site"] = train_df[ID_COL].apply(site_fn)
val_df["site"] = val_df[ID_COL].apply(site_fn)

ga_mean = train_df[GA_COL].mean()
ga_std  = train_df[GA_COL].std()
print(f"GA mean={ga_mean:.3f}  std={ga_std:.3f}")

train_df["ga_norm"] = (train_df[GA_COL] - ga_mean) / ga_std
val_df["ga_norm"] = (val_df[GA_COL] - ga_mean) / ga_std

# ---------------- DataLoaders ----------------
train_ds = SweepDS(train_df, transforms=lambda x: augment_tensor(x), train=True)
val_ds   = SweepDS(val_df, transforms=None, train=False)
train_sampler = BalancedSiteSampler(train_df, site_fn, epoch_len=len(train_df))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# ---------------- Build model ----------------
backbone, feat_dim = get_backbone()
model = GA_Model(backbone, feat_dim).to(DEVICE)

optimizer = Adam([
    {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
    {"params": model.fc.parameters(), "lr": LR_HEAD},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

l1 = nn.L1Loss()
sl1 = nn.SmoothL1Loss()
def hybrid_loss(pred_norm, gt_norm):
    return 0.5 * l1(pred_norm, gt_norm) + 0.5 * sl1(pred_norm, gt_norm)

ema_model = GA_Model(backbone, feat_dim).to(DEVICE)
ema_model.load_state_dict(model.state_dict())

def update_ema(model, ema_model, alpha=0.999):
    with torch.no_grad():
        for p, q in zip(model.parameters(), ema_model.parameters()):
            q.data.mul_(alpha).add_(p.data * (1 - alpha))

# ---------------- Training loop ----------------
best_rmse = 1e9
history = {"train_loss": [], "val_rmse": [], "val_mae": []}

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    n_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for imgs, gas, _ in pbar:
        imgs = imgs.to(DEVICE)
        gas_norm = ((gas - ga_mean) / ga_std).to(DEVICE)
        optimizer.zero_grad()
        preds_norm = model(imgs)
        loss = hybrid_loss(preds_norm, gas_norm)
        loss.backward()
        optimizer.step()
        update_ema(model, ema_model)
        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": running_loss / n_batches})
    history["train_loss"].append(running_loss / max(1, n_batches))
    scheduler.step(epoch + random.random()*0.01)

    # Validate using EMA
    ema_model.eval()
    all_p, all_l, all_ids = [], [], []
    with torch.no_grad():
        for imgs, gas, sid in val_loader:
            imgs = imgs.to(DEVICE)
            p_norm = ema_model(imgs)
            p = p_norm * ga_std + ga_mean
            all_p.append(float(p.cpu()))
            all_l.append(float(gas.item()))
            all_ids.append(sid[0])
    val_rmse = sqrt(mean_squared_error(all_l, all_p))
    val_mae = mean_absolute_error(all_l, all_p)
    history["val_rmse"].append(val_rmse)
    history["val_mae"].append(val_mae)
    print(f"Epoch {epoch}: TrainLoss={history['train_loss'][-1]:.4f}  Val MAE={val_mae:.3f}  Val RMSE={val_rmse:.3f}")

    # per-site metrics
    df_val = pd.DataFrame({"study_id": all_ids, "pred": all_p, "gt": all_l})
    df_val["site"] = df_val["study_id"].apply(site_fn)
    print("Per-site (MAE / RMSE):")
    for s, g in df_val.groupby("site"):
        mae_s = mean_absolute_error(g["gt"], g["pred"])
        rmse_s = sqrt(mean_squared_error(g["gt"], g["pred"]))
        print(f"  {s}: MAE={mae_s:.3f}, RMSE={rmse_s:.3f}")

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        torch.save(ema_model.state_dict(), os.path.join(OUT_DIR, "best_ema_model.pth"))
        out_df = pd.DataFrame({"study_id": all_ids, "predicted_ga": all_p})
        out_df.to_csv(os.path.join(OUT_DIR, "best_val_predictions.csv"), index=False)
        print("Saved best model and predictions.")

# ---------------- Final evaluation ----------------
print("\n== FINAL EVAL ==")
ema_model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_ema_model.pth")))
ema_model.eval()
all_p, all_l, all_ids = [], [], []
with torch.no_grad():
    for imgs, gas, sid in val_loader:
        imgs = imgs.to(DEVICE)
        p_norm = ema_model(imgs)
        p = p_norm * ga_std + ga_mean
        all_p.append(float(p.cpu()))
        all_l.append(float(gas.item()))
        all_ids.append(sid[0])
df_val = pd.DataFrame({"study_id": all_ids, "pred": all_p, "gt": all_l})
df_val["site"] = df_val["study_id"].apply(site_fn)
overall_mae = mean_absolute_error(df_val["gt"], df_val["pred"])
overall_rmse = sqrt(mean_squared_error(df_val["gt"], df_val["pred"]))
print(f"Overall MAE={overall_mae:.3f}  RMSE={overall_rmse:.3f}")
print("Per-site:")
for s, g in df_val.groupby("site"):
    mae_s = mean_absolute_error(g["gt"], g["pred"])
    rmse_s = sqrt(mean_squared_error(g["gt"], g["pred"]))
    print(f"  {s}: MAE={mae_s:.3f}, RMSE={rmse_s:.3f}")

df_val[["study_id", "pred"]].rename(columns={"pred": "predicted_ga"}).to_csv(os.path.join(OUT_DIR, "final_val_predictions.csv"), index=False)
pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "history.csv"), index=False)
print("Saved final_val_predictions.csv and history.csv")

# ---------------- Colab 3D CNN GA Prediction ----------------
import os
import warnings
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


# ---------------- Config ----------------
CSV_TRAIN = "final_train.csv"  # Colab uploaded
CSV_TEST  = "final_test.csv"   # Colab uploaded
SWEEP_COL = "path_nifti4"      # Using single sweep
IMG_SIZE = (64, 64, 64)        # Resize all volumes to (D,H,W)
BATCH_SIZE = 2
EPOCHS = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "out_3dcnn"
os.makedirs(OUT_DIR, exist_ok=True)

print("Device:", DEVICE)

# ---------------- Utilities ----------------
def load_nifti_as_cdhw(path):
    img = nib.load(path).get_fdata()
    img = np.array(img)
    
    # Handle common shapes
    if img.ndim == 3:
        # (H,W,D) -> (1,D,H,W)
        H,W,D = img.shape
        return img.transpose(2,0,1)[np.newaxis,...].astype(np.float32)
    
    if img.ndim == 4:
        H,W,C,D = img.shape
        if C == 1:
            return img.transpose(2,3,0,1).astype(np.float32)
        if img.shape[0]==1:
            arr = np.squeeze(img, axis=0)
            return arr.transpose(2,0,1)[np.newaxis,...].astype(np.float32)
        if img.shape[3]==1:
            arr = np.squeeze(img, axis=3)
            return arr.transpose(2,0,1)[np.newaxis,...].astype(np.float32)
    raise ValueError(f"Unsupported NIfTI shape: {img.shape}")

def prepare_tensor_from_nifti(path, img_size=IMG_SIZE):
    arr = load_nifti_as_cdhw(path)
    # normalize
    mn,mx = arr.min(), arr.max()
    arr = (arr-mn)/(mx-mn) if mx>mn else arr*0.0
    tensor = torch.from_numpy(arr)  # (C,D,H,W)
    # resize to IMG_SIZE
    with torch.no_grad():
        resized = F.interpolate(tensor.unsqueeze(0), size=img_size, mode='trilinear', align_corners=False)
    return resized.squeeze(0)

# ---------------- Dataset ----------------
class SingleSweepDataset(Dataset):
    def __init__(self, df, sweep_col=SWEEP_COL):
        self.df = df.reset_index(drop=True)
        self.sweep_col = sweep_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row.get(self.sweep_col, None)
        if not path or not os.path.exists(path):
            path = row.get("path_nifti1", None)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"No valid NIfTI for row {idx}")
        img_tensor = prepare_tensor_from_nifti(path)
        ga = float(row["ga"])
        return img_tensor, torch.tensor(ga, dtype=torch.float32)

# ---------------- Model ----------------
class Model3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64,128,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)

# ---------------- Load CSV ----------------
df = pd.read_csv(CSV_TRAIN)
df.columns = df.columns.str.strip()
df = df.dropna(subset=["ga"])
df["ga"] = pd.to_numeric(df["ga"], errors="coerce")
df = df.dropna(subset=["ga"])

median_ga = df["ga"].median()
if median_ga>50:
    df["ga"] = df["ga"]/7.0

train_df, val_df = train_test_split(df, test_size=0.12, random_state=42)
train_ds = SingleSweepDataset(train_df)
val_ds = SingleSweepDataset(val_df)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Training ----------------
model = Model3D().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

train_losses,val_losses = [],[]
for epoch in range(1,EPOCHS+1):
    model.train()
    running=0
    with tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}") as t:
        for imgs,labels in t:
            imgs,labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            t.set_postfix(train_loss=running/((t.n) if t.n>0 else 1))
    train_losses.append(running/len(train_loader))

    # Validation
    # model.eval()
    # running_val=0
    # all_preds, all_labels=[],[]
    # with torch.no_grad():
    #     for imgs, labels in val_loader:
    #         imgs,labels = imgs.to(DEVICE), labels.to(DEVICE)
    #         preds = model(imgs)
    #         running_val += criterion(preds, labels).item()
    #         all_preds.extend(preds.cpu().numpy().tolist())
    #         all_labels.extend(labels.cpu().numpy().tolist())
    # val_losses.append(running_val/len(val_loader))
    # mae = mean_absolute_error(all_labels, all_preds)
    # rmse = mean_squared_error(all_labels, all_preds, squared=False)
    # print(f"Epoch {epoch} TrainLoss {train_losses[-1]:.4f} ValLoss {val_losses[-1]:.4f}  MAE {mae:.3f} RMSE {rmse:.3f}")

# Validation
model.eval()
running_val = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs)
        running_val += criterion(preds, labels).item()

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

val_losses.append(running_val / len(val_loader))

mae = mean_absolute_error(all_labels, all_preds)

mse = mean_squared_error(all_labels, all_preds)
rmse = mse ** 0.5

print(f"Epoch {epoch} TrainLoss {train_losses[-1]:.4f} "
      f"ValLoss {val_losses[-1]:.4f}  MAE {mae:.3f} RMSE {rmse:.3f}")


# ---------------- Save model & plots ----------------
torch.save(model.state_dict(), os.path.join(OUT_DIR,"model_3dcnn.pth"))
plt.figure()
plt.plot(train_losses,label="train")
plt.plot(val_losses,label="val")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig(os.path.join(OUT_DIR,"loss_curve.png"))
print("Training done. Outputs in:", OUT_DIR)

# ---------------- Test Predictions ----------------
# ---------------- Test Predictions ----------------
if os.path.exists(CSV_TEST):
    test_df = pd.read_csv(CSV_TEST)
    test_df["ga"] = pd.to_numeric(test_df["ga"], errors="coerce")
    test_df = test_df.dropna(subset=["ga"])
    if test_df["ga"].median() > 50:
        test_df["ga"] = test_df["ga"] / 7.0

    test_ds = SingleSweepDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs).detach().cpu().numpy().tolist()  # <-- FIXED
            preds.extend(out)
            trues.extend(labels.numpy().tolist())

    # Save CSV
    res_df = pd.DataFrame({"true_ga_weeks": trues, "pred_ga_weeks": preds})
    res_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    # Plot
    plt.figure(figsize=(5,5))
    plt.scatter(trues, preds, alpha=0.6)
    plt.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--')
    plt.xlabel("True GA (weeks)")
    plt.ylabel("Pred GA (weeks)")
    plt.savefig(os.path.join(OUT_DIR, "true_vs_pred.png"))

    print("Test predictions saved.")


    # Save CSV and scatter plot
    res_df = pd.DataFrame({"true_ga_weeks":trues,"pred_ga_weeks":preds})
    res_df.to_csv(os.path.join(OUT_DIR,"test_predictions.csv"), index=False)
    plt.figure(figsize=(5,5))
    plt.scatter(trues,preds,alpha=0.6)
    plt.plot([min(trues),max(trues)],[min(trues),max(trues)],'r--')
    plt.xlabel("True GA (weeks)")
    plt.ylabel("Pred GA (weeks)")
    plt.savefig(os.path.join(OUT_DIR,"true_vs_pred.png"))
    print("Test predictions saved.")

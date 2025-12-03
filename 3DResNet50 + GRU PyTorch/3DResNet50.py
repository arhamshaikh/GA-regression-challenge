import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm
import os
from torchvision.models.video import r3d_18  # lightweight 3D CNN; can change to r3d_50 if GPU allows

# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "final_valid.csv"
BATCH_SIZE = 1      # small batch for memory
NUM_WORKERS = 0
NUM_FRAMES = 8      # use more frames for better temporal context
IMG_SIZE = (32, 128, 128)  # (D,H,W), adjust if GPU allows

# -----------------------------
# Dataset
# -----------------------------
class VideoDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.paths_columns = [f'path_nifti{i}' for i in range(1, NUM_FRAMES+1)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        volumes = []

        for col in self.paths_columns:
            path = row[col]
            if not os.path.exists(path):
                img = np.zeros((1, *IMG_SIZE), dtype=np.float32)
            else:
                nifti = nib.load(path)
                img = nifti.get_fdata()
                img = torch.tensor(img, dtype=torch.float32)
                if img.ndim == 3:
                    img = img.unsqueeze(0)  # add channel dimension
                # interpolate to IMG_SIZE
                img = F.interpolate(img.unsqueeze(0), size=IMG_SIZE, mode='trilinear', align_corners=False).squeeze(0)
            volumes.append(img)

        x = torch.stack(volumes, dim=0)  # [NUM_FRAMES, 1, D, H, W]
        y = torch.tensor(row['ga'], dtype=torch.float32)
        return x, y, row['study_id']

# -----------------------------
# Model
# -----------------------------
class ResNetGRU(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        # 3D CNN backbone
        self.backbone = r3d_18(weights=None)  # lightweight; replace with r3d_50(weights='KINETICS400') if GPU allows
        self.backbone.stem[0] = nn.Conv3d(1, 64, kernel_size=7, stride=(1,2,2), padding=(3,3,3), bias=False)
        self.backbone.fc = nn.Identity()  # remove classifier, output features
        self.gru = nn.GRU(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [B, F, C, D, H, W]
        B, F, C, D, H, W = x.shape
        x = x.view(B*F, C, D, H, W)
        feats = self.backbone(x)  # [B*F, 512]
        feats = feats.view(B, F, -1)  # [B, F, 512]
        out, _ = self.gru(feats)
        out = self.fc(out[:, -1, :])
        return out.squeeze(1)

# -----------------------------
# Training
# -----------------------------
def train_and_save(num_epochs=50):
    dataset = VideoDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = ResNetGRU().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler()  # mixed precision

    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(loader)
        for X, y, study_ids in loop:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                pred = model(X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            torch.cuda.empty_cache()

        # Save intermediate predictions each epoch if needed
        # You can add code here to save model checkpoints
    def save_predictions(model, dataset, csv_name="3dresnet50predictions.csv"):
            model.eval()
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
            results = []

            with torch.no_grad():
                for idx in tqdm(range(len(dataset)), desc="Saving predictions"):
                    X, y = dataset[idx]
                    X = X.unsqueeze(0).to(DEVICE)
                    pred = model(X).item()
                    study_id = dataset.data.iloc[idx]['study_id']
                    results.append({"study_id": study_id, "predicted_ga": pred})

            pd.DataFrame(results).to_csv(csv_name, index=False)
            print(f"Predictions saved to {csv_name}")

if __name__ == "__main__":
    train_and_save(num_epochs=50)

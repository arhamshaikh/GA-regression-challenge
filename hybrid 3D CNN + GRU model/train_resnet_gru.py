import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm
import os

# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "final_valid.csv"
BATCH_SIZE = 1   # small batch to avoid memory crash
NUM_WORKERS = 0  # 0 if CPU memory limited
NUM_FRAMES = 4   # number of nifti volumes per study
IMG_SIZE = (32, 64, 64)  # downsampled (D, H, W)
NUM_EPOCHS = 50

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
                # fallback: zeros if missing
                img = torch.zeros((1, *IMG_SIZE), dtype=torch.float32)
            else:
                nifti = nib.load(path)
                img = nifti.get_fdata()
                img = torch.tensor(img, dtype=torch.float32)

                # Ensure shape [C,D,H,W] for Conv3d
                if img.ndim == 3:  # [H,W,D]
                    img = img.permute(2,0,1).unsqueeze(0)  # [1,D,H,W]
                elif img.ndim == 4:  # [H,W,D,C]
                    img = img.permute(3,2,0,1)  # [C,D,H,W]
                    if img.shape[0] != 1:
                        img = img.mean(0, keepdim=True)  # reduce channels to 1

                # interpolate to IMG_SIZE
                img = F.interpolate(img.unsqueeze(0), size=IMG_SIZE, mode='trilinear', align_corners=False).squeeze(0)

            volumes.append(img)

        x = torch.stack(volumes, dim=0)  # [NUM_FRAMES, 1, D, H, W]
        y = torch.tensor(row['ga'], dtype=torch.float32)
        return x, y

# -----------------------------
# Model
# -----------------------------
class ResNetGRU(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        # simple 3D CNN
        self.cnn = nn.Sequential(
            nn.Conv3d(1,16,3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,stride=1,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.gru = nn.GRU(input_size=32, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self, x):
        # x: [B, NUM_FRAMES, C, D, H, W]
        B, F, C, D, H, W = x.shape
        x = x.view(B*F, C, D, H, W)
        features = self.cnn(x).view(B, F, -1)  # [B, F, 32]
        out, _ = self.gru(features)
        out = self.fc(out[:, -1, :])
        return out.squeeze(1)

# -----------------------------
# Training
# -----------------------------
def train_and_save(num_epochs=NUM_EPOCHS):
    dataset = VideoDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = ResNetGRU().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for X, y in loop:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                pred = model(X)
                loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            torch.cuda.empty_cache()

    # save model
    torch.save(model.state_dict(), "resnet_gru_model.pth")
    print("Model saved as resnet_gru_model.pth")

def save_predictions(model, dataset, csv_name="hybrid3dcnnpredictions.csv"):
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
    train_and_save()


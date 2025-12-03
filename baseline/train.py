import torch
from torch.utils.data import DataLoader
from model_baseline import BaselineGA
from dataset import GADataset
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Load Data
# -----------------------------
train_dataset = GADataset("final_train.csv", "/mnt/Data/pakistan/arhamcode/GA-regression-challenge/dataset")
val_dataset = GADataset("final_valid.csv", "/mnt/Data/pakistan/arhamcode/GA-regression-challenge/dataset")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

# -----------------------------
# Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineGA().to(device)

criterion = nn.L1Loss()            # MAE
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# Training
# -----------------------------
for epoch in range(10):
    model.train()
    total_loss = 0

    for sweeps, label in train_loader:
        sweeps = [s.to(device) for s in sweeps]
        label = label.to(device)

        pred = model(sweeps)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train MAE: {total_loss / len(train_loader):.4f}")

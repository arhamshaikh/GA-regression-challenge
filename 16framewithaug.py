import torch
from torch.utils.data import DataLoader
import pandas as pd
from model import NEJMbaseline
from data import SweepEvalDataset, imagenet_transform

# -----------------------------
# Config
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = "checkpoints/best_model.pth"   
val_csv = '/mnt/Data/hackathon/final_valid.csv'
batch_size = 8
n_sweeps_val = 8
output_csv = "validation_predictions.csv"

# -----------------------------
# Load validation dataset
# -----------------------------
val_dataset = SweepEvalDataset(
    val_csv,
    n_sweeps=n_sweeps_val,
    transform=imagenet_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

# -----------------------------
# Load study IDs from CSV
# -----------------------------
val_ids = pd.read_csv(val_csv)['study_id'].tolist()

# -----------------------------
# Load model + checkpoint
# -----------------------------
model = NEJMbaseline().to(device)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# Run inference
# -----------------------------
all_preds = []

with torch.no_grad():
    for sweeps, labels in val_loader:  
        B, S, T, C, H, W = sweeps.shape
        sweeps = sweeps.to(device).view(B, S * T, C, H, W)
        outputs, _ = model(sweeps)
        outputs = outputs.squeeze(1).cpu().numpy()
        all_preds.extend(outputs)

# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame({
    'study_id': val_ids,
    'predicted_ga': all_preds
})

df.to_csv(output_csv, index=False)

print(f"✅ Predictions saved to {output_csv}")

import torch
from torch.utils.data import DataLoader
import pandas as pd
from model import NEJMbaseline
from data import SweepEvalDataset, imagenet_transform

# -----------------------------
# Config
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = "checkpoints/best_model.pth"   # Your trained model checkpoint
test_csv = '/mnt/Data/hackathon/final_test_split.csv'  # Your test CSV
output_csv = "final_test_predictions_days.csv"   # Output CSV

# Memory-safe settings
batch_size = 1       # Reduce if OOM
n_sweeps_test = 2    # Reduce if OOM
num_workers = 2      # Reduce if OOM

# -----------------------------
# Load TEST dataset
# -----------------------------
test_dataset = SweepEvalDataset(
    csv_path=test_csv,
    n_sweeps=n_sweeps_test,
    transform=imagenet_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

# -----------------------------
# Load study IDs
# -----------------------------
test_ids = pd.read_csv(test_csv)['study_id'].tolist()

# -----------------------------
# Load model + checkpoint
# -----------------------------
# Make sure NEJMbaseline class is imported from model.py
model = NEJMbaseline().to(device)

# Load checkpoint safely (state_dict)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# Run inference
# -----------------------------
all_preds = []

with torch.no_grad():
    for sweeps, _ in test_loader:
        # sweeps: (B, S, T, C, H, W)
        B, S, T, C, H, W = sweeps.shape

        # Flatten sweeps for model: (B, S*T, C, H, W)
        sweeps = sweeps.to(device).view(B, S * T, C, H, W)

        outputs, _ = model(sweeps)
        outputs = outputs.squeeze(1).cpu().numpy()

        all_preds.extend(outputs)

# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame({
    'study_id': test_ids,
    'predicted_ga': all_preds
})

df.to_csv(output_csv, index=False)

print("✅ DONE (IN DAYS)!")
print(f"📄 Predictions saved to: {output_csv}")


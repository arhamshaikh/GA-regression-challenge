import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


class GADataset(Dataset):
    def __init__(self, csv_file, data_root):
        df = pd.read_csv(csv_file)
        self.study_ids = df["study_id"].tolist()
        self.labels = df["ga"].tolist()
        self.data_root = data_root

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

    def load_sweep(self, sweep_path):
        arr = np.load(sweep_path)            # (T,H,W) or (T,H,W,3)

        if arr.ndim == 3:                    # grayscale → convert to 3 channels
            arr = np.stack([arr]*3, axis=-1)

        frames = [self.transform(arr[i]) for i in range(arr.shape[0])]
        frames = torch.stack(frames)         # (T,3,224,224)
        return frames

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        study_dir = os.path.join(self.data_root, study_id)

        sweeps = []
        for n in range(1, 9):                # 8 sweeps
            p = os.path.join(study_dir, f"sweep_{n}.npy")
            sweeps.append(self.load_sweep(p))  # each sweep = (T,3,224,224)

        return sweeps, label

    def __len__(self):
        return len(self.study_ids)

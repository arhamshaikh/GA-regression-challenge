import torch
import torch.nn as nn
import random
from torchvision.models import resnet18
import torch.nn.functional as F

# -------- uniform sample --------
def sample_uniform(num_frames, k=16):
    idx = torch.linspace(0, num_frames-1, steps=min(k, num_frames)).long()
    return idx

# -------- Model --------
class BaselineGA(nn.Module):
    def __init__(self):
        super().__init__()

        res = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(res.children())[:-1])  # 512-dim
        self.dim = 512

        self.attention = nn.Linear(self.dim, 1)      # frame score
        self.regressor = nn.Linear(self.dim, 1)      # GA prediction

    def forward(self, sweeps):
        # sweeps = list of 8 tensors, each (T,3,224,224)

        # 1. Random sweep
        idx = random.randint(0, 7)
        frames = sweeps[idx]
        T = frames.shape[0]

        # 2. Uniform 16-frame sampling
        chosen = sample_uniform(T, k=16)
        frames = frames[chosen]                      # (16,3,224,224)

        feats = []
        for f in frames:
            f = f.unsqueeze(0)
            z = self.feature_extractor(f).view(1, -1)  # (1,512)
            feats.append(z)

        feats = torch.cat(feats, dim=0)              # (16,512)

        # 3. Attention
        att = self.attention(feats)                  # (16,1)
        w = torch.softmax(att, dim=0)

        # 4. Weighted feature
        combined = torch.sum(w * feats, dim=0)       # (512,)

        # 5. Regression
        out = self.regressor(combined)               # (1)
        return out

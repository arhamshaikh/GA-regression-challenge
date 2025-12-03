import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ----------------------------
# Attention module
# ----------------------------
class WeightedAverageAttention(nn.Module):
    """
    Computes a weighted average of temporal features using learned attention scores.
    features: (B, T, D)
    """
    def __init__(self, feature_dim=512, reduced_dim=128):
        super().__init__()
        self.W = nn.Linear(feature_dim, 64)
        self.V = nn.Linear(64, 1)
        self.Q = nn.Linear(feature_dim, reduced_dim)
        self.norm = nn.LayerNorm(reduced_dim)

    def forward(self, features):
        # features: (B, T, D)
        attn_scores = self.V(torch.tanh(self.W(features)))  # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)        # (B, T, 1)

        reduced_features = self.Q(features)                 # (B, T, reduced_dim)
        reduced_features = self.norm(reduced_features)

        weighted_sum = torch.sum(attn_weights * reduced_features, dim=1)  # (B, reduced_dim)
        return weighted_sum, attn_weights.squeeze(-1)  # (B, reduced_dim), (B, T)


# ----------------------------
# Model definition
# ----------------------------
class NEJMbaseline(nn.Module):
    """
    ResNet18-based regression model with attention over time (frames).
    """
    def __init__(self,
                 reduced_dim=128,
                 fine_tune_backbone=True,
                 backbone='resnet18',
                 pretrained=True,
                 dropout_p=0.3):
        super().__init__()

        # Handle torchvision weights API
        if backbone == 'resnet18':
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                resnet = models.resnet18(weights=weights)
            except AttributeError:
                resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            try:
                weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
                resnet = models.resnet34(weights=weights)
            except AttributeError:
                resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove final FC, keep conv features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = feature_dim

        # Optionally freeze backbone
        if not fine_tune_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.attention = WeightedAverageAttention(
            feature_dim=self.feature_dim,
            reduced_dim=reduced_dim
        )

        # Stronger head with dropout
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(reduced_dim, reduced_dim)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(reduced_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C, H, W)
        Returns:
            output: Predicted value (B, 1)
            attn_weights: Temporal attention weights (B, T)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        features = self.feature_extractor(x)                # (B*T, 512, 1, 1)
        features = features.view(B, T, self.feature_dim)    # (B, T, 512)

        aggregated, attn_weights = self.attention(features) # (B, reduced_dim), (B, T)

        h = self.dropout(aggregated)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        output = self.fc_out(h)                             # (B, 1)

        return output, attn_weights

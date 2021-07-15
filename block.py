# Transformer Building block implementation
import torch.nn as nn
from attention import AttentionModel
from mlp import MLP

class BuildingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4.0, include_bias = True, dropout_p = 0.5, attention_p = 0.5):
        super(BuildingBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = AttentionModel(dim, num_heads, include_bias, attention_p, dropout_p)
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)

        self.hidden_features = int(dim * mlp_ratio)

        self.mlp = MLP(dim, self.hidden_features, dim)

    def forward(self, x):
        # x shape: (n_samples, n_patches + 1, dim)
        x = x + self.attention(self.norm1(x)) # x = x + [for Resedual Connection]
        x = x + self.mlp(self.norm2(x))
        return x
# Integrate the whole thing
import torch
import torch.nn as nn
from patch_embeddings import PatchEmbedding
from block import BuildingBlock

class VisionTransformer(nn.Module):
    def __init__(self, image_size=384, patch_size=16, input_channels=3, num_classes=1000, embedding_dims=768, depth=12, num_heads=12, mlp_ratio=4.0, include_bias = True, dropout_p = 0.5, attention_p = 0.5):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, input_channels, embedding_dims) # instance of patch embedding model
        self.cls = nn.Parameter(torch.zeros(1, 1, embedding_dims))
        self.positional_embeddings = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.num_patches, embedding_dims)) # to get the exact position of a given patch in the image
        self.pos_drop = nn.Dropout(dropout_p)

        self.blocks = nn.ModuleList([ BuildingBlock(embedding_dims, num_heads, mlp_ratio, include_bias, dropout_p, attention_p) for transformer in range(depth) ])

        self.norm = nn.LayerNorm(embedding_dims, eps=1e-6)
        self.head = nn.Linear(embedding_dims, num_classes)

    def forward(self, x):
        # x shape: (n_samples, in_channels, img_size, img_size)
        n_samples = x.shape[0]
        x = self.patch_embedding(x)
        cls = self.cls.expand(n_samples, -1, -1) # shape: (n_samples, 1, embedding_dims)
        x = torch.cat((cls, x), dim = 1) # Concatination -> shape(n_samples, 1 + n_patches, embedding_dims)
        x = x + self.positional_embeddings
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        cls_final = x[:, 0]

        x = self.head(cls_final)

        return x
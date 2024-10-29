import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: (batch_size, embed_dim, num_patches**0.5, num_patches**0.5)
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        qkv = self.qkv(x)  # Shape: (batch_size, num_patches, embed_dim * 3)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, batch_size, num_heads, num_patches, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # Each shape: (batch_size, num_heads, num_patches, head_dim)

        energy = torch.einsum("bnqd,bnkd->bnqk", q, k)  # Shape: (batch_size, num_heads, num_patches, num_patches)

        attention = torch.nn.functional.softmax(energy / self.scale, dim=-1)  # Shape: (batch_size, num_heads, num_patches, num_patches)

        out = torch.einsum("bnqk,bnvd->bnqd", attention, v)  # Shape: (batch_size, num_heads, num_patches, head_dim)

        out = out.reshape(batch_size, num_patches, embed_dim)  # Shape: (batch_size, num_patches, embed_dim)

        out = self.fc_out(out)  # Shape: (batch_size, num_patches, embed_dim)

        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT_Classifier(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=32, num_layers=3, num_heads=2, mlp_dim=64, num_classes=1, dropout=0.1):
        super(ViT_Classifier, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        for _ in range(num_layers)])

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.positional_embedding
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]
        out = self.fc(cls_token_final)

        # Sigmoid to map the quantum output to a probability between 0 and 1 # ONLY HQM
        out = torch.sigmoid(out) 

        return out

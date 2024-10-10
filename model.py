import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_classes=10, dim=512, depth=6, heads=8, mlp_dim=1024):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.Transformer(dim, heads, depth, depth, mlp_dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels, height // 8, 8, width // 8, 8)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(batch_size, (height // 8) * (width // 8), -1)
        x = self.patch_embedding(x)

        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.position_embedding

        x = self.transformer(x, x)
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)

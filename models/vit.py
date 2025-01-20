import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop_ratio: float = 0.,
                 proj_drop_ratio: float = 0.):
        """
        多头自注意力模块
        :param embed_dims: 输入的特征维度
        :param num_heads: 注意力头的数量
        :param qkv_bias: 是否添加偏置
        :param qk_scale: QK 缩放因子
        :param attn_drop_ratio: 注意力得分的丢弃概率
        :param proj_drop_ratio: 输出的丢弃概率
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims ** -0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # (batch_size, num_patches + 1, embed_dims)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dims)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, embed_dims: int, mlp_ratio: float = 4.0, drop_rate: float = 0.0):
        """
        多层感知机模块
        :param embed_dims: 输入和输出的特征维度
        :param mlp_ratio: 隐藏层维度与输入维度的放大比例
        :param drop_rate: 丢弃率
        """
        super().__init__()
        hidden_dims = int(embed_dims * mlp_ratio)
        self.fc1 = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dims, embed_dims)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        """
        Transformer Block 模块
        :param embed_dims: 输入的特征维度
        :param num_heads: 注意力头的数量
        :param mlp_ratio: MLP 的放大比例
        :param drop_path: 路径丢弃率
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = Attention(embed_dims, num_heads)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.mlp = MLP(embed_dims, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 15,
                 embed_dims: int = 384,
                 depth: int = 6,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0):
        """
        Vision Transformer 实现
        :param image_size: 输入图片大小
        :param patch_size: 划分 patch 的大小
        :param num_classes: 分类数量
        :param embed_dims: 嵌入的特征维度
        :param depth: Transformer Block 的层数
        :param num_heads: 注意力头的数量
        :param mlp_ratio: MLP 的放大比例
        :param drop_rate: 丢弃率
        """
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dims, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dims))

        self.blocks = nn.Sequential(
            *[Block(embed_dims, num_heads, mlp_ratio, drop_rate) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dims)
        self.cls_head = nn.Linear(embed_dims, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dims)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, num_patches + 1, embed_dims)
        x += self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # 取出 cls_token 对应的特征

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.cls_head(x)
        return x

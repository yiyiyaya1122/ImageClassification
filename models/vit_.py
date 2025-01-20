import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Attention(nn.Cell):
    def __init__(self, embed_dims, num_heads=12):
        """
        self.qkv: 对输入的 tokens 进行线性映射，将结果分解得到 q, k, v
        self.proj: 对 attention 运算的结果进行线性映射
        """
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads

        assert self.head_dims * self.num_heads == embed_dims

        # scale=1/sqrt(E//num_heads)
        self.scale = self.head_dims**-0.5
        # W^Q, W^K,W^V
        self.q_proj = nn.Dense(embed_dims, embed_dims, has_bias=False)
        self.k_proj = nn.Dense(embed_dims, embed_dims, has_bias=False)
        self.v_proj = nn.Dense(embed_dims, embed_dims, has_bias=False)
        # W^O
        self.out_proj = nn.Dense(embed_dims, embed_dims)

    def construct(self, x):
        """Multihead Attention
        Args:
            x (Tensor): shape (B, t, E)
        Returns:
            x (Tensor): shape (B, t, E)
        """
        B, t, E = x.shape
        C = E // self.num_heads
        # === TODO ===
        # ...
        return x


class Block(nn.Cell):
    def __init__(self, embed_dims, num_heads, mlp_ratio=4.0, qkv_bias=False):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm((embed_dims,))
        self.attn = Attention(embed_dims, num_heads)
        self.norm2 = nn.LayerNorm((embed_dims,))
        self.mlp = nn.SequentialCell(
            [
                nn.Dense(embed_dims, int(embed_dims * mlp_ratio)),
                nn.GELU(),
                nn.Dense(int(embed_dims * mlp_ratio), embed_dims),
            ]
        )

    def construct(self, x):
        """Transformer Block
        Args:
            x (Tensor): shape (B, t, E)
        Returns:
            x (Tensor): shape (B, t, E)
        """
        # === TODO ===
        # ...
        return x


class ViT(nn.Cell):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 15,
        embed_dims: int = 384,
        depth: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        """
        Args:
            image_size (int): 图片尺寸
            patch_size (int): 划分的 patch 大小
            num_classes (int): 分类数
            embed_dims (int): 嵌入维度
            depth (int): transformer 层数
            num_heads (int): 多头注意力的头数
            mlp_ratio (float): MLP 层对 embed_dims 的放大比例
        """
        super(ViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        # 图片 -> Patches
        self.patch_embed = nn.Conv2d(
            3, embed_dims, kernel_size=patch_size, stride=patch_size, has_bias=True
        )
        # Position Embedding
        self.pos_embed = ms.Parameter(ops.zeros((1, num_patches + 1, embed_dims)))
        # 用于分类的 cls_token
        self.cls_token = ms.Parameter(ops.zeros((1, 1, embed_dims)))
        # Transformer Blocks
        self.blocks = nn.SequentialCell(
            [Block(embed_dims, num_heads, mlp_ratio) for _ in range(depth)]
        )
        # 分类器
        self.cls_head = nn.Dense(embed_dims, num_classes)

    def construct(self, x):
        """Transformer Encoder
        Args:
            x (Tensor): [B, C, H, H], H 为图片大小
        Returns:
            logits (Tensor): [B, num_classes]
        """
        B = x.shape[0]
        # === TODO ===
        # ...
        return x

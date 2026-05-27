"""
Structure Enhancement Network (基于 Baseline 改进)

核心设计：
- StructureEnhancer放在Swin-Tiny的Stage1(blocks)之后、PatchMerging之前
- StructureEnhancer是SE-Net风格的通道注意力模块（只校准通道，不改变空间）
- Photo和SVG共用同一个网络骨干，无模态差异
- 仅使用Triplet Loss训练

Swin-Tiny前向流程（修复版）：
  patch_embed    → [B, 3136, 96]      (56×56 tokens)
  layers[0].blocks → [B, 3136, 96]    (Stage1 blocks，分辨率不变)
  [StructureEnhancer插入点]            (在PatchMerging之前，对96维特征做通道增强)
  layers[0].downsample → [B, 784, 192] (PatchMerging, 28×28)
  layers[1] → [B, 196, 384]          (14×14)
  layers[2] → [B, 49, 768]           (7×7)
  layers[3] → [B, 49, 768]          (7×7)
  PositionInvariantDescriptor → [B, embed_dim]
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import timm


# =============================================================================
# 工具函数
# =============================================================================

def pad_to_square(img: torch.Tensor) -> torch.Tensor:
    """
    输入：[C, H, W] float tensor，值域[0,1]
    输出：[C, S, S]，S=max(H,W)，白色(1.0)填充，居中
    """
    _, h, w = img.shape
    if h == w:
        return img
    s = max(h, w)
    padded = torch.ones(img.shape[0], s, s, dtype=img.dtype, device=img.device)
    top  = (s - h) // 2
    left = (s - w) // 2
    padded[:, top:top+h, left:left+w] = img
    return padded


# =============================================================================
# StructureEnhancer（SE-Net风格）
# =============================================================================

class StructureEnhancerSENet(nn.Module):
    """
    结构增强模块（SE-Net风格）

    核心思想：
    - 只校准通道权重，不改变空间结构
    - 用全局池化（gap）计算通道注意力
    - 残差增强：enhanced = feat * weight

    位置：放在Swin-Tiny的Stage1(blocks)之后、PatchMerging之前
    """

    def __init__(self, channels: int = 96, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W] 输入特征图
        Returns:
            enhanced: [B, C, H, W] 增强后的特征图
        """
        gap = feat.mean(dim=(2, 3))          # [B, C]
        weight = self.mlp(gap)                # [B, C]
        enhanced = feat * weight.unsqueeze(-1).unsqueeze(-1)
        return enhanced

    def get_sigmoid_weights(self, feat: torch.Tensor) -> torch.Tensor:
        """获取 sigmoid 输出（用于指标监控，无状态，可直接调用）"""
        gap = feat.mean(dim=(2, 3))
        return self.mlp(gap)  # [B, C], sigmoid 输出，值域 (0, 1)


# =============================================================================
# 位置无关描述符（修复版）
# =============================================================================

class PositionInvariantDescriptor(nn.Module):
    """
    位置无关描述符

    设计原则：
    - 只用全局统计量，不做任何空间分箱（spatial bin会破坏位置无关性）
    - mean: 受符号占比影响（占比越大越准确）
    - std: 几乎不受占比影响
    - max: 最鲁棒，占比变化时最强激活的channel值基本不变
    - 拼接后经MLP投影到embed_dim

    为什么不用 min：与 max 冗余，且 min 对背景噪声更敏感
    为什么不用 spatial bin：同一符号放在左上角和右下角会激活不同bin，
                           破坏位置无关性
    """

    def __init__(self, in_channels: int = 768, embed_dim: int = 64):
        super().__init__()
        # 3×in_channels → embed_dim，使用两层MLP增强表达能力
        self.proj = nn.Sequential(
            nn.Linear(in_channels * 3, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W] 特征图

        Returns:
            descriptor: [B, embed_dim] L2归一化的位置无关描述符
        """
        # mean: [B, C]
        mean = feat.mean(dim=(2, 3))
        # std: [B, C]
        std = feat.std(dim=(2, 3))
        # max: [B, C]，flatten后取每通道最大值
        max_val = feat.flatten(2).max(dim=2)[0]

        stats = torch.cat([mean, std, max_val], dim=1)   # [B, 3*C]
        out   = self.proj(stats)                          # [B, embed_dim]
        return F.normalize(out, p=2, dim=1)


# =============================================================================
# 完整网络
# =============================================================================

class SVGStructureEnhanceNet(nn.Module):
    """
    Structure Enhancement Network

    架构（修复版）：
    Photo → patch_embed → Stage1_blocks → StructureEnhancer → PatchMerging
            → Stage2/3/4 → PositionInvariantDescriptor → embedding

    关键设计：
    - StructureEnhancer放在Stage1(blocks)之后、PatchMerging之前
      （只有浅层特征才有足够的空间结构信息）
    - StructureEnhancer在推理时仍然生效，增强通道权重
    - Photo和SVG共用同一个网络骨干，无模态差异
    - PositionInvariantDescriptor放在最后，不破坏位置无关性
    """

    def __init__(
        self,
        embed_dim: int = 64,
        stage1_channels: int = 96,
        dropout: float = 0.1,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        use_pretrained: bool = True,
        pretrained_path: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.stage1_channels = stage1_channels

        # Swin-Tiny Backbone（global_pool="" 输出完整特征图）
        if pretrained_path:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                num_classes=0,
                global_pool='',
            )
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=False)
            logging.info(f"Loaded pretrained weights from: {pretrained_path}")
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=use_pretrained,
                num_classes=0,
                global_pool='',
            )

        self.backbone_out_channels = self.backbone.num_features  # 768

        # StructureEnhancer（SE-Net风格，放在Stage1(blocks)之后、PatchMerging之前）
        self.structure_enhancer = StructureEnhancerSENet(
            channels=stage1_channels,
            reduction=16,
        )

        # 位置无关描述符
        self.descriptor = PositionInvariantDescriptor(
            in_channels=self.backbone_out_channels,
            embed_dim=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单张图像 → embedding（直接返回 tensor，供 TripletNet 直接使用）

        Args:
            x: [B, 3, 224, 224] 输入图像

        Returns:
            embedding: [B, embed_dim] L2 归一化的 embedding
        """
        return self._to_embedding(x)

    def extract_stage1_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取 StructureEnhancer 的输入特征图。

        返回值形状：[B, C, H, W]，用于训练指标统计。
        """
        x = self.backbone.patch_embed(x)
        for blk in self.backbone.layers[0].blocks:
            x = blk(x)

        B, H, W, C = x.shape
        return x.permute(0, 3, 1, 2).contiguous()

    def _to_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        单张图像 → embedding

        前向流程：
            patch_embed → Stage1_blocks → StructureEnhancer → PatchMerging
            → Stage2/3/4 → PositionInvariantDescriptor → L2norm → [B, embed_dim]
        """
        x_2d = self.extract_stage1_feature_map(x)
        x_2d = self.structure_enhancer(x_2d)                # [B, 96, 56, 56]
        x = x_2d.permute(0, 2, 3, 1).contiguous()        # [B, 56, 56, 96]

        # PatchMerging: [B, 56, 56, 96] → [B, 28, 28, 192]
        x = self.backbone.layers[0].downsample(x)

        # Stage2, 3, 4
        x = self.backbone.layers[1](x)
        x = self.backbone.layers[2](x)
        x = self.backbone.layers[3](x)

        # Final: [B, 7, 7, 768] → [B, 768, 7, 7]
        final_feat = x.permute(0, 3, 1, 2)                 # [B, 768, 7, 7]

        # PositionInvariantDescriptor: [B, 768, 7, 7] → [B, embed_dim]
        emb = self.descriptor(final_feat)                   # [B, embed_dim]
        return emb


# =============================================================================
# TripletNet包装
# =============================================================================

class TripletNet(nn.Module):
    """三路共享网络的TripletNet（embedding_net.forward() 直接返回 tensor）"""

    def __init__(self, embedding_net: nn.Module):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        """返回三个 embedding tensor"""
        anchor_emb   = self.embedding_net(anchor)
        positive_emb = self.embedding_net(positive)
        negative_emb = self.embedding_net(negative)
        return anchor_emb, positive_emb, negative_emb

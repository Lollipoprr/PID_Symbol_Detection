from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TripletNet",
    "EmbeddingNet",
]


class TripletNet(nn.Module):
    def __init__(self, embedding_net: nn.Module):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, a, p, n):
        eA = self.embedding_net(a)
        eP = self.embedding_net(p)
        eN = self.embedding_net(n)
        return (
            F.pairwise_distance(eA, eP, p=2),
            F.pairwise_distance(eA, eN, p=2),
            eA,
            eP,
            eN,
        )


class EmbeddingNet(nn.Module):
    _DEFAULT_BACKBONE = "swin_tiny_patch4_window7_224"
    _DEFAULT_CKPT = Path(
        "/media/wit/SSD_5151/wxr/Long-CLIP/PID_Symbol_Detection/"
        "swin_tiny_patch4_window7_224"
    )
    _CANDIDATE_FILES = [
        "pytorch_model.bin",
        "model.bin",
        "model.pth",
        "model.ckpt",
        "model.safetensors",
    ]

    def __init__(
        self,
        embedding_size: int,
        backbone_name: str | None = None,
        checkpoint_path: str | Path | None = None,
        use_pretrained: bool = True,
        num_queries: int = 4,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        reduction: int = 16,
        sge_groups: int = 32,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        backbone_name = backbone_name or self._DEFAULT_BACKBONE
        checkpoint_path = Path(checkpoint_path or self._DEFAULT_CKPT)
        load_local_ckpt = use_pretrained and checkpoint_path.exists()

        self.backbone = timm.create_model(
            backbone_name,
            num_classes=0,
            global_pool="avg",
            pretrained=use_pretrained and not load_local_ckpt,
        )

        if load_local_ckpt:
            try:
                ckpt_path = self._resolve_ckpt_path(checkpoint_path)
                state_dict = self._load_state_dict(ckpt_path)
                if isinstance(state_dict, dict) and "model" in state_dict:
                    state_dict = state_dict["model"]
                missing, unexpected = self.backbone.load_state_dict(
                    state_dict, strict=False
                )
                if missing:
                    print(f"[EmbeddingNet] missing={len(missing)} e.g. {missing[:3]}")
                if unexpected:
                    print(f"[EmbeddingNet] unexpected={len(unexpected)} e.g. {unexpected[:3]}")
                print(f"[EmbeddingNet] loaded local weights: {ckpt_path}")
            except Exception as err:
                print(f"[EmbeddingNet] local load failed ({err}), fallback to timm pretrained")
        elif use_pretrained and not checkpoint_path.exists():
            print(f"[EmbeddingNet] warning: {checkpoint_path} not found, fallback to timm pretrained")

        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        return x

    def _resolve_ckpt_path(self, path: Path) -> Path:
        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(f"{path} is neither file nor directory")
        for fname in self._CANDIDATE_FILES:
            c = path / fname
            if c.is_file():
                return c
        raise FileNotFoundError(f"No checkpoint file found under {path}")

    def _load_state_dict(self, ckpt_path: Path):
        if ckpt_path.suffix == ".safetensors":
            from safetensors import safe_open

            sd = {}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
            return sd
        return torch.load(ckpt_path, map_location="cpu")

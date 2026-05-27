from pipeline.base import BasePipeline
from utils.bbox_utils import BBoxUtils
from utils.class_crops_builder import ClassCropsBuilder
from utils.stage2_trainer import FewShotTrainer
from utils.stage2_Siamese_Network import TripletNet, EmbeddingNet
from utils.stage2_triplets_generator import EpisodicTripletDatasetFromDir
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2


# ── pad到正方形（白色填充）────────────────────────────────────────────────────
def pad_to_square(img: torch.Tensor) -> torch.Tensor:
    """
    输入：[C, H, W] float tensor，值域[0,1]
    输出：[C, S, S]，S=max(H,W)，白色(1.0)填充，居中
    """
    _, h, w = img.shape
    if h == w:
        return img
    s      = max(h, w)
    padded = torch.ones(img.shape[0], s, s, dtype=img.dtype, device=img.device)
    top    = (s - h) // 2
    left   = (s - w) // 2
    padded[:, top:top+h, left:left+w] = img
    return padded


class Stage2BaselinePipeline(BasePipeline):
    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path)
        self.dataset_builder = None

    def validate(self) -> bool:
        paths       = self.get_data_paths()
        model_paths = self.get_model_paths()
        if not model_paths['stage1_class_agnostic_weights_dir'].exists():
            raise FileNotFoundError("Stage 1 model directory not found")
        if not paths['raw_images_dir'].exists() or \
           not paths['raw_class_aware_labels_dir'].exists():
            raise FileNotFoundError("Raw data directories not found")
        return True

    def prepare_symbol_crops(self) -> None:
        paths = self.get_data_paths()
        BBoxUtils.extract_bbox_crops(
            image_dir=paths['raw_images_dir'],
            label_dir=paths['raw_class_aware_labels_dir'],
            output_dir=paths['symbol_crops_dir']
        )

    def prepare_train_data(self) -> None:
        """Split symbol crops into train and val directories per class."""
        config = self.get_stage2_baseline_config()
        paths  = self.get_data_paths()

        train_val_split = tuple(
            float(x) for x in config.get("train_val_split", [0.8, 0.2])
        )

        self.dataset_builder = ClassCropsBuilder(
            crops_root_dir=paths['symbol_crops_dir']
        )
        self.dataset_builder.split_train_val(
            train_dir=paths['train_dir'],
            val_dir=paths['val_dir'],
            train_val_split=train_val_split,
            seed=42,
        )

    def train_model(self, resume: bool = False) -> None:
        paths       = self.get_data_paths()
        model_paths = self.get_model_paths()
        config      = self.get_stage2_baseline_config()

        # ── 模型 ──────────────────────────────────────────────────────────
        embedding_net = EmbeddingNet(
            embedding_size=config['embedding_size'],
            use_pretrained=True,
            num_queries=config.get('num_queries', 4),
            sge_groups=config.get('sge_groups', 32),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_net.to(device)

        for p in embedding_net.backbone.parameters():
            p.requires_grad = True

        model = TripletNet(embedding_net)
        model.to(device)

        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"[Stage2] 参数统计: trainable={trainable_params:,}/{total_params:,}")

        # ── 超参 ──────────────────────────────────────────────────────────
        train_val_split = [float(x) for x in config.get("train_val_split", [0.8, 0.2])]
        epochs      = int(config.get("epochs", 20))
        batch_size  = int(config.get("batch_size", 64))
        warmup_epochs      = int(config.get("warmup_epochs", 2))
        # SVG 跨域损失已禁用（用户已删除 stage2_svg 相关脚本）

        learning_rate = float(config.get("learning_rate", 0.00003))
        weight_decay = float(config.get("weight_decay", 0.05))
        criterion = nn.TripletMarginLoss(margin=config.get('margin', 1.0), p=2)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # ── Transform ─────────────────────────────────────────────────────
        # 真实图训练：只保留 pad_to_square + resize + normalize，不做额外增强
        base_transform = v2.Compose([
            v2.Lambda(pad_to_square),
            v2.Resize(size=(224, 224), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        apn_transform = base_transform

        # ── Triplet DataLoader ─────────────────────────────────────────────
        max_per_class = int(config.get("max_samples_per_class", 400))
        semi_hard_candidates = int(config.get("semi_hard_candidates", 10))

        # ── 打印关键超参 ────────────────────────────────────────────────────
        print("=" * 65)
        print("【Baseline 训练配置】")
        print(f"  batch_size        : {batch_size}")
        print(f"  learning_rate     : {learning_rate}")
        print(f"  weight_decay     : {weight_decay}")
        print(f"  epochs           : {epochs}")
        print(f"  warmup_epochs    : {warmup_epochs}")
        print(f"  margin           : {config.get('margin', 1.0)}")
        print(f"  max_per_class    : {max_per_class}")
        print(f"  semi_hard_candidates: {semi_hard_candidates}")
        print(f"  train_dir        : {paths['train_dir']}")
        print(f"  val_dir          : {paths['val_dir']}")
        print("=" * 65)

        train_dataset = EpisodicTripletDatasetFromDir(
            paths['train_dir'], apn_transform,
            max_per_class=max_per_class,
            semi_hard_candidates=semi_hard_candidates,
            pn_root_dir=paths['train_dir'],
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )

        # val_dataloader 从 val_dir 创建（不再是 episode 划分）
        val_dataloader = None
        if paths['val_dir'].exists():
            val_dataset = EpisodicTripletDatasetFromDir(
                paths['val_dir'], apn_transform,
                max_per_class=max_per_class,
                semi_hard_candidates=semi_hard_candidates,
                pn_root_dir=paths['val_dir'],
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False,
            )
        else:
            print("[Stage2] ⚠️ val_dir 不存在，val_dataloader 设为 None")

        # ── 真实val评估器 ──────────────────────────────────────────────────
        from utils.stage2_trainer import RealValEvaluator
        real_val_evaluator = RealValEvaluator(
            train_dir         = paths['train_dir'],
            val_dir           = paths['val_dir'],
            device            = device,
            max_per_class     = 20,
            prototype_temperature = config.get('prototype_temperature', 0.1),
            use_adaptive_temp     = config.get('prototype_adaptive_temp', True),
        )

        # ── 训练 ──────────────────────────────────────────────────────────
        output_dir = model_paths['stage2_weights_dir']
        trainer = FewShotTrainer(
            model, train_dataloader, val_dataloader, output_dir,
            warmup_epochs=warmup_epochs,
            real_val_evaluator=real_val_evaluator,
        )
        print(
            f"[Baseline] 开始训练: {epochs} epochs, "
            f"batch_size={batch_size}, max_per_class={max_per_class}, "
            f"semi_hard_candidates={semi_hard_candidates}"
        )
        trainer.train(epochs, criterion, optimizer,
                      base_lr=learning_rate, resume=resume)

    def run(self) -> None:
        if not self.validate():
            raise ValueError("Pipeline validation failed")
        print("Extracting symbol crops...")
        self.prepare_symbol_crops()
        print("Preparing training data...")
        self.prepare_train_data()
        print("Training baseline model...")
        self.train_model()
        print("Stage 2 baseline pipeline completed successfully")
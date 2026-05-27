import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Any, Optional, Literal, Union, Callable
from torchvision.io import decode_image
from torchvision.transforms import v2
from utils.stage2_triplets_generator import decode_image_from_path
from utils.helpers import get_image_files, resolve_image_paths
from utils.bbox_utils import BBoxUtils
from utils.stage2_Siamese_Network import EmbeddingNet, TripletNet
from utils.svg_helpers import pad_to_square_pil, EVAL_TRANSFORM as _EVAL_TRANSFORM
import cv2
from PIL import Image
import numpy as np
from pathlib import Path as _Path

# ════════════════════════════════════════════════════════════════════════════════════════
# compute_metric_support
# 支持 SVG 度量：提供 SVG 版本的 compute_svg_metric_support 封装
# ════════════════════════════════════════════════════════════════════════════════════════
def compute_metric_support(
    embeddings: torch.Tensor,
    prototype_matrix: torch.Tensor,
    distance_metric: str = "euclidean",
    metric_temperature: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    计算 query embedding 与各类原型的度量支持度

    Args:
        embeddings:        [B, D] 待匹配的 query embedding
        prototype_matrix:  [C, D] 类别原型，dim-0 顺序与 class_ids 一致
        distance_metric:   "euclidean" 或 "cosine"
        metric_temperature: softmax 温度参数

    Returns:
        Dict{'distance_like', 'metric_probs', 'nearest_distance',
        'distance_gap', 'nearest_index', 'metric_conf'}
    """
    temperature = max(float(metric_temperature), 1e-6)

    if distance_metric == "euclidean":
        distance_like = torch.norm(embeddings.unsqueeze(1) - prototype_matrix.unsqueeze(0), dim=2).squeeze(1)  # [B, C]
        metric_probs = F.softmax(-distance_like / temperature, dim=1)  # [B, C]
        sorted_values, sorted_indices = torch.sort(distance_like, dim=1, descending=False)
        nearest_distance = sorted_values[:, 0:1]  # [B, 1]
        gap_values = sorted_values[:, 1:] - sorted_values[:, :1]  # [B, C-1]
        distance_gap = gap_values[:, 0:1] if gap_values.shape[1] > 0 else torch.full_like(nearest_distance, float("inf"))
    else:  # cosine
        similarities = F.cosine_similarity(embeddings.unsqueeze(1), prototype_matrix.unsqueeze(0), dim=2)  # [B, C]
        metric_probs = F.softmax(similarities / temperature, dim=1)  # [B, C]
        sorted_values, sorted_indices = torch.sort(similarities, dim=1, descending=True)
        nearest_distance = (1.0 - sorted_values[:, 0:1]).clamp(min=0.0)  # [B, 1]
        gap_values = sorted_values[:, :1] - sorted_values[:, 1:]  # [B, C-1]
        distance_gap = gap_values[:, 0:1] if gap_values.shape[1] > 0 else torch.full_like(nearest_distance, float("inf"))
        distance_like = 1.0 - similarities

    nearest_index = sorted_indices[:, 0:1]  # [B, 1]
    metric_conf = metric_probs.gather(1, nearest_index)  # [B, 1]

    return {
        "distance_like": distance_like,
        "metric_probs": metric_probs,
        "nearest_distance": nearest_distance,
        "distance_gap": distance_gap,
        "nearest_index": nearest_index,
        "metric_conf": metric_conf,
    }

# ════════════════════════════════════════════════════════════════════════════════════════
# calibrate_known_support_thresholds
# 基于已知样本自适应校准各类阈值
# ════════════════════════════════════════════════════════════════════════════════════════
def calibrate_known_support_thresholds(
    proto_distances: List[float],
    proto_gaps: List[float],
    metric_confs: List[float],
    clf_confs: Optional[List[float]] = None,
    clf_margins: Optional[List[float]] = None,
    base_thresholds: Optional[Dict[str, Any]] = None,
    proto_distance_quantile: float = 0.95,
    proto_gap_quantile: float = 0.10,
    metric_conf_quantile: float = 0.10,
    clf_conf_quantile: float = 0.10,
    clf_margin_quantile: float = 0.10,
) -> Dict[str, Any]:
    """
    基于已知样本自适应校准各类阈值
    计算各类指标的百分位数作为自适应阈值

    Args:
        proto_distances:   各类原型距离列表
        proto_gaps:        各类原型 gap 列表
        metric_confs:      度量置信度列表（top-1 softmax prob）
        clf_confs:         分类器 top-1 置信度
        clf_margins:      分类器 margin 阈值
        base_thresholds:  基础阈值字典

    Returns:
        校准后的阈值字典
    """
    import numpy as _np

    _arr_dist = _np.array(proto_distances, dtype=np.float32)
    _arr_gaps = _np.array(proto_gaps, dtype=np.float32)
    _arr_conf = _np.array(metric_confs, dtype=np.float32)

    result = dict(base_thresholds) if base_thresholds else {}

    if len(_arr_dist) >= 2:
        result["proto_distance_thresh"] = float(_np.percentile(_arr_dist, proto_distance_quantile * 100))
    if len(_arr_gaps) >= 2:
        result["proto_gap_thresh"] = float(_np.percentile(_arr_gaps, proto_gap_quantile * 100))
    if len(_arr_conf) >= 2:
        result["metric_conf_thresh"] = float(_np.percentile(_arr_conf, metric_conf_quantile * 100))

    if clf_confs is not None:
        _arr_clf_conf = _np.array(clf_confs, dtype=np.float32)
        if len(_arr_clf_conf) >= 2:
            result["clf_conf_thresh"] = float(_np.percentile(_arr_clf_conf, clf_conf_quantile * 100))

    if clf_margins is not None:
        _arr_clf_margin = _np.array(clf_margins, dtype=np.float32)
        if len(_arr_clf_margin) >= 2:
            result["clf_margin_thresh"] = float(_np.percentile(_arr_clf_margin, clf_margin_quantile * 100))

    return result

# ════════════════════════════════════════════════════════════════════════════════════════
# pad_to_square_tensor
# 来自 stage2_baseline_train.py 的 pad_to_square 函数
#   输入[C, H, W] float tensor，范围[0,1]
#   输出[C, S, S]，S=max(H,W)，空白处填充(1.0)
# ════════════════════════════════════════════════════════════════════════════════════════
def pad_to_square_tensor(img: torch.Tensor) -> torch.Tensor:
    """将 [C, H, W] tensor pad 为正方形，保持原始内容"""
    _, h, w = img.shape
    if h == w:
        return img
    s      = max(h, w)
    padded = torch.ones(img.shape[0], s, s, dtype=img.dtype, device=img.device)
    top    = (s - h) // 2
    left   = (s - w) // 2
    padded[:, top:top+h, left:left+w] = img
    return padded

# 与 apn_transform 相同的图像预处理流程
# 输入：pad_to_square → Resize(224) → Normalize → 输出：标准化张量
# 输入：pad_to_square → Resize(224) → Normalize
_INFERENCE_TRANSFORM = v2.Compose([
    v2.Lambda(pad_to_square_tensor),
    v2.Resize(size=(224, 224), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class EmbeddingClassifier:
    def __init__(
        self,
        model_path: Union[Path, str],
        transform: Optional[Callable] = None,
        model_type: Literal["disentangle", "triplet"] = "disentangle",
    ) -> None:
        """
        Initialize the Few-Shot Class Predictor.

        加载 TripletNet 推理模型，兼容常见 checkpoint 结构。

        Args:
            model_path: Path to the pre-trained model
            transform: 可选 transform，若为 None 则使用默认_INFERENCE_TRANSFORM
            model_type: 强制指定模型类型 ("disentangle"=新架构, "triplet"=旧架构)
        """
        # 检测 CUDA 是否真的可用（is_available 可能返回 True 但实际不可用）
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                _t = torch.zeros(1, device="cuda")
                del _t
                self.device = torch.device("cuda")
            except Exception:
                self.device = torch.device("cpu")

        self.model_type = model_type
        self._is_new_arch = False
        self._is_svg_ca_model = False

        loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        print("[EmbeddingClassifier] 加载标准 TripletNet 模型")
        embedding_net = EmbeddingNet(embedding_size=64, sge_groups=32)
        self.fewshot_model = TripletNet(embedding_net)

        state_to_load = loaded
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_to_load = loaded["model_state_dict"]

        if isinstance(state_to_load, dict):
            keys = list(state_to_load.keys())
            has_prefixed = any(k.startswith("embedding_net.") for k in keys)
            if has_prefixed:
                load_state = state_to_load
            else:
                load_state = {f"embedding_net.{k}": v for k, v in state_to_load.items()}
            self.fewshot_model.load_state_dict(load_state, strict=False)

        self.fewshot_model.to(self.device)
        self.fewshot_model.eval()

        # 对于SVG-CA模型，直接使用fewshot_model
        if not hasattr(self, '_is_svg_ca_model') or not self._is_svg_ca_model:
            self.embedding_model = self.fewshot_model.embedding_net
        else:
            self.embedding_model = self.fewshot_model
        self.embedding_model.eval()

        self._svg_centers_matrix: Optional[torch.Tensor] = None

        # 处理transform：包含Normalize的pad+resize预处理
        # 如果提供了自定义transform则使用它
        self.transform = transform
        self.logger = logging.getLogger(__name__)

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """执行 pad_to_square → Resize(224) → Normalize

        Args:
            img: [C, H, W] float tensor，范围[0,1]，用于pad/resize

        Returns:
            [C, 224, 224] 标准化后的tensor
        """
        return _INFERENCE_TRANSFORM(img)

    def extract_embedding(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """提取图像 embedding，仅返回标准化 raw embedding。"""
        with torch.no_grad():
            raw_emb = self.embedding_model(image)  # [B, D]
            raw_emb = F.normalize(raw_emb, p=2, dim=1)
            return raw_emb

    def _compute_embedding_from_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.embedding_model(image)
        return embedding

    def _compute_embedding_from_image_path(self, image_path: str) -> torch.Tensor:
        """从图像路径计算 embedding

        调用 compute_class_prototypes 之前必须先调用此方法
        支持 decode 和 float/255 的 pad_to_square → Resize(224) → Normalize
        用于训练图的批量特征提取
        """
        image = decode_image_from_path(image_path)  # [C, H, W] uint8
        image = image.float() / 255.0               # [C, H, W] float [0,1]
        image = self._preprocess(image)             # [C, 224, 224] 标准化
        image = image.unsqueeze(0).to(self.device)  # [1, C, 224, 224]
        with torch.no_grad():
            embedding = self.embedding_model(image)
        return embedding

    def _compute_embeddings_batch(self, image_paths: List[str]) -> torch.Tensor:
        """批量计算图像 embedding，batch_size 自适应调整

        Args:
            image_paths: 图像路径列表

        Returns:
            [N, D] embedding张量
        """
        batch_size = getattr(self, "batch_size", 64)
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[i:i + batch_size]
            tensors = []
            for path in chunk_paths:
                image = decode_image_from_path(path)
                image = image.float() / 255.0
                image = self._preprocess(image)
                tensors.append(image)
            batch = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                raw_embs = self.embedding_model(batch)  # [B, D]
                raw_embs = F.normalize(raw_embs, p=2, dim=1)

                embeddings = raw_embs

            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def adaptive_temperature(self, n_samples: int, base_temp: float = 0.1) -> float:
        """根据样本数量自适应调整温度参数"""
        if n_samples < 200:
            return base_temp * 3.0
        elif n_samples < 1000:
            return base_temp * 1.5
        else:
            return base_temp

    def compute_weighted_class_prototypes(
        self,
        train_dir: Path,
        base_temperature: float = 0.1,
        use_adaptive_temp: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        计算加权类别原型

        Step 1 — 加权采样：1/sqrt(N) 先验权重
                   即 权重~0.1，增强少数样本的影响力
        Step 2 — 相似度加权：0.3温度的cosine softmax
                   即 利用相似度重新加权，增强代表性样本
        Step 3 — 加权平均：计算最终原型

        同时计算 P99 阈值存入 self.class_radii
        供 predict_class_from_prototypes 拒识使用

        Args:
            train_dir:       训练图目录，子文件夹名为class_id
            base_temperature:  基础温度参数（默认0.3）
            use_adaptive_temp: 是否启用自适应温度（默认启用）

        Returns:
            Dict[class_id -> prototype_tensor [1, D]]
        """
        train_dir = Path(train_dir)
        if not train_dir.exists() or not train_dir.is_dir():
            raise ValueError(f"Invalid training directory: {train_dir}")

        class_prototypes_dict = {}
        # 全局统计用于计算 P99 阈值
        all_intra_distances: List[float] = []
        # 存储每类的 P99 阈值
        self.class_radii: Dict[int, float] = {}
        # 全局能量分数收集用于 P99 计算
        all_energy_scores: List[float] = []
        # 存储 embedding 用于后续能量计算
        _support_embeddings: Dict[int, torch.Tensor] = {}  # class_id -> [N, D]

        for class_folder in train_dir.iterdir():
            if not class_folder.is_dir():
                continue
            class_label = int(class_folder.name)
            image_paths = get_image_files(class_folder)
            if not image_paths:
                self.logger.warning(f"No images found in {class_folder}. Skipping.")
                continue

            all_embeddings = self._compute_embeddings_batch(image_paths)  # [N, D] raw
            all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)  # normalize for distance consistency
            n_samples = all_embeddings.shape[0]

            class_mean = all_embeddings.mean(dim=0, keepdim=True)   # [1, D]
            class_mean = torch.nn.functional.normalize(class_mean, p=2, dim=1)

            # Step 1: 1/sqrt(N) 先验 + 加权采样
            prior_weights = torch.ones(n_samples)
            prior_weights = prior_weights / prior_weights.sum()
            inv_sqrt_n = n_samples ** 0.5
            prior_weights = prior_weights / inv_sqrt_n
            prior_weights = prior_weights / prior_weights.sum()

            # Step 2: 相似度加权 + cosine softmax概率
            # 【修复】温度从固定0.3改为自适应：
            #   - tail类(N<200): T=0.5，保护少数样本不被极端权重压垮
            #   - mid类(N<1000): T=0.3，平衡
            #   - head类(N>=1000): T=0.1，精确区分高置信样本
            # 原固定T=0.3过于aggressive，softmax后最大权重趋近1，
            # 与prior_weights相乘后少数样本主导，导致tail类原型质量下降
            if use_adaptive_temp:
                temperature = self.adaptive_temperature(n_samples, base_temperature)
            else:
                temperature = 0.5  # 温和温度，平衡头部和尾部类
            similarities = (all_embeddings @ class_mean.T).squeeze(1)     # [N]
            sim_weights = torch.softmax(similarities / temperature, dim=0)    # [N]

            # Step 3: 加权平均
            combined_weights = prior_weights * sim_weights
            combined_weights = combined_weights / combined_weights.sum()

            weighted_mean = (all_embeddings * combined_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
            weighted_mean = torch.nn.functional.normalize(weighted_mean, p=2, dim=1)

            class_prototypes_dict[class_label] = weighted_mean.detach().to(self.device)

            # 计算类内距离用于确定各类的 P90/P95/P99 阈值
            # embedding 已归一化，与推理时保持一致，阈值落在 [0, √2] 范围内
            proto_cpu = weighted_mean.cpu()  # [1, D] 单位球面
            intra_dists = torch.cdist(all_embeddings, proto_cpu).squeeze(1)  # [N]
            intra_dists_np = intra_dists.numpy()
            all_intra_distances.extend(intra_dists_np.tolist())

            # 计算各类 P99（类内距离的99分位数）
            if n_samples >= 10:
                class_p99 = float(np.percentile(intra_dists_np, 99))
            else:
                # 样本不足时使用 max*1.5 作为 P99 的粗略估计
                class_p99 = float(intra_dists_np.max()) * 1.5
            self.class_radii[class_label] = class_p99
            _support_embeddings[class_label] = all_embeddings.cpu()

        # 全局 P99：所有类内距离的 99 分位数
        # 即 99% 的类内样本距离 < 该值
        # 当某类 P99 > 全局 P99 时说明该类内样本较分散
        import numpy as _np
        all_intra_arr = _np.array(all_intra_distances)
        self.global_p90 = float(_np.percentile(all_intra_arr, 90))
        self.global_p95 = float(_np.percentile(all_intra_arr, 95))
        self.global_p99 = float(_np.percentile(all_intra_arr, 99))

        # 计算能量分数（计划文件公式：E = -T · log(1 - max_sim + ε)）
        # 相似度高 → 能量高 → 已知类
        _T = 0.1  # 使用较小的温度使能量分布更集中
        _proto_matrix = torch.cat([p.cpu() for _, p in sorted(class_prototypes_dict.items())], dim=0)  # [C, D]
        # 对每个支持集样本计算其能量分数
        for class_label, sup_embs in _support_embeddings.items():
            sup_embs_n = F.normalize(sup_embs, p=2, dim=1)  # [N, D]
            # 计算与所有原型的相似度
            sims = sup_embs_n @ _proto_matrix.T  # [N, C]
            max_sims, _ = sims.max(dim=1)  # [N], 取最大相似度
            # 能量分数：E = -T · log(1 - max_sim + ε)
            eps = 1e-6
            energy = -_T * torch.log(1 - max_sims + eps)  # [N]
            all_energy_scores.extend(energy.tolist())

        self._energy_T = _T
        _energy_arr = _np.array(all_energy_scores)
        self.energy_p90 = float(_np.percentile(_energy_arr, 90))
        self.energy_p95 = float(_np.percentile(_energy_arr, 95))
        self.energy_p99 = float(_np.percentile(_energy_arr, 99))
        # 设置能量阈值：使用 P95，即 95% 置信度
        self.energy_threshold = self.energy_p95

        self.logger.info(
            f"[compute_weighted_class_prototypes] 计算完成，共 {len(class_prototypes_dict)} 个类别\n"
            f"  先验权重=1/sqrt(N) + 相似度温度=0.3\n"
            f"  类内距离：P90={self.global_p90:.4f}  "
            f"P95={self.global_p95:.4f}  P99={self.global_p99:.4f}\n"
            f"  能量分数(T={_T:.4f})：P90={self.energy_p90:.4f}  "
            f"P95={self.energy_p95:.4f}  P99={self.energy_p99:.4f}\n"
            f"  能量阈值(用P95)= {self.energy_threshold:.4f}  用于 bbox 拒识"
        )
        return class_prototypes_dict

    def compute_class_prototypes(self, train_dir: Path) -> Dict[int, torch.Tensor]:
        """
        计算类别原型，默认使用 prior=1/sqrt(N) + 自适应温度
        同时计算 P99 阈值存入 self.global_p99
        """
        return self.compute_weighted_class_prototypes(
            train_dir,
            base_temperature=0.1,
            use_adaptive_temp=True,
        )

    def compute_simple_class_prototypes(
        self,
        train_dir: Path,
    ) -> Dict[int, torch.Tensor]:
        """
        计算简单平均类别原型（消融实验用：不使用加权）

        与 compute_weighted_class_prototypes 的区别：
          - 无 1/sqrt(N) 先验权重采样
          - 无相似度温度 softmax 加权
          - 直接对同类所有样本 embedding 求算术平均

        同时计算 P99 阈值存入 self.class_radii / self.global_p99
        （与加权版本保持一致的阈值统计结构，供拒识使用）

        Args:
            train_dir: 训练图目录，子文件夹名为 class_id

        Returns:
            Dict[class_id -> prototype_tensor [1, D]]
        """
        train_dir = Path(train_dir)
        if not train_dir.exists() or not train_dir.is_dir():
            raise ValueError(f"Invalid training directory: {train_dir}")

        class_prototypes_dict: Dict[int, torch.Tensor] = {}
        all_intra_distances: List[float] = []
        self.class_radii: Dict[int, float] = {}
        all_energy_scores: List[float] = []
        _support_embeddings: Dict[int, torch.Tensor] = {}

        for class_folder in train_dir.iterdir():
            if not class_folder.is_dir():
                continue
            class_label = int(class_folder.name)
            image_paths = get_image_files(class_folder)
            if not image_paths:
                self.logger.warning(f"No images found in {class_folder}. Skipping.")
                continue

            all_embeddings = self._compute_embeddings_batch(image_paths)  # [N, D] raw
            all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)  # normalize for distance consistency
            n_samples = all_embeddings.shape[0]

            # 简单平均：直接算术平均，不做任何加权
            class_mean = all_embeddings.mean(dim=0, keepdim=True)  # [1, D]
            class_mean = torch.nn.functional.normalize(class_mean, p=2, dim=1)

            class_prototypes_dict[class_label] = class_mean.detach().to(self.device)

            # 计算类内距离（与加权版本保持一致，用于阈值）
            # embedding 已归一化，与推理时保持一致，阈值落在 [0, √2] 范围内
            proto_cpu = class_mean.cpu()  # [1, D] 单位球面
            intra_dists = torch.cdist(all_embeddings, proto_cpu).squeeze(1)  # [N]
            intra_dists_np = intra_dists.numpy()
            all_intra_distances.extend(intra_dists_np.tolist())

            if n_samples >= 10:
                class_p99 = float(np.percentile(intra_dists_np, 99))
            else:
                class_p99 = float(intra_dists_np.max()) * 1.5
            self.class_radii[class_label] = class_p99
            _support_embeddings[class_label] = all_embeddings.cpu()

        import numpy as _np
        all_intra_arr = _np.array(all_intra_distances)
        self.global_p90 = float(_np.percentile(all_intra_arr, 90))
        self.global_p95 = float(_np.percentile(all_intra_arr, 95))
        self.global_p99 = float(_np.percentile(all_intra_arr, 99))

        # 能量分数（计划文件公式：E = -T · log(1 - max_sim + ε)）
        _T = 0.1  # 与加权版本保持一致
        _proto_matrix = torch.cat(
            [p.cpu() for _, p in sorted(class_prototypes_dict.items())], dim=0
        )
        for class_label, sup_embs in _support_embeddings.items():
            sup_embs_n = F.normalize(sup_embs, p=2, dim=1)
            # 计算与所有原型的相似度
            sims = sup_embs_n @ _proto_matrix.T  # [N, C]
            max_sims, _ = sims.max(dim=1)  # [N], 取最大相似度
            # 能量分数：E = -T · log(1 - max_sim + ε)
            eps = 1e-6
            energy = -_T * torch.log(1 - max_sims + eps)  # [N]
            all_energy_scores.extend(energy.tolist())

        self._energy_T = _T
        _energy_arr = _np.array(all_energy_scores)
        self.energy_p90 = float(_np.percentile(_energy_arr, 90))
        self.energy_p95 = float(_np.percentile(_energy_arr, 95))
        self.energy_p99 = float(_np.percentile(_energy_arr, 99))
        self.energy_threshold = self.energy_p95

        self.logger.info(
            f"[compute_simple_class_prototypes] 计算完成（简单平均，无加权），"
            f"共 {len(class_prototypes_dict)} 个类别\n"
            f"  类内距离：P90={self.global_p90:.4f}  "
            f"P95={self.global_p95:.4f}  P99={self.global_p99:.4f}\n"
            f"  能量分数(T={_T:.4f})：P90={self.energy_p90:.4f}  "
            f"P95={self.energy_p95:.4f}  P99={self.energy_p99:.4f}"
        )
        return class_prototypes_dict

    def predict_class_from_prototypes(
        self,
        img_src: Union[str, Path, List[str], List[Path]],
        labels_dir: Union[str, Path],
        class_prototypes_dict: Dict[int, torch.Tensor],
        output_dir: Path,
        distance_metric: Literal["euclidean", "cosine"] = "euclidean",
        rejection_threshold: Optional[float] = None,
        rejection_mode: Literal["distance_only"] = "distance_only",
        save_detailed_results: bool = True,
    ):
        """
        基于 stage1 输出的 class-agnostic bbox 进行类别预测（最近邻 + 欧氏距离）。

        开集检测策略：distance > rejection_threshold → 拒识（Unknown）

        Args:
            img_src:               图像目录或图像路径列表
            labels_dir:            stage1 输出的 class-agnostic 标注目录
            class_prototypes_dict: 类别原型字典（来自 compute_class_prototypes）
            output_dir:            预测结果输出目录
            rejection_threshold:    拒识阈值（默认 0.25），distance > threshold 时拒识
            save_detailed_results: 是否保存详细结果

        拒识的 bbox 使用 class_id = -1

        Returns:
            每张图像的 predicted_labels 列表
        """
        _threshold_value = float('inf')
        if rejection_threshold is not None:
            _threshold_value = float(rejection_threshold)
        elif hasattr(self, 'global_p99') and self.global_p99 is not None:
            _threshold_value = float(self.global_p99)
        else:
            _threshold_value = 0.25
        self.logger.info(
            f"[模式] mode={rejection_mode}：distance>{_threshold_value:.4f}时拒识"
        )

        # 统计变量
        total_bboxes    = 0
        rejected_bboxes = 0
        predicted_labels = []   # 所有图像的预测结果
        self._diag_distances: List[float] = []  # 用于诊断的距离列表
        _detailed_results: List[Dict] = []  # 用于详细结果输出

        # 获取图像路径列表
        if isinstance(img_src, (list, tuple)):
            image_paths = [Path(p) for p in img_src]
        else:
            image_paths = get_image_files(img_src)

        if not image_paths:
            self.logger.error(f"No images found in {img_src}.")
            return

        for image_path in image_paths:
            self.logger.info(f"Processing image: {image_path}")
            if not Path(image_path).exists():
                self.logger.error(f"Image not found: {image_path}")
                continue

            label_path  = Path(labels_dir) / (Path(image_path).stem + ".txt")
            output_path = Path(output_dir) / (Path(image_path).stem + ".txt")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(image_path))
            image_height, image_width = image.shape[0], image.shape[1]

            try:
                predicted_labels = []
                yolo_preds = BBoxUtils.get_bboxes_array_from_file(label_path)

                for line in yolo_preds:
                    total_bboxes += 1
                    class_id, xc, yc, w, h = line[:5]
                    x_min = int((xc - w / 2) * image_width)
                    y_min = int((yc - h / 2) * image_height)
                    x_max = int((xc + w / 2) * image_width)
                    y_max = int((yc + h / 2) * image_height)

                    cropped_image = BBoxUtils.crop_image(image, x_min, y_min, x_max, y_max)
                    if cropped_image is None or cropped_image.size == 0:
                        total_bboxes -= 1
                        continue

                    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                    cropped_tensor = torch.from_numpy(
                        np.array(pil_image)
                    ).permute(2, 0, 1).float() / 255.0

                    cropped_tensor = self._preprocess(cropped_tensor)
                    cropped_tensor = cropped_tensor.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        raw_emb = self.embedding_model(cropped_tensor)
                    raw_emb = F.normalize(raw_emb, p=2, dim=1)

                    object_embedding = raw_emb

                    if distance_metric == "euclidean":
                        all_distances: List[Tuple[int, float]] = []
                        for cls_id, prototype in class_prototypes_dict.items():
                            proto_2d = prototype.view(1, -1)
                            query_2d = object_embedding.view(1, -1)
                            distance = torch.cdist(query_2d, proto_2d).item()
                            all_distances.append((int(cls_id), distance))

                        all_distances.sort(key=lambda x: x[1])
                        min_distance = all_distances[0][1]
                        predicted_class = all_distances[0][0]
                        self._diag_distances.append(float(min_distance))

                        reject = min_distance > _threshold_value

                        if reject:
                            rejected_bboxes += 1
                            predicted_labels.append(
                                f"-1 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                            )
                        else:
                            predicted_labels.append(
                                f"{predicted_class} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                            )

                        if save_detailed_results:
                            _detailed_results.append({
                                'image_path': str(image_path),
                                'image_name': Path(image_path).name,
                                'predicted_class': -1 if reject else int(predicted_class),
                                'true_class': -1,
                                'distance': float(min_distance),
                                'is_unknown': reject,
                                'embedding': object_embedding.squeeze(0).cpu().tolist(),
                                'top3': [(int(cls_id), float(dist)) for cls_id, dist in all_distances[:3]],
                                'proto_distances': {int(cid): float(d) for cid, d in all_distances},
                                'bbox': {'xc': float(xc), 'yc': float(yc), 'w': float(w), 'h': float(h)},
                            })
                    elif distance_metric == "cosine":
                        all_similarities: List[Tuple[int, float]] = []
                        for cls_id, prototype in class_prototypes_dict.items():
                            proto_2d = prototype.view(1, -1)
                            query_2d = object_embedding.view(1, -1)
                            similarity = torch.nn.functional.cosine_similarity(
                                query_2d, proto_2d
                            ).item()
                            all_similarities.append((int(cls_id), similarity))

                        all_similarities.sort(key=lambda x: x[1], reverse=True)
                        max_similarity = all_similarities[0][1]
                        predicted_class = all_similarities[0][0]
                        cosine_distance = 1.0 - max_similarity
                        self._diag_distances.append(float(cosine_distance))

                        reject = cosine_distance > _threshold_value

                        if reject:
                            rejected_bboxes += 1
                            predicted_labels.append(
                                f"-1 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                            )
                        else:
                            predicted_labels.append(
                                f"{predicted_class} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                            )

                        if save_detailed_results:
                            _detailed_results.append({
                                'image_path': str(image_path),
                                'image_name': Path(image_path).name,
                                'predicted_class': -1 if reject else int(predicted_class),
                                'true_class': -1,
                                'distance': float(cosine_distance),
                                'is_unknown': reject,
                                'embedding': object_embedding.squeeze(0).cpu().tolist(),
                                'top3': [(int(cls_id), float(sim)) for cls_id, sim in all_similarities[:3]],
                                'proto_distances': {int(cid): float(1.0 - sim) for cid, sim in all_similarities},
                                'bbox': {'xc': float(xc), 'yc': float(yc), 'w': float(w), 'h': float(h)},
                            })

                with open(output_path, 'w') as f:
                    for label in predicted_labels:
                        f.write(label + '\n')
                self.logger.info(f"Predicted labels written to: {output_path}")

            except FileNotFoundError:
                self.logger.error(f"Label file not found: {label_path}")
                continue

        if hasattr(self, "_diag_distances") and self._diag_distances:
            import numpy as np_module
            arr = np_module.array(self._diag_distances)
            self.logger.info(
                f"[诊断] 距离统计（共 {len(arr)} bbox）：\n"
                f"  min={arr.min():.4f}  P10={np_module.percentile(arr, 10):.4f}  "
                f"P25={np_module.percentile(arr, 25):.4f}\n"
                f"  P50={np_module.percentile(arr, 50):.4f}  P75={np_module.percentile(arr, 75):.4f}\n"
                f"  P90={np_module.percentile(arr, 90):.4f}  P95={np_module.percentile(arr, 95):.4f}\n"
                f"  P99={np_module.percentile(arr, 99):.4f}  max={arr.max():.4f}\n"
                f"  阈值：distance_only, threshold={_threshold_value:.4f}"
            )

        # 输出最终统计
        if total_bboxes > 0:
            reject_rate = rejected_bboxes / total_bboxes * 100
            self.logger.info(
                f"\n[总结]\n"
                f"  总 bbox 数：        {total_bboxes}\n"
                f"  拒识 bbox (id=-1)：{rejected_bboxes}  ({reject_rate:.1f}%)\n"
                f"  接受 bbox：         {total_bboxes - rejected_bboxes}\n"
                f"  拒识模式：         {rejection_mode}\n"
                f"  阈值设置：     distance_only(dist>{_threshold_value:.4f})"
            )

        # 保存详细结果到 CSV 和 JSON
        if save_detailed_results and _detailed_results:
            from datetime import datetime
            import json as _json

            # 按类别整理 prototype 矩阵供 embedding 质量评估使用
            sorted_class_ids = sorted(class_prototypes_dict.keys())
            prototype_matrix = torch.cat(
                [class_prototypes_dict[cid] for cid in sorted_class_ids], dim=0
            ).cpu().tolist()

            # 创建输出目录
            output_dir_path = Path(output_dir)
            detailed_dir = output_dir_path.parent / 'detailed_results'
            detailed_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 1. 保存 JSON
            json_path = detailed_dir / f'predictions_{timestamp}.json'
            # 保存完整结果（包含 bbox 用于评估）
            summary = {
                'total': len(_detailed_results),
                'unknown_count': sum(1 for r in _detailed_results if r['is_unknown']),
                'known_count': sum(1 for r in _detailed_results if not r['is_unknown']),
                'rejection_mode': rejection_mode,
                'rejection_threshold': _threshold_value,
                'distance_metric': distance_metric,
                'class_ids': sorted_class_ids,
                'prototypes': prototype_matrix,
                'embedding_dim': len(prototype_matrix[0]) if prototype_matrix else 0,
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                _json.dump({'summary': summary, 'predictions': _detailed_results}, f, indent=2, ensure_ascii=False)
            self.logger.info(f"[详细结果] JSON 已保存: {json_path}")

            # 2. 保存 CSV
            csv_path = detailed_dir / f'predictions_{timestamp}.csv'
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("image_name,predicted_class,distance,gap,is_unknown,top3\n")
                for r in _detailed_results:
                    top3_str = ";".join([f"cls{c[0]}:{c[1]:.4f}" for c in r.get('top3', [])])
                    gap_val = float(r.get('gap', 0.0))
                    f.write(
                        f"{r['image_name']},{r['predicted_class']},"
                        f"{r['distance']:.4f},{gap_val:.4f},{r['is_unknown']},{top3_str}\n"
                    )
        return predicted_labels

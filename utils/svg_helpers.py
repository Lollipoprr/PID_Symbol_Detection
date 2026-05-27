"""SVG rendering utilities shared across modules."""
import io
from pathlib import Path

from PIL import Image
import cairosvg
import torch
import torch.nn.functional as F
from torchvision.transforms import v2


def pad_to_square_pil(img: Image.Image, fill: int = 255) -> Image.Image:
    """PIL 图像 pad 到正方形（白色填充，居中）。"""
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    new_img = Image.new("RGB", (s, s), (fill, fill, fill))
    new_img.paste(img, ((s - w) // 2, (s - h) // 2))
    return new_img


def pad_to_square_tensor(img: torch.Tensor) -> torch.Tensor:
    """[C,H,W] float tensor pad 到正方形，白色填充，居中。"""
    _, h, w = img.shape
    if h == w:
        return img
    s = max(h, w)
    padded = torch.ones(img.shape[0], s, s, dtype=img.dtype, device=img.device)
    top  = (s - h) // 2
    left = (s - w) // 2
    padded[:, top:top+h, left:left+w] = img
    return padded


EVAL_TRANSFORM = v2.Compose([
    v2.Lambda(pad_to_square_tensor),
    v2.Resize(size=(224, 224), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def render_svg_with_bg(svg_path: Path, out_size: int = 224) -> Image.Image:
    """将 SVG 渲染为 PNG 图像并添加白色背景。

    透明区域填充白色，确保符号主体正确显示。
    """
    png_data = cairosvg.svg2png(
        url=str(svg_path),
        output_width=out_size,
        output_height=out_size,
    )
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("RGB")
    return img


def encode_image(
    img: Image.Image,
    model: torch.nn.Module,
    device: str,
    transform=None,
) -> torch.Tensor:
    """将单张 PIL 图像编码为 embedding（L2 归一化）。"""
    if transform is None:
        transform = EVAL_TRANSFORM
    img_padded = pad_to_square_pil(img).resize((224, 224))
    t = torch.from_numpy(
        __import__("numpy").array(img_padded)
    ).permute(2, 0, 1).float() / 255.0
    t = transform(t).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model(t).squeeze(0)
    return F.normalize(e.unsqueeze(0), p=2, dim=1).squeeze(0)


# ── Class map utilities (migrated from svg_guided_trainer.py) ─────────────────

def load_class_map(path: str | Path | None = None) -> dict:
    """加载 {class_id: svg_name} 映射表。"""
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "class_svg_map.json"
    else:
        path = Path(path)

    if not path.exists():
        return {}

    import json
    with open(path) as f:
        raw = json.load(f)

    result = {}
    for k, v in raw.items():
        try:
            result[int(k)] = str(v)
        except (ValueError, TypeError):
            pass
    return result


def is_svg_png_dir(path: Path) -> bool:
    """判断目录是否为 svg_to_png/{class_id}/ 格式。"""
    if not path.is_dir():
        return False
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        return False
    return all(_is_valid_class_id(d.name) for d in subdirs)


def _is_valid_class_id(name: str) -> bool:
    """判断子目录名是否为合法的 class_id（整数）。"""
    try:
        int(name)
        return True
    except ValueError:
        return False

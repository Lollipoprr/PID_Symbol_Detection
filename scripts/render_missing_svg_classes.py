#!/usr/bin/env python3
"""
修复并渲染缺失的SVG类别 (v11 - 使用export-area-drawing)

问题：某些SVG文件有transform导致内容偏移
解决方案：使用Inkscape的--export-area-drawing选项导出图形内容区域
"""

import subprocess
import tempfile
import re
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class SVGVariant:
    """SVG变体配置"""
    stroke_width: float
    scale: float
    offset_x: float
    offset_y: float
    tag: str

    @property
    def filename(self) -> str:
        return f"sw{self.stroke_width}_s{self.scale}_px{int(self.offset_x*10)}_py{int(self.offset_y*10)}"


def generate_variants(per_class: int, seed: int = 42) -> List[SVGVariant]:
    """生成指定数量的变体配置"""
    import random
    random.seed(seed)

    stroke_widths = [0.2, 0.3, 0.4, 0.5, 0.6]
    scales = [0.5, 0.6, 0.8, 1.0, 1.2]

    base_variants = []
    for sw in stroke_widths:
        for sc in scales:
            base_variants.append(SVGVariant(
                stroke_width=sw,
                scale=sc,
                offset_x=0.0,
                offset_y=0.0,
                tag=f"sw{str(sw).replace('.', '')}_s{str(sc).replace('.', '')}"
            ))

    variants = base_variants.copy()

    if per_class > len(base_variants):
        extra_needed = per_class - len(base_variants)
        for _ in range(extra_needed):
            sw = random.choice(stroke_widths)
            sc = random.choice(scales)
            ox = random.uniform(-0.2, 0.2)
            oy = random.uniform(-0.2, 0.2)
            variants.append(SVGVariant(
                stroke_width=sw,
                scale=sc,
                offset_x=ox,
                offset_y=oy,
                tag=f"sw{str(sw).replace('.', '')}_s{str(sc).replace('.', '')}_ox{int(ox*100)}_oy{int(oy*100)}"
            ))

    return variants[:per_class]


def modify_svg_for_variant(svg_text: str, variant: SVGVariant) -> str:
    """修改SVG文本以应用变体参数"""
    sw = str(variant.stroke_width)

    # 修改style中的stroke-width
    svg_modified = re.sub(r'stroke-width:\s*[^;]+', f'stroke-width:{sw}', svg_text)

    # 修改stroke-width属性
    svg_modified = re.sub(r'stroke-width="[^"]*"', f'stroke-width="{sw}"', svg_modified)

    # 设置fill为none
    svg_modified = re.sub(r'fill="[^"]*"', 'fill="none"', svg_modified)

    # 确保stroke为黑色
    svg_modified = re.sub(r'stroke="#ffffff"', 'stroke="#000000"', svg_modified)

    # 修改viewBox实现缩放和偏移
    if variant.scale != 1.0 or variant.offset_x != 0 or variant.offset_y != 0:
        vb_pattern = r'viewBox\s*=\s*["\']([^"\']+)["\']'
        vb_match = re.search(vb_pattern, svg_modified)

        if vb_match:
            parts = vb_match.group(1).split()
            if len(parts) == 4:
                try:
                    vb_x, vb_y, vb_w, vb_h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

                    new_w = vb_w / variant.scale
                    new_h = vb_h / variant.scale

                    center_x = vb_x + vb_w / 2 + variant.offset_x * vb_w
                    center_y = vb_y + vb_h / 2 + variant.offset_y * vb_h

                    new_x = center_x - new_w / 2
                    new_y = center_y - new_h / 2

                    new_vb = f"{new_x:.2f} {new_y:.2f} {new_w:.2f} {new_h:.2f}"
                    svg_modified = re.sub(vb_pattern, f'viewBox="{new_vb}"', svg_modified)
                except ValueError:
                    pass

    return svg_modified


def render_with_inkscape(svg_text: str, width: int = 512) -> Optional[Image.Image]:
    """使用Inkscape渲染SVG文本，返回RGBA图像"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', encoding='utf-8', delete=False) as f:
        f.write(svg_text)
        svg_temp_path = f.name

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        png_temp_path = tmp.name

    try:
        # 使用export-area-drawing导出图形内容区域
        cmd = [
            'inkscape',
            '--export-filename', png_temp_path,
            '--export-width', str(width),
            '--export-area-drawing',
            svg_temp_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return None

        img = Image.open(png_temp_path)
        return img.convert('RGBA')

    except Exception:
        return None
    finally:
        for p in [svg_temp_path, png_temp_path]:
            if Path(p).exists():
                Path(p).unlink()


def post_process_image(img: Image.Image, size: int = 224) -> Image.Image:
    """
    后处理图像
    
    使用白色背景合成，透明线条显示为白色
    """
    # 用白色背景合成
    white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    white_bg.paste(img, mask=img.split()[3])

    # 转换为RGB
    result = white_bg.convert('RGB')

    # 缩放到目标尺寸
    result = result.resize((size, size), Image.LANCZOS)

    return result


def render_svg_to_pil(
    svg_path: str,
    variant: SVGVariant,
    size: int = 224,
    export_size: int = 512,
) -> Optional[Image.Image]:
    """渲染SVG为PIL图像"""
    with open(svg_path, "r", encoding="utf-8", errors="ignore") as f:
        original_svg = f.read()

    # 修改SVG以应用变体
    modified_svg = modify_svg_for_variant(original_svg, variant)

    # 使用Inkscape渲染
    img = render_with_inkscape(modified_svg, width=export_size)

    if img is None:
        return None

    # 后处理
    return post_process_image(img, size)


def main():
    parser = argparse.ArgumentParser(description="修复并渲染缺失的SVG类别")
    parser.add_argument("--svg_dir", type=str, default="/media/wit/SSD_5151/wxr/Long-CLIP/PID_Symbol_Detection/originalsvg", help="SVG文件所在目录")
    parser.add_argument("--output_dir", type=str, default="/media/wit/HDD_16T/wxr/experiment/data/processed/stage_2/svg_to_png", help="输出目录")
    parser.add_argument("--per_class", type=int, default=25, help="每类生成的变体数量")
    parser.add_argument("--size", type=int, default=224, help="输出图像尺寸")
    parser.add_argument("--export_size", type=int, default=512, help="内部渲染尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 需要修复的类别
    missing_classes = {
        '3': 'OPC.svg',
        '6': '放空管.svg',
        '9': '阀门.svg',
        '14': '排放.svg',
        '17': '阀门10.svg',
        '36': '流量孔板.svg',
        '53': '电磁阀1.svg'
    }

    svg_dir = Path(args.svg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = generate_variants(args.per_class, seed=args.seed)

    success, failed = 0, 0

    print(f"开始渲染 {len(missing_classes)} 个缺失类别...")
    print(f"每类生成 {len(variants)} 个变体")
    print(f"渲染尺寸: {args.export_size}, 输出尺寸: {args.size}")
    print("=" * 60)

    for class_id, svg_filename in missing_classes.items():
        svg_path = svg_dir / svg_filename
        class_dir = output_dir / class_id
        class_dir.mkdir(parents=True, exist_ok=True)

        if not svg_path.exists():
            print(f"[跳过] class_id={class_id} 文件不存在: {svg_path}")
            failed += 1
            continue

        class_success = 0
        class_errors = []

        for idx, variant in enumerate(variants):
            try:
                img = render_svg_to_pil(
                    str(svg_path),
                    variant,
                    size=args.size,
                    export_size=args.export_size
                )

                if img is None:
                    if len(class_errors) < 2:
                        class_errors.append("Inkscape渲染失败")
                    continue

                out_path = class_dir / f"{idx}_{variant.filename}.png"
                img.save(out_path, optimize=True)
                class_success += 1
            except Exception as e:
                if len(class_errors) < 2:
                    class_errors.append(f"{type(e).__name__}: {str(e)[:60]}")
                continue

        if class_success > 0:
            print(f"[完成] class_id={class_id} ({svg_filename}): {class_success}/{len(variants)} 张")
            success += 1
        else:
            print(f"[失败] class_id={class_id} ({svg_filename})")
            for err in class_errors:
                print(f"       错误: {err}")
            failed += 1

    print("=" * 60)
    print(f"[完成] 成功: {success} 类, 失败: {failed} 类")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()

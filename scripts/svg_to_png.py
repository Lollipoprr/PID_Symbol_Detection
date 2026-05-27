#!/usr/bin/env python3
"""
生成干净的 SVG 渲染数据集（Phase 2a SVG预训练专用）

设计目标：
  - 学习"干净的几何+拓扑"特征
  - 提供丰富的类内变体（不同stroke、尺寸、位置）
  - 不添加任何噪声（SVG本身是干净的）

生成策略：
  1. stroke_width 变体：细/中/粗线条
  2. viewBox 缩放变体：模拟符号大小差异
  3. 位置偏移变体：符号在画布中的不同位置
  4. 总计：每类生成多种变体，覆盖几何多样性

输出目录结构：
  svg_to_png/<class_id>/
    0_stroke0.5_scale1.0_pos0.png
    1_stroke0.5_scale1.0_pos1.png
    ...

运行示例：
  python src/scripts/generate_svg_clean_dataset.py --per_class 100
"""

import io
import random
import argparse
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from PIL import Image
import cairosvg
import zipfile
import xml.etree.ElementTree as ET
import re


@dataclass
class SVGVariant:
    """SVG变体配置"""
    stroke_width: float
    scale: float
    offset_x: float  # 相对偏移
    offset_y: float
    tag: str

    @property
    def filename(self) -> str:
        return f"sw{self.stroke_width}_s{self.scale}_px{int(self.offset_x*10)}_py{int(self.offset_y*10)}"


# ── 变体生成策略 ──────────────────────────────────────────────────────────────

def generate_variants(per_class: int, seed: int = 42) -> List[SVGVariant]:
    """生成指定数量的变体配置"""
    random.seed(seed)

    stroke_widths = [0.2, 0.3, 0.4, 0.5, 0.6]
    # scale 保持为 1.0，不修改 viewBox
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


# ── 解析映射表.xlsx ──────────────────────────────────────────────────────────

def parse_mapping_table(xlsx_path: str) -> List[Tuple[str, str]]:
    """返回 [(class_id, 中文名称), ...]，按 class_id 升序排列"""
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(xlsx_path) as z:
        strings = []
        ss_root = ET.fromstring(z.read("xl/sharedStrings.xml").decode("utf-8"))
        for si in ss_root.findall("x:si", ns):
            strings.append("".join(t.text or "" for t in si.findall(".//x:t", ns)))

        sheet_root = ET.fromstring(z.read("xl/worksheets/sheet1.xml").decode("utf-8"))
        rows = []
        for row in sheet_root.findall(".//x:row", ns):
            row_data = []
            for c in row.findall("x:c", ns):
                t_attr = c.get("t", "")
                v = c.find("x:v", ns)
                if v is not None:
                    if t_attr == "s":
                        row_data.append(strings[int(v.text)])
                    else:
                        row_data.append(v.text)
                else:
                    row_data.append("")
            rows.append(row_data)

    result = []
    for row in rows[1:]:
        if len(row) >= 2 and row[0] and row[1]:
            result.append((row[0], row[1]))
    result.sort(key=lambda x: int(x[0]))
    return result


# ── SVG渲染核心 ──────────────────────────────────────────────────────────────

def modify_viewbox(svg_text: str, scale: float, offset_x: float, offset_y: float) -> str:
    """
    通过修改 viewBox 实现缩放和偏移。
    对于 Inkscape SVG，先移除 transform 把内容移回原位，再调整 viewBox。
    """
    vb_pattern = r'viewBox\s*=\s*["\']([^"\']+)["\']'
    vb_match = re.search(vb_pattern, svg_text)
    if not vb_match:
        return svg_text

    parts = vb_match.group(1).split()
    if len(parts) != 4:
        return svg_text

    vb_x, vb_y, vb_w, vb_h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

    try:
        root = ET.fromstring(svg_text.encode('utf-8'))
    except ET.ParseError:
        return svg_text

    # 找到并处理 transform
    translate_x, translate_y = 0, 0
    transform_group = None
    for elem in root.iter():
        t = elem.get('transform')
        if t and 'translate' in t:
            m = re.search(r'translate\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)', t)
            if m:
                translate_x, translate_y = float(m.group(1)), float(m.group(2))
                transform_group = elem
                break

    # 如果有 transform，移除它并调整内容位置
    if transform_group is not None:
        # 移除 transform 属性
        del transform_group.attrib['transform']
        # 把 translate 值加到所有子元素上
        for elem in transform_group.iter():
            x_val = elem.get('x')
            y_val = elem.get('y')
            if x_val is not None:
                try:
                    elem.set('x', str(float(x_val) + translate_x))
                except ValueError:
                    pass
            if y_val is not None:
                try:
                    elem.set('y', str(float(y_val) + translate_y))
                except ValueError:
                    pass
            # 处理 path d 属性
            d = elem.get('d')
            if d:
                nums = re.findall(r'([+-]?\d+(?:\.\d+)?)', d)
                new_d = d
                idx = 0
                for num in nums:
                    if idx % 2 == 0:  # x 坐标
                        new_d = new_d.replace(num, str(float(num) + translate_x), 1)
                    else:  # y 坐标
                        new_d = new_d.replace(num, str(float(num) + translate_y), 1)
                    idx += 1
                elem.set('d', new_d)
            # 处理 polygon points
            pts = elem.get('points')
            if pts:
                new_pts = []
                for pair in pts.split():
                    coords = pair.split(',')
                    if len(coords) == 2:
                        try:
                            px = str(float(coords[0]) + translate_x)
                            py = str(float(coords[1]) + translate_y)
                            new_pts.append(f"{px},{py}")
                        except ValueError:
                            new_pts.append(pair)
                elem.set('points', ' '.join(new_pts))
        # 更新 svg_text
        svg_text = ET.tostring(root, encoding='unicode', xml_declaration=False)
        svg_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_text

    # 现在用修正后的逻辑计算新的 viewBox
    # scale > 1: viewBox 变小 -> 内容放大
    # scale < 1: viewBox 变大 -> 内容缩小
    new_w = vb_w / scale
    new_h = vb_h / scale

    # 应用偏移
    center_x = vb_x + vb_w / 2 + offset_x * vb_w
    center_y = vb_y + vb_h / 2 + offset_y * vb_h

    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2

    new_vb = f"{new_x:.2f} {new_y:.2f} {new_w:.2f} {new_h:.2f}"

    return re.sub(vb_pattern, f'viewBox="{new_vb}"', svg_text)


def render_svg_to_pil(
    svg_path: str,
    variant: SVGVariant,
    size: int = 224,
    bg_gray: int = 255,
) -> Image.Image:
    """渲染SVG为PIL图像"""
    with open(svg_path, "r", encoding="utf-8", errors="ignore") as f:
        svg_text = f.read()

    is_inkscape = ("sodipodi" in svg_text or "inkscape" in svg_text or 'mm"' in svg_text)

    svg_fixed = _inject_stroke_width(svg_text, variant.stroke_width)

    if variant.scale != 1.0 or variant.offset_x != 0 or variant.offset_y != 0:
        svg_fixed = modify_viewbox(svg_fixed, variant.scale, variant.offset_x, variant.offset_y)

    if is_inkscape:
        png_bytes = cairosvg.svg2png(bytestring=svg_fixed.encode("utf-8"), output_width=448)
    else:
        png_bytes = cairosvg.svg2png(bytestring=svg_fixed.encode("utf-8"), output_width=448)

    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGB", img.size, (bg_gray, bg_gray, bg_gray))
    bg.paste(img, mask=img.split()[3])
    img = bg

    w, h = img.size
    maxd = max(w, h)
    canvas = Image.new("RGB", (maxd, maxd), (255, 255, 255))
    canvas.paste(img, ((maxd - w) // 2, (maxd - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def _inject_stroke_width(svg_text: str, stroke_width: float) -> str:
    """用正则方式修改 stroke-width"""
    sw = str(stroke_width)

    # 修改 style 中的 stroke-width
    svg_fixed = re.sub(
        r'stroke-width:\s*[^;]+',
        f'stroke-width:{sw}',
        svg_text
    )

    # 设置 fill 为 none
    svg_fixed = re.sub(r'fill="[^"]*"', 'fill="none"', svg_fixed)

    # 确保 stroke 为黑色
    svg_fixed = re.sub(r'stroke="#ffffff"', 'stroke="#000000"', svg_fixed)
    svg_fixed = re.sub(r"stroke='#ffffff'", "stroke='#000000'", svg_fixed)

    # 修改 stroke-width 属性
    if 'stroke-width' in svg_fixed:
        svg_fixed = re.sub(r'stroke-width="[^"]*"', f'stroke-width="{sw}"', svg_fixed)
    else:
        # 在 stroke 属性后添加 stroke-width
        svg_fixed = re.sub(r'(stroke="[^"]*")', rf'\1 stroke-width="{sw}"', svg_fixed)

    return svg_fixed


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成干净的SVG渲染数据集（Phase 2a SVG预训练专用）")
    parser.add_argument("--xlsx", type=str, default="/media/wit/HDD_16T/wxr/experiment/映射表.xlsx", help="映射表.xlsx路径")
    parser.add_argument("--svg_dir", type=str, default="/media/wit/SSD_5151/wxr/Long-CLIP/PID_Symbol_Detection/originalsvg", help="SVG文件所在目录")
    parser.add_argument("--output_dir", type=str, default="/media/wit/HDD_16T/wxr/experiment/data/processed/stage_2/svg_to_png", help="输出目录")
    parser.add_argument("--per_class", type=int, default=25, help="每类生成的变体数量")
    parser.add_argument("--size", type=int, default=224, help="输出图像尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dry_run", action="store_true", help="仅预览生成计划，不实际渲染")

    args = parser.parse_args()

    mapping = parse_mapping_table(args.xlsx)
    print(f"[INFO] 共读取 {len(mapping)} 个类别")

    variants = generate_variants(args.per_class, seed=args.seed)
    print(f"[INFO] 每类生成 {len(variants)} 个变体")
    print(f"       - stroke_width: {[v.stroke_width for v in variants[:5]]}...")
    print(f"       - scale: {[round(v.scale, 2) for v in variants[:5]]}...")
    print(f"       - 总计约 {len(mapping) * len(variants)} 张图像")

    if args.dry_run:
        print("\n[DRY RUN] 预览生成计划:")
        for i, (class_id, name) in enumerate(mapping[:3]):
            print(f"  {class_id}: {name} -> {len(variants)} 张")
        print(f"  ... (共 {len(mapping)} 类)")
        return

    svg_dir = Path(args.svg_dir)
    svg_files = {f.stem: f for f in svg_dir.glob("*.svg")}
    print(f"[INFO] 找到 {len(svg_files)} 个SVG文件")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success, skipped, failed = 0, 0, 0

    for class_id, chinese_name in mapping:
        svg_path = svg_files.get(chinese_name)
        if svg_path is None:
            skipped += 1
            print(f"  [跳过] class_id={class_id} '{chinese_name}' 未找到SVG")
            continue

        class_dir = output_dir / class_id
        class_dir.mkdir(parents=True, exist_ok=True)

        class_success = 0
        for idx, variant in enumerate(variants):
            try:
                img = render_svg_to_pil(str(svg_path), variant, size=args.size, bg_gray=255)
                out_path = class_dir / f"{idx}_{variant.filename}.png"
                img.save(out_path, optimize=True)
                class_success += 1
            except Exception as e:
                print(f"  [错误] class_id={class_id} {variant.filename}: {e}")
                continue

        success += 1
        print(f"  [完成] {class_id}: {chinese_name} -> {class_success}/{len(variants)} 张")

    print(f"\n{'='*60}")
    print(f"[完成] 总计:")
    print(f"  - 成功: {success} 类")
    print(f"  - 跳过: {skipped} 类 (SVG缺失)")
    print(f"  - 失败: {failed} 类")
    print(f"  - 总图像数: {success * len(variants)} 张")
    print(f"  - 输出目录: {output_dir}")
    print(f"{'='*60}")

    import json
    config_path = output_dir / "generation_config.json"
    config = {
        "per_class": args.per_class,
        "variants_count": len(variants),
        "stroke_widths": list(set(v.stroke_width for v in variants)),
        "scales": list(set(v.scale for v in variants)),
        "size": args.size,
        "seed": args.seed,
        "total_images": success * len(variants),
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] 配置已保存: {config_path}")


if __name__ == "__main__":
    main()

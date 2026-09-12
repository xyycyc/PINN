from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\Desktop\code\ai_model")
OUT = ROOT / "人工智能截图整改_可直接替换"

FONT_CN = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_CN_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_MONO = Path(r"C:\Windows\Fonts\consola.ttf")
FONT_MONO_BOLD = Path(r"C:\Windows\Fonts\consolab.ttf")

WHITE = "#FFFFFF"
INK = "#172033"
MUTED = "#5E6B7A"
BLUE = "#1F4E79"
BLUE_2 = "#2F75B5"
BLUE_PALE = "#EAF2F8"
CYAN_PALE = "#EEF7FA"
GREEN = "#256B3F"
GREEN_PALE = "#EAF5EE"
ORANGE = "#9A4B00"
ORANGE_PALE = "#FFF3E6"
PURPLE = "#5B3D8B"
PURPLE_PALE = "#F2EEFA"
PANEL = "#F7F9FC"
PANEL_2 = "#F1F4F8"
BORDER = "#B9C5D1"
GRID = "#D9E0E7"


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO_BOLD if mono and bold else FONT_MONO if mono else FONT_CN_BOLD if bold else FONT_CN
    return ImageFont.truetype(str(path), size=size)


def text_width(draw: ImageDraw.ImageDraw, value: str, fnt: ImageFont.FreeTypeFont) -> float:
    return draw.textlength(value, font=fnt)


def fit_font(
    draw: ImageDraw.ImageDraw,
    value: str,
    max_width: int,
    start_size: int,
    min_size: int,
    *,
    bold: bool = False,
    mono: bool = False,
) -> ImageFont.FreeTypeFont:
    for size in range(start_size, min_size - 1, -1):
        candidate = font(size, bold=bold, mono=mono)
        if text_width(draw, value, candidate) <= max_width:
            return candidate
    return font(min_size, bold=bold, mono=mono)


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = INK,
) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), value, font=fnt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2 - bbox[1]), value, font=fnt, fill=fill)


def draw_bullet_lines(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    lines: list[str],
    fnt: ImageFont.FreeTypeFont,
    *,
    color: str = INK,
    gap: int = 13,
) -> int:
    bullet_r = max(3, fnt.size // 9)
    for line in lines:
        draw.ellipse((x, y + fnt.size // 2 - bullet_r, x + 2 * bullet_r, y + fnt.size // 2 + bullet_r), fill=BLUE_2)
        draw.text((x + 3 * bullet_r + 8, y), line, font=fnt, fill=color)
        y += fnt.size + gap
    return y


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = BLUE_2,
    width: int = 7,
) -> None:
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    length = width * 3.1
    spread = math.pi / 7
    p1 = (x1 - length * math.cos(angle - spread), y1 - length * math.sin(angle - spread))
    p2 = (x1 - length * math.cos(angle + spread), y1 - length * math.sin(angle + spread))
    draw.polygon([(x1, y1), p1, p2], fill=color)


def save_png(image: Image.Image, filename: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    image.save(path, format="PNG", dpi=(300, 300), optimize=True)
    return path


def manifest_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_allocations(data: dict) -> str:
    parts = []
    for item in data.get("constituent_material_catalog", []):
        name = str(item.get("material_name", "material"))
        count = int(item.get("sampled_point_count", 0))
        parts.append(f"{name} {count:,}")
    return " + ".join(parts)


def draw_manifest_figure(
    *,
    width: int,
    height: int,
    manifest_path: Path,
    directory_name: str,
    material_display_name: str,
    filename: str,
) -> Path:
    data = manifest_data(manifest_path)
    img = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(img)
    margin = max(28, width // 55)
    gap = max(22, width // 70)
    heading_size = max(30, min(38, height // 20))
    heading_font = font(heading_size, bold=True)
    mono_size = max(21, min(27, height // 29))
    tree_font = font(mono_size, mono=True)
    label_font = font(max(22, min(28, height // 27)), mono=True)
    value_font = font(max(24, min(30, height // 25)))
    small_font = font(max(20, min(25, height // 31)))
    small_mono = font(max(19, min(24, height // 32)), mono=True)

    title_y = margin - 3
    draw.text((margin, title_y), "数据目录", font=heading_font, fill=INK)
    left_width = int(width * 0.37)
    right_x = margin + left_width + gap
    right_width = width - right_x - margin
    draw.text((right_x, title_y), "train_manifest.json 关键字段", font=heading_font, fill=INK)

    panel_y = margin + heading_size + 18
    panel_bottom = height - margin
    draw.rounded_rectangle((margin, panel_y, margin + left_width, panel_bottom), radius=16, fill=PANEL, outline=BORDER, width=2)
    draw.rounded_rectangle((right_x, panel_y, right_x + right_width, panel_bottom), radius=16, fill=WHITE, outline=BORDER, width=2)

    tree_lines = [
        "database/",
        "└─ data_process/",
        f"   └─ {directory_name}/",
        "      ├─ temperature_fields/",
        "      ├─ waveforms/",
        "      ├─ manifest.json",
        "      ├─ mesh_audit.json",
        "      ├─ sampling_index.npz",
        "      ├─ split_config.json",
        "      ├─ train_manifest.json  [current]",
        "      ├─ validation_manifest.json",
        "      └─ test_manifest.json",
    ]
    tree_x = margin + 22
    tree_y = panel_y + 26
    available_tree_width = left_width - 44
    longest = max(tree_lines, key=len)
    tree_font = fit_font(draw, longest, available_tree_width, tree_font.size, 18, mono=True)
    tree_gap = max(7, (panel_bottom - tree_y - 25 - tree_font.size * len(tree_lines)) // max(1, len(tree_lines) - 1))
    for idx, line in enumerate(tree_lines):
        color = BLUE if "train_manifest" in line else INK
        current_font = font(tree_font.size, mono=True, bold="train_manifest" in line)
        draw.text((tree_x, tree_y), line, font=current_font, fill=color)
        tree_y += tree_font.size + tree_gap

    sample_catalog = data.get("sample_material_catalog", [{}])[0]
    temp = data.get("temperature_summary", {})
    waveform = data.get("waveform_contract", {})
    rows = [
        ("dataset_label", str(data.get("dataset_label", ""))),
        ("material", material_display_name),
        ("case_count", f'{int(sample_catalog.get("case_count", temp.get("case_count", 0))):,} 组'),
        ("temperature", f'{temp.get("min_k", "-")}–{temp.get("max_k", "-")} K'),
        ("point_count", f'{int(temp.get("point_count", 0)):,} 个固定物理节点'),
        ("node_allocation", format_allocations(data)),
        ("waveform", f'crop_length={waveform.get("crop_length", "-")}; resampled={str(waveform.get("resampled", False)).lower()}'),
    ]
    flow_h = max(96, height // 7)
    rows_top = panel_y + 14
    rows_bottom = panel_bottom - flow_h - 10
    row_h = (rows_bottom - rows_top) / len(rows)
    key_x = right_x + 22
    value_x = right_x + int(right_width * 0.34)
    for idx, (key, value) in enumerate(rows):
        y0 = int(rows_top + idx * row_h)
        y1 = int(rows_top + (idx + 1) * row_h)
        if idx % 2 == 0:
            draw.rectangle((right_x + 2, y0, right_x + right_width - 2, y1), fill=PANEL)
        if idx:
            draw.line((right_x + 15, y0, right_x + right_width - 15, y0), fill=GRID, width=1)
        key_f = fit_font(draw, key, int(right_width * 0.31) - 28, label_font.size, 18, mono=True)
        value_f = fit_font(draw, value, int(right_width * 0.64) - 35, value_font.size, 19)
        key_bbox = draw.textbbox((0, 0), key, font=key_f)
        val_bbox = draw.textbbox((0, 0), value, font=value_f)
        draw.text((key_x, (y0 + y1 - (key_bbox[3] - key_bbox[1])) / 2 - key_bbox[1]), key, font=key_f, fill=BLUE)
        draw.text((value_x, (y0 + y1 - (val_bbox[3] - val_bbox[1])) / 2 - val_bbox[1]), value, font=value_f, fill=INK)

    flow_y = panel_bottom - flow_h
    draw.line((right_x + 15, flow_y, right_x + right_width - 15, flow_y), fill=BORDER, width=2)
    draw.text((right_x + 22, flow_y + 10), "样本记录映射", font=small_font, fill=MUTED)
    box_y0 = flow_y + 42
    box_y1 = panel_bottom - 16
    box_gap = 24
    inner_x0 = right_x + 22
    inner_x1 = right_x + right_width - 22
    box_w = int((inner_x1 - inner_x0 - box_gap * 2) / 3)
    labels = ["waveforms/*.npy", "train_manifest.json", "temperature_fields/*.npz"]
    fills = [CYAN_PALE, BLUE_PALE, GREEN_PALE]
    outlines = [BLUE_2, BLUE, GREEN]
    for i, label in enumerate(labels):
        x0 = inner_x0 + i * (box_w + box_gap)
        x1 = x0 + box_w
        draw.rounded_rectangle((x0, box_y0, x1, box_y1), radius=10, fill=fills[i], outline=outlines[i], width=2)
        label_f = fit_font(draw, label, box_w - 16, small_mono.size, 16, mono=True)
        draw_centered(draw, (x0, box_y0, x1, box_y1), label, label_f, outlines[i])
        if i < 2:
            arrow(draw, (x1 + 4, (box_y0 + box_y1) // 2), (x1 + box_gap - 4, (box_y0 + box_y1) // 2), color=MUTED, width=4)

    return save_png(img, filename)


def draw_inference_summary() -> Path:
    width, height = 1654, 334
    img = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(img)
    margin = 28
    header_font = font(34, bold=True)
    status_font = font(23, bold=True)
    label_font = font(22)
    value_font = font(40, bold=True)
    detail_font = font(19)
    mono_font = font(19, mono=True)

    draw.text((margin, 18), "AI温度场重构推理任务验证摘要", font=header_font, fill=INK)
    status_box = (width - 330, 16, width - margin, 59)
    draw.rounded_rectangle(status_box, radius=20, fill=GREEN_PALE, outline=GREEN, width=2)
    draw_centered(draw, status_box, "正常完成  exit_code = 0", status_font, GREEN)

    y0, y1 = 76, 216
    gap = 18
    card_w = (width - 2 * margin - 3 * gap) // 4
    cards = [
        ("4个温度场总耗时", "0.248766 s", "连续推理统计"),
        ("平均单场耗时", "62.19 ms", "即 0.0622 s/场"),
        ("模型参数量", "1,441,744", "参数文件约 5.50 MiB"),
        ("GPU显存峰值", "42.04 MiB", "44,078,592 bytes"),
    ]
    for i, (label, value, detail) in enumerate(cards):
        x0 = margin + i * (card_w + gap)
        x1 = x0 + card_w
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=PANEL, outline=BORDER, width=2)
        draw.text((x0 + 18, y0 + 13), label, font=label_font, fill=MUTED)
        value_f = fit_font(draw, value, card_w - 36, value_font.size, 29, bold=True)
        draw.text((x0 + 18, y0 + 48), value, font=value_f, fill=BLUE)
        draw.text((x0 + 18, y1 - 34), detail, font=detail_font, fill=INK)

    bar_y0, bar_y1 = 234, height - 20
    draw.rounded_rectangle((margin, bar_y0, width - margin, bar_y1), radius=12, fill=BLUE_PALE, outline=BLUE_2, width=2)
    draw.text((margin + 18, bar_y0 + 10), "benchmark", font=font(20, bold=True, mono=True), fill=BLUE)
    draw.text((margin + 175, bar_y0 + 10), "warm-up = 1  |  repeated runs = 3", font=mono_font, fill=INK)
    draw.text((margin + 18, bar_y0 + 43), "checkpoint: .../two_steady_wumu_2026_7_12_1858/wumu.pt", font=mono_font, fill=MUTED)
    manifest_text = "manifest: .../wumu_case_temperature_field/test_manifest.json"
    manifest_font = fit_font(draw, manifest_text, 650, mono_font.size, 15, mono=True)
    draw.text((width - margin - text_width(draw, manifest_text, manifest_font), bar_y0 + 43), manifest_text, font=manifest_font, fill=MUTED)

    return save_png(img, "图3.55_AI推理耗时日志_白底摘要.png")


def draw_collaboration_diagram() -> Path:
    width, height = 1677, 962
    img = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(img)
    margin = 40
    header_font = font(38, bold=True)
    box_title_font = font(32, bold=True)
    body_font = font(27)
    small_font = font(23)
    mono_font = font(22, mono=True)
    mono_bold = font(22, mono=True, bold=True)

    draw.text((margin, 24), "调用链与数据方向", font=header_font, fill=INK)
    draw.text((width - margin, 34), "Python推理 → C ABI → Fortran指标计算 → Python归档", font=small_font, fill=MUTED, anchor="ra")

    top = 92
    box_h = 300
    left = (margin, top, 570, top + box_h)
    bridge = (670, top, 1007, top + box_h)
    right = (1107, top, width - margin, top + box_h)

    draw.rounded_rectangle(left, radius=18, fill=BLUE_PALE, outline=BLUE_2, width=3)
    draw.rounded_rectangle(bridge, radius=18, fill=PANEL_2, outline=MUTED, width=3)
    draw.rounded_rectangle(right, radius=18, fill=ORANGE_PALE, outline=ORANGE, width=3)

    draw.text((left[0] + 24, left[1] + 18), "Python人工智能模块", font=box_title_font, fill=BLUE)
    draw.text((left[0] + 24, left[1] + 66), "model/predict.py", font=mono_bold, fill=MUTED)
    draw_bullet_lines(
        draw,
        left[0] + 28,
        left[1] + 110,
        ["读取波形、manifest 与 checkpoint", "预处理并执行模型前向推理", "形成 temp_true 与 temp_pred"],
        body_font,
        gap=16,
    )

    bridge_title = "Python–Fortran桥接"
    bridge_title_font = fit_font(draw, bridge_title, bridge[2] - bridge[0] - 44, box_title_font.size, 24, bold=True)
    draw.text((bridge[0] + 22, bridge[1] + 18), bridge_title, font=bridge_title_font, fill=INK)
    draw.text((bridge[0] + 22, bridge[1] + 66), "fortran/native_bridge.py", font=fit_font(draw, "fortran/native_bridge.py", bridge[2] - bridge[0] - 44, mono_bold.size, 17, mono=True, bold=True), fill=MUTED)
    bridge_lines = ["np.float64", "C-contiguous", "ctypes.CDLL", "指针 + n"]
    y = bridge[1] + 115
    for line in bridge_lines:
        bx = (bridge[0] + 30, y, bridge[2] - 30, y + 38)
        draw.rounded_rectangle(bx, radius=8, fill=WHITE, outline=BORDER, width=2)
        bridge_line_font = mono_font if line != "指针 + n" else font(mono_font.size)
        draw_centered(draw, bx, line, bridge_line_font, INK)
        y += 47

    draw.text((right[0] + 24, right[1] + 18), "Fortran数值计算模块", font=box_title_font, fill=ORANGE)
    draw.text((right[0] + 24, right[1] + 66), "fortran/ai_numeric.f90", font=mono_bold, fill=MUTED)
    draw_bullet_lines(
        draw,
        right[0] + 28,
        right[1] + 110,
        ["ISO_C_BINDING 与 bind(C) 导出接口", "逐点计算预测误差", "返回 MAE、RMSE、Max error"],
        body_font,
        gap=16,
        color=INK,
    )

    arrow(draw, (left[2] + 12, top + 150), (bridge[0] - 12, top + 150), color=BLUE_2, width=7)
    draw.text(((left[2] + bridge[0]) // 2, top + 105), "数组", font=small_font, fill=MUTED, anchor="mm")
    arrow(draw, (bridge[2] + 12, top + 150), (right[0] - 12, top + 150), color=ORANGE, width=7)
    draw.text(((bridge[2] + right[0]) // 2, top + 105), "C ABI", font=small_font, fill=MUTED, anchor="mm")

    result_y0, result_y1 = 425, 526
    draw.rounded_rectangle((margin, result_y0, width - margin, result_y1), radius=16, fill=GREEN_PALE, outline=GREEN, width=3)
    arrow(draw, (right[2] - 40, result_y0 - 8), (right[2] - 40, result_y0 + 28), color=GREEN, width=6)
    draw.text((margin + 26, result_y0 + 16), "返回与归档", font=font(30, bold=True), fill=GREEN)
    result_text = "Fortran 返回 mae、rmse、max_error  →  Python 写入 metrics.json，并生成误差统计、图表和报告"
    result_f = fit_font(draw, result_text, width - 2 * margin - 260, 28, 22)
    draw.text((margin + 220, result_y0 + 28), result_text, font=result_f, fill=INK)

    code_y0, code_y1 = 558, height - margin
    draw.rounded_rectangle((margin, code_y0, width - margin, code_y1), radius=18, fill=PANEL, outline=BORDER, width=2)
    draw.text((margin + 24, code_y0 + 16), "关键接口与逻辑（精简展示）", font=font(31, bold=True), fill=INK)
    divider_x = width // 2
    draw.line((divider_x, code_y0 + 64, divider_x, code_y1 - 18), fill=BORDER, width=2)

    draw.text((margin + 24, code_y0 + 70), "Python调用", font=font(27, bold=True), fill=BLUE)
    python_lines = [
        "pred = model(preprocess(waveform))",
        "true = ascontiguousarray(temp_true, float64)",
        "pred = ascontiguousarray(pred, float64)",
        "mae, rmse, max_err =",
        "    compute_prediction_metrics(true, pred)",
        "metrics[...] = mae, rmse, max_err",
    ]
    py_y = code_y0 + 116
    py_font = fit_font(draw, max(python_lines, key=len), divider_x - margin - 70, 23, 17, mono=True)
    for line in python_lines:
        draw.text((margin + 28, py_y), line, font=py_font, fill=INK)
        py_y += py_font.size + 12

    draw.text((divider_x + 24, code_y0 + 70), "Fortran接口与计算逻辑", font=font(27, bold=True), fill=ORANGE)
    fortran_lines = [
        "subroutine compute_prediction_metrics(...) &",
        "  bind(C, name='compute_prediction_metrics')",
        "e_i = y_pred(i) - y_true(i)",
        "MAE       = sum(abs(e_i)) / n",
        "RMSE      = sqrt(sum(e_i * e_i) / n)",
        "Max error = max(abs(e_i))",
    ]
    ft_y = code_y0 + 116
    ft_font = fit_font(draw, max(fortran_lines, key=len), width - margin - divider_x - 70, 23, 17, mono=True)
    for line in fortran_lines:
        draw.text((divider_x + 28, ft_y), line, font=ft_font, fill=INK)
        ft_y += ft_font.size + 15

    return save_png(img, "图3.56_Python与Fortran协同_流程伪代码.png")


def write_readme(paths: list[Path]) -> None:
    text = """# 人工智能章节黑底截图替换清单

原文件 `D:\\Desktop\\技术报告合稿-0725_终版.docx` 未作任何修改。

本文件夹提供 5 幅可一对一替换的白底图片：

1. 图 3.47：W30Mo70 训练数据目录与 manifest，白底目录树加关键字段摘要。
2. 图 3.55：AI 推理耗时日志，改为可核查的白底运行摘要。
3. 图 3.56：Python 与 Fortran 协同，改为调用链和精简伪代码。
4. 图 3.57：金属基复合材料 manifest，白底目录树加关键字段摘要。
5. 图 3.58：硅基复合材料 manifest，白底目录树加关键字段摘要。

替换时保留原图题，不需要改正文。建议在 Word 中锁定纵横比并沿用原图显示宽度：图 3.47 约 14.80 cm，图 3.55 约 14.00 cm，图 3.56 约 14.20 cm，图 3.57 约 14.50 cm，图 3.58 约 14.80 cm。

图片均为 300 ppi 白底 PNG。插入前建议在“文件 → 选项 → 高级 → 图像大小和质量”中针对当前文档选择“高保真”，并勾选“不压缩文件中的图像”。

内容依据当前工程中的 train_manifest.json、推理日志截图、fortran/native_bridge.py、model/predict.py 和 fortran/ai_numeric.f90 整理；图 3.56 中非接口声明部分标为精简逻辑展示，不替代源代码交付。

## 文件及像素尺寸

"""
    for path in paths:
        with Image.open(path) as im:
            text += f"- `{path.name}`：{im.width} × {im.height} px，300 ppi\n"
    (OUT / "替换说明.md").write_text(text, encoding="utf-8-sig")


def main() -> None:
    paths = [
        draw_manifest_figure(
            width=1748,
            height=782,
            manifest_path=ROOT / "database/data_process/wumu_case_temperature_field/train_manifest.json",
            directory_name="wumu_case_temperature_field",
            material_display_name="W30Mo70 钨钼多层材料",
            filename="图3.47_W30Mo70训练数据目录与manifest_白底重绘.png",
        ),
        draw_inference_summary(),
        draw_collaboration_diagram(),
        draw_manifest_figure(
            width=1712,
            height=817,
            manifest_path=ROOT / "database/data_process/metal_case_temperature_field/train_manifest.json",
            directory_name="metal_case_temperature_field",
            material_display_name="W基体/SiC纤维金属基复合材料",
            filename="图3.57_金属基复合材料manifest_白底重绘.png",
        ),
        draw_manifest_figure(
            width=1748,
            height=715,
            manifest_path=ROOT / "database/data_process/silicon_case_temperature_field/train_manifest.json",
            directory_name="silicon_case_temperature_field",
            material_display_name="SiC纤维/CVI-SiC基体硅基复合材料",
            filename="图3.58_硅基复合材料manifest_白底重绘.png",
        ),
    ]
    write_readme(paths)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

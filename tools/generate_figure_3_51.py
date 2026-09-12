from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "图3.51_单图交付"
OUTPUT_PATH = OUTPUT_DIR / "图3.51_checkpoint更新与参数冻结机制配置_严格同信息版.png"

WIDTH = 1800
HEIGHT = 1320

FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def draw_file_icon(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, color: str) -> None:
    width = round(size * 0.72)
    fold = round(size * 0.24)
    draw.polygon(
        [
            (x, y),
            (x + width - fold, y),
            (x + width, y + fold),
            (x + width, y + size),
            (x, y + size),
        ],
        fill="#ffffff",
        outline=color,
        width=2,
    )
    draw.line((x + width - fold, y, x + width - fold, y + fold, x + width, y + fold), fill=color, width=2)


def draw_chevron(draw: ImageDraw.ImageDraw, x: int, y: int, open_state: bool, color: str) -> None:
    if open_state:
        points = [(x, y), (x + 12, y), (x + 6, y + 8)]
    else:
        points = [(x, y), (x + 8, y + 6), (x, y + 12)]
    draw.polygon(points, fill=color)


def draw_code_file_mark(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str) -> None:
    if kind == "pt":
        draw.arc((x, y, x + 22, y + 22), start=205, end=355, fill="#bb5a18", width=3)
        draw.arc((x + 4, y + 4, x + 18, y + 18), start=205, end=355, fill="#bb5a18", width=3)
        draw.ellipse((x + 1, y + 15, x + 6, y + 20), fill="#bb5a18")
    elif kind == "json":
        draw.text((x - 2, y - 5), "{}", font=font(27, True), fill="#5a7f18")
    else:
        draw.ellipse((x + 2, y + 2, x + 20, y + 20), outline="#2e6f91", width=3)
        draw.text((x + 8, y - 1), "i", font=font(18, True), fill="#2e6f91")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(image)

    text = "#17202a"
    secondary = "#52606d"
    line = "#d7dde3"
    panel = "#fbfcfd"
    selection = "#e9edf1"
    blue = "#255f78"

    # The two file rows are transcribed only from the upper part of the source image.
    draw.rounded_rectangle((45, 38, WIDTH - 45, 252), radius=15, fill=panel, outline=line, width=2)
    file_rows = (
        ("2026_7_16_0127.pt", "2026/7/16 1:28", "PT 文件", "5,648 KB"),
        ("ai_model.pt", "2026/7/14 13:12", "PT 文件", "5,663 KB"),
    )
    row_font = font(31)
    for index, (name, timestamp, file_type, size) in enumerate(file_rows):
        y = 76 + index * 84
        draw_file_icon(draw, 82, y - 5, 38, secondary)
        draw.text((142, y - 4), name, font=row_font, fill=text)
        draw.text((725, y - 4), timestamp, font=row_font, fill=secondary)
        draw.text((1200, y - 4), file_type, font=row_font, fill=secondary)
        draw.text((1510, y - 4), size, font=row_font, fill=secondary)

    # The directory entries below are transcribed only from the lower part of the source image.
    tree_left = 180
    tree_top = 300
    tree_right = WIDTH - 180
    tree_bottom = HEIGHT - 42
    draw.rounded_rectangle((tree_left, tree_top, tree_right, tree_bottom), radius=15, fill=panel, outline=line, width=2)

    entries = (
        (0, "result", "folder", True, False, False),
        (1, "predict", "folder", False, False, False),
        (1, "smoke_runtime", "folder", False, False, False),
        (1, "train", "folder", True, False, False),
        (2, "checkpoint", "folder", True, False, False),
        (3, "metal_matrix_waveforms_post0", "folder", False, False, False),
        (3, "one_steady_metal_2026_5_15_2339", "folder", False, False, False),
        (3, "one_steady_metal_matrix_2026_5_16_0155", "folder", False, False, False),
        (3, "point_field_demo_smoke", "folder", False, False, False),
        (3, "point_field_smoke", "folder", False, False, False),
        (3, "qianyi_check", "folder", True, True, False),
        (4, "CVISiC.pt", "pt", False, False, False),
        (4, "qianyi_check_material_router.json", "json", False, False, True),
        (4, "README.md", "readme", False, False, True),
        (4, "WSiC.pt", "pt", False, False, False),
        (4, "wumu.pt", "pt", False, False, False),
    )

    tree_font = font(29)
    tree_font_bold = font(29, True)
    line_height = 57
    start_y = tree_top + 26
    indent = 46
    base_x = tree_left + 38

    # Subtle tree guides reproduce hierarchy without adding labels or relationships.
    for depth, first_row, last_row in ((1, 1, 15), (2, 4, 15), (3, 5, 15), (4, 11, 15)):
        x = base_x + depth * indent - 22
        y1 = start_y + first_row * line_height + 5
        y2 = start_y + last_row * line_height + 34
        draw.line((x, y1, x, y2), fill="#e1e6ea", width=2)

    for row, (depth, label, kind, is_open, selected, untracked) in enumerate(entries):
        y = start_y + row * line_height
        if selected:
            draw.rounded_rectangle(
                (tree_left + 16, y - 6, tree_right - 16, y + line_height - 7),
                radius=7,
                fill=selection,
                outline="#b8c0c7",
                width=1,
            )

        x = base_x + depth * indent
        if kind == "folder":
            draw_chevron(draw, x, y + 14, is_open, secondary)
            label_x = x + 27
            label_color = blue if label in {"result", "predict", "smoke_runtime", "train", "checkpoint"} else text
            draw.text((label_x, y + 2), label, font=tree_font_bold if selected else tree_font, fill=label_color)
        else:
            draw_code_file_mark(draw, x + 2, y + 11, kind)
            label_x = x + 37
            label_color = blue if kind in {"json", "readme"} else secondary
            draw.text((label_x, y + 2), label, font=tree_font, fill=label_color)
            if untracked:
                draw.text((tree_right - 64, y + 2), "U", font=tree_font_bold, fill=blue)

    image.save(OUTPUT_PATH, format="PNG", dpi=(300, 300), optimize=True)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMPOSITE = ROOT / "_inspection_fig348" / "gui_source_1133x1011_21306fa162.png"
OUTPUT_DIR = ROOT / "图3.48_单图交付"
OUTPUT_PATH = OUTPUT_DIR / "图3.48_数据库结果目录及数据库构建GUI_重拼高清版.png"

WIDTH = 1800
HEIGHT = 1565
FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def draw_chevron(draw: ImageDraw.ImageDraw, x: int, y: int, expanded: bool, color: str) -> None:
    if expanded:
        points = ((x, y), (x + 13, y), (x + 6, y + 8))
    else:
        points = ((x, y), (x + 8, y + 6), (x, y + 12))
    draw.polygon(points, fill=color)


def draw_tree_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    entries: tuple[tuple[int, str, bool, bool], ...],
) -> None:
    left, top, right, bottom = bounds
    panel_fill = "#fbfcfd"
    border = "#cbd4dc"
    guide = "#dfe5ea"
    text = "#26343f"
    root_text = "#245f77"
    selection = "#e7ebef"

    draw.rounded_rectangle(bounds, radius=14, fill=panel_fill, outline=border, width=2)
    row_height = 35
    start_y = top + 17
    base_x = left + 28
    indent = 36
    tree_font = font(24)
    tree_font_bold = font(24, True)

    max_depth = max(depth for depth, *_ in entries)
    for depth in range(1, max_depth + 1):
        first = next((i for i, entry in enumerate(entries) if entry[0] >= depth), None)
        last = next((i for i in range(len(entries) - 1, -1, -1) if entries[i][0] >= depth), None)
        if first is not None and last is not None:
            x = base_x + depth * indent - 17
            draw.line(
                (x, start_y + first * row_height + 6, x, start_y + last * row_height + 27),
                fill=guide,
                width=2,
            )

    for index, (depth, label, expanded, selected) in enumerate(entries):
        y = start_y + index * row_height
        if selected:
            draw.rounded_rectangle(
                (left + 10, y - 3, right - 10, y + row_height - 3),
                radius=5,
                fill=selection,
                outline="#bcc6ce",
                width=1,
            )
        x = base_x + depth * indent
        draw_chevron(draw, x, y + 9, expanded, "#66737d")
        draw.text(
            (x + 24, y - 1),
            label,
            font=tree_font_bold if depth == 0 else tree_font,
            fill=root_text if depth == 0 else text,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(canvas)

    # These entries are the exact names and hierarchy visible in the two original top crops.
    database_entries = (
        (0, "database", True, False),
        (1, "cache \\ preprocess", False, True),
        (1, "data_process", False, False),
        (1, "preprocess", False, False),
        (1, "raw", False, False),
        (1, "rule", False, False),
        (1, "smoke_runtime", False, False),
    )
    result_entries = (
        (0, "result", True, False),
        (1, "predict", False, True),
        (1, "smoke_runtime", False, False),
        (1, "train", True, False),
        (2, "checkpoint", False, True),
        (2, "report", False, False),
    )

    draw_tree_panel(draw, (48, 30, 867, 315), database_entries)
    draw_tree_panel(draw, (933, 30, 1752, 315), result_entries)

    # Use the largest historical composite found in the report drafts.  The crop below is
    # only the original GUI window; no pixels, labels, or values are synthesized.
    with Image.open(SOURCE_COMPOSITE) as source:
        gui = source.convert("RGB").crop((17, 258, 1117, 1004))

    target_width = 1704
    target_height = round(gui.height * target_width / gui.width)
    gui = gui.resize((target_width, target_height), Image.Resampling.LANCZOS)
    gui = gui.filter(ImageFilter.UnsharpMask(radius=0.55, percent=85, threshold=2))

    gui_x = (WIDTH - target_width) // 2
    gui_y = 350
    draw.rectangle(
        (gui_x - 2, gui_y - 2, gui_x + target_width + 1, gui_y + target_height + 1),
        fill="#c4ccd3",
    )
    canvas.paste(gui, (gui_x, gui_y))

    canvas.save(OUTPUT_PATH, format="PNG", dpi=(300, 300), optimize=True)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

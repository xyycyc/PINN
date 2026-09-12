from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "_inspection_docx" / "media"
OUTPUT_DIR = ROOT / "人工智能截图整改_严格同信息版"


@dataclass(frozen=True)
class FigureSpec:
    figure: str
    source_name: str
    output_name: str
    target_size: tuple[int, int]


FIGURES = (
    FigureSpec("图3.47", "image48.png", "图3.47_严格同信息_白底高清.png", (1748, 782)),
    FigureSpec("图3.55", "image56.png", "图3.55_严格同信息_白底高清.png", (1654, 334)),
    FigureSpec("图3.56", "image57.png", "图3.56_严格同信息_白底高清.png", (1677, 962)),
    FigureSpec("图3.57", "image58.png", "图3.57_严格同信息_白底高清.png", (1712, 817)),
    FigureSpec("图3.58", "image59.png", "图3.58_严格同信息_白底高清.png", (1748, 715)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def histogram_percentile(histogram: list[int], fraction: float) -> int:
    total = sum(histogram)
    threshold = total * fraction
    running = 0
    for value, count in enumerate(histogram):
        running += count
        if running >= threshold:
            return value
    return 255


def make_print_light(source: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Convert the same source pixels to a print-friendly light grayscale image."""
    grayscale = source.convert("L")
    histogram = grayscale.histogram()
    low = histogram_percentile(histogram, 0.01)
    high = histogram_percentile(histogram, 0.995)
    if high <= low:
        low, high = 0, 255

    # Reverse luminance while stretching only the source screenshot's own range.
    # This is a pixel-only operation: it introduces no text, symbols, or labels.
    scale = 255.0 / (high - low)
    lookup = []
    for value in range(256):
        normalized = max(0.0, min(255.0, (value - low) * scale))
        lookup.append(round(255.0 - normalized))
    light = grayscale.point(lookup)

    if light.size != target_size:
        light = light.resize(target_size, Image.Resampling.LANCZOS)

    # Mild sharpening compensates for the source screenshots' low effective DPI.
    return light.filter(ImageFilter.UnsharpMask(radius=0.65, percent=105, threshold=3))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in FIGURES:
        source_path = SOURCE_DIR / spec.source_name
        output_path = OUTPUT_DIR / spec.output_name
        with Image.open(source_path) as source:
            original_size = source.size
            result = make_print_light(source, spec.target_size)
            result.save(output_path, format="PNG", dpi=(300, 300), optimize=True)

        print(
            f"{spec.figure}\t{spec.source_name}\t{original_size[0]}x{original_size[1]}"
            f"\t{spec.output_name}\t{spec.target_size[0]}x{spec.target_size[1]}"
            f"\tsource_sha256={sha256(source_path)}\toutput_sha256={sha256(output_path)}"
        )


if __name__ == "__main__":
    main()

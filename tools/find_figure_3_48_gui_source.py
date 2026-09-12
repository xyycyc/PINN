from __future__ import annotations

import io
import hashlib
import math
import zipfile
from pathlib import Path

from PIL import Image, ImageFilter


ROOT = Path(r"D:\Desktop\code\ai_model")
TARGET_PATH = ROOT / "_inspection_docx" / "media" / "image49.png"
INSPECTION_DIR = ROOT / "_inspection_fig348"

DOCX_FILES = tuple(
    sorted(
        {
    Path(r"D:\Desktop\技术报告合稿-0725_终版.docx"),
    Path(r"D:\Desktop\待交付\基于人工智能的温度场反演方法研究情况.docx"),
    Path(r"D:\Desktop\待交付\技术报告合稿初稿-0725_熊.docx"),
    Path(r"D:\Desktop\待交付\技术报告合稿初稿-0725.docx"),
    Path(r"D:\Desktop\待交付\技术报告合稿-0725_终版.docx"),
    Path(r"D:\Desktop\待交付\模块三-程序测试记录表.docx"),
    ROOT / "文本" / "AI温度场反演测试大纲.docx",
    ROOT / "修改.docx",
    ROOT / "ai反演.docx",
    ROOT / "content.docx",
        }
        | set(Path(r"D:\Desktop\待交付").glob("*.docx"))
        | set(ROOT.glob("*.docx"))
    )
)


def normalized_vector(image: Image.Image, size: tuple[int, int] = (96, 66)) -> list[float]:
    gray = image.convert("L").resize(size, Image.Resampling.LANCZOS)
    gray = gray.filter(ImageFilter.GaussianBlur(0.45))
    values = [float(value) for value in gray.getdata()]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    deviation = math.sqrt(variance) or 1.0
    return [(value - mean) / deviation for value in values]


def correlation(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right)) / len(left)


def main() -> None:
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(TARGET_PATH) as composite:
        # Exact lower GUI rectangle in the already-composited Figure 3.48 image.
        target = composite.crop((10, 163, 704, 635))
        target.save(INSPECTION_DIR / "figure_3_48_lower_gui_crop.png")
        target_vector = normalized_vector(target)
        target_aspect = target.width / target.height

    candidates: list[tuple[float, float, str, str, tuple[int, int], bytes]] = []
    for docx_path in DOCX_FILES:
        if not docx_path.exists():
            continue
        try:
            with zipfile.ZipFile(docx_path) as archive:
                for member in archive.namelist():
                    if not member.startswith("word/media/"):
                        continue
                    try:
                        payload = archive.read(member)
                        with Image.open(io.BytesIO(payload)) as candidate:
                            if candidate.width < 500 or candidate.height < 300:
                                continue
                            aspect = candidate.width / candidate.height
                            aspect_delta = abs(aspect - target_aspect) / target_aspect
                            if aspect_delta > 0.25:
                                continue
                            score = correlation(target_vector, normalized_vector(candidate))
                            candidates.append((score, aspect_delta, str(docx_path), member, candidate.size, payload))
                    except Exception:
                        continue
        except zipfile.BadZipFile:
            continue

    candidates.sort(reverse=True)
    extracted_hashes: set[str] = set()
    for score, aspect_delta, docx, member, size, payload in candidates[:30]:
        print(
            f"score={score:.6f}\taspect_delta={aspect_delta:.4f}\t"
            f"size={size[0]}x{size[1]}\tdocx={docx}\tmember={member}"
        )
        if score >= 0.67:
            digest = hashlib.sha256(payload).hexdigest()
            if digest not in extracted_hashes:
                extracted_hashes.add(digest)
                suffix = Path(member).suffix.lower() or ".bin"
                extracted_path = INSPECTION_DIR / f"gui_source_{size[0]}x{size[1]}_{digest[:10]}{suffix}"
                extracted_path.write_bytes(payload)


if __name__ == "__main__":
    main()

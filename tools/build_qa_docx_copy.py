from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("replacement_dir", type=Path)
    args = parser.parse_args()

    replacements = {
        "word/media/image48.png": args.replacement_dir / "图3.47_W30Mo70训练数据目录与manifest_白底重绘.png",
        "word/media/image56.png": args.replacement_dir / "图3.55_AI推理耗时日志_白底摘要.png",
        "word/media/image57.png": args.replacement_dir / "图3.56_Python与Fortran协同_流程伪代码.png",
        "word/media/image58.png": args.replacement_dir / "图3.57_金属基复合材料manifest_白底重绘.png",
        "word/media/image59.png": args.replacement_dir / "图3.58_硅基复合材料manifest_白底重绘.png",
    }
    missing = [str(path) for path in replacements.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing replacement files: " + ", ".join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    replaced: set[str] = set()
    with zipfile.ZipFile(args.source, "r") as src, zipfile.ZipFile(args.output, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            replacement = replacements.get(info.filename)
            if replacement is not None:
                data = replacement.read_bytes()
                replaced.add(info.filename)
            dst.writestr(info, data)

    expected = set(replacements)
    if replaced != expected:
        raise RuntimeError(f"Replacement mismatch: expected {sorted(expected)}, got {sorted(replaced)}")
    print(args.output)


if __name__ == "__main__":
    main()

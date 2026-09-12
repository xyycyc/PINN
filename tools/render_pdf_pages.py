from __future__ import annotations

import argparse
from pathlib import Path

import pypdfium2 as pdfium


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("pages", nargs="+", type=int, help="1-based page numbers")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    document = pdfium.PdfDocument(args.pdf)
    scale = args.dpi / 72.0
    for page_number in args.pages:
        if not 1 <= page_number <= len(document):
            raise ValueError(f"Page {page_number} outside 1..{len(document)}")
        page = document[page_number - 1]
        bitmap = page.render(scale=scale, rotation=0)
        image = bitmap.to_pil()
        out = args.out_dir / f"page-{page_number}.png"
        image.save(out, dpi=(args.dpi, args.dpi))
        print(out)


if __name__ == "__main__":
    main()

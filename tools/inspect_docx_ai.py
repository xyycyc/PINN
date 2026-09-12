from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image, ImageStat


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def qn(prefix: str, name: str) -> str:
    return f"{{{NS[prefix]}}}{name}"


def node_text(node: etree._Element) -> str:
    parts: list[str] = []
    for child in node.iter():
        if child.tag in {qn("w", "t"), qn("w", "delText"), qn("w", "instrText")}:
            parts.append(child.text or "")
        elif child.tag == qn("w", "tab"):
            parts.append("\t")
        elif child.tag in {qn("w", "br"), qn("w", "cr")}:
            parts.append("\n")
    return "".join(parts).strip()


def paragraph_details(p: etree._Element, rels: dict[str, str]) -> dict:
    style_nodes = p.xpath("./w:pPr/w:pStyle", namespaces=NS)
    style = style_nodes[0].get(qn("w", "val"), "") if style_nodes else ""
    outline_nodes = p.xpath("./w:pPr/w:outlineLvl", namespaces=NS)
    outline = outline_nodes[0].get(qn("w", "val"), "") if outline_nodes else ""
    page_break = bool(p.xpath(".//w:br[@w:type='page'] | .//w:lastRenderedPageBreak", namespaces=NS))
    images: list[dict] = []
    for blip in p.xpath(".//a:blip", namespaces=NS):
        rid = blip.get(qn("r", "embed")) or blip.get(qn("r", "link")) or ""
        drawing = blip
        while drawing is not None and drawing.tag != qn("w", "drawing"):
            drawing = drawing.getparent()
        extent_nodes = drawing.xpath(".//wp:extent", namespaces=NS) if drawing is not None else []
        docpr_nodes = drawing.xpath(".//wp:docPr", namespaces=NS) if drawing is not None else []
        extent = extent_nodes[0] if extent_nodes else None
        docpr = docpr_nodes[0] if docpr_nodes else None
        images.append(
            {
                "rid": rid,
                "target": rels.get(rid, ""),
                "width_inches": round(int(extent.get("cx")) / 914400, 3) if extent is not None else None,
                "height_inches": round(int(extent.get("cy")) / 914400, 3) if extent is not None else None,
                "name": docpr.get("name", "") if docpr is not None else "",
                "title": docpr.get("title", "") if docpr is not None else "",
                "descr": docpr.get("descr", "") if docpr is not None else "",
            }
        )
    return {
        "kind": "paragraph",
        "style": style,
        "outline": outline,
        "text": node_text(p),
        "page_break": page_break,
        "images": images,
    }


def table_details(tbl: etree._Element, rels: dict[str, str]) -> dict:
    rows: list[list[str]] = []
    images: list[dict] = []
    for tr in tbl.xpath("./w:tr", namespaces=NS):
        row: list[str] = []
        for tc in tr.xpath("./w:tc", namespaces=NS):
            row.append(" / ".join(filter(None, (node_text(p) for p in tc.xpath(".//w:p", namespaces=NS)))))
            for p in tc.xpath(".//w:p", namespaces=NS):
                images.extend(paragraph_details(p, rels)["images"])
        rows.append(row)
    flat = " || ".join(" | ".join(row) for row in rows)
    return {"kind": "table", "style": "", "outline": "", "text": flat, "page_break": False, "images": images, "rows": rows}


def image_metrics(path: Path) -> dict:
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
        thumb = rgb.copy()
        thumb.thumbnail((600, 600))
        gray = thumb.convert("L")
        hist = gray.histogram()
        total = max(1, sum(hist))
        mean = ImageStat.Stat(gray).mean[0]
        dark_50 = sum(hist[:51]) / total
        dark_100 = sum(hist[:101]) / total
        white_245 = sum(hist[245:]) / total
        edge: list[int] = []
        w, h = gray.size
        if w and h:
            edge.extend(gray.crop((0, 0, w, 1)).getdata())
            edge.extend(gray.crop((0, h - 1, w, h)).getdata())
            edge.extend(gray.crop((0, 0, 1, h)).getdata())
            edge.extend(gray.crop((w - 1, 0, w, h)).getdata())
        return {
            "pixel_width": im.width,
            "pixel_height": im.height,
            "format": im.format or "",
            "mode": im.mode,
            "dpi_x": (im.info.get("dpi") or (None, None))[0],
            "dpi_y": (im.info.get("dpi") or (None, None))[1],
            "mean_luminance": round(mean, 2),
            "dark_fraction_50": round(dark_50, 4),
            "dark_fraction_100": round(dark_100, 4),
            "white_fraction_245": round(white_245, 4),
            "edge_median_luminance": round(statistics.median(edge), 2) if edge else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = args.out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.docx) as zf:
        document = etree.fromstring(zf.read("word/document.xml"))
        rel_tree = etree.fromstring(zf.read("word/_rels/document.xml.rels"))
        rels = {
            rel.get("Id", ""): rel.get("Target", "")
            for rel in rel_tree.xpath("./pr:Relationship", namespaces=NS)
        }
        media_names = sorted(name for name in zf.namelist() if name.startswith("word/media/") and not name.endswith("/"))
        for name in media_names:
            (media_dir / Path(name).name).write_bytes(zf.read(name))

    body = document.find(qn("w", "body"))
    entries: list[dict] = []
    for body_index, child in enumerate(body or []):
        if child.tag == qn("w", "p"):
            item = paragraph_details(child, rels)
        elif child.tag == qn("w", "tbl"):
            item = table_details(child, rels)
        else:
            continue
        item["body_index"] = body_index
        entries.append(item)

    target_to_occurrences: dict[str, list[dict]] = {}
    for idx, entry in enumerate(entries):
        for image in entry["images"]:
            target = image["target"]
            occurrence = {
                "entry_index": idx,
                "body_index": entry["body_index"],
                "container_text": entry["text"],
                **image,
            }
            target_to_occurrences.setdefault(target, []).append(occurrence)

    inventory: list[dict] = []
    for media_name in media_names:
        target = "media/" + Path(media_name).name
        path = media_dir / Path(media_name).name
        row = {"target": target, "file_name": path.name, "file_bytes": path.stat().st_size}
        try:
            row.update(image_metrics(path))
        except Exception as exc:
            row["metric_error"] = f"{type(exc).__name__}: {exc}"
        row["occurrences"] = target_to_occurrences.get(target, [])
        inventory.append(row)

    (args.out_dir / "structure.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "image_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = [
        "file_name", "file_bytes", "pixel_width", "pixel_height", "format", "mode", "dpi_x", "dpi_y",
        "mean_luminance", "dark_fraction_50", "dark_fraction_100", "white_fraction_245",
        "edge_median_luminance", "occurrence_count", "body_indices", "display_sizes_inches",
    ]
    with (args.out_dir / "image_inventory.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for row in inventory:
            occ = row.get("occurrences", [])
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in csv_fields},
                    "occurrence_count": len(occ),
                    "body_indices": ",".join(str(x["body_index"]) for x in occ),
                    "display_sizes_inches": ";".join(f'{x.get("width_inches")}x{x.get("height_inches")}' for x in occ),
                }
            )

    nonempty = [i for i, entry in enumerate(entries) if entry["text"] or entry["images"]]
    context_lines: list[str] = []
    for row in inventory:
        for occurrence in row.get("occurrences", []):
            idx = occurrence["entry_index"]
            pos = nonempty.index(idx) if idx in nonempty else 0
            neighbor_ids = nonempty[max(0, pos - 4): min(len(nonempty), pos + 5)]
            context_lines.append(
                f'## {row["file_name"]} body={occurrence["body_index"]} '
                f'pixels={row.get("pixel_width")}x{row.get("pixel_height")} '
                f'display={occurrence.get("width_inches")}x{occurrence.get("height_inches")}in '
                f'mean={row.get("mean_luminance")} dark100={row.get("dark_fraction_100")}'
            )
            for neighbor_idx in neighbor_ids:
                neighbor = entries[neighbor_idx]
                marker = ">" if neighbor_idx == idx else " "
                text = neighbor["text"].replace("\n", " ")
                context_lines.append(f'{marker} [{neighbor["body_index"]}] {neighbor["style"]}: {text[:500]}')
            context_lines.append("")
    (args.out_dir / "image_context.md").write_text("\n".join(context_lines), encoding="utf-8-sig")

    keywords = ("人工智能", "机器学习", "深度学习", "神经网络", "AI", "训练", "伪代码", "代码")
    print(f"entries={len(entries)} media={len(media_names)} occurrences={sum(len(x.get('occurrences', [])) for x in inventory)}")
    print("KEYWORD ENTRIES")
    for entry in entries:
        text = entry["text"].replace("\n", " ")
        if any(keyword.lower() in text.lower() for keyword in keywords):
            print(f'[{entry["body_index"]}] {entry["style"]}: {text[:400]}')
    print("DARK IMAGES")
    for row in inventory:
        if row.get("mean_luminance", 255) < 115 or row.get("dark_fraction_100", 0) > 0.55:
            print(
                row["file_name"],
                f'{row.get("pixel_width")}x{row.get("pixel_height")}',
                f'mean={row.get("mean_luminance")}',
                f'dark100={row.get("dark_fraction_100")}',
                "body=" + ",".join(str(x["body_index"]) for x in row.get("occurrences", [])),
            )


if __name__ == "__main__":
    main()

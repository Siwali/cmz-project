"""Create a tiny manually reviewed example set from the processed dataset ZIP.

Selected sequences are explicit so that the demo does not inherit every frame
label blindly from the source clip. Detector boxes are shown as automatic
proposals; the dish ROI on E03 is a manually drawn example region.
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

EXAMPLES = [
    ("E03", 5.6, "Eating"), ("E03", 6.6, "Eating"), ("E03", 7.6, "Eating"),
    ("L03", 3.3, "Locomotion"), ("L03", 4.1, "Locomotion"), ("L03", 4.3, "Locomotion"),
    ("R05", 37.6, "Resting"), ("R05", 38.6, "Resting"), ("R05", 39.6, "Resting"),
]
# Polygon around the visible dish in E03's 640x640 letterboxed full-scene frame.
E03_DISH_ROI = [(207, 277), (261, 275), (279, 289), (266, 305), (218, 308), (203, 294)]
COLORS = {"Eating": (20, 190, 60), "Locomotion": (40, 120, 255), "Resting": (200, 40, 190)}


def full_box(row):
    scale = 640 / max(float(row.width), float(row.height))
    left = (640 - round(float(row.width) * scale)) // 2
    top = (640 - round(float(row.height) * scale)) // 2
    return [float(row.box_x1)*scale+left, float(row.box_y1)*scale+top,
            float(row.box_x2)*scale+left, float(row.box_y2)*scale+top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = []
    tiles = []
    with zipfile.ZipFile(args.zip) as zf:
        m = pd.read_csv(zf.open("metadata/manifest.csv"), low_memory=False)
        for video_id, sec, label in EXAMPLES:
            g = m[m.video_id == video_id]
            row = g.iloc[(g.timestamp_sec - sec).abs().argmin()]
            image = Image.open(io.BytesIO(zf.read(row.full_path))).convert("RGB")
            d = ImageDraw.Draw(image)
            if video_id == "E03":
                d.polygon(E03_DISH_ROI, outline=(0, 220, 255), width=4)
            box_ok = all(pd.notna(v) for v in [row.box_x1, row.box_y1, row.box_x2, row.box_y2])
            if box_ok:
                d.rectangle(full_box(row), outline=COLORS[label], width=4)
            # Caption sits in the letterbox margin, away from the scene.
            d.rectangle([0, 0, 250, 24], fill="black")
            d.text((5, 5), f"{label} | {video_id} | {row.timestamp_sec:.1f}s", fill="white")
            if video_id == "E03":
                d.rectangle([0, 24, 240, 44], fill="black")
                d.text((5, 27), "cyan = food dish ROI; green = civet box", fill=(0, 220, 255))
            time_tag = f"{row.timestamp_sec:.1f}".replace(".", "p")
            name = f"{label}_{video_id}_{time_tag}s.jpg"
            image.save(out / name, quality=94)
            # Use coordinates in the 640x640 processed full-frame space.
            bbox = full_box(row) if box_ok else [None] * 4
            labels.append({
                "image": name, "video_id": video_id,
                "timestamp_sec": round(float(row.timestamp_sec), 3),
                "behavior_label": label, "label_basis": "manually reviewed short sequence",
                "bbox_source": "automatic detector proposal; visually checked" if box_ok else "missing",
                "bbox_x1": round(bbox[0], 1) if bbox[0] is not None else "",
                "bbox_y1": round(bbox[1], 1) if bbox[1] is not None else "",
                "bbox_x2": round(bbox[2], 1) if bbox[2] is not None else "",
                "bbox_y2": round(bbox[3], 1) if bbox[3] is not None else "",
                "tray_roi": str(E03_DISH_ROI) if video_id == "E03" else "",
            })
            tile = image.copy()
            tile.thumbnail((300, 300))
            tiles.append((label, video_id, float(row.timestamp_sec), tile))

    cols, tile_w, tile_h = 3, 300, 324
    sheet = Image.new("RGB", (cols*tile_w, 3*tile_h+32), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 7), "Small manually reviewed behavior annotation demo | 3 frames per class", fill="black")
    for i, (label, video_id, sec, image) in enumerate(tiles):
        x, y = (i % cols)*tile_w, 32+(i//cols)*tile_h
        sheet.paste(image, (x, y))
        ImageDraw.Draw(sheet).text((x+4, y+302), f"{label} | {video_id} | {sec:.1f}s", fill="black")
    sheet.save(out / "manual_annotation_demo.jpg", quality=92)
    pd.DataFrame(labels).to_csv(out / "manual_annotation_demo.csv", index=False)
    print(f"Wrote {len(labels)} reviewed frames and CSV to {out}")
    print("Drinking is not present in this ZIP; examples are Eating, Locomotion, Resting.")


if __name__ == "__main__":
    main()

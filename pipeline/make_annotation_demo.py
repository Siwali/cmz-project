"""Create a small, human-reviewable annotation demo from the processed ZIP.

The source labels are inherited from clip folders; this tool presents them as
provisional frame labels and draws the dataset's detector boxes for inspection.
It does not claim the labels or boxes have been manually verified.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CLASSES = ("Eating", "Locomotion", "Resting")
COLORS = {"Eating": (20, 160, 40), "Locomotion": (20, 90, 220), "Resting": (170, 50, 180)}


def choose_frames(manifest: pd.DataFrame, n: int) -> pd.DataFrame:
    m = manifest.copy()
    m = m[m["readable"].astype(bool)]
    m = m[m[["box_x1", "box_y1", "box_x2", "box_y2"]].notna().all(axis=1)]
    if "person_in_frame" in m:
        m = m[~m["person_in_frame"].fillna(False).astype(bool)]
    quality = [c for c in ("too_dark", "too_bright", "low_contrast", "blurry") if c in m]
    if quality:
        m = m[~m[quality].fillna(False).astype(bool).any(axis=1)]
    if "md_conf" in m:
        m = m.sort_values("md_conf", ascending=False)
    chosen = []
    for label in CLASSES:
        pool = m[m.label == label]
        if "motion_pct" in pool:
            # Prefer visually representative moments: active frame changes for
            # Eating/Locomotion, minimal changes for Resting.
            pool = pool.sort_values(
                "motion_pct", ascending=(label == "Resting")
            )
        elif "md_conf" in pool:
            pool = pool.sort_values("md_conf", ascending=False)
        # First take one high-quality frame per clip, then fill remaining slots
        # with temporally separated frames from clips with multiple examples.
        first = pool.groupby("video_id", sort=True, as_index=False).head(1)
        if "motion_pct" in first:
            first = first.sort_values("motion_pct", ascending=(label == "Resting"))
        elif "md_conf" in first:
            first = first.sort_values("md_conf", ascending=False)
        selected = list(first.head(n).index)
        remaining = pool.drop(index=selected).copy()
        for idx in selected:
            row = pool.loc[idx]
            remaining = remaining[
                ~((remaining.video_id == row.video_id) &
                  ((remaining.timestamp_sec - row.timestamp_sec).abs() < 5))
            ]
        for idx in remaining.index:
            if len(selected) >= n:
                break
            selected.append(idx)
            row = pool.loc[idx]
            remaining = remaining[
                ~((remaining.video_id == row.video_id) &
                  ((remaining.timestamp_sec - row.timestamp_sec).abs() < 5))
            ]
        chosen.append(pool.loc[selected].copy())
    return pd.concat(chosen, ignore_index=True)


def annotated_frame(zf: zipfile.ZipFile, row, size: int = 256) -> Image.Image:
    img = Image.open(io.BytesIO(zf.read(row.full_path))).convert("RGB")
    scale = 640 / max(float(row.width), float(row.height))
    left = (640 - round(float(row.width) * scale)) // 2
    top = (640 - round(float(row.height) * scale)) // 2
    box = [
        float(row.box_x1) * scale + left,
        float(row.box_y1) * scale + top,
        float(row.box_x2) * scale + left,
        float(row.box_y2) * scale + top,
    ]
    color = COLORS[row.label]
    d = ImageDraw.Draw(img)
    d.rectangle(box, outline=color, width=5)
    img = img.resize((size, size))
    tile = Image.new("RGB", (size, size + 48), "white")
    tile.paste(img, (0, 0))
    d = ImageDraw.Draw(tile)
    d.text((5, size + 3), f"{row.label} | {row.video_id} | {row.timestamp_sec:.1f}s", fill="black")
    conf = getattr(row, "md_conf", None)
    if conf is not None and pd.notna(conf):
        d.text((5, size + 23), f"detector confidence: {float(conf):.2f}", fill="#555555")
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="cmz_processed_dataset.zip")
    ap.add_argument("--out", required=True, help="output folder (keep out of public git)")
    ap.add_argument("--per-class", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as zf:
        manifest = pd.read_csv(zf.open("metadata/manifest.csv"), low_memory=False)
        selected = choose_frames(manifest, args.per_class)
        selected.to_csv(out / "annotation_demo_manifest.csv", index=False)
        tiles = []
        review_rows = []
        for row in selected.itertuples(index=False):
            tile = annotated_frame(zf, row)
            tiles.append(tile)
            safe_time = f"{float(row.timestamp_sec):06.1f}".replace(".", "p")
            image_name = f"{row.label}_{row.video_id}_{safe_time}s.jpg"
            tile.save(out / image_name, quality=94)
            review_rows.append({
                "image": image_name,
                "video_id": row.video_id,
                "second": round(float(row.timestamp_sec), 1),
                "proposed_label_from_clip": row.label,
                "confirmed_label": "",
                "box_correct": "",
                "notes": "",
            })
        pd.DataFrame(review_rows).to_csv(out / "annotation_demo_review.csv", index=False)

    cols = args.per_class
    rows = len(CLASSES)
    sheet = Image.new("RGB", (cols * 256, rows * 304 + 35), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), "Provisional frame labels from clip labels; boxes are automatic and need review", fill="black")
    for i, tile in enumerate(tiles):
        x = (i % cols) * 256
        y = 30 + (i // cols) * 304
        sheet.paste(tile, (x, y))
    sheet.save(out / "annotation_demo_contact_sheet.jpg", quality=92)
    print(f"Wrote {len(selected)} provisional examples to {out}")
    print(selected.groupby("label").size().to_string())
    print(f"Contact sheet: {out / 'annotation_demo_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()

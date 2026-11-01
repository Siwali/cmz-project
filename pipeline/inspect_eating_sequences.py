"""Build full-scene timelines for the ZIP's Eating clips for human review."""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=1.0, help="seconds between displayed frames")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        m = pd.read_csv(zf.open("metadata/manifest.csv"), low_memory=False)
        m = m[(m.label == "Eating") & m.readable.astype(bool)].copy()
        m["sec"] = m.timestamp_sec.round().astype(int)
        m = m.sort_values("timestamp_sec").drop_duplicates(["video_id", "sec"])
        tile_w, tile_h, cols, rows_per_page = 192, 216, 5, 5
        per_page = cols * rows_per_page
        for vid, group in m.groupby("video_id", sort=True):
            group = group.sort_values("sec")
            for page_idx, offset in enumerate(range(0, len(group), per_page), 1):
                page = group.iloc[offset:offset + per_page]
                sheet = Image.new("RGB", (cols * tile_w, rows_per_page * tile_h + 32), "white")
                d = ImageDraw.Draw(sheet)
                d.text((6, 6), f"{vid}: source clip label Eating | page {page_idx} | full scene", fill="black")
                for i, r in enumerate(page.itertuples(index=False)):
                    im = Image.open(io.BytesIO(zf.read(r.full_path))).convert("RGB")
                    # Draw the detector's box after mapping raw coordinates to 640-square letterbox coordinates.
                    vals = [r.box_x1, r.box_y1, r.box_x2, r.box_y2]
                    if all(pd.notna(v) for v in vals):
                        scale = 640 / max(float(r.width), float(r.height))
                        left = (640 - round(float(r.width) * scale)) // 2
                        top = (640 - round(float(r.height) * scale)) // 2
                        box = [float(r.box_x1)*scale+left, float(r.box_y1)*scale+top,
                               float(r.box_x2)*scale+left, float(r.box_y2)*scale+top]
                        ImageDraw.Draw(im).rectangle(box, outline=(20, 160, 40), width=4)
                    im.thumbnail((tile_w, tile_w))
                    x, y = (i % cols) * tile_w, 32 + (i // cols) * tile_h
                    sheet.paste(im, (x, y))
                    ImageDraw.Draw(sheet).text((x + 4, y + tile_w + 2), f"{r.sec}s  motion={r.motion_pct:.2f}%", fill="black")
                sheet.save(out / f"{vid}_timeline_p{page_idx}.jpg", quality=90)
            print(vid, len(group), "one-second frames")
    print("wrote full-scene review sheets to", out)


if __name__ == "__main__":
    main()

"""Contact sheets for checking the clip labels of the processed dataset.

One thumbnail per second of each video, in time order (the civet crop when there is one,
otherwise the whole frame with a red border), 80 thumbnails per page. Look through the
sheets and list wrong stretches in review_edits.csv:

  video_id,start_s,end_s,action
  R03,40,55,drop                  # civet walks away inside a "Resting" clip
  E02,0,8,relabel:Locomotion

Usage: python make_review_sheets.py --base /content/CMZ [--out /content/CMZ/review_sheets]
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

COLS, ROWS, THUMB = 10, 8, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="folder holding metadata/ and processed/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out) if args.out else base / "review_sheets"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(base / "metadata" / "manifest.csv", low_memory=False)
    m = m[m["readable"] == True]  # noqa: E712

    per_page = COLS * ROWS
    for video_id, g in m.groupby("video_id"):
        g = g.sort_values("timestamp_sec")
        g = g.assign(sec=g["timestamp_sec"].astype(int)).drop_duplicates("sec")
        label = g["label"].iloc[0]
        for page, start in enumerate(range(0, len(g), per_page), 1):
            chunk = g.iloc[start:start + per_page]
            sheet = Image.new("RGB", (COLS * THUMB, ROWS * (THUMB + 14) + 16), "white")
            d = ImageDraw.Draw(sheet)
            d.text((4, 2), f"{video_id}  label={label}  page {page}", fill="black")
            for i, r in enumerate(chunk.itertuples()):
                x, y = (i % COLS) * THUMB, 16 + (i // COLS) * (THUMB + 14)
                has_crop = isinstance(r.crop_path, str) and r.crop_path
                src = base / (r.crop_path if has_crop else r.full_path)
                try:
                    im = Image.open(src).convert("RGB").resize((THUMB, THUMB))
                except OSError:
                    continue
                if not has_crop:
                    ImageDraw.Draw(im).rectangle([0, 0, THUMB - 1, THUMB - 1], outline="red", width=3)
                sheet.paste(im, (x, y))
                d.text((x + 2, y + THUMB), f"{r.sec}s", fill="black")
            sheet.save(out / f"{video_id}_p{page}.jpg", quality=85)
        print(video_id, label, f"{len(g)} s")

    template = out / "review_edits.csv"
    if not template.exists():
        template.write_text("video_id,start_s,end_s,action\n")
    print(f"sheets and review_edits.csv template in {out}")


if __name__ == "__main__":
    main()

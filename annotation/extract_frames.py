"""Extract still frames for annotation from windows listed in a CSV.

windows.csv columns: video, start_s, end_s, tag
  video   path to the source video (elapsed seconds are counted from its start)
  start_s, end_s   window in elapsed seconds
  tag     free text (e.g. day_active, night_box, empty_negative); written to the manifest

Frames are named <video stem>_s<elapsed second, 6 digits>.jpg, so the elapsed
second (and with it the CCTV clock time) can always be recovered from the name.

Usage: python extract_frames.py windows.csv --out frames --every 1
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd


def extract(video, start_s, end_s, out_dir, every_s):
    stem = Path(video).stem
    pattern = str(out_dir / f"{stem}_s%06d.jpg")
    # fps=1/every_s: frame k of the output is at start_s + k*every_s. Number frames by elapsed second.
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", str(start_s), "-t", str(end_s - start_s), "-i", str(video),
           "-vf", f"fps=1/{every_s}", "-q:v", "2",
           "-start_number", str(start_s), pattern]
    if every_s != 1:
        raise SystemExit("only --every 1 is supported (frame numbers are elapsed seconds)")
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob(f"{stem}_s*.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--every", type=int, default=1, help="seconds between frames")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in pd.read_csv(args.windows).itertuples():
        before = set(out_dir.glob("*.jpg"))
        extract(w.video, int(w.start_s), int(w.end_s), out_dir, args.every)
        for f in sorted(set(out_dir.glob("*.jpg")) - before):
            rows.append({"file": f.name, "video": Path(w.video).name,
                         "elapsed_s": int(f.stem.rsplit("_s", 1)[1]), "tag": w.tag})
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"{len(rows)} frames -> {out_dir}")


if __name__ == "__main__":
    main()

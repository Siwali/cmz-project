"""Cut short labelled clips out of the long CCTV videos, laid out for the frame-extraction
notebook (CMZ_Frame_Extraction_and_Preprocessing.ipynb):

  <base>/raw_videos/<behavior>/<video_id>.mp4    one clip per timeline event
  <base>/metadata/videos.csv                     pre-filled with the camera and notes
  <base>/metadata/video_splits.csv               time-based train/test split

Clock times in the timeline are the CCTV clock; every recording starts at 00:00:00, so
clock time = elapsed time in the file.

Usage:
  python make_clips.py annotation/timeline_events.csv --raw /content/drive/MyDrive/CMZprojectphase2/data/raw \
      --base /content/drive/MyDrive/CMZ [--dry-run]
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd

SKIP_BEHAVIORS = {"not_visible", "resting_in_box", "unsure"}  # not clip classes (hidden / unknown)
MIN_CLIP_S = 3
# First-pass classes: three folders, finer behaviors are merged or left out (see --all-classes)
THREE_CLASSES = {"locomotion": "locomotion", "feeding": "feeding",
                 "standing_still": "still", "sitting": "still", "resting_lying": "still"}
TEST_FROM_S = 45 * 60  # events starting at or after 00:45:00 are the held-out test visit


def to_s(clock):
    h, m, s = (int(x) for x in clock.split(":"))
    return h * 3600 + m * 60 + s


def find_video(raw, camera):
    hits = sorted(Path(raw).rglob(f"{camera}_*.mp4"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {camera}_*.mp4 under {raw}, found {len(hits)}")
    return hits[0]


def cut(video, start_s, end_s, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(start_s), "-i", str(video), "-t", str(end_s - start_s),
         "-map", "0:v:0", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        check=True,
    )


def merge_csv(path, new, key="video_id"):
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        new = pd.concat([old, new[~new[key].isin(old[key])]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--raw", required=True, help="folder holding cam1..cam4 with the long videos")
    ap.add_argument("--base", required=True, help="the notebook's BASE_DIR (the CMZ folder)")
    ap.add_argument("--all-classes", action="store_true",
                    help="keep every behavior as its own class instead of the 3-class set")
    ap.add_argument("--dry-run", action="store_true", help="list the clips without cutting")
    args = ap.parse_args()

    base = Path(args.base)
    events = pd.read_csv(args.events, dtype=str, keep_default_na=False)
    videos, splits, skipped = [], [], []
    for e in events.itertuples():
        why = None
        label = e.behavior if args.all_classes else THREE_CLASSES.get(e.behavior)
        if e.behavior in SKIP_BEHAVIORS:
            why = "not a clip class"
        elif label is None:
            why = "not in the 3-class set"
        elif not e.clock_start or not e.clock_end:
            why = "no start/end time"
        elif to_s(e.clock_end) - to_s(e.clock_start) < MIN_CLIP_S:
            why = f"shorter than {MIN_CLIP_S} s"
        if why:
            skipped.append(f"{e.camera} {e.clock_start or '?'} {e.behavior}: {why}")
            continue

        start, end = to_s(e.clock_start), to_s(e.clock_end)
        video_id = f"{e.camera}_{e.clock_start.replace(':', '')}"
        rel = f"{label}/{video_id}.mp4"
        print(f"{video_id:<14} {label:<20} {end - start:>4} s  ({e.behavior})")
        if not args.dry_run:
            dst = base / "raw_videos" / rel
            if not dst.exists():
                cut(find_video(args.raw, e.camera), start, end, dst)
        note = f"{e.behavior}: {e.notes}" + (f" [check: {e.check}]" if e.check else "")
        videos.append({"video_id": video_id, "label": label, "file": rel, "camera": e.camera,
                       "start_sec": "", "end_sec": "", "notes": note})
        splits.append({"video_id": video_id, "label": label,
                       "split": "test" if start >= TEST_FROM_S else "train", "fold": -1})

    print(f"\n{len(videos)} clips; skipped {len(skipped)}:")
    for s in skipped:
        print("  ", s)
    summary = pd.DataFrame(splits)
    print("\nclips per label and split:")
    print(pd.crosstab(summary.label, summary.split).to_string())

    if not args.dry_run:
        merge_csv(base / "metadata" / "videos.csv", pd.DataFrame(videos))
        merge_csv(base / "metadata" / "video_splits.csv", summary.astype({"fold": str}))


if __name__ == "__main__":
    main()

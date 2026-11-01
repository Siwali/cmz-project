"""Run a trained YOLO model over a CCTV video and write the per-second CSV that
civet_rules.py reads: second, clock_time, visible, zone, behavior, confidence, motion_px.

The model's class names are used as the behavior labels (see ANNOTATION_GUIDE.md).
One frame per second is sampled. Zones come from polygons in a ROI json:
  {"box_entrance": [[x, y], ...], "door": [...], "water_bowl": [...], ...}
with x, y normalized to 0-1 (see pipeline/rois/); the first polygon (in the order
below) containing the box centre wins, otherwise the zone is "floor".

Usage:
  python detect_video.py A04.mp4 best.pt rois_A04.json --clock 00:00:00 \
      --start-s 0 --end-s 300 --out out/A04_0000_0300.csv
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ZONE_ORDER = ["box_entrance", "box_body", "door", "water_bowl", "tray", "tire_bed"]


def load_rois(path, width, height):
    """Polygons are stored normalized (0-1), so one file fits any video resolution."""
    raw = json.loads(Path(path).read_text())
    scale = np.array([width, height])
    return {name: np.round(np.array(raw[name]) * scale).astype(np.int32)
            for name in ZONE_ORDER if name in raw}


def zone_of(rois, cx, cy):
    for name, poly in rois.items():
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
            return name
    return "floor"


def detect(model, frames, conf):
    """Best civet detection per frame as (behavior, conf, cx, cy) or None."""
    out = []
    for r in model(frames, conf=conf, verbose=False):
        if len(r.boxes) == 0:
            out.append(None)
            continue
        i = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
        out.append((r.names[int(r.boxes.cls[i])], float(r.boxes.conf[i]),
                    (x1 + x2) / 2, (y1 + y2) / 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("weights")
    ap.add_argument("rois")
    ap.add_argument("--clock", default="00:00:00", help="CCTV clock time at second 0 of the video")
    ap.add_argument("--start-s", type=int, default=0)
    ap.add_argument("--end-s", type=int, required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    model = YOLO(args.weights)
    rois = load_rois(args.rois, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    stride = int(round(cap.get(cv2.CAP_PROP_FPS)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_s * stride)

    found = {}  # second -> detection tuple or None
    frames, secs = [], []

    def flush():
        for s, d in zip(secs, detect(model, frames, args.conf)):
            found[s] = d
        frames.clear()
        secs.clear()

    for s in range(args.start_s, args.end_s):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        secs.append(s)
        for _ in range(stride - 1):
            cap.grab()
        if len(frames) == args.batch:
            flush()
    if frames:
        flush()

    base = datetime.strptime(args.clock, "%H:%M:%S")
    rows, prev = [], None
    for s in range(args.start_s, args.end_s):
        d = found.get(s)
        row = {"second": s, "clock_time": (base + timedelta(seconds=s)).strftime("%H:%M:%S"),
               "visible": 0, "zone": "", "behavior": "", "confidence": 0.0, "motion_px": 0.0}
        if d is not None:
            behavior, c, cx, cy = d
            row.update(visible=1, zone=zone_of(rois, cx, cy), behavior=behavior,
                       confidence=round(c, 3))
            if prev is not None:
                row["motion_px"] = round(float(np.hypot(cx - prev[0], cy - prev[1])), 1)
            prev = (cx, cy)
        else:
            prev = None
        rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(rows)} s, {sum(r['visible'] for r in rows)} with a detection")


if __name__ == "__main__":
    main()

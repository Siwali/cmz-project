"""Shared data helpers for the 3-class behavior model, built on the processed dataset
(<base>/metadata/manifest.csv + <base>/processed/full640 + crop224).

All boxes are converted to full640 coordinates (the 640x640 letterboxed frames), so crops
can be re-cut tighter than the 224 crops that came with the dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["Eating", "Locomotion", "Resting"]
STATUS = {"Eating": "Active", "Locomotion": "Active", "Resting": "Inactive"}
RULES_NAME = {"Eating": "feeding", "Locomotion": "locomotion", "Resting": "resting_lying"}
FULL = 640


def raw_box_to_full(df, cols=("box_x1", "box_y1", "box_x2", "box_y2")):
    """Map boxes from raw-frame pixels to the 640x640 letterboxed frame."""
    scale = FULL / df[["width", "height"]].max(axis=1)
    left = (FULL - (df["width"] * scale).round()) // 2
    top = (FULL - (df["height"] * scale).round()) // 2
    x1, y1, x2, y2 = (df[c] for c in cols)
    return x1 * scale + left, y1 * scale + top, x2 * scale + left, y2 * scale + top


def apply_edits(m, edits_csv):
    """review_edits.csv: video_id,start_s,end_s,action  (action = drop | relabel:<Class>)."""
    if not edits_csv or not Path(edits_csv).exists():
        return m
    edits = pd.read_csv(edits_csv, dtype={"video_id": str}).dropna(subset=["video_id", "action"])
    m = m.copy()
    m["_drop"] = False
    for e in edits.itertuples():
        hit = (m.video_id == e.video_id) & (m.timestamp_sec >= e.start_s) & (m.timestamp_sec <= e.end_s)
        if e.action == "drop":
            m.loc[hit, "_drop"] = True
        elif str(e.action).startswith("relabel:"):
            new = e.action.split(":", 1)[1].strip()
            if new not in CLASSES:
                raise ValueError(f"unknown class in review_edits.csv: {new}")
            m.loc[hit, "label"] = new
        else:
            raise ValueError(f"unknown action in review_edits.csv: {e.action}")
    return m[~m["_drop"]].drop(columns="_drop")


def load_frames(base, edits_csv=None, redetect_csv=None):
    """Manifest rows with full640 boxes (friend's detector, then re-detections), review edits applied."""
    base = Path(base)
    m = pd.read_csv(base / "metadata" / "manifest.csv", low_memory=False)
    m = m[m["readable"] == True].copy()  # noqa: E712
    m["bx1"], m["by1"], m["bx2"], m["by2"] = raw_box_to_full(m)
    m["box_source"] = np.where(m["bx1"].notna(), "first_pass", "")
    if redetect_csv and Path(redetect_csv).exists():
        r = pd.read_csv(redetect_csv).drop_duplicates("full_path").set_index("full_path")
        miss = m["bx1"].isna() & m["full_path"].isin(r.index)
        for col in ("bx1", "by1", "bx2", "by2"):
            m.loc[miss, col] = m.loc[miss, "full_path"].map(r[col])
        m.loc[miss, "box_source"] = "redetect"
    return apply_edits(m, edits_csv)


def select_training(m):
    """Frames with a civet box, no person, and no quality flag."""
    bad = m[["too_dark", "too_bright", "low_contrast", "blurry"]].astype(bool).any(axis=1)
    ok = m["bx1"].notna() & ~m["person_in_frame"].astype(bool) & ~bad & m["label"].isin(CLASSES)
    return m[ok].reset_index(drop=True)


def tight_crop(img, box, margin=0.1, size=224, jitter=0.0, rng=None, pad=114):
    """Square crop around the civet with `margin` on each side (0.1 = 10 %)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(max(x2 - x1, y2 - y1) * (1 + 2 * margin), 32)
    if jitter and rng is not None:
        cx += rng.uniform(-jitter, jitter) * side
        cy += rng.uniform(-jitter, jitter) * side
        side *= 1 + rng.uniform(-jitter, jitter)
    s = int(round(side))
    left, top = int(round(cx - s / 2)), int(round(cy - s / 2))
    canvas = Image.new("RGB", (s, s), (pad,) * 3)
    canvas.paste(img, (-left, -top))
    return canvas.resize((size, size), Image.BILINEAR)


def shortcut_prediction(m):
    """Baseline that never looks at the civet: infrared -> Locomotion, still picture -> Resting, else Eating."""
    return np.where(m["lighting"] == "ir_night", "Locomotion",
                    np.where(m["near_duplicate"].astype(bool), "Resting", "Eating"))


def to_rules_csv(pred, video_id, out_path):
    """Per-second table in the format civet_rules.py reads (visible=0 where nothing was predicted)."""
    g = pred[pred.video_id == video_id].copy()
    g["second"] = g["timestamp_sec"].astype(int)
    probs = g.groupby("second")[[f"p_{c}" for c in CLASSES]].mean()
    probs = probs.reindex(range(0, int(probs.index.max()) + 1))
    seen_s = probs.notna().any(axis=1)
    best = probs.fillna(-1).idxmax(axis=1).str.replace("p_", "", regex=False)
    rows = []
    for s, label in best.items():
        seen = bool(seen_s.loc[s])
        rows.append({
            "second": s,
            "clock_time": f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}",
            "visible": int(seen), "zone": "",
            "behavior": RULES_NAME[label] if seen else "",
            "confidence": round(float(probs.loc[s].max()), 3) if seen else 0.0,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)

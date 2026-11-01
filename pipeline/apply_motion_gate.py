r"""Apply a motion gate to a run's CNN predictions, then rebuild the observation sheet.

A single-frame CNN confuses a stationary eating civet with a resting one. But a resting
civet is still, while an eating one moves (head, paws, chewing). This measures the
frame-to-frame motion inside the civet box and re-calls any 'Resting' second whose motion
exceeds --motion-thresh as Active (Eating or Locomotion, whichever the CNN preferred).

It runs as a post-process over run_local_behavior.py / notebook-03 output
(raw_frame_predictions.csv + the sampled frames), so no detector re-run is needed. It
writes the gated predictions, a smoothed gated observation sheet, and a gated preview MP4.

The threshold is hand-calibrated on daytime 1-fps clips of one camera. On that data, true
rest sits near motion 3.5-3.8, calm eating near 5, active eating 8+; grooming/paw-licking
overlaps calm eating (~4.5). 4.5 catches most eating and treats grooming as active.
Recalibrate for other sampling rates or infrared, where sensor noise raises the floor.

Usage:
  python apply_motion_gate.py --run local_runs\day_eating [--motion-thresh 4.5]
  python apply_motion_gate.py --run OUT --frames-dir /content/raw_pilot_frames  # notebook layout
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_observation_sheet import STATUS, build_sheet  # noqa: E402


def _crop_gray(path, box):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (64, 64)).astype(np.float32)


def box_motion(frames, pred):
    """Mean abs pixel diff (0-255) between a frame and the previous one, inside the civet box."""
    motion = [np.nan] * len(pred)
    for i in range(1, len(pred)):
        r0, r1 = pred.iloc[i - 1], pred.iloc[i]
        if not (np.isfinite(r1.x1) and np.isfinite(r0.x1)):
            continue
        a = _crop_gray(frames[i], (r1.x1, r1.y1, r1.x2, r1.y2))
        b = _crop_gray(frames[i - 1], (r1.x1, r1.y1, r1.x2, r1.y2))  # same box, previous frame
        if a is not None and b is not None:
            motion[i] = float(np.abs(a - b).mean())
    return motion


def gate_label(behavior, motion, p_eating, p_loco, thresh):
    """Only Resting flips, and only when the box moves more than the threshold."""
    if behavior == "Resting" and np.isfinite(motion) and motion >= thresh:
        return "Eating" if p_eating >= p_loco else "Locomotion"
    return behavior


def write_gated_preview(frames, pred, path, classify_conf=0.80):
    """Annotated preview using the gated label; box coloured by status, motion shown."""
    first = cv2.imread(str(frames[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    color = {"Active": (0, 220, 0), "Inactive": (220, 120, 0), "Unknown": (0, 180, 255)}
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, (w, h))
    for fp, r in zip(frames, pred.itertuples()):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        status = STATUS.get(r.behavior_gated, "Unknown")
        if np.isfinite(r.x1):
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color[status], 3)
        mot = f"{r.motion:.1f}" if np.isfinite(r.motion) else "-"
        text = f"{r.cctv_time} | {status}: {r.behavior_gated} | motion {mot} | MD {r.md_conf:.2f}"
        cv2.rectangle(frame, (0, 0), (min(w, 1180), 38), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with raw_frame_predictions.csv")
    ap.add_argument("--frames-dir", default=None, help="sampled frames (default <run>/frames)")
    ap.add_argument("--motion-thresh", type=float, default=4.5)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--min-bout", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=2)
    args = ap.parse_args()

    run = Path(args.run)
    frames_dir = Path(args.frames_dir) if args.frames_dir else run / "frames"
    pred = pd.read_csv(run / "raw_frame_predictions.csv").reset_index(drop=True)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) != len(pred):
        raise SystemExit(f"{len(frames)} frames in {frames_dir} vs {len(pred)} rows in {run}")

    pred["motion"] = box_motion(frames, pred)
    pred["behavior_gated"] = [gate_label(r.behavior, r.motion, r.p_Eating, r.p_Locomotion, args.motion_thresh)
                              for r in pred.itertuples()]
    pred.to_csv(run / "raw_frame_predictions_gated.csv", index=False)
    write_gated_preview(frames, pred, run / "raw_behavior_preview_gated.mp4")

    rows = [{"second": int(round(r.elapsed_s)), "behavior": r.behavior_gated,
             "clock": r.cctv_time, "conf": float(r.behavior_conf) if pd.notna(r.behavior_conf) else None}
            for r in pred.itertuples()]
    per_second, bouts = build_sheet(rows, args.smooth_window, args.min_bout, args.max_gap)
    pd.DataFrame(per_second).to_csv(run / "behavior_timeline_gated.csv", index=False)
    bt = pd.DataFrame(bouts)
    bt.to_csv(run / "observation_sheet_gated.csv", index=False)

    d = pred.dropna(subset=["motion"])
    med, mean, mx, p90 = d.motion.median(), d.motion.mean(), d.motion.max(), np.percentile(d.motion, 90)
    print(f"\n=== {run.name} ===  thresh={args.motion_thresh}  ({len(d)} pairs)")
    print(f"motion: median {med:.2f}  mean {mean:.2f}  p90 {p90:.2f}  max {mx:.2f}")
    print("raw CNN counts :", pred["behavior"].value_counts().to_dict())
    print("gated counts   :", pred["behavior_gated"].value_counts().to_dict())
    print(f"flipped Resting->Active: {int((pred.behavior != pred.behavior_gated).sum())}")
    print("\ngated observation sheet (time, status, behavior):")
    print(bt.to_string(index=False))
    print("\nTime by status (s):")
    print(bt.groupby("status")["duration_s"].sum().to_string())
    print(f"\nsaved gated outputs + preview to {run.resolve()}")


if __name__ == "__main__":
    main()

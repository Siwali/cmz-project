r"""Deployed inference: YOLO-seg civet tracker + ResNet behavior classifier, full frame rate.

Option A (two models):
  1. a YOLO-seg model (1 class: civet) detects and TRACKS the civet with ByteTrack, every frame;
  2. the 3-class behavior ResNet (train_behavior.py) classifies each tracked crop
     (Eating / Locomotion / Resting);
  3. a motion gate + smoothing turn the per-frame stream into an observation sheet;
  4. a full-fps annotated preview is written - a stable box that follows the civet, labelled
     with its track id and behavior.

Unlike run_local_behavior.py (which samples 1 fps and re-detects each second independently), this
tracks at the video's native frame rate, so the box no longer jumps. The YOLO model replaces generic
MegaDetector.

The observation sheet is still built at 1 Hz (majority behavior per second), so the motion gate reuses
the same hand-calibrated threshold (4.5) as the 1-fps pipeline. The PREVIEW is full fps.

Example:
  python pipeline/run_tracked_behavior.py \
    --video clip.mp4 \
    --civet-model civet_yolo_segmentation/weights/best.pt \
    --behavior-model pipeline/model/model.pt \
    --out local_runs/tracked_demo

Smoke test with a generic COCO seg model (detections will be poor - just checks the plumbing):
  python pipeline/run_tracked_behavior.py --video clip.mp4 --civet-model yolo11n-seg.pt \
    --behavior-model pipeline/model/model.pt --out local_runs/_smoke --any-class --max-frames 90
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behavior_data import CLASSES, tight_crop               # noqa: E402
from make_observation_sheet import STATUS, build_sheet        # noqa: E402
from apply_motion_gate import gate_label                      # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
STATUS_COLOR = {"Active": (0, 220, 0), "Inactive": (220, 120, 0), "Unknown": (0, 180, 255)}


def load_behavior(model_path, device):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return m.to(device).eval()


def classify(behavior, frame_rgb, box, device):
    crop = tight_crop(Image.fromarray(frame_rgb), box, margin=0.1, size=224)
    with torch.no_grad():
        probs = torch.softmax(behavior(TF(crop).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    k = int(probs.argmax())
    return CLASSES[k], float(probs[k]), {c: float(p) for c, p in zip(CLASSES, probs)}


def box_gray(frame_bgr, box):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    g = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 64)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--civet-model", required=True, help="YOLO-seg .pt (1 class civet), or a generic seg model for a smoke test")
    ap.add_argument("--behavior-model", required=True, help="ResNet model.pt from train_behavior.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence")
    ap.add_argument("--classify-conf", type=float, default=0.50, help="only classify when YOLO conf >= this")
    ap.add_argument("--behavior-every", type=int, default=3, help="run the ResNet every N frames (reuse between)")
    ap.add_argument("--motion-thresh", type=float, default=4.5)
    ap.add_argument("--cctv-start", default="2026-01-01 00:00:00")
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--min-bout", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=2)
    ap.add_argument("--any-class", action="store_true", help="accept any YOLO class (smoke test with COCO models)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = whole video)")
    args = ap.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    base = datetime.fromisoformat(args.cctv_start)

    yolo = YOLO(args.civet_model)
    behavior = load_behavior(args.behavior_model, device)

    writer = None
    per_frame, sec_gray = [], {}
    last = {"behavior": "Unknown", "conf": 0.0, "probs": {c: 0.0 for c in CLASSES}}

    stream = yolo.track(source=args.video, stream=True, persist=True, tracker="bytetrack.yaml",
                        conf=args.conf, verbose=False)
    for fi, res in enumerate(stream):
        if args.max_frames and fi >= args.max_frames:
            break
        frame = res.orig_img                      # BGR, native resolution
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out / "tracked_preview.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        t_s = fi / fps

        box = tid = None
        b = res.boxes
        if b is not None and b.xyxy is not None and len(b) > 0:
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy().reshape(-1)
            cls = b.cls.cpu().numpy().reshape(-1) if b.cls is not None else np.zeros(len(xyxy))
            ids = b.id.cpu().numpy().reshape(-1) if b.id is not None else np.full(len(xyxy), -1)
            keep = np.ones(len(xyxy), bool) if args.any_class else (cls == 0)
            if keep.any():
                j = np.where(keep)[0][conf[keep].argmax()]
                box, tid, bconf = xyxy[j], int(ids[j]), float(conf[j])

        row = {"frame": fi, "second": int(t_s), "time_s": t_s,
               "cctv_time": (base + timedelta(seconds=t_s)).strftime("%H:%M:%S"),
               "track_id": tid if tid is not None else -1,
               "behavior": "Unknown", "behavior_conf": 0.0,
               "p_Eating": 0.0, "p_Locomotion": 0.0, "p_Resting": 0.0,
               "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan}

        if box is not None:
            x1, y1, x2, y2 = (float(v) for v in box)
            row.update(x1=x1, y1=y1, x2=x2, y2=y2)
            if bconf >= args.classify_conf:
                if fi % max(1, args.behavior_every) == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bl, bc, bp = classify(behavior, rgb, (x1, y1, x2, y2), device)
                    last = {"behavior": bl, "conf": bc, "probs": bp}
                row.update(behavior=last["behavior"], behavior_conf=last["conf"],
                           **{f"p_{c}": last["probs"][c] for c in CLASSES})
            if row["second"] not in sec_gray:
                g = box_gray(frame, box)
                if g is not None:
                    sec_gray[row["second"]] = g

        per_frame.append(row)

        # full-fps overlay
        status = STATUS.get(row["behavior"], "Unknown")
        if box is not None:
            c = STATUS_COLOR[status]
            cv2.rectangle(frame, (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"])), c, 3)
            cv2.putText(frame, f"ID {row['track_id']}", (int(row["x1"]), int(row["y1"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        txt = f"{row['cctv_time']} | {status}: {row['behavior']} {row['behavior_conf']:.2f}"
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 900), 34), (0, 0, 0), -1)
        cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    if writer is not None:
        writer.release()
    pf = pd.DataFrame(per_frame)
    pf.to_csv(out / "tracked_frame_predictions.csv", index=False)
    print(f"processed {len(pf)} frames at {fps:.1f} fps; wrote tracked_preview.mp4")

    # ---- 1 Hz observation sheet (majority behavior per second) + motion gate ----
    secs = sorted(int(s) for s in pf["second"].unique())
    rows, gated = [], []
    prev_g = None
    for s in secs:
        sub = pf[(pf["second"] == s) & (pf["behavior"] != "Unknown")]
        if len(sub):
            beh = sub["behavior"].mode().iat[0]
            conf = float(sub.loc[sub["behavior"] == beh, "behavior_conf"].mean())
            pe = float(sub["p_Eating"].mean()); pl = float(sub["p_Locomotion"].mean())
        else:
            beh, conf, pe, pl = "Unknown", 0.0, 0.0, 0.0
        g = sec_gray.get(s)
        motion = float(np.abs(g - prev_g).mean()) if (g is not None and prev_g is not None) else np.nan
        if g is not None:
            prev_g = g
        beh_g = gate_label(beh, motion, pe, pl, args.motion_thresh)
        clk = (base + timedelta(seconds=s)).strftime("%H:%M:%S")
        rows.append({"second": s, "behavior": beh_g, "clock": clk, "conf": conf})
        gated.append({"second": s, "cctv_time": clk, "behavior_cnn": beh, "behavior_gated": beh_g,
                      "motion": motion, "status": STATUS.get(beh_g, "Unknown")})
    pd.DataFrame(gated).to_csv(out / "behavior_per_second.csv", index=False)
    per_second, bouts = build_sheet(rows, args.smooth_window, args.min_bout, args.max_gap)
    bt = pd.DataFrame(bouts)
    bt.to_csv(out / "observation_sheet.csv", index=False)
    print("\nObservation sheet (time, status, behavior):")
    print(bt.to_string(index=False) if len(bt) else "(no classified seconds)")
    print(f"\nsaved outputs to {out.resolve()}")


if __name__ == "__main__":
    main()

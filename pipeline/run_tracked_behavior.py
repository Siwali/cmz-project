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
from ethogram_sheets import write_sheets                      # noqa: E402

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


def near_tray(box, tray, frac=0.3):
    """True if the civet box touches or is within frac*civet_diag of the (fixed) tray box."""
    cx1, cy1, cx2, cy2 = box
    bx1, by1, bx2, by2 = tray
    dx = max(bx1 - cx2, cx1 - bx2, 0.0)
    dy = max(by1 - cy2, cy1 - by2, 0.0)
    gap = (dx * dx + dy * dy) ** 0.5
    diag = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
    return gap <= frac * diag


def centroid_speed(hist, t_s, cx, cy, window_s=0.5):
    """Civet-centroid speed (px/s) over ~window_s, using the oldest centroid within the window."""
    ref = next((e for e in hist if t_s - e[0] <= window_s), None)
    if ref is None:
        return 0.0
    dt = t_s - ref[0]
    return (((cx - ref[1]) ** 2 + (cy - ref[2]) ** 2) ** 0.5 / dt) if dt > 1e-3 else 0.0


def tray_adjust(behavior, probs, box, tray, speed, translate_px):
    """Tray-region rule: at the tray while not walking -> Eating; away from the tray, never Eating.
    Complements the motion gate (which only flips Resting->Active) by fixing the daytime
    Eating->Resting collapse at the feeding tray. 'tray' is None -> rule off."""
    if tray is None or box is None:
        return behavior
    at = near_tray(box, tray)
    if at and speed < translate_px:
        return "Eating"
    if (not at) and behavior == "Eating":
        return max((c for c in CLASSES if c != "Eating"), key=lambda c: probs.get(c, 0.0))
    return behavior


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
    ap.add_argument("--tray-box", default=None,
                    help="fixed food-tray box 'x1,y1,x2,y2' in native pixels; enables the tray-region "
                         "rule (at tray + not walking -> Eating; away from tray -> never Eating). "
                         "Per camera; omit to disable. cam3 example: 688,505,792,586")
    ap.add_argument("--tray-translate", type=float, default=35.0,
                    help="civet-centroid speed (px/s) above which it is 'walking past' the tray, so "
                         "not forced to Eating")
    ap.add_argument("--cctv-start", default="2026-01-01 00:00:00")
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--min-bout", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=2)
    ap.add_argument("--any-class", action="store_true", help="accept any YOLO class (smoke test with COCO models)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = whole video)")
    ap.add_argument("--no-preview", action="store_true",
                    help="skip the annotated preview video (second pass); useful for long videos where "
                         "only the observation sheet is needed")
    args = ap.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tray_box = None
    if args.tray_box:
        tray_box = [float(v) for v in args.tray_box.split(",")]
        if len(tray_box) != 4:
            raise SystemExit("--tray-box must be 'x1,y1,x2,y2'")
        print("tray-region rule ON, tray box:", tray_box)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    base = datetime.fromisoformat(args.cctv_start)

    yolo = YOLO(args.civet_model)
    behavior = load_behavior(args.behavior_model, device)

    per_frame = []
    cent_hist = []          # (t_s, cx, cy) recent civet centroids, for the tray rule's walking test
    sec_motion = {}         # second -> in-box motion vs ~1 s earlier (same box, both frames)
    sec_seen = set()
    prev_sec = None         # (frame_bgr, box, second): first sampled frame of the previous second
    last = {"behavior": "Unknown", "conf": 0.0, "probs": {c: 0.0 for c in CLASSES}}

    stream = yolo.track(source=args.video, stream=True, persist=True, tracker="bytetrack.yaml",
                        conf=args.conf, verbose=False)
    for fi, res in enumerate(stream):
        if args.max_frames and fi >= args.max_frames:
            break
        frame = res.orig_img                      # BGR, native resolution
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
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            speed = centroid_speed(cent_hist, t_s, cx, cy)   # px/s over the recent window
            cent_hist.append((t_s, cx, cy))
            cent_hist[:] = [e for e in cent_hist if t_s - e[0] <= 1.0]
            if bconf >= args.classify_conf:
                if fi % max(1, args.behavior_every) == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bl, bc, bp = classify(behavior, rgb, (x1, y1, x2, y2), device)
                    last = {"behavior": bl, "conf": bc, "probs": bp}
                beh = tray_adjust(last["behavior"], last["probs"], (x1, y1, x2, y2),
                                  tray_box, speed, args.tray_translate)
                row.update(behavior=beh, behavior_conf=last["conf"],
                           **{f"p_{c}": last["probs"][c] for c in CLASSES})
            # motion: crop the CURRENT box from this frame and from the previous second's frame
            # (same box region, ~1 s apart) - a moving civet leaves the box, a resting one does not
            s = row["second"]
            if s not in sec_seen:
                a = box_gray(frame, box)
                if a is not None:
                    if prev_sec is not None:
                        bprev = box_gray(prev_sec[0], box)
                        if bprev is not None:
                            sec_motion[s] = float(np.abs(a - bprev).mean())
                    prev_sec = (frame.copy(), box, s)
                    sec_seen.add(s)

        per_frame.append(row)

    pf = pd.DataFrame(per_frame)
    pf.to_csv(out / "tracked_frame_predictions.csv", index=False)
    print(f"processed {len(pf)} frames at {fps:.1f} fps")

    # ---- 1 Hz observation sheet (majority behavior per second) + motion gate ----
    secs = sorted(int(s) for s in pf["second"].unique())
    rows, gated = [], []
    for s in secs:
        sub = pf[(pf["second"] == s) & (pf["behavior"] != "Unknown")]
        if len(sub):
            beh = sub["behavior"].mode().iat[0]
            conf = float(sub.loc[sub["behavior"] == beh, "behavior_conf"].mean())
            pe = float(sub["p_Eating"].mean()); pl = float(sub["p_Locomotion"].mean())
        else:
            beh, conf, pe, pl = "Unknown", 0.0, 0.0, 0.0
        motion = sec_motion.get(s, np.nan)
        beh_g = gate_label(beh, motion, pe, pl, args.motion_thresh)
        clk = (base + timedelta(seconds=s)).strftime("%H:%M:%S")
        rows.append({"second": s, "behavior": beh_g, "clock": clk, "conf": conf})
        gated.append({"second": s, "cctv_time": clk, "behavior_cnn": beh, "behavior_gated": beh_g,
                      "motion": motion, "status": STATUS.get(beh_g, "Unknown")})
    pd.DataFrame(gated).to_csv(out / "behavior_per_second.csv", index=False)
    per_second, bouts = build_sheet(rows, args.smooth_window, args.min_bout, args.max_gap)
    bt = pd.DataFrame(bouts)
    bt.to_csv(out / "observation_sheet.csv", index=False)
    if len(bt):                                   # Phase-1 ethogram tables (duration + eating frequency)
        write_sheets(bt, out)

    # ---- preview (second pass): annotate each frame with the FINAL gated label, so the preview
    # matches the observation sheet (tray rule + motion gate), not the raw per-frame CNN ----
    sec_gated = {g["second"]: g["behavior_gated"] for g in gated}
    cap2 = cv2.VideoCapture(args.video) if not args.no_preview else None
    writer = None
    for row in ([] if args.no_preview else per_frame):
        ok, frame = cap2.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out / "tracked_preview.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        gb = sec_gated.get(row["second"], row["behavior"])
        status = STATUS.get(gb, "Unknown")
        if not (isinstance(row["x1"], float) and np.isnan(row["x1"])):
            c = STATUS_COLOR[status]
            cv2.rectangle(frame, (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"])), c, 3)
            cv2.putText(frame, f"ID {row['track_id']}", (int(row["x1"]), int(row["y1"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        txt = f"{row['cctv_time']} | {status}: {gb}"
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 900), 34), (0, 0, 0), -1)
        cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    if cap2 is not None:
        cap2.release()
    if writer is not None:
        writer.release()

    print("\nObservation sheet (time, status, behavior):")
    print(bt.to_string(index=False) if len(bt) else "(no classified seconds)")
    print(f"\nsaved outputs + preview to {out.resolve()}")


if __name__ == "__main__":
    main()

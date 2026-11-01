r"""Run the full raw-video behavior pipeline locally (VS Code), no Colab, no ffmpeg.

Samples one frame per second from a CCTV clip with OpenCV, runs the civet detector
(MegaDetector), classifies high-confidence crops with the trained ResNet, writes the
per-second predictions, an annotated preview MP4, and the smoothed observation sheet.

This mirrors notebook 03 (cells 4-6) but as a script. It needs the local inference env
(Python 3.12) from requirements-infer.txt, and a model.pt downloaded from Drive.

Example (PowerShell, from the repo root):
  .\.venv-infer\Scripts\python.exe pipeline\run_local_behavior.py ^
    --video "E:\College\Image Processing\Phase2\Dataset\...\A03_20260405140327.mp4" ^
    --model "E:\path\to\raw_pilot_model_reviewed\model.pt" ^
    --start-s 6330 --end-s 6475 --cctv-start "2026-04-05 14:03:27" ^
    --out local_runs\day_eating
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behavior_data import CLASSES, tight_crop          # noqa: E402
from make_observation_sheet import build_sheet          # noqa: E402
from apply_motion_gate import box_motion, gate_label, write_gated_preview  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def sample_frames(video, start_s, end_s, sample_fps, frames_dir):
    """Write one frame every 1/sample_fps seconds between start_s and end_s using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("*.jpg"):
        p.unlink()
    step = 1.0 / sample_fps
    paths = []
    n = int(round((end_s - start_s) * sample_fps))
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((start_s + i * step) * fps)))
        ok, frame = cap.read()
        if not ok:
            break
        out = frames_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(out), frame)
        paths.append(out)
    cap.release()
    if not paths:
        raise RuntimeError("No frames sampled; check --video and the interval.")
    return paths, fps


def load_classifier(model_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def main():
    import cv2
    from PytorchWildlife.models import detection as pw_detection

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True, help="model.pt from train_behavior.py")
    ap.add_argument("--start-s", type=int, required=True)
    ap.add_argument("--end-s", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cctv-start", default="2026-01-01 00:00:00", help="clock at elapsed second 0")
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--classify-conf", type=float, default=0.80)
    ap.add_argument("--low-conf", type=float, default=0.15)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--motion-thresh", type=float, default=4.5, help="box-motion above this re-calls Resting as Active")
    ap.add_argument("--no-motion", action="store_true", help="skip the motion gate (CNN only)")
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--min-bout", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    frame_paths, fps = sample_frames(args.video, args.start_s, args.end_s, args.sample_fps, out / "frames")
    print(f"sampled {len(frame_paths)} frames at {args.sample_fps} fps (source {fps:.1f} fps)")

    md = pw_detection.MegaDetectorV6Apache(device=device, pretrained=True, version="MDV6-apa-rtdetr-e")
    results = md.batch_image_detection(str(out / "frames"), batch_size=8 if device == "cuda" else 2)
    if len(results) != len(frame_paths):
        raise RuntimeError(f"Detector returned {len(results)} results for {len(frame_paths)} frames")

    classifier = load_classifier(args.model, device)
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    base = datetime.fromisoformat(args.cctv_start)

    per_frame, low = [], []
    for idx, (fp, result) in enumerate(zip(frame_paths, results)):
        frame = cv2.imread(str(fp))
        det = result.get("detections")
        boxes = np.asarray(det.xyxy) if det is not None else np.empty((0, 4))
        scores = np.asarray(getattr(det, "confidence", [])).reshape(-1) if det is not None else np.array([])
        cats = np.asarray(getattr(det, "class_id", [])).reshape(-1) if det is not None else np.array([])
        animal = [(b, float(s)) for b, s, c in zip(boxes, scores, cats)
                  if int(c) == 0 and float(s) >= args.low_conf]
        animal.sort(key=lambda x: x[1], reverse=True)
        elapsed = args.start_s + idx / args.sample_fps
        clock = (base + timedelta(seconds=elapsed)).strftime("%H:%M:%S")
        row = {"elapsed_s": elapsed, "cctv_time": clock, "md_conf": animal[0][1] if animal else 0.0,
               "behavior": "Unknown", "behavior_conf": 0.0,
               "p_Eating": 0.0, "p_Locomotion": 0.0, "p_Resting": 0.0,
               "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan}
        if animal:
            box, conf = animal[0]
            x1, y1, x2, y2 = map(float, box)
            row.update(x1=x1, y1=y1, x2=x2, y2=y2)
            low.append({"elapsed_s": elapsed, "cctv_time": clock, "md_conf": conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            if conf >= args.classify_conf:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop = tight_crop(Image.fromarray(rgb), (x1, y1, x2, y2), margin=args.margin)
                with torch.no_grad():
                    probs = torch.softmax(classifier(tf(crop).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
                k = int(probs.argmax())
                row.update(behavior=CLASSES[k], behavior_conf=float(probs[k]))
                for cls, p in zip(CLASSES, probs):
                    row[f"p_{cls}"] = float(p)
        per_frame.append(row)

    pred = pd.DataFrame(per_frame)

    # Motion gate: a single-frame CNN confuses eating with resting; box-motion rescues it.
    if args.no_motion:
        pred["motion"] = np.nan
        pred["behavior_gated"] = pred["behavior"]
    else:
        pred["motion"] = box_motion(frame_paths, pred)
        pred["behavior_gated"] = [gate_label(r.behavior, r.motion, r.p_Eating, r.p_Locomotion, args.motion_thresh)
                                  for r in pred.itertuples()]

    pred.to_csv(out / "raw_frame_predictions.csv", index=False)
    pd.DataFrame(low).to_csv(out / "low_confidence_detection_candidates.csv", index=False)
    print("\nper-second behavior (raw CNN):")
    print(pred["behavior"].value_counts(dropna=False).to_string())
    if not args.no_motion:
        print("\nper-second behavior (motion-gated):")
        print(pred["behavior_gated"].value_counts(dropna=False).to_string())

    _write_preview(cv2, frame_paths, per_frame, out / "raw_behavior_preview.mp4", args.sample_fps, args.classify_conf)
    if not args.no_motion:
        write_gated_preview(frame_paths, pred, out / "raw_behavior_preview_gated.mp4")

    rows = [{"second": int(round(r.elapsed_s)), "behavior": r.behavior_gated,
             "clock": r.cctv_time, "conf": float(r.behavior_conf) if pd.notna(r.behavior_conf) else None}
            for r in pred.itertuples()]
    per_second, bouts = build_sheet(rows, args.smooth_window, args.min_bout, args.max_gap)
    pd.DataFrame(per_second).to_csv(out / "behavior_timeline_smoothed.csv", index=False)
    bt = pd.DataFrame(bouts)
    bt.to_csv(out / "observation_sheet.csv", index=False)
    print("\nObservation sheet (time, status, behavior):")
    print(bt.to_string(index=False))
    print("\nTime by status (s):")
    print(bt.groupby("status")["duration_s"].sum().to_string())
    print(f"\nsaved outputs to {out.resolve()}")


def _write_preview(cv2, frame_paths, per_frame, path, sample_fps, classify_conf):
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(sample_fps, 1), (w, h))
    for fp, row in zip(frame_paths, per_frame):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        if np.isfinite(row["x1"]):
            color = (0, 220, 0) if row["md_conf"] >= classify_conf else (0, 180, 255)
            cv2.rectangle(frame, (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"])), color, 3)
        text = f"{row['cctv_time']} | {row['behavior']} {row['behavior_conf']:.2f} | MD {row['md_conf']:.2f}"
        cv2.rectangle(frame, (0, 0), (min(w, 1100), 38), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    print("saved preview:", path)


if __name__ == "__main__":
    main()

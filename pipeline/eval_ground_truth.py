"""Compare model output on raw footage with hand-made CVAT ground truth.

Ground truth: a CVAT "CVAT for images 1.1" export (annotations.xml, or the .zip that holds it) with
one `civet` box per frame and a `behavior` attribute (eating / active_other / resting /
inactive_other / unknown). Frame names carry the elapsed second (..._s002130.jpg).

Predictions: raw_frame_predictions.csv from notebook 03 (elapsed_s, cctv_time, md_conf, behavior,
x1..y2). Optional: without --pred the script only summarises the ground truth.

Reports detector recall/precision against the hand-drawn boxes, how often the model gave a label,
the behavior confusion, Eating vs not-Eating, and Active/Inactive agreement.

Usage: python eval_ground_truth.py --gt civeteating.zip [--pred raw_frame_predictions.csv] --out <folder>
"""
import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

GT_STATUS = {"eating": "Active", "active_other": "Active", "resting": "Inactive",
             "inactive_other": "Inactive", "unknown": "Unknown"}
MODEL_STATUS = {"Eating": "Active", "Locomotion": "Active", "Resting": "Inactive", "Unknown": "Unknown"}


def read_gt(path):
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            xml = z.read(next(n for n in z.namelist() if n.endswith(".xml")))
    else:
        xml = path.read_bytes()
    rows = []
    for im in ET.fromstring(xml).iter("image"):
        m = re.search(r"_s(\d+)", im.get("name", ""))
        if not m:
            continue
        row = {"second": int(m.group(1)), "gt_visible": False, "gt_behavior": "no_civet"}
        box = next((b for b in im.findall("box") if b.get("label") == "civet"), None)
        if box is not None:
            behavior = next((a.text for a in box.findall("attribute") if a.get("name", "").lower() == "behavior"), None)
            row.update(gt_visible=True, gt_behavior=(behavior or "unknown").strip().lower(),
                       gx1=float(box.get("xtl")), gy1=float(box.get("ytl")),
                       gx2=float(box.get("xbr")), gy2=float(box.get("ybr")))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("second").reset_index(drop=True)


def clock(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def runs(gt):
    """Consecutive seconds with the same ground-truth behavior."""
    out, start = [], 0
    labels, secs = gt["gt_behavior"].tolist(), gt["second"].tolist()
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start] or secs[i] != secs[i - 1] + 1:
            out.append({"start": clock(secs[start]), "end": clock(secs[i - 1] + 1),
                        "seconds": secs[i - 1] + 1 - secs[start], "behavior": labels[start]})
            start = i
    return pd.DataFrame(out)


def iou(r):
    ix = max(0.0, min(r.gx2, r.x2) - max(r.gx1, r.x1))
    iy = max(0.0, min(r.gy2, r.y2) - max(r.gy1, r.y1))
    inter = ix * iy
    union = (r.gx2 - r.gx1) * (r.gy2 - r.gy1) + (r.x2 - r.x1) * (r.y2 - r.y1) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="CVAT export: annotations.xml or the .zip")
    ap.add_argument("--pred", default=None, help="raw_frame_predictions.csv from notebook 03")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.5, help="overlap needed to count a detector box as correct")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt = read_gt(args.gt)
    print(f"ground truth: {len(gt)} frames, {int(gt.gt_visible.sum())} with a civet box")
    print(gt["gt_behavior"].value_counts().to_string())
    print("\nground-truth timeline (elapsed time = CCTV time when the video starts at 00:00:00):")
    print(runs(gt).to_string(index=False))
    if not args.pred:
        return

    pred = pd.read_csv(args.pred)
    pred["second"] = pred["elapsed_s"].round().astype(int)
    d = gt.merge(pred, on="second", how="inner")
    print(f"\n{len(d)} frames in both files")

    has_pred_box = d["x1"].notna()
    d["iou"] = [iou(r) if (r.gt_visible and pd.notna(r.x1)) else 0.0 for r in d.itertuples()]
    hit = d.gt_visible & has_pred_box & (d["iou"] >= args.iou)
    det = {
        "civet frames": int(d.gt_visible.sum()),
        "detector recall (IoU>=%.1f)" % args.iou: round(hit.sum() / max(d.gt_visible.sum(), 1), 3),
        "detector precision": round(hit.sum() / max(has_pred_box.sum(), 1), 3),
        "boxes on empty frames": int((has_pred_box & ~d.gt_visible).sum()),
        "mean IoU when both have a box": round(d.loc[d.gt_visible & has_pred_box, "iou"].mean(), 3),
    }
    print("\ndetector vs hand-drawn boxes:")
    for k, v in det.items():
        print(f"  {k}: {v}")

    known = d[d.gt_behavior.isin(GT_STATUS) & (d.gt_behavior != "unknown")]
    labelled = known[known["behavior"] != "Unknown"]
    print(f"\nmodel gave a behavior on {len(labelled)} of {len(known)} frames with a known ground-truth behavior")
    if len(labelled):
        print("\nground truth (rows) vs model (columns):")
        print(pd.crosstab(labelled.gt_behavior, labelled.behavior).to_string())
        eat_gt, eat_m = labelled.gt_behavior == "eating", labelled.behavior == "Eating"
        print(f"\nEating: recall {(eat_gt & eat_m).sum() / max(eat_gt.sum(), 1):.2f}, "
              f"precision {(eat_gt & eat_m).sum() / max(eat_m.sum(), 1):.2f}")
        st_gt, st_m = labelled.gt_behavior.map(GT_STATUS), labelled.behavior.map(MODEL_STATUS)
        print(f"Active/Inactive agreement: {(st_gt == st_m).mean():.2f} on {len(labelled)} frames")
    d.to_csv(out / "ground_truth_vs_model.csv", index=False)
    print(f"\nsaved {out / 'ground_truth_vs_model.csv'}")


if __name__ == "__main__":
    main()

"""Run the trained behavior classifier on a hand-labelled 640x640 clip and score resting.

This is for a CVAT "CVAT for images 1.1" export whose frames are the processed 640x640
letterboxed images (names like R01_f000300.jpg) with one `civet` box per frame and a
`behavior` attribute. Unlike eval_ground_truth.py, the frames are already full640, so the
boxes are used as-is (no raw->full640 mapping) and there is no second/CCTV mapping.

The ground truth here only distinguishes active_other vs resting, while the model predicts
Eating / Locomotion / Resting, so the honest question is the Active/Inactive one:
  - of the resting frames, how many does the model call Resting (resting recall)?
  - of the active frames, how many does it wrongly call Resting (false resting)?

Needs the actual frame images (the .zip holds only annotations.xml); point --frames-root at the
dataset folder that holds them (they are found by file name, anywhere under it).

Usage (Colab):
  python eval_resting_clip.py --gt annotation/civetresting.zip \
      --model /content/drive/MyDrive/CMZprojectphase2/data/behavior_run1/model.pt \
      --frames-root /content/CMZ/processed --out /content/drive/MyDrive/CMZprojectphase2/data/resting_eval
"""
import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

from behavior_data import CLASSES, STATUS, tight_crop

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# The clip's labels, collapsed to the Active/Inactive status the model can be judged on.
GT_STATUS = {"eating": "Active", "active_other": "Active",
             "resting": "Inactive", "inactive_other": "Inactive"}


def read_gt(path):
    """One row per labelled frame: file name, behavior, and the box in 640x640 pixels."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            xml = z.read(next(n for n in z.namelist() if n.endswith(".xml")))
    else:
        xml = path.read_bytes()
    rows = []
    for im in ET.fromstring(xml).iter("image"):
        box = next((b for b in im.findall("box") if b.get("label") == "civet"), None)
        if box is None:
            continue  # no civet drawn -> nothing to classify
        behavior = next((a.text for a in box.findall("attribute")
                         if a.get("name", "").lower() == "behavior"), None)
        rows.append({"name": Path(im.get("name", "")).name,
                     "gt_behavior": (behavior or "unknown").strip().lower(),
                     "bx1": float(box.get("xtl")), "by1": float(box.get("ytl")),
                     "bx2": float(box.get("xbr")), "by2": float(box.get("ybr"))})
    return pd.DataFrame(rows)


def index_frames(root):
    """Map each image file name under `root` to its full path (first match wins)."""
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.name not in idx:
            idx[p.name] = p
    return idx


class CropSet(Dataset):
    def __init__(self, df, paths, margin):
        self.df, self.paths, self.margin = df.reset_index(drop=True), paths, margin
        self.tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.paths[r["name"]]).convert("RGB")
        crop = tight_crop(img, (r.bx1, r.by1, r.bx2, r.by2), self.margin, jitter=0.0, rng=None)
        return self.tf(crop), i


def predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def frame_num(name):
    m = re.search(r"_f(\d+)", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="CVAT export: annotations.xml or the .zip")
    ap.add_argument("--model", required=True, help="model.pt saved by train_behavior.py (a state dict)")
    ap.add_argument("--frames-root", required=True, help="folder that holds the clip's frame images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.1, help="crop margin; match train_behavior.py (0.1)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    gt = read_gt(args.gt).sort_values("name", key=lambda s: s.map(frame_num)).reset_index(drop=True)
    print(f"ground truth: {len(gt)} labelled frames")
    print(gt["gt_behavior"].value_counts().to_string())

    idx = index_frames(args.frames_root)
    gt["found"] = gt["name"].isin(idx)
    if not gt["found"].all():
        missing = gt.loc[~gt["found"], "name"].tolist()
        print(f"\n{len(missing)} frame image(s) not found under {args.frames_root}, e.g. {missing[:3]}")
        gt = gt[gt["found"]].reset_index(drop=True)
    if gt.empty:
        raise SystemExit("no frame images found; check --frames-root")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    paths = {n: idx[n] for n in gt["name"]}
    probs = predict(model, DataLoader(CropSet(gt, paths, args.margin), batch_size=args.batch), device)
    gt["pred"] = [CLASSES[k] for k in probs.argmax(1)]
    for k, c in enumerate(CLASSES):
        gt[f"p_{c}"] = probs[:, k]
    gt["gt_status"] = gt["gt_behavior"].map(GT_STATUS)
    gt["pred_status"] = gt["pred"].map(STATUS)

    print("\nground truth (rows) vs model (columns), per frame:")
    print(pd.crosstab(gt.gt_behavior, gt.pred).reindex(columns=CLASSES, fill_value=0).to_string())

    rest = gt[gt.gt_behavior == "resting"]
    act = gt[gt.gt_status == "Active"]
    if len(rest):
        print(f"\nresting recall: {(rest.pred == 'Resting').mean():.2f} "
              f"({int((rest.pred == 'Resting').sum())}/{len(rest)} resting frames called Resting)")
    if len(act):
        print(f"false resting on active frames: {(act.pred == 'Resting').mean():.2f} "
              f"({int((act.pred == 'Resting').sum())}/{len(act)} active frames wrongly called Resting)")
    known = gt[gt.gt_status.notna()]
    if len(known):
        print(f"Active/Inactive agreement: {(known.gt_status == known.pred_status).mean():.2f} "
              f"on {len(known)} frames")

    gt.to_csv(out / "resting_eval.csv", index=False)
    print(f"\nsaved {out / 'resting_eval.csv'}")


if __name__ == "__main__":
    main()

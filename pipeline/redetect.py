"""Second detection pass on the processed full640 frames where the first detector found no civet.

The first pass (MDV6-yolov10-e) missed the civet in most infrared clips. This runs the
MDV6-apa-rtdetr-e model, which worked on infrared in the A04 pilot, with a lower threshold,
and writes boxes in full640 coordinates to <base>/metadata/redetect.csv.

Needs a GPU for speed. Usage: python redetect.py --base /content/CMZ [--conf 0.15]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PytorchWildlife.models import detection as pw_detection
from tqdm.auto import tqdm

ANIMAL = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--version", default="MDV6-apa-rtdetr-e")
    args = ap.parse_args()

    base = Path(args.base)
    m = pd.read_csv(base / "metadata" / "manifest.csv", low_memory=False)
    todo = m[(m["readable"] == True) & m["box_x1"].isna()]  # noqa: E712
    print(f"{len(todo)} frames without a civet box")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pw_detection.MegaDetectorV6Apache(device=device, pretrained=True, version=args.version)

    rows = []
    for path in tqdm(todo["full_path"]):
        rgb = np.asarray(Image.open(base / path).convert("RGB"))
        det = model.single_image_detection(rgb, det_conf_thres=args.conf)["detections"]
        animal = det.class_id == ANIMAL
        if not animal.any():
            continue
        best = int(np.argmax(np.where(animal, det.confidence, -1)))
        x1, y1, x2, y2 = (float(v) for v in det.xyxy[best])
        rows.append({"full_path": path, "bx1": x1, "by1": y1, "bx2": x2, "by2": y2,
                     "md_conf": float(det.confidence[best])})

    out = base / "metadata" / "redetect.csv"
    pd.DataFrame(rows, columns=["full_path", "bx1", "by1", "bx2", "by2", "md_conf"]).to_csv(out, index=False)
    print(f"found a civet in {len(rows)} of {len(todo)} frames -> {out}")


if __name__ == "__main__":
    main()

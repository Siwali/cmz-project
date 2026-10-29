"""Train a 3-class behavior classifier (Eating / Locomotion / Resting) on tight civet crops.

Reads the processed dataset, applies review_edits.csv, uses first-pass boxes plus the
re-detections, and reports test results next to a "never looks at the civet" baseline, per
lighting condition and per video.

Two modes:
  default  train on the dataset's train split, report on its test split (a handful of videos)
  --cv     5-fold cross-validation by video using the `fold` column: every video is tested once,
           so the score does not hang on 3 test videos

Usage:
  python train_behavior.py --base /content/CMZ --edits /content/drive/MyDrive/CMZprojectphase2/data/review_edits.csv \
      --out /content/drive/MyDrive/CMZprojectphase2/data/behavior_run1 [--cv --epochs 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

from behavior_data import (CLASSES, STATUS, load_frames, select_training, shortcut_prediction,
                           tight_crop, to_rules_csv)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def add_noise(x):
    return (x + 0.03 * torch.randn_like(x)).clamp(0, 1)


def random_gamma(x):
    return x.clamp(1e-4, 1) ** float(np.random.uniform(0.7, 1.4))


class CropSet(Dataset):
    def __init__(self, base, df, train, margin):
        self.base, self.df, self.train, self.margin = Path(base), df.reset_index(drop=True), train, margin
        aug = [T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.4, contrast=0.4),
               T.RandomAffine(degrees=10, scale=(0.9, 1.1))] if train else []
        tail = [T.ToTensor()]
        if train:
            tail += [T.RandomApply([T.Lambda(random_gamma)], p=0.5), T.RandomApply([T.Lambda(add_noise)], p=0.5)]
        self.tf = T.Compose(aug + tail + [T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.base / r.full_path).convert("RGB")
        rng = np.random.default_rng() if self.train else None
        crop = tight_crop(img, (r.bx1, r.by1, r.bx2, r.by2), self.margin,
                          jitter=0.08 if self.train else 0.0, rng=rng)
        return self.tf(crop), CLASSES.index(r.label)


def predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def confusion(true, pred):
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(true, pred):
        cm[CLASSES.index(t), CLASSES.index(p)] += 1
    return pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])


def macro_f1(cm):
    m = cm.to_numpy()
    f1 = []
    for k in range(len(CLASSES)):
        tp, fp, fn = m[k, k], m[:, k].sum() - m[k, k], m[k, :].sum() - m[k, k]
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1))


def report(name, d, pred_col):
    cm = confusion(d.label, d[pred_col])
    acc = float((d.label == d[pred_col]).mean())
    print(f"\n{name}: accuracy {acc:.3f}, macro-F1 {macro_f1(cm):.3f}  ({len(d)} rows)")
    print(cm.to_string())
    return {"accuracy": acc, "macro_f1": macro_f1(cm), "rows": int(len(d))}


def fit(base, parts, args, device):
    """Train on parts['train'], keep the epoch with the best val macro-F1, return the model."""
    counts = parts["train"].label.value_counts().reindex(CLASSES).fillna(1)
    weights = torch.tensor((counts.sum() / (len(CLASSES) * counts)).to_numpy(), dtype=torch.float32, device=device)
    train_loader = DataLoader(CropSet(base, parts["train"], True, args.margin), batch_size=args.batch,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(CropSet(base, parts["val"], False, args.margin), batch_size=args.batch, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        val = parts["val"].assign(pred=[CLASSES[k] for k in predict(model, val_loader, device).argmax(1)])
        f1 = macro_f1(confusion(val.label, val.pred))
        print(f"  epoch {epoch}: train loss {total / len(parts['train']):.4f}  val macro-F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_state


def with_predictions(model, base, d, args, device):
    p = predict(model, DataLoader(CropSet(base, d, False, args.margin), batch_size=args.batch, num_workers=2), device)
    d = d.assign(pred=[CLASSES[k] for k in p.argmax(1)], shortcut=shortcut_prediction(d))
    for k, c in enumerate(CLASSES):
        d[f"p_{c}"] = p[:, k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--edits", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.1, help="crop margin around the box (0.1 = 10 %%)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--redetect-min-conf", type=float, default=0.3,
                    help="ignore re-detected boxes below this confidence (they include plants and shadows)")
    ap.add_argument("--freeze-backbone", action="store_true", help="train only the final layer (steadier on tiny data)")
    ap.add_argument("--cv", action="store_true", help="5-fold cross-validation by video instead of the fixed split")
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    print("device:", device)

    redetect_csv = base / "metadata" / "redetect.csv"
    print("redetect.csv found:", redetect_csv.exists(), "->", redetect_csv)
    frames = load_frames(base, args.edits, redetect_csv, args.redetect_min_conf)
    data = select_training(frames)
    print("frames with a usable civet box, per label and split:")
    print(pd.crosstab(data.label, data.split, margins=True).to_string())
    print("box source:", data.box_source.value_counts().to_dict())

    tests = []
    if args.cv:
        folds = sorted(f for f in data.fold.unique() if f >= 0)
        for k in folds:
            nxt = folds[(folds.index(k) + 1) % len(folds)]
            parts = {"train": data[~data.fold.isin([k, nxt])], "val": data[data.fold == nxt],
                     "test": data[data.fold == k]}
            print(f"\n=== fold {k}: test videos {sorted(parts['test'].video_id.unique())}, "
                  f"val videos {sorted(parts['val'].video_id.unique())} ===")
            if any(d.empty or d.label.nunique() < 2 for d in (parts["train"], parts["val"])) or parts["test"].empty:
                print("  skipped: a split is empty or has a single class")
                continue
            model, _ = fit(base, parts, args, device)
            tests.append(with_predictions(model, base, parts["test"], args, device))
        test = pd.concat(tests)
    else:
        parts = {s: data[data.split == s] for s in ("train", "val", "test")}
        for s, d in parts.items():
            if d.empty:
                raise SystemExit(f"no usable frames in the {s} split")
        model, best_state = fit(base, parts, args, device)
        torch.save(best_state, out / "model.pt")
        test = with_predictions(model, base, parts["test"], args, device)

    keep = ["video_id", "label", "split", "fold", "lighting", "timestamp_sec", "window_id", "pred", "shortcut",
            "box_source"] + [f"p_{c}" for c in CLASSES]
    test[keep].to_csv(out / "predictions.csv", index=False)

    title = "CROSS-VALIDATION (every video tested once)" if args.cv else "TEST"
    metrics = {"mode": "cv" if args.cv else "fixed split", "args": vars(args)}
    metrics["model"] = report(f"{title}, model (every frame)", test, "pred")
    metrics["shortcut_baseline"] = report(f"{title}, baseline that ignores the civet", test, "shortcut")
    for light, d in test.groupby("lighting"):
        metrics[f"model_{light}"] = report(f"{title}, model, {light} only", d, "pred")

    per_video = test.assign(ok=test.label == test.pred).groupby(["label", "video_id", "lighting"]).agg(
        frames=("ok", "size"), correct_pct=("ok", lambda s: round(100 * s.mean()))).reset_index()
    print("\nper video:")
    print(per_video.to_string(index=False))
    per_video.to_csv(out / "per_video.csv", index=False)

    sec = (test.assign(second=test.timestamp_sec.astype(int))
           .groupby(["video_id", "second", "label"])[[f"p_{c}" for c in CLASSES]].mean().reset_index())
    sec["pred"] = [CLASSES[k] for k in sec[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)]
    metrics["per_second"] = report(f"{title}, model, one prediction per second", sec, "pred")
    confusion(test.label, test.pred).to_csv(out / "confusion.csv")
    sec["status"] = sec.pred.map(STATUS)
    sec.to_csv(out / "per_second.csv", index=False)

    rules_dir = out / "rules"
    rules_dir.mkdir(exist_ok=True)
    for vid in test.video_id.unique():
        to_rules_csv(test, vid, rules_dir / f"{vid}.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nsaved predictions, metrics and per-second tables in {out}")


if __name__ == "__main__":
    main()

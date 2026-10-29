"""Add frame-to-frame motion to the CNN's per-second predictions.

A single-frame CNN cannot see movement, but movement is the main evidence for Active vs Inactive.
This takes the out-of-fold predictions written by `train_behavior.py --cv` (predictions.csv), adds
the motion measured in the manifest, and fits a small logistic regression per second. It is tested
with the same folds (train on 4 folds of videos, test on the 5th), so every video is scored by a
model that never saw it.

Variants reported:
  cnn only           argmax of the CNN probabilities
  motion only        logistic regression on motion features
  cnn + motion       logistic regression on CNN probabilities + motion
  cnn + motion + light   same, plus an infrared flag (can lean on lighting)
  baseline           infrared -> Locomotion, still -> Resting, else Eating

Usage: python stack_motion.py --base /content/CMZ --pred <run>/behavior_cv/predictions.csv --out <run>/behavior_cv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from behavior_data import CLASSES, STATUS
from civet_rules import enforce_min_bout, to_runs

P_COLS = [f"p_{c}" for c in CLASSES]
MOTION_COLS = ["motion_log", "motion_max_log", "still_frac"]


def per_second_table(base, pred_csv):
    p = pd.read_csv(pred_csv)
    m = pd.read_csv(Path(base) / "metadata" / "manifest.csv", low_memory=False)
    m = m[["video_id", "timestamp_sec", "motion_pct", "near_duplicate"]].drop_duplicates(["video_id", "timestamp_sec"])
    d = p.merge(m, on=["video_id", "timestamp_sec"], how="left")
    d["second"] = d["timestamp_sec"].astype(int)
    d["still"] = d["near_duplicate"].astype(float)
    g = d.groupby(["video_id", "second"]).agg(
        label=("label", "first"), fold=("fold", "first"), lighting=("lighting", "first"),
        shortcut=("shortcut", lambda s: s.mode().iat[0]),
        motion_mean=("motion_pct", "mean"), motion_max=("motion_pct", "max"), still_frac=("still", "mean"),
        **{c: (c, "mean") for c in P_COLS}).reset_index()
    g["motion_log"] = np.log1p(g["motion_mean"].fillna(0))
    g["motion_max_log"] = np.log1p(g["motion_max"].fillna(0))
    g["still_frac"] = g["still_frac"].fillna(0.5)
    g["is_ir"] = (g["lighting"] == "ir_night").astype(float)
    return g


def macro_f1(true, pred):
    f1 = []
    for c in CLASSES:
        tp = ((true == c) & (pred == c)).sum()
        fp = ((true != c) & (pred == c)).sum()
        fn = ((true == c) & (pred != c)).sum()
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1))


def score(name, g, pred):
    true = g["label"].to_numpy()
    pred = np.asarray(pred)
    ai_true = np.array([STATUS[t] for t in true])
    ai_pred = np.array([STATUS[x] for x in pred])
    out = {"variant": name, "accuracy": float((true == pred).mean()), "macro_f1": macro_f1(true, pred),
           "active_inactive_acc": float((ai_true == ai_pred).mean())}
    for light, tag in (("ir_night", "ir"), ("color_day", "day")):
        for cls in ("Resting", "Locomotion", "Eating"):
            sel = (g["lighting"].to_numpy() == light) & (true == cls)
            if sel.any():
                out[f"recall_{cls}_{tag}"] = float((pred[sel] == cls).mean())
    return out


def cross_fit(g, cols, folds):
    pred = np.empty(len(g), dtype=object)
    for k in folds:
        test = (g["fold"] == k).to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(g.loc[~test, cols], g.loc[~test, "label"])
        pred[test] = model.predict(g.loc[test, cols])
    return pred


def clock(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def bout_table(g, pred, min_bout):
    """time, status, behavior per video: consecutive seconds with the same status (runs shorter than
    min_bout seconds are absorbed by their neighbours, as in civet_rules.py). Seconds with no
    detection count as Unknown."""
    rows = []
    for vid, v in g.assign(pred=pred).sort_values("second").groupby("video_id"):
        first, last = int(v["second"].min()), int(v["second"].max())
        full = v.set_index("second")["pred"].reindex(range(first, last + 1))
        states = ["Unknown" if not isinstance(x, str) else STATUS[x] for x in full]
        for state, s, e in enforce_min_bout(to_runs(states), min_bout):
            seg = full.iloc[s:e].dropna()
            rows.append({"video_id": vid, "start_time": clock(first + s), "end_time": clock(first + e),
                         "duration_s": e - s, "status": state,
                         "behavior": seg.mode().iat[0] if len(seg) else "not_visible"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="folder holding metadata/manifest.csv")
    ap.add_argument("--pred", required=True, help="predictions.csv from train_behavior.py --cv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-bout", type=int, default=5, help="shortest Active/Inactive stretch kept in the table (s)")
    args = ap.parse_args()

    g = per_second_table(args.base, args.pred)
    folds = sorted(f for f in g["fold"].unique() if f >= 0)
    print(f"{len(g)} seconds from {g.video_id.nunique()} videos, folds {folds}")

    variants = {
        "baseline (lighting + stillness)": g["shortcut"].to_numpy(),
        "cnn only": np.array(CLASSES)[g[P_COLS].to_numpy().argmax(1)],
        "motion only": cross_fit(g, MOTION_COLS, folds),
        "cnn + motion": cross_fit(g, P_COLS + MOTION_COLS, folds),
        "cnn + motion + light": cross_fit(g, P_COLS + MOTION_COLS + ["is_ir"], folds),
    }
    rows = [score(name, g, pred) for name, pred in variants.items()]
    table = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\nper-second results (every video scored by a model that never saw it):")
    print(table.round(3).to_string())

    best = "cnn + motion"
    cm = pd.crosstab(g["label"], pd.Series(variants[best], name="predicted"))
    print(f"\nconfusion for '{best}':")
    print(cm.to_string())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g.assign(**{f"pred_{k.split(' (')[0].replace(' + ', '_').replace(' ', '_')}": v for k, v in variants.items()}) \
        .to_csv(out / "per_second_stacked.csv", index=False)
    bouts = bout_table(g, variants[best], args.min_bout)
    bouts.to_csv(out / "behavior_table.csv", index=False)
    print(f"\nbehavior table (time, status, behavior), first rows of {len(bouts)}:")
    print(bouts.head(12).to_string(index=False))
    table.to_csv(out / "stacked_metrics.csv")
    (out / "stacked_metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()

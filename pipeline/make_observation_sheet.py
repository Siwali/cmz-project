"""Turn raw_frame_predictions.csv (one row per sampled second) into an observation sheet.

A frame-by-frame classifier flickers: a stationary civet can read Eating one second and
Resting the next. This script smooths the per-second behavior with a centred rolling
majority vote, bridges brief detector dropouts, drops sub-threshold bouts, then collapses
consecutive seconds into behavior bouts with start/end clock time, duration, and the
Active/Inactive status from the ethogram.

Unknown (no civet box above the classify threshold) is kept as its own state and is never
counted as Resting; only Unknown runs shorter than --max-gap, sitting between two bouts of
the same behavior, are bridged.

Usage:
  python make_observation_sheet.py --pred raw_frame_predictions.csv --out <folder> \
      [--smooth-window 5] [--min-bout 3] [--max-gap 2]
"""
import argparse
from collections import Counter
from pathlib import Path

# Eating/Locomotion are Active; Resting is Inactive; a missing detection stays Unknown.
STATUS = {"Eating": "Active", "Locomotion": "Active", "Resting": "Inactive", "Unknown": "Unknown"}
UNKNOWN = "Unknown"


def smooth_labels(labels, window):
    """Centred rolling majority vote. Ties keep the original label (no flip without support)."""
    if window <= 1:
        return list(labels)
    k = window // 2
    n = len(labels)
    out = []
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        counts = Counter(labels[lo:hi])
        top, top_n = counts.most_common(1)[0]
        tie = sum(1 for v in counts.values() if v == top_n) > 1
        out.append(labels[i] if tie else top)
    return out


def runs(labels):
    """List of (start_index, end_index_exclusive, label) for consecutive equal labels."""
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, labels[start]))
            start = i
    return out


def bridge_unknown(labels, max_gap):
    """Relabel a short Unknown run to the behavior on both sides when they agree."""
    labels = list(labels)
    segs = runs(labels)
    for j, (s, e, lab) in enumerate(segs):
        if lab != UNKNOWN or (e - s) > max_gap:
            continue
        if 0 < j < len(segs) - 1:
            prev_lab, next_lab = segs[j - 1][2], segs[j + 1][2]
            if prev_lab == next_lab and prev_lab != UNKNOWN:
                for i in range(s, e):
                    labels[i] = prev_lab
    return labels


def merge_short_bouts(labels, min_bout):
    """Absorb any run shorter than min_bout into its longer neighbour, repeating until stable.

    Unknown runs are left alone so that real no-detection gaps survive; short behavior
    flickers are the ones this removes.
    """
    labels = list(labels)
    changed = True
    while changed:
        changed = False
        segs = runs(labels)
        if len(segs) <= 1:
            break
        for j, (s, e, lab) in enumerate(segs):
            if lab == UNKNOWN or (e - s) >= min_bout:
                continue
            left = segs[j - 1] if j > 0 else None
            right = segs[j + 1] if j < len(segs) - 1 else None
            cand = [c for c in (left, right) if c and c[2] != UNKNOWN]
            if not cand:
                continue
            winner = max(cand, key=lambda c: c[1] - c[0])[2]
            for i in range(s, e):
                labels[i] = winner
            changed = True
            break
    return labels


def to_bouts(seconds, labels, clocks, confs):
    """Collapse to bouts with clock span, duration (s) and mean confidence."""
    bouts = []
    for s, e, lab in runs(labels):
        span = [confs[i] for i in range(s, e) if confs[i] is not None]
        bouts.append({
            "start_elapsed_s": seconds[s],
            "end_elapsed_s": seconds[e - 1] + 1,
            "start_time": clocks[s],
            "end_time": _plus_one_second(clocks[e - 1]),
            "duration_s": seconds[e - 1] + 1 - seconds[s],
            "status": STATUS.get(lab, "Unknown"),
            "behavior": lab,
            "mean_conf": round(sum(span) / len(span), 3) if span else 0.0,
        })
    return bouts


def _plus_one_second(hhmmss):
    try:
        h, m, s = (int(x) for x in str(hhmmss).split(":"))
    except ValueError:
        return hhmmss
    t = (h * 3600 + m * 60 + s + 1) % 86400
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def build_sheet(rows, smooth_window, min_bout, max_gap):
    """rows: list of dicts with second, behavior, clock, conf. Returns (per_second, bouts)."""
    rows = sorted(rows, key=lambda r: r["second"])
    seconds = [r["second"] for r in rows]
    clocks = [r["clock"] for r in rows]
    confs = [r["conf"] for r in rows]
    raw = [r["behavior"] if r["behavior"] in STATUS else UNKNOWN for r in rows]

    smoothed = smooth_labels(raw, smooth_window)
    smoothed = bridge_unknown(smoothed, max_gap)
    smoothed = merge_short_bouts(smoothed, min_bout)

    per_second = [{"elapsed_s": seconds[i], "clock_time": clocks[i],
                   "behavior_raw": raw[i], "behavior_smoothed": smoothed[i],
                   "status": STATUS.get(smoothed[i], "Unknown")} for i in range(len(rows))]
    return per_second, to_bouts(seconds, smoothed, clocks, confs)


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="raw_frame_predictions.csv from notebook 03")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth-window", type=int, default=5, help="odd window for the majority vote (seconds)")
    ap.add_argument("--min-bout", type=int, default=3, help="drop/merge behavior bouts shorter than this (s)")
    ap.add_argument("--max-gap", type=int, default=2, help="bridge Unknown gaps up to this long between equal bouts (s)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred)
    pred["second"] = pred["elapsed_s"].round().astype(int)
    pred = pred.drop_duplicates("second").sort_values("second")
    rows = [{"second": int(r.second),
             "behavior": str(r.behavior),
             "clock": str(r.cctv_time),
             "conf": float(r.behavior_conf) if pd.notna(r.behavior_conf) else None}
            for r in pred.itertuples()]

    per_second, bouts = build_sheet(rows, args.smooth_window, args.min_bout, args.max_gap)
    ps = pd.DataFrame(per_second)
    bt = pd.DataFrame(bouts)
    ps.to_csv(out / "behavior_timeline_smoothed.csv", index=False)
    bt.to_csv(out / "observation_sheet.csv", index=False)

    flips_raw = int((ps["behavior_raw"] != ps["behavior_raw"].shift()).sum())
    flips_sm = int((ps["behavior_smoothed"] != ps["behavior_smoothed"].shift()).sum())
    print(f"{len(ps)} seconds; label changes: {flips_raw} raw -> {flips_sm} smoothed")
    print("\nObservation sheet (time, status, behavior):")
    print(bt.to_string(index=False))
    print("\nTime by status (s):")
    print(bt.groupby("status")["duration_s"].sum().to_string())
    print("\nTime by behavior (s):")
    print(bt.groupby("behavior")["duration_s"].sum().to_string())
    print(f"\nsaved {out / 'observation_sheet.csv'} and {out / 'behavior_timeline_smoothed.csv'}")


if __name__ == "__main__":
    main()

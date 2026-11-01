"""Rule layer: per-second detections -> Duration table and Frequency table.

Input CSV (one row per second, contiguous): second, clock_time (HH:MM:SS),
visible (0/1), zone, behavior.

Usage: python civet_rules.py fake_detections.csv --out out
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ACTIVE = {"locomotion", "feeding", "drinking", "grooming", "sniffing_exploring",
          "urine", "feces", "territorial_marking"}
INACTIVE = {"standing_still", "sitting", "resting_lying", "resting_in_box"}
FREQ_BEHAVIORS = ["feeding", "drinking", "grooming", "territorial_marking", "urine", "feces"]

MIN_BOUT_S = 60       # shorter Active/Inactive runs are absorbed by their neighbours
BOX_ZONE = "box_entrance"
BOX_REST_S = 600      # hidden this long after the box entrance -> resting_in_box
EVENT_GAP_S = 5       # same behavior closer together than this is one event
MIN_EVENT_S = 2       # shorter detections are ignored in the Frequency table


def load(path):
    df = pd.read_csv(path, keep_default_na=False)
    df["visible"] = df["visible"].astype(int)
    if not (df["second"].diff().dropna() == 1).all():
        raise ValueError("'second' must be contiguous, one row per second")
    return df.reset_index(drop=True)


def apply_box_rule(df, box_zone=BOX_ZONE, box_rest_s=BOX_REST_S):
    """Label hidden stretches: resting_in_box if lost at the box entrance for
    >= box_rest_s, otherwise not_visible."""
    beh = df["behavior"].to_numpy(dtype=object).copy()
    vis = df["visible"].to_numpy()
    zone = df["zone"].to_numpy(dtype=object)
    last_zone = None
    i, n = 0, len(df)
    while i < n:
        if vis[i]:
            last_zone = zone[i]
            i += 1
            continue
        j = i
        while j < n and not vis[j]:
            j += 1
        resting = last_zone == box_zone and (j - i) >= box_rest_s
        beh[i:j] = "resting_in_box" if resting else "not_visible"
        i = j
    out = df.copy()
    out["behavior"] = beh
    return out


def to_status(behavior):
    if behavior in ACTIVE:
        return "Active"
    if behavior in INACTIVE:
        return "Inactive"
    return "Unknown"


def to_runs(states):
    """[state, start, end) runs over a list of states."""
    runs, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            runs.append([states[start], start, i])
            start = i
    return runs


def enforce_min_bout(runs, min_len=MIN_BOUT_S):
    """Repeatedly absorb the shortest run (< min_len) into its longer neighbour."""
    runs = [r[:] for r in runs]
    while len(runs) > 1:
        k = min(range(len(runs)), key=lambda i: (runs[i][2] - runs[i][1], i))
        if runs[k][2] - runs[k][1] >= min_len:
            break
        if k == 0:
            tgt = 1
        elif k == len(runs) - 1:
            tgt = k - 1
        else:
            left = runs[k - 1][2] - runs[k - 1][1]
            right = runs[k + 1][2] - runs[k + 1][1]
            tgt = k - 1 if left >= right else k + 1
        runs[tgt][1] = min(runs[tgt][1], runs[k][1])
        runs[tgt][2] = max(runs[tgt][2], runs[k][2])
        del runs[k]
        merged = [runs[0]]
        for r in runs[1:]:
            if r[0] == merged[-1][0]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        runs = merged
    return runs


def clock(base, offset_s):
    return (base + timedelta(seconds=int(offset_s))).strftime("%H:%M:%S")


def build_bouts(df, min_bout_s=MIN_BOUT_S):
    base = datetime.strptime(df["clock_time"].iloc[0], "%H:%M:%S")
    states = [to_status(b) for b in df["behavior"]]
    rows = []
    for state, s, e in enforce_min_bout(to_runs(states), min_bout_s):
        main = df["behavior"].iloc[s:e].value_counts().idxmax()
        rows.append({"start_time": clock(base, s), "end_time": clock(base, e),
                     "duration_s": e - s, "status": state, "main_behavior": main})
    return pd.DataFrame(rows)


def duration_sheet(bouts):
    """Chronological rows: each Inactive bout is paired with the Active bout that
    immediately follows it; Unknown bouts are skipped and break the pairing."""
    recs = bouts.to_dict("records")
    rows, i = [], 0
    while i < len(recs):
        cur = recs[i]
        if cur["status"] == "Unknown":
            i += 1
            continue
        row = {"No": len(rows) + 1, "Inactive_start": "", "Inactive_stop": "",
               "Active_start": "", "Active_stop": ""}
        if cur["status"] == "Inactive":
            row["Inactive_start"], row["Inactive_stop"] = cur["start_time"], cur["end_time"]
            nxt = recs[i + 1] if i + 1 < len(recs) else None
            if nxt is not None and nxt["status"] == "Active":
                row["Active_start"], row["Active_stop"] = nxt["start_time"], nxt["end_time"]
                i += 1
        else:
            row["Active_start"], row["Active_stop"] = cur["start_time"], cur["end_time"]
        rows.append(row)
        i += 1
    total = bouts.groupby("status")["duration_s"].sum()
    rows.append({"No": "Total",
                 "Inactive_start": f"{total.get('Inactive', 0) / 60:.1f} min", "Inactive_stop": "",
                 "Active_start": f"{total.get('Active', 0) / 60:.1f} min", "Active_stop": ""})
    return pd.DataFrame(rows)


def frequency_sheet(df, event_gap_s=EVENT_GAP_S, min_event_s=MIN_EVENT_S):
    base = datetime.strptime(df["clock_time"].iloc[0], "%H:%M:%S")
    events = []
    for b in FREQ_BEHAVIORS:
        secs = df.index[df["behavior"] == b].to_numpy()
        if len(secs) == 0:
            continue
        start = prev = secs[0]
        for s in list(secs[1:]) + [None]:
            if s is None or s - prev - 1 > event_gap_s:
                if prev - start + 1 >= min_event_s:
                    events.append({"behavior": b, "hour": (base + timedelta(seconds=int(start))).hour})
                if s is not None:
                    start = s
            if s is not None:
                prev = s
    first_h = base.hour
    last_h = (base + timedelta(seconds=len(df) - 1)).hour
    hours = list(range(first_h, last_h + 1))
    labels = {h: f"{h % 24:02d}.01-{(h + 1) % 24:02d}.00" for h in hours}
    ev = pd.DataFrame(events, columns=["behavior", "hour"])
    table = (ev.groupby(["hour", "behavior"]).size().unstack(fill_value=0)
             .reindex(index=hours, columns=FREQ_BEHAVIORS, fill_value=0))
    table.index = [labels[h] for h in table.index]
    table.index.name = "Time"
    table.loc["Total"] = table.sum()
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="out")
    ap.add_argument("--min-bout", type=int, default=MIN_BOUT_S)
    ap.add_argument("--box-rest", type=int, default=BOX_REST_S)
    ap.add_argument("--event-gap", type=int, default=EVENT_GAP_S)
    args = ap.parse_args()

    df = apply_box_rule(load(args.csv), box_rest_s=args.box_rest)
    bouts = build_bouts(df, args.min_bout)
    sheet = duration_sheet(bouts)
    freq = frequency_sheet(df, event_gap_s=args.event_gap)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bouts.to_csv(out / "bouts.csv", index=False)
    sheet.to_csv(out / "duration_sheet.csv", index=False)
    freq.to_csv(out / "frequency_sheet.csv")

    print("== Bouts ==\n", bouts.to_string(index=False))
    print("\n== Duration sheet ==\n", sheet.to_string(index=False))
    unknown_min = bouts.loc[bouts["status"] == "Unknown", "duration_s"].sum() / 60
    print(f"(Unknown / not visible, not counted above: {unknown_min:.1f} min)")
    print("\n== Frequency sheet ==\n", freq.to_string())


if __name__ == "__main__":
    main()

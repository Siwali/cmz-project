"""Turn an observation sheet (bouts) into the Phase-1 ethogram tables:

  Duration sheet  - chronological rows pairing each Inactive span with the Active
                    span that follows it, plus Total Inactive / Active minutes.
  Frequency sheet - Eating events per clock hour (one row per hour + Total).

Input: observation_sheet.csv from run_tracked_behavior.py, columns
  start_time, end_time, duration_s, status, behavior
(status in {Active, Inactive, Unknown}; behavior in {Eating, Locomotion, Resting, Unknown}).

The frequency sheet is deliberately single-behavior (Eating only); the zoo ethogram's
other active behaviours (drinking/grooming/urine/feces) are out of this model's scope.

Usage: python ethogram_sheets.py <run_dir_or_observation_sheet.csv> [--out <dir>]
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

FREQ_BEHAVIOR = "Eating"


def _coalesce(runs):
    """Merge neighbouring runs that share a status (after a blip has been reassigned)."""
    out = []
    for r in runs:
        if out and out[-1]["status"] == r["status"]:
            out[-1]["end_time"] = r["end_time"]
            out[-1]["duration_s"] += r["duration_s"]
        else:
            out.append(dict(r))
    return out


def _merge_status_runs(bouts):
    """Collapse consecutive bouts of the same status into maximal Active/Inactive/Unknown spans."""
    runs = [{"status": r["status"], "start_time": r["start_time"],
             "end_time": r["end_time"], "duration_s": r["duration_s"]}
            for r in bouts.to_dict("records")]
    return _coalesce(runs)


def _smooth_spans(runs, min_span):
    """Absorb each span shorter than min_span seconds into its longer Active/Inactive
    neighbour (shortest first). This smooths brief Active<->Inactive flips and bridges
    short not-visible dropouts. Not-visible (Unknown) spans of min_span seconds or longer
    are kept as genuine gaps, so real absences survive and only the noise is merged."""
    if min_span <= 0:
        return runs
    runs = [dict(r) for r in runs]
    while len(runs) > 1:
        cand = None
        for i, r in enumerate(runs):
            if r["duration_s"] >= min_span:               # long enough to stand on its own
                continue
            neigh = [(j, runs[j]) for j in (i - 1, i + 1)
                     if 0 <= j < len(runs) and runs[j]["status"] in ("Active", "Inactive")]
            if neigh and (cand is None or r["duration_s"] < cand[1]):
                cand = (i, r["duration_s"], neigh)
        if cand is None:
            break
        i, _, neigh = cand
        _, target = max(neigh, key=lambda t: t[1]["duration_s"])
        runs[i]["status"] = target["status"]
        runs = _coalesce(runs)
    return runs


def duration_sheet(bouts, min_span=0):
    """Each Inactive span is paired with the Active span that immediately follows it;
    Unknown (civet not detected) spans are skipped and break the pairing. With
    min_span > 0, brief Active/Inactive flips are first smoothed into their neighbours."""
    runs = _smooth_spans(_merge_status_runs(bouts), min_span)
    rows, i = [], 0
    while i < len(runs):
        cur = runs[i]
        if cur["status"] == "Unknown":
            i += 1
            continue
        row = {"No": len(rows) + 1, "Inactive_start": "", "Inactive_stop": "",
               "Active_start": "", "Active_stop": ""}
        if cur["status"] == "Inactive":
            row["Inactive_start"], row["Inactive_stop"] = cur["start_time"], cur["end_time"]
            nxt = runs[i + 1] if i + 1 < len(runs) else None
            if nxt is not None and nxt["status"] == "Active":
                row["Active_start"], row["Active_stop"] = nxt["start_time"], nxt["end_time"]
                i += 1
        else:
            row["Active_start"], row["Active_stop"] = cur["start_time"], cur["end_time"]
        rows.append(row)
        i += 1
    tot = {"Active": 0, "Inactive": 0}
    for r in runs:                                # totals from the (smoothed) spans the sheet shows
        if r["status"] in tot:
            tot[r["status"]] += r["duration_s"]
    rows.append({"No": "Total",
                 "Inactive_start": f"{tot['Inactive'] / 60:.1f} min", "Inactive_stop": "",
                 "Active_start": f"{tot['Active'] / 60:.1f} min", "Active_stop": ""})
    return pd.DataFrame(rows)


def frequency_sheet(bouts, behavior=FREQ_BEHAVIOR):
    """Count <behavior> events (one bout = one event) per clock hour."""
    def hour_of(t):
        return datetime.strptime(t, "%H:%M:%S").hour

    start_h = hour_of(bouts["start_time"].iloc[0])
    end_h = hour_of(bouts["end_time"].iloc[-1])
    hours = list(range(start_h, end_h + 1)) or [start_h]
    counts = {h: 0 for h in hours}
    for t in bouts.loc[bouts["behavior"] == behavior, "start_time"]:
        counts[hour_of(t)] = counts.get(hour_of(t), 0) + 1
    label = {h: f"{h % 24:02d}.01-{(h + 1) % 24:02d}.00" for h in hours}
    rows = [{"Time": label[h], behavior: counts[h]} for h in hours]
    rows.append({"Time": "Total", behavior: sum(counts.values())})
    return pd.DataFrame(rows)


def write_sheets(bouts, out_dir, min_span=0):
    """Write duration_sheet.csv + frequency_sheet.csv next to the observation sheet."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dur = duration_sheet(bouts, min_span)
    freq = frequency_sheet(bouts)
    dur.to_csv(out / "duration_sheet.csv", index=False)
    freq.to_csv(out / "frequency_sheet.csv", index=False)
    return dur, freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a run directory or an observation_sheet.csv")
    ap.add_argument("--out", default=None, help="output dir (default: alongside the input)")
    ap.add_argument("--min-span", type=int, default=0,
                    help="smooth the duration sheet: absorb Active/Inactive spans shorter than "
                         "this many seconds into their neighbour (0 = off)")
    args = ap.parse_args()

    p = Path(args.path)
    csv = p / "observation_sheet.csv" if p.is_dir() else p
    bouts = pd.read_csv(csv)
    out = args.out or csv.parent
    dur, freq = write_sheets(bouts, out, args.min_span)

    runs = _smooth_spans(_merge_status_runs(bouts), args.min_span)   # remaining not-visible after smoothing
    unknown_min = sum(r["duration_s"] for r in runs if r["status"] == "Unknown") / 60
    print("== Duration sheet ==")
    print(dur.to_string(index=False))
    print(f"(Unknown / not visible, not counted above: {unknown_min:.1f} min)")
    print("\n== Frequency sheet (Eating events per hour) ==")
    print(freq.to_string(index=False))
    print(f"\nsaved duration_sheet.csv + frequency_sheet.csv to {Path(out).resolve()}")


if __name__ == "__main__":
    main()

"""Make a fake per-second detection CSV (2 h starting 15:00:00) to test civet_rules.py.

The scenario has known answers, listed in EXPECTED at the bottom, so the rule
layer's output can be checked by eye.

Usage: python make_fake_detections.py fake_detections.csv
"""
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

START = datetime(2026, 4, 5, 15, 0, 0)
TOTAL_S = 7200

# (start_s, end_s, behavior, zone); behavior None = civet not visible
SCENARIO = [
    (0, 1800, "resting_lying", "tire_bed"),
    (1800, 1810, "locomotion", "floor"),          # 10 s blip: absorbed by the 1-min rule
    (1810, 2400, "resting_lying", "tire_bed"),
    (2400, 2430, "locomotion", "floor"),
    (2430, 2445, "feeding", "tray"),              # feeding event 1
    (2445, 2460, "sniffing_exploring", "tray"),
    (2460, 2475, "feeding", "tray"),              # feeding event 2 (15 s gap)
    (2475, 2485, "drinking", "water_bowl"),
    (2485, 2540, "grooming", "floor"),
    (2540, 2600, "locomotion", "floor"),
    (2600, 2606, "urine", "floor"),
    (2606, 2700, "standing_still", "floor"),
    (2700, 3000, "locomotion", "floor"),
    (3000, 3010, "locomotion", "box_entrance"),
    (3010, 3910, None, None),                     # hidden 15 min after box entrance -> resting_in_box
    (3910, 3940, "locomotion", "floor"),
    (3940, 3950, "drinking", "water_bowl"),
    (3950, 4000, "sniffing_exploring", "floor"),
    (4000, 4010, "locomotion", "box_entrance"),
    (4010, 4190, None, None),                     # hidden only 3 min -> not_visible
    (4190, 4250, "locomotion", "floor"),
    (4250, 4260, "locomotion", "door"),
    (4260, 5000, None, None),                     # left through the door -> not_visible
    (5000, 5100, "locomotion", "floor"),
    (5100, 5105, "feces", "floor"),
    (5105, 6000, "resting_lying", "tire_bed"),
    (6000, 6008, "grooming", "tire_bed"),         # 8 s blip: counted as an event, absorbed in Duration
    (6008, 7200, "resting_lying", "tire_bed"),
]

EXPECTED = """
Expected (approx.):
  Frequency 15.01-16.00: feeding 2, drinking 1, grooming 1, urine 1
  Frequency 16.01-17.00: drinking 1, feces 1, grooming 1
  resting_in_box bout starting 15:50:10 (Inactive), 15 min
  not_visible (Unknown) at 16:06:50 (3 min) and 16:11:00 (12 min)
  the 10 s locomotion at 15:30:00 and 8 s grooming at 16:40:00 do NOT split the Inactive bouts
"""


def main(out_path):
    rng = np.random.default_rng(0)
    rows = []
    for start, end, behavior, zone in SCENARIO:
        for s in range(start, end):
            rows.append({
                "second": s,
                "clock_time": (START + timedelta(seconds=s)).strftime("%H:%M:%S"),
                "visible": int(behavior is not None),
                "zone": zone or "",
                "behavior": behavior or "",
                "confidence": round(float(rng.uniform(0.8, 0.99)), 2) if behavior else 0.0,
            })
    df = pd.DataFrame(rows)
    assert len(df) == TOTAL_S

    # detector flicker: ~1% single-second dropouts away from the box entrance
    flicker = (rng.random(len(df)) < 0.01) & df["visible"].eq(1) & df["zone"].ne("box_entrance")
    df.loc[flicker, ["visible", "zone", "behavior", "confidence"]] = [0, "", "", 0.0]

    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(df)} rows, {int(flicker.sum())} flicker dropouts")
    print(EXPECTED)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fake_detections.csv")

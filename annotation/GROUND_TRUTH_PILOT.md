# Raw footage ground-truth pilot

This is a starter annotation batch for the Cam 3 segment CCTV 00:35:30–00:37:10 (elapsed 2130–2230 seconds). It is for checking the annotation workflow and the visible Eating/other-activity transitions. It does not contain a confirmed `resting` example and is not enough by itself to train or validate the final model.

## 1. Extract the frames in Colab

Mount the friend-shared `CMZprojectphase2` folder in My Drive, then run from the cloned repository:

In a Colab cell, create the one-window CSV (this avoids depending on an unpushed local file), then extract the frames:

```python
import pandas as pd
from pathlib import Path

windows = Path("/content/ground_truth_pilot_windows.csv")
pd.DataFrame([{
    "video": "cam3/A03_20260405000000.mp4",
    "start_s": 2130,
    "end_s": 2230,
    "tag": "a03_pilot_eating_active"
}]).to_csv(windows, index=False)

!python /content/cmz-project/annotation/extract_frames.py "{windows}" \
  --root /content/drive/MyDrive/CMZprojectphase2/data/raw \
  --out /content/drive/MyDrive/CMZprojectphase2/data/annotations/ground_truth_pilot_frames \
  --every 1
```

This creates one frame per second with the elapsed second in each filename. Keep these frames and exports in the private Drive; never commit them.

## 2. Annotate in CVAT

Create a task named `A03_ground_truth_pilot_003530_003710`, upload the extracted JPGs as an image task, and use `cvat_labels_ground_truth.json` for the label schema.

- Draw a tight rectangle around the civet's visible body in each frame. Do not draw boxes around the tray, tire bed, doors, plants, or sleeping box.
- Set the `behavior` attribute on each civet box to `eating`, `active_other`, `resting`, `inactive_other`, or `unknown`.
- Use `eating` only when food taking/chewing is visible; mere proximity or sniffing is not enough.
- Use `resting` for visible lying/curling/resting. Use `inactive_other` for standing or sitting still. A hidden civet in the sleeping box has no visible civet box; its 10-minute resting inference is a separate ROI/time rule. Use `active_other` for locomotion, grooming, drinking, or other active behavior not labeled Eating.
- If the civet is absent, leave the frame without a civet box. If visible but behavior is unclear, use `unknown` instead of guessing.
- Correct every detector proposal by hand. Model predictions are not ground truth.

The `status` is derived later: `eating`/`active_other` -> Active; `resting`/`inactive_other` -> Inactive; `unknown` stays Unknown. The frame annotation stores visible behavior; bout duration and frequency rules are applied after review.

## 3. Add scene ROIs separately

Food-tray and sleeping-box polygons are scene ROIs, not civet training boxes. Draw the sleeping-box ROI for Cameras 2 and 3 and a food-tray ROI for the relevant camera/setup. Do not annotate a tire-bed ROI. If the tray moves, redraw its ROI for that recording/setup. The polygon labels are listed in `cvat_labels_ground_truth_rois.json`.

## 4. Build a useful dataset before training

After this workflow works, add separately reviewed footage containing clear Resting, empty/background frames, and varied civet positions/lighting. Keep complete time blocks or recordings aside for validation/test; never randomly split adjacent frames from the same event across train and test. The first 100-frame pilot is a workflow demonstration, not an accuracy claim.

Export annotations with CVAT's image annotation XML so the `behavior` attribute is retained. Avoid a detector-only YOLO export for the behavior labels; that export may discard the attribute.
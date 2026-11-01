# Annotation guide (CVAT)

## 1. Setup in CVAT
1. Create **two projects** per the schemas in this folder:
   - `civet-boxes` uses `cvat_labels_civet.json` (one rectangle label `civet`, with attribute `behavior`).
   - `civet-rois` uses `cvat_labels_rois.json` (polygons). Draw these **once per camera** on a single frame; they are not training data.
2. Load frames from `extract_frames.py` as tasks of ~100 frames. Keep the file names: they encode the elapsed second.
3. Pre-annotate with MegaDetector or the first YOLO model, then **correct every box by hand**.

## 2. Box rules
- One tight box around the whole visible body (head to tail base; include the tail only if it is clearly visible).
- Partly behind the plant or hay: box the visible part and tick `partly_hidden`.
- No civet visible: no box (these frames are the negatives). Do not box the sleeping box, shadows, or hay.
- Night IR frames get the same rules.

## 3. Behavior attribute (what to pick)
| Value | Pick it when | Status |
|---|---|---|
| `locomotion` | walking or running, body clearly moving to a new place | Active |
| `feeding` | head in the tray AND chewing/taking food visible | Active |
| `drinking` | head at the bowl AND licking, tongue or splash visible | Active |
| `sniffing_exploring` | head at tray, bowl, floor or wall, no sign of eating or drinking | Active |
| `grooming` | licking, scratching or cleaning its own body | Active |
| `territorial_marking` | clear marking posture (rare; leave out if unsure) | Active |
| `urine`, `feces` | the act is visible | Active |
| `standing_still` | standing, not moving, no other behavior | Inactive |
| `sitting` | sitting, not moving | Inactive |
| `resting_lying` | lying or curled up, including on the tire bed | Inactive |
| `unsure` | cannot tell: **use it instead of guessing** | Unknown |

Notes:
- Sniffing the tray without clear eating is `sniffing_exploring`, not `feeding`.
- Judge by the neighbouring frames (frames are 1 s apart): the same posture with the body moving is active.
- `resting_in_box` and `not_visible` are **not** labelled by hand. They come from the box ROI rule, when there is no box to draw.

## 4. What to extract (first batch, about 400 frames)
Write one row per window in `windows.csv` (`video,start_s,end_s,tag`):

| Share | Content | Tag |
|---|---|---|
| ~40% | activity: walking, tray, bowl, grooming (use screening candidate clips; the 00:40:35 water-bowl event is one) | `active_*` |
| ~15% | resting on the tire bed | `rest_tire` |
| ~15% | sleeping-box entrance: entering, leaving, plant occlusion | `box_*` |
| ~10% | tray present, with the civet near it | `tray_*` |
| ~10% | empty room (negatives) | `empty` |
| rest | doorways, other cameras (A01 to A03) | `door_*`, `cam_*` |

- At least 40% of frames from **night IR**, and frames from every camera.
- Aim for **at least 50 boxes per behavior** before training the behavior model. If one has fewer, merge it into a neighbour for now (for example `grooming` into `sniffing_exploring`) and split it later.
- Take whole 1 fps stretches (10 to 60 s) instead of isolated frames: behavior depends on context.

## 5. Splits (do this before training)
- Never split by random frame.
- Train: most recordings. Validation: separate time blocks. Test: a whole recording or day that the model never sees, with at least one night block.

## 6. Quality check
- A second pass on 10% of the frames: re-label and compare. Disagreement on `feeding` vs `sniffing_exploring` is the most likely, so fix the written rules if it shows up.
- Export from CVAT as **"CVAT for images 1.1"** (XML). It keeps the `behavior` attribute. The YOLO export would drop it, so convert the XML ourselves later.

## 7. Commands
```bash
python annotation/extract_frames.py windows.csv --out frames --every 1
```

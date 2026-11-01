# CMZ — Civet Behavior Detection

Detect a large spotted civet's behavior (**Eating / Locomotion / Resting**) from zoo overhead CCTV,
and produce an **observation sheet** (time → Active/Inactive → behavior) per the Chiang Mai Zoo
ethogram.

**Pipeline:** image pre-processing → custom **YOLO-seg** civet detector + **ByteTrack** tracker →
**ResNet18** behavior classifier → **motion gate + food-tray rule** → observation sheet + annotated
preview.

---

## Tutorial

Two paths, pick the one you need.

### A) Quick start — just try the model (≈5 min)

Run **`notebooks/07_behavior_showcase.ipynb`** in Google Colab. It loads the trained models and runs
the system on short clips of each behavior (day and night), showing the observation sheet + an
annotated preview for each, plus a combined results table.

**Before running, put these on Google Drive** under a shared folder `CMZprojectphase2/`:

| file on Drive | what it is |
|---------------|------------|
| `models/civet_bowl_seg_best.pt` | the YOLO-seg civet detector (rename the trained `best.pt` to this) |
| `models/model.pt` | the 3-class behavior ResNet |
| `data/raw/cam3/A03_20260405140327.mp4` | daytime video |
| `data/raw/cam3/A03_20260405000000.mp4` | night (infrared) video |

Then in Colab:
1. Add a shortcut to the shared `CMZprojectphase2` folder under **My Drive**.
2. Open notebook 07, **Runtime → Run all**.
3. Results appear inline and are saved to `outputs/behavior_showcase/` on Drive.

That's it — no training needed.

### B) Full pipeline — reproduce from raw video

Run the notebooks in order. Each is self-contained for its stage.

| # | notebook | stage | output |
|---|----------|-------|--------|
| 01 | `01_video_audit` | inventory the raw CCTV videos | video list / stats |
| — | *(pre-processing)* | clean raw frames: inpaint text → grayscale → median → CLAHE → letterbox 640 → validate | `cmz_processed_dataset` (640×640 grayscale) + manifest |
| 02 | `02_train_behavior` | train the 3-class behavior ResNet on processed crops | `model.pt` |
| 04 | `04_sam3_pixel_annotation_clean` | SAM 3 pixel masks for the civet (+ food bowl), single run | masks |
| 05 | `05_sam3_batch_annotation_from_processed` | SAM 3 masks in batch across frames | mask dataset |
| 06 | `06_finalize_and_prepare_segmentation_dataset` | assemble the YOLO-seg training dataset from the masks | `civet(_bowl)_yolo_segmentation` |
| — | *(train YOLO)* | train YOLO11n-seg on that dataset (`pipeline/`, local GPU or Colab) | `best.pt` |
| 03 | `03_raw_video_behavior_pilot` | 1 fps MegaDetector pilot **+ §7 tracker showcase** | pilot outputs |
| 07 | `07_behavior_showcase` | **deployed system** on clips, day + night | observation sheets + previews |

> The behavior model and the YOLO detector are independent — you can retrain either without the other.

---

## Running the deployed system directly (no notebook)

`pipeline/run_tracked_behavior.py` is the deployment script (YOLO + ByteTrack → ResNet → motion gate
+ tray rule):

```bash
python pipeline/run_tracked_behavior.py \
  --video clip.mp4 \
  --civet-model best.pt \
  --behavior-model model.pt \
  --tray-box "706,492,808,573" \   # fixed food-tray box for Camera 3 (omit to disable the rule)
  --out out_dir
```

It writes `observation_sheet.csv`, `behavior_per_second.csv`, and a full-rate annotated
`tracked_preview.mp4`.

---

## Repo structure

```
notebooks/   Colab notebooks (audit, training, annotation, pilot, showcase)
pipeline/    the inference + training code (run_tracked_behavior.py, train_behavior.py,
             behavior_data.py, apply_motion_gate.py, …) + per-camera ROIs
annotation/  annotation guides, CVAT label configs, ground-truth CSVs
```

Models and data (videos, processed dataset) live on Google Drive, not in the repo.

---

## Models

| model | role | metric |
|-------|------|--------|
| YOLO11n-seg (civet + food_bowl) | detect/segment the civet each frame | civet mAP50 0.993, bowl 0.977 |
| ByteTrack | keep one stable track on the civet | — (tracker, not trained) |
| ResNet18 (3-class) | classify behavior from the civet crop | Eating / Locomotion / Resting |

Behavior is then corrected by a **motion gate** (still box → Resting, moving → Active) and a
**tray-region rule** (at the tray + not walking → Eating; away from the tray → never Eating).

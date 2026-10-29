# Handoff: civet behavior project (CMZ)

Last updated 2026-10-29. Deadline: **2026-10-31**. Branch: `feat/civet-zoo-rules-cvat` (pushed to `github.com/Siwali/cmz-project`).

## Goal
Detect a large spotted civet's behavior from CCTV so the zoo gets an observation sheet: Active/Inactive duration plus behavior tallies.
Right now the scope is **three classes only: Eating, Locomotion, Resting**, using **only the friend's processed dataset** (`cmz_processed_dataset.zip`, 22 short clips, 5 fps frames). Cameras do not need to be identified.

## What exists (all in this repo)
| Path | What it does | Status |
|---|---|---|
| `pipeline/civet_rules.py` | Per-second behavior CSV to Active/Inactive duration sheet (1-min minimum bout, 10-min sleeping-box rule) and hourly frequency tallies | Tested locally on fake data |
| `pipeline/make_fake_detections.py` | Fake per-second CSV with known answers | Tested |
| `pipeline/behavior_data.py` | Loads the processed dataset, applies `review_edits.csv`, tight crops, per-second rules CSV | Tested locally on the real dataset (box mapping, edits, crops, rules CSV) |
| `pipeline/make_review_sheets.py` | Contact sheets, one thumbnail per second per clip, for checking labels | Tested locally: 28 sheets for the 22 clips; red border = no civet box |
| `pipeline/redetect.py` | Second MegaDetector pass (`MDV6-apa-rtdetr-e`, conf 0.15) on frames with no civet box | Parses; needs GPU |
| `pipeline/train_behavior.py` | ResNet18 on tight crops, test report vs a "ignores the civet" baseline, per-second tables | Parses; needs GPU; **not run yet** |
| `pipeline/stack_motion.py` | Combines the CV predictions with frame-to-frame motion (logistic regression, same folds) and compares with the baseline | Tested locally on synthetic predictions + the real manifest; **not run on the real `behavior_cv/predictions.csv` yet** |
| `notebooks/02_train_behavior.ipynb` | Colab notebook that runs all of the above in order | Not run yet |
| `notebooks/01_video_audit.ipynb` | Lists long videos with duration, fps, chunks | Ran in Colab, works |
| `pipeline/make_clips.py`, `annotation/*`, `pipeline/rois/*` | Earlier path: cut clips from the long videos using `timeline_events.csv`, CVAT labels, ROI polygons per camera | Built; parked for now |
| `pipeline/detect_video.py` | YOLO output on a long video to the per-second CSV | Not run (no model yet) |

## Findings about the processed dataset (from `metadata/`)
- Classes: Eating 5 videos, Locomotion 10, Resting 7. Source video is 15 fps, 1920x1080 HEVC. `camera` column is empty.
- **Lighting is tied to the class**: Eating all colour/day, Locomotion all infrared/night, Resting 5 day + 2 infrared.
- A "model" that ignores the civet (infrared = Locomotion, still picture = Resting, else Eating) scores about 85 % train, 93 % val, **44 to 54 % test**. A real model must beat the test number, and test must be read per lighting.
- The first detector (`MDV6-yolov10-e`) missed the civet in most infrared clips (median confidence 0). Test split has only 28 usable Locomotion and 12 usable Resting frames, so `redetect.py` is essential.
- Crops include the tray (Eating) and the tire bed (Resting): a background shortcut. `train_behavior.py` re-crops tighter (10 % margin).

## How to run (Colab, GPU)
1. Put `cmz_processed_dataset.zip` in `MyDrive/CMZ/`.
2. Open `notebooks/02_train_behavior.ipynb` from the repo (or upload it) and run top to bottom.
3. Review sheets land in `MyDrive/CMZ/review_sheets/`. Write bad stretches in `MyDrive/CMZ/review_edits.csv`.
4. Results land in `MyDrive/CMZ/behavior_run1/` (`metrics.json`, `predictions.csv`, `test_per_second.csv`, `rules/`, `model.pt`).

## Run log
- **Run 1 (fixed split, conf cut-off 0.35, GPU, 8 epochs):** test accuracy 0.23 (macro-F1 0.34) vs the "ignores the civet" baseline 0.54. Eating test video E05 (daytime) was predicted Resting in 149 of 171 frames; infrared Locomotion 28/28 and Resting 0/12.
  - `box source` showed only `first_pass: 3338`: **`redetect.csv` was not in `/content/CMZ/metadata/`** after the runtime restart, so the re-detections were not used and the test set was still tiny (28 Locomotion, 12 Resting). The notebook now copies it from Drive (`MyDrive/CMZprojectphase2/data/redetect.csv`) and `train_behavior.py` prints whether it was found.
  - Val macro-F1 swung 0.99, 0.89, 0.93, 0.46, 0.61, 0.96: the val/test splits are 1 to 2 videos per class, so single-split scores are noise.
- Re-detection boxes: a spot check of 12 boxes (conf 0.18 to 0.90) found about 2 to 3 on plants and 1 on a food tray. Confidence only partly separates them (civets on the tire bed score 0.36 to 0.43). Cut-off 0.35 is a compromise.
- **Run 2 (5-fold CV by video, frozen backbone, 4 epochs, conf cut-off 0.35, GPU; redetect.csv used: 1,062 re-detected boxes + 3,338 first-pass = 4,400 frames):** output in `MyDrive/CMZprojectphase2/data/behavior_cv/`.
  - Every frame: accuracy **0.717**, macro-F1 0.684. One prediction per second: accuracy 0.741, macro-F1 0.715. Majority-class guess (Resting) = 0.594.
  - "Ignores the civet" baseline (lighting + frame-to-frame change): accuracy 0.817, macro-F1 0.810, so the baseline is **better**. Note it uses motion, which is a real behavior cue that a single-frame model cannot see.
  - Recall per class (frames): Eating 70 %, Locomotion 75 %, Resting 71 %. Daytime only: accuracy 0.768. Infrared only: Locomotion 75 % but **Resting 7 %** (R06 8 %, R07 7 %): infrared Resting is called Locomotion, i.e. the model leans on lighting and there are only 2 infrared Resting videos.
  - Weak videos: E05 10 %, E02 46 %, E03 52 %, R06 8 %, R07 7 %. Strong: E01 99 %, E04 93 %, R04 99 %, L07 100 %.
  - Conclusion: pipeline works end to end; the data (22 clips, almost no infrared Eating/Resting) cannot support a reliable classifier for the zoo's night footage. Adding motion features to the classifier, and night clips from `annotation/timeline_events.csv`, are the next steps.

## Still to do
- [ ] Run the notebook end to end in Colab and fix whatever breaks (first real run of `train_behavior.py`).
- [ ] Review the 22 clips' sheets and fill `review_edits.csv`.
- [ ] Re-run training after the edits; compare against the baseline in `metrics.json` (test, per lighting).
- [ ] Write up: 22 clips is a pipeline proof, not a final model; night Eating and Resting are barely present.
- [ ] Optional: our own timeline clips (`annotation/timeline_events.csv`, `make_clips.py`) add night Eating/Resting/Locomotion; open question whether to merge them in.

## Decisions already made
- Three classes first (Eating, Locomotion, Resting); Active = Eating + Locomotion, Inactive = Resting.
- Split by video (or by time block), never by frame. Cross-camera clips of the same moment stay in one split.
- No CVAT box annotation for now: MegaDetector gives the boxes. "Annotation" = reviewing clip labels.
- Raw zoo footage is private to the group: never commit videos, frames or crops (`.gitignore` covers them). The GitHub repo is public.

## Open questions
- Where did the friend's clips come from (days, cameras)? Do they overlap with our timeline events?
- A02 sleeping-box entry time: `4:44:30` vs `0:44:30` (in `timeline_events.csv`, entered as 4:44:30).
- Whether to merge our night clips into training.

## Environment notes
- Colab loads code with a `git clone`/`pull` of this branch (public repo, no token needed). The local PC has Python 3.15 with no ML packages; test only pandas-level code locally.
- Fork's bundled git: `C:\Users\xkidc\AppData\Local\Fork\gitInstance\2.50.1\cmd\git.exe` (git is not on PATH).

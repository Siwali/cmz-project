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

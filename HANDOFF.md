# Handoff: civet behavior project (CMZ)

Last updated 2026-10-31. **Deadline moved to Monday 2026-11-02** (three working days: Sat 31 Oct, Sun 1 Nov, Mon 2 Nov). Branch `feat/civet-zoo-rules-cvat`, pushed to `github.com/Siwali/cmz-project` (public repo; no footage in it).

## 1. Where we are in one paragraph
The pipeline runs end to end on the friend's processed dataset (22 short clips, 3 classes: Eating, Locomotion, Resting): detect the civet, crop, train a CNN, add frame-to-frame motion, write a per-second table and a time/status/behavior table. Best Active/Inactive result so far is **85.7 % per second (CNN + motion)** against 82.1 % for a lighting+stillness baseline and 78.3 % for the CNN alone, from 5-fold cross-validation by video. **No manual annotation has been done yet**: clip labels are the friend's folder names, boxes come from MegaDetector, and `review_edits.csv` is still the empty template. The user chose **Option 1: review the labels of the weakest videos, re-run, compare.**

## 2. NEXT ACTION (Option 1: label review, about 45 min)
1. Open the contact sheets in Drive: `MyDrive/CMZprojectphase2/data/review_sheets/`. Start with the weakest videos: **E05, E02, E03, R05, R06, R07** (then the rest if time allows). One thumbnail per second; red border = no civet box; the number under it is the second.
2. Write wrong stretches in `MyDrive/CMZprojectphase2/data/review_edits.csv`, header `video_id,start_s,end_s,action`:
   ```
   video_id,start_s,end_s,action
   E05,0,20,drop              # civet not eating / not visible
   E02,0,8,relabel:Locomotion  # label does not match what the civet does
   ```
   `drop` removes those seconds; `relabel:<Eating|Locomotion|Resting>` changes the label.
3. In Colab (GPU runtime), run in this order (notebook `notebooks/02_train_behavior.ipynb` has these cells):
   - clone/pull: `git -C /content/cmz-project fetch origin feat/civet-zoo-rules-cvat` then `git -C /content/cmz-project reset --hard origin/feat/civet-zoo-rules-cvat`
   - unzip the data to local disk (`/content/CMZ`) from `MyDrive/CMZprojectphase2/data/cmz_processed_dataset.zip`
   - copy `redetect.csv` from Drive into `/content/CMZ/metadata/` (**must exist, otherwise the re-detected boxes are silently skipped**; the log prints `redetect.csv found: True`)
   - train with cross-validation, writing to a NEW folder so run 2 is kept:
     `python pipeline/train_behavior.py --base /content/CMZ --edits <Drive>/review_edits.csv --out <Drive>/behavior_cv_reviewed --cv --epochs 4 --freeze-backbone --redetect-min-conf 0.35`
   - motion step: `python pipeline/stack_motion.py --base /content/CMZ --pred <Drive>/behavior_cv_reviewed/predictions.csv --out <Drive>/behavior_cv_reviewed`
4. Compare `stacked_metrics.csv` (new) with the Run 3 table in section 5. Report honestly whether label review helped.
5. Write up (section 7) and commit/push.

## 2b. Plan for the extra time (deadline Mon 2 Nov)
- **Sat 31 Oct:** Option 1 above (label review of the weakest videos, re-run CV + motion, compare with Run 3).
- **Sun 1 Nov: add night data (the main weakness).** Cut our own clips from the long videos with `pipeline/make_clips.py` (12 clips from `annotation/timeline_events.csv`: night Eating 2, Locomotion 6, Resting 4; class names now match the friend's: Eating/Locomotion/Resting), run them through the friend's preprocessing (`CMZ_Frame_Extraction_and_Preprocessing.ipynb`: text masking, MegaDetector crops, manifest), concatenate their manifest with `cmz_processed_dataset`'s, give the new videos folds, and re-run CV. Expected gain: infrared Eating/Resting examples. NOT STARTED; needs a merge step (new `fold` values, `camera` column, unique video_ids such as `A03_003545`). Caution: A02 and A03 clips of the same moment must share a fold.
- **Mon 2 Nov:** final write-up (outline in section 7), final commit/push, results tables and `behavior_table.csv` for the report. Keep the morning free for last fixes.
- If time is short, skip the Sunday step and say in the report that night data is the next step.

## 3. Repo structure (`E:\College\Image Processing\cmz-project`)
```
HANDOFF.md                         this file
.gitignore                         ignores videos, images, weights, out/ runs/ data/
annotation/
  ANNOTATION_GUIDE.md              CVAT guide (parked: not used with the processed dataset)
  cvat_labels_civet.json, cvat_labels_rois.json
  extract_frames.py                frames from long videos (--root for the video folder)
  windows.csv, timeline_events.csv hand-written timeline of 4 cameras (A01-A04, night, 35-50 min)
notebooks/
  01_video_audit.ipynb             lists the long videos (Ran in Colab, works)
  02_train_behavior.ipynb          the whole Colab run for the processed dataset
pipeline/
  behavior_data.py                 manifest loading, review edits, tight crops, rules CSV
  make_review_sheets.py            contact sheets for the label review
  redetect.py                      2nd MegaDetector pass (GPU; took 1.5 h on CPU)
  train_behavior.py                ResNet18 on crops; --cv, --freeze-backbone, --redetect-min-conf
  stack_motion.py                  CNN + motion; writes behavior_table.csv
  civet_rules.py                   per-second CSV to Active/Inactive sheet + hourly tallies (1-min rule, sleeping-box rule)
  make_fake_detections.py          test data for civet_rules.py
  make_clips.py                    cuts clips from long videos using timeline_events.csv (parked)
  detect_video.py                  YOLO on a long video to per-second CSV (parked, no model)
  rois/rois_A01..A04.json          normalized ROI polygons per camera (door, box_entrance, box_body, tray, tire_bed, water_bowl)
```

## 4. Drive structure (`MyDrive/CMZprojectphase2/`)
```
data/
  cmz_processed_dataset.zip        friend's dataset (571 MB), contains metadata/ and processed/
  manifest.csv, frames_raw.csv, video_inventory.csv, ...   copies of the metadata CSVs
  redetect.csv                     2nd-pass boxes (1,534 of 1,536 frames; weak ones filtered by --redetect-min-conf)
  review_sheets/                   contact sheets (28 JPGs)
  review_edits.csv                 EMPTY TEMPLATE: to be filled in (next action)
  behavior_run1/                   run 1 (fixed split)
  behavior_cv/                     runs 2 and 3: predictions.csv, per_video.csv, per_second.csv, per_second_stacked.csv, stacked_metrics.csv, behavior_table.csv, rules/
  raw/cam1..cam4/                  long CCTV videos (A01..A04, 1.2 to 7 h each, 25 fps, night IR)
```
Colab working copies (lost on restart): `/content/cmz-project` (code) and `/content/CMZ` (unzipped dataset).

## 5. Results so far
**Dataset findings.** Eating 5 videos (all colour/day), Locomotion 10 (all infrared), Resting 7 (5 day + 2 infrared). Lighting is tied to the class, so a model can cheat. The first-pass detector (`MDV6-yolov10-e`) missed the civet in most infrared frames; `redetect.py` (`MDV6-apa-rtdetr-e`, conf 0.15) fixed most of that but boxes some plants and a food tray (about 2-3 of 12 sampled), so weak boxes are dropped (`--redetect-min-conf 0.35`).

**Run 1, fixed split.** Test accuracy 0.23 vs baseline 0.54. Invalid: re-detections were not loaded (`box source: first_pass 3338` only), and the split has 1-2 videos per class.

**Run 2, 5-fold CV by video (4,400 frames, 1,062 re-detected boxes).** Frame accuracy 0.717 (macro-F1 0.684); per second 0.741 (0.715). Majority guess 0.594. Recall Eating 70 %, Locomotion 75 %, Resting 71 %. Infrared Resting only 7 % (R06 8 %, R07 7 %). Weak videos: E05 10 %, E02 46 %, E03 52 %, R06, R07. Strong: E01 99 %, E04 93 %, R04 99 %, L07 100 %.

**Run 3, CNN + motion (`stack_motion.py`, 954 seconds, same folds).**

| variant | accuracy | macro-F1 | Active/Inactive | Resting recall (infrared) |
|---|---|---|---|---|
| baseline (lighting + stillness) | 0.821 | 0.815 | 0.821 | 0.000 |
| CNN only | 0.741 | 0.715 | 0.783 | 0.063 |
| motion only | 0.714 | 0.602 | 0.848 | 0.825 |
| **CNN + motion** | 0.780 | 0.726 | **0.857** | 0.508 |
| CNN + motion + lighting flag | 0.823 | 0.813 | 0.823 | 0.032 |

Reading: motion carries most of the Active/Inactive signal and recovers infrared Resting without lighting. The lighting flag restores 3-class accuracy by using the shortcut (infrared Resting collapses to 3 %), so report CNN + motion as the main number. The baseline's 3-class lead is inflated by the dataset's lighting/class link. Only 22 clips: differences of a few points are suggestive, not proven.

## 6. Decisions and constraints
- Three classes (Eating, Locomotion, Resting); Active = Eating + Locomotion, Inactive = Resting. Camera identity not needed.
- Only the friend's processed dataset for training (user's choice). Our own night timeline clips (`timeline_events.csv`, `make_clips.py`) are parked; they would add night Eating/Resting.
- Split by video (5 folds from `fold`), never by frame.
- "Annotation" here = reviewing clip labels (CVAT box annotation is not planned: boxes come from MegaDetector).
- Zoo footage is private to the group: never commit videos, frames or crops. The repo is public; the raw Drive folder may be link-shared: the user was told to check.
- Rule layer for the long videos (1-min minimum bout, 10-min sleeping-box rule at the A02/A03 `box_entrance`) exists in `civet_rules.py` but is not yet fed with real model output.

## 7. Write-up outline
1. Aim and data: 22 short clips, 3 classes; clip-level labels, automatic boxes, no manual frame annotation (until the label review).
2. Method: MegaDetector boxes (two passes), tight crops, ResNet18 (frozen backbone), motion features, per-second table, status table.
3. Results: Run 2 and Run 3 tables above (plus the post-review run when done).
4. Limitations: tiny dataset; lighting tied to class; no infrared Eating, only 2 infrared Resting videos; plant/tray false boxes; labels per clip; zoo night footage not tested.
5. Next steps: add night clips from the timeline, manual box/label annotation, feed per-second output into `civet_rules.py` for the zoo observation sheet.

## 8. Open questions
- Where did the friend's clips come from (days, cameras)? Overlap with the timeline events?
- A02 box entry time: `4:44:30` (as entered) vs `0:44:30`.
- Whether to merge our night clips into training later.

## 9. Environment and working notes
- Local PC: Windows, Python 3.15 (no ML packages); git is not on PATH: use `C:\Users\xkidc\AppData\Local\Fork\gitInstance\2.50.1\cmd\git.exe`. Test only pandas/PIL-level code locally; training needs a Colab T4 GPU.
- Colab pulls code with `fetch` + `reset --hard origin/feat/civet-zoo-rules-cvat` (simple `pull` once failed silently).
- The user asks for confirmation before commits and pushes: they answer "Commit and push" each time.

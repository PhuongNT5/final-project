# Paper Method Verification - Onal & Dandil, Unsafe-Net

Source: `paper/paper.txt` (343 lines), full re-read. All line numbers = actual file lines (cat -n).

## 1. Dataset / YOLO v4 training claims - CONFIRMED w/ one internal inconsistency

| Claim | Verdict | Evidence |
|---|---|---|
| 566 train / 125 test raw video segments, 8 classes | CONFIRMED | L121 ("566... training... 125... test"), Table 2 breakdown L134-196, sums to 566/125 exactly |
| 2262 train images / 329 test images labelled for YOLO v4 (Table 3) | CONFIRMED | L205 ("2262 images"), per-class rows L211-267 sum to 2262 (526+180+590+200+351+82+252+81) and 329 |
| Table 3 images span all 8 classes (4 unsafe + 4 safe) | CONFIRMED | Table 3 rows list all 8 class names incl. Safe Walkway, Authorized Intervention, Closed Panel Cover, Safe Carrying (L215-261) |
| YOLO v4 trained at imgsz 416, batch 65, subdivisions 32, 2 GPUs, mAP ~97% | CONFIRMED | L323 verbatim |
| YOLO v4 used at **nc=4** | CONFIRMED (as stated) but see inconsistency below | L323: "the number of classes was 4" |

**Inconsistency (not resolvable from text alone):** L323 ties nc=4 to the *same* "2262 images in the training set" figure that Table 3 (L205-267) explicitly breaks into 8 classes. Either (a) genuine paper error, or (b) Table 3's 2262/329 counts describe the full labelling effort (used to report labelling stats and/or a separate 8-class QA detector), while the operational detector actually wired into the pipeline was reconfigured/retrained at nc=4 (consistent with L312, which states the ConvLSTM stage only ever receives the 4 unsafe classes). Text does not disambiguate. Flag as documented ambiguity, not as an error we can adjudicate.

## 2. Frame reduction + object-wise decomposition (L305-306, L312, L325)

- L305-306: "YOLO v4 first identifies possible frames containing unsafe behaviour from video sequences and discards the rest... a reduced video stream is obtained."
- L312: "This architecture takes as input the videos from the YOLO v4 network containing the unsafe actions Safe Walkway Violation, Unauthorized Intervention, Opened Panel Cover and Carrying Overload with Forklift." I.e. YOLO's detection classes = the 4 ConvLSTM target labels, literally.
- L325 (Fig 9c + Fig 10 worked example): a 105-frame/3.5s clip is reduced to 40 frames/1.33s by YOLO, then "transmitted to the ConvLSTM network in three parts... including the Safe Walkway Violation, Opened Panel Cover and Unauthorized Intervention unsafe behaviour classes and only one action in each video. These procedures are applied in the same way to each video in the test set."

Confirmed: one clip -> N single-behaviour sub-videos, N = number of distinct unsafe-object classes YOLO detected in it. Sub-video membership is defined by YOLO's own predicted class, not by an independent temporal/actor segmentation.

## 3. THE CRITICAL QUESTION - does this leak the label into the input?

**Our reading (documented, not an accusation):** yes, structurally. Reasoning, in order of strength:

1. YOLO's detection head classes are literally the 4 target behaviour labels (L312). Frame-level curation ("keep frames where object of class X was detected, discard the rest") is therefore *performed using the same taxonomy ConvLSTM is later scored on*. A reduced/decomposed sub-video is, by construction, "the frames YOLO thinks belong to class X" - ConvLSTM is then asked to confirm/guess class X on a video already curated for homogeneity in X.
2. L48 states YOLO output is used to build "a video segment... for objects belonging to unsafe action classes within these frames" - again the segment is built around the detected class, not an independent motion cue.
3. This is architecturally different from typical action recognition where the classifier alone maps raw clip -> label; here a class-aware pre-filter partitions the input before the classifier runs.

**Strongest counter-argument:** the exact match between the ConvLSTM test tally (64 = 32+11+13+8, L334) and the *original* Table 2 per-class raw-clip test counts (32+11+13+8=64, L138-162) shows the final evaluation is scored one prediction per originally-annotated raw clip, not per YOLO-generated sub-video. If the Fig-10-style 3-way decomposition applied universally and each sub-video were separately scored, N would exceed 64 for any clip with >1 unsafe behaviour visible. It doesn't. This means either (a) the decomposition-into-multiple-sub-videos is a rare/illustrative case (most test clips already contain one behaviour after expert pre-segmentation, L62: clips were built "by identifying the safe and unsafe behaviours in the presence of experts" - i.e. single-behaviour by construction before YOLO ever runs), or (b) when a clip is split, only the sub-video matching its pre-assigned ground-truth label is scored and the rest are silently dropped. Text does not say which. This weakens the "N-inflation" / "discard-the-wrong-branches" version of the leakage claim, but does **not** address point 1-3 above: the *frame content* fed to ConvLSTM for the scored clip is still filtered by YOLO's own class-specific detections, so within-clip information is still class-conditioned before ConvLSTM sees it.

**Verdict:** text supports the "structural" reading (frame curation is class-conditioned) strongly and directly (L312, L48, L305-306). Text is ambiguous/silent on the "N-inflation" strong form of the claim, and the 64-count arithmetic argues against that stronger form. Recommend the thesis phrase this precisely: *"the detector's class assignment participates in constructing the classifier's input, which differs from scoring an untrimmed raw clip"* - not *"the paper's numbers are inflated"* (arithmetic contradicts that) and not *"this is misconduct"* (it's a stated design choice, described plainly in the text, intended - per L306 - to improve ConvLSTM accuracy and speed).

## 4. ConvLSTM split + Tables 5-7 - CONFIRMED, and the 85.78%/95.81% discrepancy is resolved (not a paper error)

- L327: "452 videos were selected for the training set and 64 videos were selected for the test set." CONFIRMED.
- L334 (Table 5, no reduction): errors = 1 (Safe Walkway Violation, out of 32), 2 (Unauthorized Intervention, /11), 3 (Opened Panel Cover, /13), 1 (Carrying Overload, /8). Total 7 errors -> **57/64 correct**.
- L334 (Table 6, with reduction/Unsafe-Net): errors = 0, 1, 1, 0 same order -> **62/64 correct**.
- Table 7 (L340) reports "85.78% Acc" (no reduction) and "95.81% Acc" (with reduction, the paper's headline number). We recomputed both as **macro-averaged per-class recall** to check consistency:
  - No reduction: mean(31/32, 9/11, 10/13, 7/8) = mean(0.9688, 0.8182, 0.7692, 0.875) = **0.8578** - matches 85.78% exactly.
  - With reduction: mean(32/32, 10/11, 12/13, 8/8) = mean(1, 0.9091, 0.9231, 1) = **0.9581** - matches 95.81% exactly.
- This confirms the paper's headline "95.81%" is macro-recall, while "62/64 = 96.88%" is top-1/micro accuracy - two different, both internally consistent, metrics from the same confusion counts. Not an error; just two metrics reported without always naming which is which. Matches the framing already used in `plan.md`.

## 5. Metrics defined in paper (L317-322)

AP, mAP, IoU, Precision, Recall for YOLO stage; Accuracy = (TP+TN)/(TP+TN+FP+FN) (Eq. 8) for ConvLSTM stage, applied per-class then macro-averaged (see #4). No per-class F1, no confusion matrix numbers beyond Tables 5/6, no confidence intervals, no significance test anywhere in the paper.

## Unresolved questions
- Whether Fig-10's 3-way decomposition is the general procedure or a rare illustrative case - text says "applied in the same way to each video in the test set" (L325) but the 64-count arithmetic implies most test clips are already single-behaviour. Cannot fully reconcile from text alone.
- Whether Table 3's 2262-image/8-class figure and L323's nc=4 refer to the same trained model or two different detectors.

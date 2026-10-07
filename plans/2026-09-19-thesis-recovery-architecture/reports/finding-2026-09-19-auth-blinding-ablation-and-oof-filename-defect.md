# Finding: the zone result survives class-blinding + an OOF-filename overwrite defect

Date: 2026-09-19. Phase-02 step 8 ablation. **Zero test touches.** ~90 s CPU.

## 1. The ablation

Question a committee will certainly ask: *is your "zone sub-task generalises" result actually
relational reasoning, or just a `Person_Authorized` lookup?* (`zone_ok` carries 23.1% P_Auth tokens
vs 2.8% for `restricted_zone` — an 8x enrichment, so the shortcut is plausible.)

Built `tokens_bin8authblind_{train,test}.pt` — detector classes 0 and 1 merged into a single generic
`Person` class (21,730 train + 4,445 test tokens relabelled), everything else byte-identical. Re-ran
E3 LCO-691, 5 seeds, 60 epochs, `drop_conflict` — the exact baseline recipe.

| variant | BA | safe rec | unsafe rec | panel-disclosed BA | safe rec |
|---|--:|--:|--:|--:|--:|
| baseline (`bin8_`) | 69.6% | 51.4% | 87.7% | **76.6%** | 60.1% |
| **auth-BLINDED** | **68.9%** | 47.4% | 90.4% | **75.2%** | 55.2% |

| sub-class | n | baseline | blinded | delta |
|---|--:|--:|--:|--:|
| 0 walkway_violation | 207 | 94.2% | 96.6% | +2.4 |
| 1 restricted_zone | 107 | 86.9% | 89.7% | +2.8 |
| 2 panel_open | 142 | 73.9% | 78.2% | +4.2 |
| 3 carry_overload | 56 | 100.0% | 100.0% | 0.0 |
| 4 walkway_ok | 75 | 36.0% | 29.3% | -6.7 |
| 5 zone_ok | 38 | 76.3% | 71.1% | **-5.3** |
| 6 panel_closed | 32 | 12.5% | 12.5% | 0.0 |
| 7 carry_safe | 30 | 100.0% | 100.0% | 0.0 |

## 2. Verdict — the zone result SURVIVES

Removing the authorized/unauthorized distinction costs **0.7 pp BA / 1.4 pp panel-disclosed BA**. The
zone pair moves asymmetrically and slightly: `zone_ok` -5.3 pp, `restricted_zone` **+2.8 pp** — i.e.
the model merely shifts a little toward "Unsafe", it does not lose the ability to separate the pair.

**Size check, stated plainly: `zone_ok` has n=38, so -5.3 pp is 2 clips.** Single 5-seed run, no CI.
This is a *descriptive* result that fails to falsify, not a positive significance claim. Wording it
may support: *the zone sub-task's cross-camera performance does not depend on the person-class
shortcut; it is carried by the geometry channel (`iou_restricted` separates the pair 4.4x, 0.027 vs
0.119).* Wording it must NOT support: any claim that blinding is provably harmless.

This is the one place in the dataset where the relation-token idea does what the thesis says it does,
and it holds up under the obvious attack. It is the positive result.

Consistent secondary observation: blinding *improved* three unsafe sub-classes (+2.4/+2.8/+4.2). With
only 365 `P_Auth` instances against 4,497 `P_Unauth` (and detector P_Auth recall 0.899), the auth
channel is weak and noisy; removing it slightly de-noises the majority direction.

## 3. Code defect found: OOF dumps are written to an UNTAGGED filename

`run_kfold_relation.py:143` builds the path as
`enc_{arch}{dsuf}{csuf}_lco-691_headline-{dedupe}_oof.json` — **the token tag is not in the name**,
although the companion `.pt` at `:146` *does* carry it
(`lco_oof_softmax_{tag}binary_{arch}...pt`). Any two runs differing only in `--tag` therefore collide
and the second silently overwrites the first.

**This happened during this ablation**: the blinded run clobbered
`enc_E3_lco-691_headline-drop_conflict_oof.json`, the E3 baseline dump. It is gitignored
(`.gitignore:12 sweep_results/**/*.json`) so git could not restore it.

**Restored, no data lost**, by reconstructing from the surviving tagged tensor
`lco_oof_softmax_bin8_binary_E3_drop_conflict.pt`; the restored file reproduces the baseline exactly
(safe recall 51.4%, BA 69.6%) and carries a `restored_note` field recording the provenance.

**Provenance of every other dump verified.** All six `enc_*_lco-691_headline-drop_conflict_oof.json`
cross-checked per-clip against their tagged `.pt` companions: all six report `tag=bin8_`, all six
MATCH with zero per-clip disagreements. **Every number in the two preceding finding reports is
therefore secured against this defect.**

### Required fix (Lane B, before any further LCO run)
Put the tag in the JSON path, exactly as the `.pt` path already does. Until that lands, **never run
two tags at the same `--arch`/`--dedupe` without moving the JSON aside first.** Also worth adding:
write `tag` into the JSON body (the `.pt` has it, the JSON does not), so a dump is self-describing.

## Unresolved questions

1. Does the -5.3 pp on `zone_ok` (2 clips) survive more seeds? Cheap to answer (~90 s/run) and worth
   3-5 repeats before any wording is frozen, given the whole positive result leans here.
2. Should the blinded variant become a permanent reported row? It costs ~90 s and directly answers
   the obvious committee question — recommend yes, as a descriptive ablation with the n=38 caveat
   printed beside it.
3. How many other artifacts in `sweep_results/` share this untagged-filename pattern? Only the LCO
   JSON was checked; the k-fold driver and the ensemble scripts were not audited.

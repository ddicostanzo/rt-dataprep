# Spec: a meaningful `retained` signal (#305)

## Problem
The first real cohort run (v0.3.0, OSC job 8533071: 651 brachy plans at 320x320x256, 1 mm, `--allow-crop`) logged 247
plans with `retained < 0.9`. The crop cut nothing that matters:
- 212 of the low mask files are the body outline, which is bigger than the crop box. Every body-named mask in the cohort
  (640) has `roi_interpreted_type` `EXTERNAL`, and no other mask does.
- 143 plans have a summed-dose `retained < 0.9` (min 0.771), yet voxels >= 5 Gy were retained 1.000 in every plan and voxels
  >= 1 Gy min 0.961. Summed dose is dominated by the low-dose tail.

The plan-level value (manifest.jsonl `retained`, the run log's low-retained count) is meant to be a quality filter, and
today it flags plans that are fine.

## Decisions (owner, 2026-11-12)
1. **Dose `retained` measures the region at or above 1 Gy.** A dose file's series.json `retained` is the fraction of its
   native-grid voxels with dose >= 1 Gy whose centres fall inside the final grid (1 Gy is a module constant). Dose is
   always Gy (convert refuses other `DoseUnits`). A dose with no voxel >= 1 Gy, or that does not cross the final grid, is
   exactly `1.0`, as today for an uncut file. `crosses_final` still decides `cropped` on any nonzero voxel, so a dose cut
   only below 1 Gy counts as cropped yet has `retained` `1.0`. The threshold is applied by thresholding the native dose
   before the same voxel-centre count `native_retained` does today; mask `retained` is unchanged.
2. **Body outlines leave the plan-level minimum.** A mask whose series.json `dicom.roi_interpreted_type` is exactly the
   string `EXTERNAL` keeps
   its own per-file `retained` (still volume-based) in series.json, but is left out of:
   - the plan's lowest `retained` that preprocess uses for the run log's `with retained < 0.9` count, and
   - manifest.jsonl's plan `retained` key.
   A plan whose only `retained` values are EXTERNAL masks gets no manifest `retained` key, the same as a run with no values.
   One shared predicate serves the preprocess plan minimum (preprocess.py, end of `preprocess_plan`) and manifest `_retained`.
3. **Signal only.** Mask `retained` stays volume-based. `LOW_RETAINED` stays 0.9. `--allow-crop` still never refuses on
   `retained`, and the `cropped` count and report column are unchanged (a cut body still counts as cropped).
   README's `--allow-crop` and manifest sections describe the new meanings.

## Behaviour
- Unchanged: which files are cut, refused (a dose/mask emptied by the crop), or written; the series.json key set; the
  manifest key order; non-geometric runs (blank `retained`, no manifest key); beam volumes (no `retained`).
- Trees preprocessed by v0.3.0 keep their summed-dose values until `preprocess --overwrite` is run again; no marker
  tells the two meanings apart, so the next release note says to re-run (owner decision). Re-running `manifest` on such a
  tree already drops EXTERNAL masks from the plan minimum.

## Acceptance tests (synthetic data only)
Tests build NIfTI + series.json plan trees as the existing `retained` tests do; tests 3-5 give every row a `dicom`
object (all-or-none per series.json) with the mask's `roi_interpreted_type`.
1. A dose whose sub-1 Gy tail crosses the final grid but whose >= 1 Gy region lies inside gets `retained` `1.0` in series.json.
2. A dose whose >= 1 Gy region is partly cut gets the voxel fraction of that region, not the summed-dose fraction.
3. A plan with a cut EXTERNAL mask and uncut other files: the EXTERNAL mask's series.json `retained` is below 1.0, the
   manifest line's `retained` is 1.0, and the run log counts 0 plans with retained < 0.9.
4. A cut non-EXTERNAL mask still lowers the manifest `retained` and the log count, as today.
5. Manifest on a series.json whose EXTERNAL mask has the lowest value omits it from the minimum (old trees).
6. The threshold is inclusive: a dose whose only nonzero voxel is exactly 1.0 Gy, centred outside the final grid but
   interpolated into it, gets `retained` `0.0` (with `>` it would be `1.0`). The existing
   `test_a_retained_of_zero_alone_does_not_refuse_when_the_output_still_holds_voxels` is this test, unchanged.

## Deferred
- A mask whose `dicom` is missing or whose `roi_interpreted_type` aggregates to `{"distinct": [...]}` counts as non-EXTERNAL.
- Excluding EXTERNAL masks from the `cropped` count/report column.
- Dose files that are one fraction or one beam (`DoseSummationType` FRACTION/BEAM): 1 Gy means something different there. Superseded by #307: the threshold is now min(1 Gy, 10% of the dose's max).
- Configurable dose threshold (CLI/`PreprocessSteps`), or a relative (% of max / prescription) threshold.
- Refusing a plan when the crop cuts dose above the threshold.

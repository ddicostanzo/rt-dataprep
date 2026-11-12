# Findings — brachy-dwells (#282)

Spec: docs/specs/2026-11-11-brachy-dwells.md (REVIEWED). Plan: docs/plans/brachy-dwells/plan.json.

Summary: convert writes `dwell.parquet` for HDR/PDR brachy RTPLANs (one row per paired-CP dwell, zero-time dwells
kept) through a new `dwell_table` module with the one fixed reason REASON_DWELL, in a fifth catch; preprocess/manifest
carry the new kind `dwell` (position_i/j/k re-indexed on geometric runs). Pattern: ion_plan.parquet (PR #296).

Order: task-01 dwell_table + synthetic writer → task-02 preprocess/manifest → task-03 convert → task-04 docs/config.

Carry-over gotchas:
- Commit gate: ≤ 5 files per commit (findings.md counts); private-name mentions need CLAUDE_ALLOW_PRIVATE_API=1 listed in the body.
- `uv run --group dev ...` only; no bare python. Full suite ~4 min.
- Synthetic DICOM via save_as/dcmread; SYNTHETIC_ identifiers; no UIDs/values in log assertions; DS values ≤ 16 chars.
- Temp files only in your own scratchpad (tmp-guard blocks elsewhere).

## task-01
Files: `src/rt_dataprep/dwell_table.py` (new), `tests/test_dwell_table.py` (new), `tests/synthetic_dicom.py` (write_rtplan/_channel).
- Public API: `DWELL_SCHEMA`, `REASON_DWELL`, `DWELL_WEIGHT_TOLERANCE`, `PAIR_ATOL_MM`, `dwell_rows(plan, ct)`, `to_table`, `write_dwell_table`.
- Writer: `write_rtplan(treatment_type=None, sources=None)`; channel dicts take `total_time`, `final_weight`, `source_number`,
  `movement_type`, `pulses=(n, interval)`, `tags`, `point_tags={cp index: tags}`; `sd.brachy_source(number, **tags)` builds a SourceSequence item.
  Positions and weights are now written as DS strings through `_ds` (repr when <= 16 chars, else `.10g`), also for old callers: values unchanged.
- Decisions: duplicate-ChannelNumber check keys on (setup number, channel number) over channels WITH control points only (the deferred N3 question);
  every `SourceSequence` item must carry `SourceNumber`; pair check allows `PAIR_ATOL_MM + 1e-9` so a pair 0.1 mm apart is not refused by float
  subtraction noise (0.1001 still is); `ControlPointIndex` is required on CP 2k only; `NumberOfPulses`/`PulseRepetitionInterval` are read on HDR too
  (null when absent) and required on PDR.
- Gotchas: tests mutate the READ-BACK header (`edit(plan)`) rather than the writer; a non-finite position goes through `point_tags` (DS 'nan'
  written unvalidated). Tests read via `read_header_core` (no quiet block). The AST test names `_pydicom_quiet` (private-name flag at commit).
  Multi-value test wraps the plan in a `.get`-only `Listy` that returns plain lists, since assigning a list to a pydicom element converts it back.

## review-1 (wave, task-01)
PASS, no BLOCKING. Fixed: W1 (final-0-with-time check had no isolating test; added, mutation-checked), N3 rename,
N5/N6 spec wording. Deferred to the 2.9 filing: N2 (no test isolates the decrease check; the flat and sum checks also
catch it), N4 (PDR pulse-interval test sits outside the parametrized case, wording only).

## task-02
Files: `src/rt_dataprep/preprocess.py`, `src/rt_dataprep/manifest.py`, `tests/test_preprocess.py`, `tests/test_manifest.py`.
- `dwell` is in `KINDS`, both keep-the-name tuples and the geometric branch, which now picks `(schema, prefix)` per kind from a dict;
  `index_plan_table(source, path, final, schema=plan_table.SCHEMA, prefix='isocenter')` (`'position'` for dwell). `REASON_PLAN_INDEX`
  names plan.parquet, ion_plan.parquet and dwell.parquet (still contains 'plan table').
- manifest `_files` gains `dwell` after `ion_plan`.
- Tests: `add_dwell(tmp_path, folder, kind, name, table)` in test_preprocess builds the table from `test_dwell_table.plan_of` (HDR, positions on the
  CT grid of `write_plan`); `DWELL_NAME` is a literal there because `conv.DWELL_NAME` arrives in task-03 (switch to it then).

## review-2 (wave, task-02)
PASS, no BLOCKING. W1 (phi rule lacks dwell.parquet) is task-04's criterion. Deferred to 2.9 filing: N1 (signature-only
prefix-default test), N2 (no manifest test with both ion_plan and dwell rows). task-03: switch the test literal
DWELL_NAME in tests/test_preprocess.py to conv.DWELL_NAME if convenient.

## task-03
Files: `src/rt_dataprep/convert.py`, `tests/test_convert.py`, `tests/test_preprocess.py` (DWELL_NAME now `conv.DWELL_NAME`), this file.
- convert: `DWELL_NAME = 'dwell.parquet'` (shares the `ION_PLAN_NAME` line); fifth catch right after the ion catch
  (`_header_call(dwell_table.dwell_rows, plan, ct)`, `to_table` outside, one WARNING `convert: <REASON_DWELL>`); written after
  ion_plan.parquet, before beam_mu; series.json row kind `dwell`, modality RTPLAN, frame = plan `FrameOfReferenceUID` ('' when absent),
  dicom from (ALL, RTPLAN). The local is named `dwell_tbl` (`dwell_table` is the imported module).
- Tests (end of test_convert.py, `hdr_plan(tree, channels=None, **kw)` helper): `make_plan`'s default plan has no FrameOfReferenceUID, so the
  default fixtures cover the '' case and one test passes `frame_uid`. The default writer takes `channels=` through `plan_meta`, so no
  partial is needed. Malformed fixture = `paired=False` (odd control-point count). HDR plans produce no plan.parquet (no BeamSequence).
- A tag-level fault test needs a REAL DICOM keyword the item lacks (a non-keyword attribute is a plain AttributeError, aborting convert).
- Full suite: 1916 passed, 4 skipped, 3 deselected; ruff clean.

## review-3 (wave, task-03)
PASS, no BLOCKING; no escape path from dwell_rows found. Deferred to 2.9 filing: N1 (tag-level fault test does not
assert total WARNING count 1 / INFO REASON_MISSING_TAG).

## task-04
Files: `README.md`, `.gitignore`, `project-profile.yml`, this file.
- README: new `### dwell.parquet` after `### ion_plan.parquet`; `dwell` added to the manifest-key row (and its order) and to the series.json `kind` list.
  Column names are the `DWELL_SCHEMA` ones. The `retained` sentence (blank for CT, applicator, line, plan, beam and series) never listed ion_plan, so dwell was not added there.
- `.gitignore`: `dwell.parquet`. `project-profile.yml` phi rule: dwell.parquet in the per-file metadata list, with applicator names, dwell positions and source calibration dates.

## review-loop-pass-1
/code-review (opus) + spec axis (opus) over the PR diff: no BLOCKING, no plan bugs. Trivial fixed: README dwell section
had split beam_mu's heading from its body (moved above it); nullable columns unmarked; moving-source wording; grammar;
dwell_table.py bare tuple of two _need calls. Deferred → hardening issues (see PR body).

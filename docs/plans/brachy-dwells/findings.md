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

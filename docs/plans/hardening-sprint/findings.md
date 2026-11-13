# Findings: hardening sprint

Spec: docs/specs/2026-11-12-hardening-sprint.md. Plan: docs/plans/hardening-sprint/plan.json (six tasks; groups [01,03,05], [02,04], [06]).
Execution: sequential in the session worktree on feat/hardening-sprint (max_parallel_worktrees = 1). Each task appends `## <task-id>` with files changed, decisions, gotchas and the red-mutation runs.
Repo gate: at most 5 files per commit; `~/.claude/hooks/pre-commit.sh` must pass before each commit.

## task-01

Files changed: src/rt_dataprep/preprocess.py (index_plan_table), tests/test_preprocess.py (two tests, plus `import pathlib` and `import pyarrow as pa`).
Decisions: `data = source.read_bytes()` sits outside the try; parse is `pq.read_table(pa.BufferReader(data))` catching (OSError, ArrowInvalid, ArrowNotImplementedError, ArrowTypeError) -> False. The write stays outside the try. Docstring updated. The only other caller is the existing direct test, which passes unchanged.
Gotchas: corruption recipe `data[-24:-8] = 0xFF*16` (footer thrift just before the trailing length + PAR1) raised plain OSError in-test on the first try; no fallback to a data page or snappy block was needed. Test (b) expects the patient row ('', 'refused', REASON_IO_ERROR) (the worker's OSError handler yields a patient-level refused row, not 'failed'), and no output patient folder.
Red runs: before the change (a) failed (patient got REASON_IO_ERROR, the OSError escaped pq.read_table) and (b) failed (read_bytes never called, patient preprocessed). Both pass after.
Suite: 2062 passed, 4 skipped, 3 deselected; ruff clean; compileall clean.

## task-03

Files changed: src/rt_dataprep/beam_volume.py (`_points`), tests/test_beam_volume.py (helper `_with_setups`, three tests; test 1 parametrised over '' and absent).
Decisions: one `all(...)` kept (so the all->any mutation still targets it): an item passes if PatientPosition == the CT position, or PatientPosition is ''/None and PatientAdditionalPosition is empty/absent. A valid PatientPosition plus PatientAdditionalPosition goes through the unchanged equality check. The module deliberately has no quiet_call/try (an AST test enforces it), so plain `.get` reads stay.
Gotchas: `any` over an empty PatientSetupSequence is False, so the all->any mutation fails 27 tests (every plan without a setup sequence), not just test 3; test 3 (`test_ac8_every_setup_item_must_match_the_ct`) is among the failures.
Red runs: before the change test 1 failed for '' and None (DID NOT RAISE). all->any: red (27 failed incl. test 3). "Refuse whenever PatientAdditionalPosition is non-empty": red, only test 2 failed. Both reverted; `git diff src/` shows only the new condition.
Suite: 2066 passed, 4 skipped, 3 deselected; ruff clean; compileall clean.

## task-05

Files changed: src/rt_dataprep/convert.py (`_table` except tuple now `(pa.ArrowInvalid, pa.ArrowTypeError, OverflowError)`), tests/test_convert.py (two parametrised tests, ArrowInvalid/ArrowTypeError each: `test_an_arrow_error_in_ion_to_table_only_drops_ion_plan_parquet` via pbs_plan/REASON_ION_PLAN, `test_an_arrow_error_in_dwell_to_table_only_drops_dwell_parquet` via hdr_plan/REASON_DWELL).
Decisions: the OverflowError catch has no test (unreachable from IS-VR inputs). Tests mirror the photon template: outcome converted, parquet absent, exactly one warning with the table's reason, SYNTHETIC_VALUE not in caplog, assert_no_leak.
Gotchas: the mutation must guard None rows (`None if rows is None else _header_call(to_table, rows)`); an unguarded `_header_call(to_table, None)` also reddens the photon test on plans with no ion/dwell rows, which is noise, not a pin.
Red runs: with ion and dwell sites routed through `_header_call` (None-guarded), exactly the 4 new tests failed (outcome 'refused' != 'converted'); the photon tests stayed green. Reverted; `git diff src/` shows only the except tuple.
Suite: 2070 passed, 4 skipped, 3 deselected; ruff clean; compileall clean.

## task-02

Files changed: src/rt_dataprep/preprocess.py (image loop in preprocess_plan).
Decisions: `checked = kind != 'beam'` is set once (beam-footprint comment kept on that line); the crop-check block and the has_voxels refusal both use it. Not `kind in ('dose','mask')`. Pure refactor, no new test, no red run.
Suite: 2070 passed, 4 skipped, 3 deselected; ruff clean; compileall clean.

## task-04

Files changed: tests/test_beam_volume.py (two tests, each parametrised over the three TABLE_TOP_ANGLES keywords: `test_a_setup_beam_with_a_non_zero_table_top_angle_is_not_refused`, `test_ac8_table_top_angle_first_listed_on_a_later_control_point_is_refused`). No src/ change.
Decisions: the setup beam is `sd.photon_beam(2, kind='setup')` with the angle set on every control point, added to a valid `static()` plan. The later-CP test zeroes all three angles on CP 0 and sets the keyword on CP 1. 360.0 behavior untouched.
Red runs: dropping the SETUP `continue` in `_table_top_flat` -> exactly the 3 setup tests failed (CaseRefused). Checking only control point 0 (`[:1]`) -> exactly the 3 later-CP tests failed (DID NOT RAISE). Both reverted; `git diff src/` is empty.

## task-06

Files changed: tests/test_fluence.py, tests/test_beam_volume.py, tests/test_convert.py. No src/ change (`git diff src/` empty).
Tests added: (1) `test_aperture_is_none_for_fewer_than_two_leaf_boundaries` ([one], [none]) in test_beam_volume; (2) `test_ac8_a_not_modelled_aperture_on_a_weight_zero_control_point_is_still_refused` (blocks, mlcy); (3) `test_accumulation_is_float64_so_small_weights_are_not_lost_beside_a_large_one`; (4) `test_ac8_edge_at_exactly_200_is_inside_at_minus_x_plus_y_and_minus_y` (-X, +Y, -Y); (5) `test_ac8_a_setup_beam_sharing_a_beam_number_with_a_treatment_beam_is_accepted` (setup last, setup first); (8) `test_ac8_duplicate_treatment_beam_numbers_are_refused_by_fluence_itself`. Item 6: the AC1 outside-field assertion now checks a[0,:170], a[0,230:], a[0,:,:150], a[0,:,250:] are all zero (was :169, 231:, :149, 251:). Item 7: the beam-numbers convert test passes plan_meta frame_uid=sd.generate_uid() and asserts the compared frame UID `!= ''`.
Decisions: (3) plan_rows accepts BeamMeterset 2^25+4, so no hand-built rows; the plan is `static` with six control points, cumulative weights [0, 2^25, 2^25, 2^25+2, 2^25+2, 2^25+4], and the test asserts the row mu list [0, 2^25, 0, 2, 0, 2] before checking pixel (200, 200) == 2^25+4. (2) the weight-0 beam has weights [0, 0, 1], so row 0 has weight 0; the test first confirms the unmodified rows are accepted. (8) rows come from a [1, 2] plan, the plan passed to fluence_mu is [1, 1].
Red runs (each reverted): (1) remove `len(boundaries) < 2 or` -> both new tests failed. (2) call aperture only when weight != 0 -> blocks and mlcy failed. (3) `np.zeros(..., dtype=np.float64)` -> float32 accumulator -> the new test failed (only it). (4) `< -LIMIT` -> `<= -LIMIT` -> [-X] and [-Y] failed; `> LIMIT` -> `>= LIMIT` -> [+Y] and the existing 200/200.01 test failed. (5) drop the SETUP filter in fluence_mu's duplicate check -> both orders failed. (8) delete the duplicate check -> the new test failed.

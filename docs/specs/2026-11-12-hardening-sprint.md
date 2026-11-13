# Hardening sprint: #299 #290 #291 #295 #284 #335 #327 #325

One PR that closes eight `hardening` issues. Most of it is tests. There are four small code changes, each listed below. The real-data bullets move to #283 (see Out of scope).

## Code changes

1. **#284, preprocess.index_plan_table.** Read the file with `data = source.read_bytes()`. Any OSError there is a file-system error (permission, EIO, vanished file), so it propagates and fails the patient as today (REASON_IO_ERROR). Then parse from memory with `pq.read_table(pa.BufferReader(data))`. Any `OSError`, `pa.ArrowInvalid`, `pa.ArrowNotImplementedError` or `pa.ArrowTypeError` from the parse means the file is corrupt. The bytes are already in memory, so nothing raised there can be a file-system error, so the function returns False and only that table is dropped (REASON_PLAN_INDEX). The stress test confirmed that a full-length corrupt footer or page raises plain OSError through BufferReader. An OSError from the later write still propagates. Plan tables are small, so holding the bytes in memory costs nothing.
2. **#335, beam_volume._points.** Refuse (REASON_BEAM_VOLUME) when a PatientSetupSequence item has an empty or absent `PatientPosition` and a non-empty `PatientAdditionalPosition`: its position can't be checked against the CT. Clinical output. Under PS3.3 C.8.8.12, PatientAdditionalPosition may sit alongside a valid PatientPosition (e.g. HFS + 'arms up'). That case stays accepted and goes through the existing PatientPosition == CT check.
3. **#327, convert._table.** Add `OverflowError` to the caught tuple next to `pa.ArrowInvalid, pa.ArrowTypeError`, so that plan loses only its table. Unreachable today, so no test.
4. **#290, preprocess image loop.** Replace the two `kind != 'beam'` checks with one flag, `checked = kind != 'beam'`, set once, with the existing comment kept on it. Do not use `kind in ('dose','mask')`, because it is not exactly equivalent. Behavior is unchanged and covered by the existing tests. The beam_volume/plan_table `_refuse`/`_check` copies stay as they are, by decision.

## Tests for the code changes

- **#284:**
  - (a) A geometric preprocess run on a full-length `plan.parquet` whose footer thrift bytes are overwritten: the patient is still preprocessed, plan.parquet is left out, and exactly one REASON_PLAN_INDEX warning is logged, with no REASON_IO_ERROR. The corruption must raise plain OSError, not ArrowInvalid. Overwrite the footer metadata bytes just before the trailing `<4-byte length>PAR1`, keep the file length, and assert in the test that `pq.read_table` raises OSError and not ArrowInvalid. The test must fail on current code.
  - (b) `pathlib.Path.read_bytes` monkeypatched to raise OSError for plan.parquet: the patient gets REASON_IO_ERROR.
  - Template: `tests/test_preprocess.py::test_a_geometric_run_leaves_out_an_unreadable_or_non_schema_plan_parquet`.
- **#335:** the empty-PatientPosition + PatientAdditionalPosition item is refused. HFS + PatientAdditionalPosition on an HFS CT is accepted.

## Tests only (behavior already correct)

Each pin names the mutation that must turn it red. The implementer applies the mutation, sees the test fail, then reverts it.

- **#299 + #327 (Arrow-error tests):** the issue predates #276. `convert._table` now catches Arrow errors, so the right pin is "the plan still converts and only that table is dropped with its own fixed reason". If the call were moved into `_header_call`, the ValueError would refuse the plan with REASON_MISSING_TAG. The photon case already exists (`tests/test_convert.py` `test_an_arrow_error_in_to_table_only_drops_that_plans_parquet`). Add the same test for `ion_plan_table.to_table` (REASON_ION_PLAN, ion_plan.parquet) and `dwell_table.to_table` (its reason, dwell.parquet), parametrized over ArrowInvalid and ArrowTypeError. Builders: `pbs_plan` and `hdr_plan` in tests/test_convert.py, beside the photon test. Red mutation: wrap the `_table(...)` call in `_header_call`, or drop the try in `_table`.
- **#291:** a SETUP beam with a non-zero table-top angle is skipped (no refusal). A non-zero angle first listed on control point 1 or later is refused. 360.0 is still refused, by decision. Red mutation: let `_table_top_flat` check SETUP beams too, and check only control point 0.
- **#295:** seven bullets.
  - aperture with fewer than two leaf boundaries is refused.
  - A not-modelled aperture on a weight-0 control point is refused.
  - Accumulation stays float64. The output is cast to float32 at the end, so the test needs a sum that loses small additions only when the running total is float32. Choose cumulative MU so that the trapezoid weights are 2^24 followed by unit weights that come out as exactly 1.0, with a total added of at least 2: 2^24+1 rounds back to 2^24 at the final cast, but 2^24+2 survives. Red mutation: `np.float64` → `np.float32` in fluence_mu's accumulator.
  - An exactly-200 mm field is accepted at −X, +Y and −Y.
  - A SETUP beam may share a BeamNumber with a treatment beam.
  - The AC1 all-zero-outside checks include the bordering rows and columns.
  - The synthetic plan in the convert fluence test gets a non-empty frame UID.
- **#335:** two setup items, one matching and one not, is refused. Red mutation: `all` → `any` in the setup check.
- **#325:** call fluence_mu directly with two treatment beams that share a BeamNumber, and expect REASON_FLUENCE. The check is kept, by decision. Red mutation: delete the duplicate check in fluence_mu.

## Already done

- #325 bullets 2 and 3 (ion/dwell control-point checks, photon meterset >= 0) shipped in bbe859f (#336).

## Out of scope: moved to #283 as a comment, then closed with the PR

- #327: count photon treatment beams that carry ApplicatorSequence before the next full photon run.
- #325: verify ControlPointIndex == position on a real photon export.
- #335: narrow the setup check to referenced setups if a real plan is refused for an unused one.

## Acceptance

- Full suite passes, ruff is clean, and the pre-commit gate passes.
- Each new test fails if the behavior it pins is removed.
- The PR closes #299 #290 #291 #295 #284 #335 #327 #325, and #283 carries the three moved bullets.

## Deferred / declined (recorded, not dropped)

- 360°/near-zero table-top angle tolerance (#291): added to the #283 comment with the real-data bullets, to act on if real plans hit it.
- Sharing `_refuse`/`_check` across modules (#290): declined by the owner. Stated in the PR body.

## Orchestrator steps outside the code

- Before merge, post one comment on #283 listing the three moved real-data bullets and the 360° item.
- The PR body carries eight separate `Closes #N` lines.

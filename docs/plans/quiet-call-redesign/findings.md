# quiet_call redesign: findings

Spec: docs/specs/2026-11-04-quiet-call-redesign.md. Plan: docs/plans/quiet-call-redesign/plan.json.
Baseline at 5977e72: 483 passed, 1 skipped.

## task-01

- Red before the change (tests/test_dicom_io.py, 36 collected, 5 failed): A4 failed for KeyboardInterrupt('SYNTHETIC_x'), SystemExit('SYNTHETIC_x'), the needs-an-argument BaseException subclass and GeneratorExit (original object leaked through); A5 failed (extra fd 12 left open after the second os.dup fails). A6 and the SystemExit(3) / SystemExit() / SystemExit(None) cases passed pre-change (pins).
- Decisions: the except BaseException sits after the existing except Exception inside fn's try; it records only (type, code), and the bare object is built and raised after the with/finally blocks. SystemExit subclasses map to SystemExit; SystemExit(None) and SystemExit() both come out as SystemExit() (None code). Construction failure of type(exc)() falls back to KeyboardInterrupt().
- _fds_quiet: dups are appended one at a time inside the try; the finally restores fds 1/2 only when both dups exist, then closes whatever was made.
- Gotchas: ruff PLW (stacklevel) wants stacklevel on the thread warn in the A6 test. The scan flags dotted private refs, so the A5 test reaches _fds_quiet via getattr(dicom_io, '_fds_quiet')  # noqa: B009.
- Result: 492 passed, 1 skipped (baseline 483 + 9 new cases); ruff and compileall clean.

## task-02

- Red before the change (tests/test_convert.py, 8 new cases, 5 failed): A1 failed (DID NOT RAISE ConvertAborted: the TypeError read as REASON_DOSE_UNREADABLE); A10 failed for all four cases (NaN frame offset, and NaN in ImagePositionPatient, ImageOrientationPatient and PixelSpacing: each converted without refusing). A2 passed pre-change (pin), for the 5-value IOP, the 1-value PixelSpacing and a 2-value IPP.
- Decisions: _read_dose is extract (quiet) / arithmetic (bare) / build (quiet). Extraction checks counts 6 / 3 / 2 and np.isfinite on the plain floats, raising a bare ValueError (refused by the wrapper). offsets is used by len() alone inside extraction; indexing and subtraction start in the arithmetic step. The offsets finiteness check sits in _check_dose_header after np.diff and before the grid test, so a NaN offset gives REASON_DOSE_GRID.
- Equivalence: reorder, origin, spacing, direction and the float64 scaling are the old statements moved unchanged, so output for a well-formed RTDOSE is the same; every existing dose test passes unmodified.
- Gotcha: pydicom reads a one-value DS back as a scalar (DSfloat), not a list; the old and new code both refuse it through the TypeError from iterating it. The A2 fixture check uses np.ravel for that reason.
- Result: 500 passed, 1 skipped (492 + 8 new cases); ruff and compileall clean.

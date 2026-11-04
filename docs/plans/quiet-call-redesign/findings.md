# quiet_call redesign: findings

Spec: docs/specs/2026-11-04-quiet-call-redesign.md. Plan: docs/plans/quiet-call-redesign/plan.json.
Baseline at 5977e72: 483 passed, 1 skipped.

## task-01

- Red before the change (tests/test_dicom_io.py, 36 collected, 5 failed): A4 failed for KeyboardInterrupt('SYNTHETIC_x'), SystemExit('SYNTHETIC_x'), the needs-an-argument BaseException subclass and GeneratorExit (original object leaked through); A5 failed (extra fd 12 left open after the second os.dup fails). A6 and the SystemExit(3) / SystemExit() / SystemExit(None) cases passed pre-change (pins).
- Decisions: the except BaseException sits after the existing except Exception inside fn's try; it records only (type, code), and the bare object is built and raised after the with/finally blocks. SystemExit subclasses map to SystemExit; SystemExit(None) and SystemExit() both come out as SystemExit() (None code). Construction failure of type(exc)() falls back to KeyboardInterrupt().
- _fds_quiet: dups are appended one at a time inside the try; the finally restores fds 1/2 only when both dups exist, then closes whatever was made.
- Gotchas: ruff PLW (stacklevel) wants stacklevel on the thread warn in the A6 test. The scan flags dotted private refs, so the A5 test reaches _fds_quiet via getattr(dicom_io, '_fds_quiet')  # noqa: B009.
- Result: 492 passed, 1 skipped (baseline 483 + 9 new cases); ruff and compileall clean.

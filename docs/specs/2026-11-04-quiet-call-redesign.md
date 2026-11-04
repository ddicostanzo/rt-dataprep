# quiet_call and _header_call: bugs are loud, not refusals (#40, #20, #24)

## Problem

`dicom_io.quiet_call(fn, reason)` runs a third-party call with every output channel shut and turns any `Exception` into
`CaseRefused(reason)` with nothing of the original kept. That is right for third-party failures, whose messages name
paths (vault: [[third-party-loader-errors-and-warnings-embed-phi-paths]]). It is wrong for our own code: a coding bug
inside a quiet closure reads as a per-case refusal instead of aborting the run (#40). The one closure with substantial
arithmetic of ours is `_read_dose` in convert (geometry, scaling, reordering). Before the OSC parity run, refusal counts
have to be trustworthy: a bug must be loud, not a plausible refusal count.

#20 adds smaller defects in the same wrapper: a `BaseException` from `fn` passes through with its original object
(message included), `_fds_quiet` leaks fds when the second `os.dup` fails, and Python 3.14 context-aware warnings could
let a worker thread's warning bypass the filter. #24: the A2.10 PHI test's truncated-CT half cannot catch a quiet-wrapper
regression, because ITK's error for a truncated slice names no path.

## Decisions (owner, 2026-11-04 interview)

- **Why now:** backlog, before the parity run. No incident; the smallest change that makes bugs loud.
- **#40 approach:** split closures. A quiet closure holds third-party access, value extraction and deliberate raises
  only; our arithmetic runs outside, where a bug reaches the existing abort path (convert: `REASON_INTERNAL`, logged as
  exception class name + `crash_site`, run raises `ConvertAborted`). `quiet_call` keeps its broad except.
- **Split scope:** `_read_dose` only, plus the rule written into `quiet_call`'s docstring. `_place_struct` and `_run`
  (masks) stay: their failures are almost all third-party attribute access on bad data. Preprocess already passes
  `REASON_INTERNAL` to `quiet_call`, so its failures abort already.
- **Bad data after the split:** validated inside the quiet extraction. Every value the outside arithmetic needs is
  checked there (counts), so a malformed RTDOSE still refuses with `REASON_DOSE_UNREADABLE`, and a failure outside is a
  bug by construction.
- **BaseException:** re-raised as a bare object of the same kind: `KeyboardInterrupt()`; `SystemExit(code)` when the
  code is an int (or None), else `SystemExit(1)`; any other non-`Exception` as bare `type(exc)()`. Raised after the
  except block, nothing chained.
- **Thread safety:** deferred to #2. Docstring states quiet_call is process-wide and not thread-safe (fds 1/2 cannot be
  per thread); #2 gets a comment that parallelism must be process-based or serialize quiet_call.
- **Fold-ins:** #24 (A2.10 test with a CT read failure whose raw ITK message names the path), the `_fds_quiet` fd leak,
  and a test pinning that a warning from a thread started inside `quiet_call` is not emitted.
- **`_header_call` (stress-test round, owner: include):** the same masking as #40 for header-only steps
  (`_assemble`, `_check_volume`, `_check_contour_images`, `_dose_candidates`, `_check_dose_header`, `masks.classify`,
  `_series_rows`). Not rewritten: (1) each `REASON_MISSING_TAG` refusal logs the exception class name and `crash_site`
  (no message), so a bug clustering at one line is visible in the parity logs; (2) an `AttributeError` or `KeyError`
  whose raising frame (innermost traceback frame) is not in pydicom propagates instead of refusing, reaching the abort
  path. Verified: a missing tag (`ds.X`, `ds['X']`) raises from `pydicom.dataset`. `TypeError`, `ValueError`,
  `IndexError` stay refusals, since our frames raise them for bad data (`int(None)`, an empty sequence).
- **Dose finiteness (stress-test round, owner: fix here):** NaN frame offsets pass `_check_dose_header`'s uniformity
  check and `SetSpacing` accepts NaN, so a malformed RTDOSE could resample into a wrong dose image. Non-finite offsets
  refuse with `REASON_DOSE_GRID`; non-finite IOP/IPP/PixelSpacing in the `_read_dose` extraction refuse with
  `REASON_DOSE_UNREADABLE`.
- **Merge:** /dev may push, open the PR, run one review round, wait for Codex's 👍/review, resolve its threads and
  squash-merge.

## Behaviour

### B1. `_read_dose` split (convert.py)

1. **Quiet extraction** (`quiet_call(..., REASON_DOSE_UNREADABLE)`): `dcmread`, `pixel_array` as float64 times
   `float(DoseGridScaling)`, the frame-count check against `offsets`, and `ImageOrientationPatient`,
   `ImagePositionPatient`, `PixelSpacing` read as plain Python floats with their counts (6, 3, 2) and finiteness
   checked. The step returns only floats and an ndarray (no pydicom object leaves it). Any failure,
   deliberate or third-party, refuses as today.
2. **Our arithmetic, unquieted:** row/column/normal cosines, the reorder for a negative step, origin, spacing and
   direction tuples. Pure numpy on the extracted values; a failure here propagates as itself.
3. **Quiet build** (`quiet_call(..., REASON_DOSE_UNREADABLE)`): `GetImageFromArray`, `SetSpacing`/`SetDirection`/
   `SetOrigin`, `resample_dose`. ITK judges degenerate geometry here, as today.

Output for every well-formed RTDOSE is unchanged: the existing dose tests (A2.1, summation, orientation, negative and
absolute offsets) pass unmodified.

### B2. `quiet_call` contract (dicom_io.py docstring)

States the closure rule (third-party access, value extraction, deliberate raises; no arithmetic or logic of ours), that
`fn`'s `Exception`s all become `CaseRefused(reason)`, the `BaseException` handling, and that it swaps process-wide state
and is not thread-safe (#2).

### B3. BaseException

As decided above. If constructing the bare object fails (`BaseExceptionGroup`, a custom class with required args), a
bare `KeyboardInterrupt()` is raised instead: an interrupt is still an interrupt, and nothing of the original leaks. The
existing `MemoryError` / ITK out-of-memory path is unchanged.

### B4. `_fds_quiet` fd leak

If `os.dup(2)` fails after `os.dup(1)` succeeded, the first duplicate and the sink are closed before the error
propagates. No fd is left open by any failure in setup.

### B5. `_header_call` (convert.py)

The except tuple is unchanged. On a caught `AttributeError`/`KeyError` whose innermost traceback frame's module is not
`pydicom` or a `pydicom.*` submodule, the exception propagates as itself (it is ours, and the abort path logs only class
and site). An `AttributeError` raised from pydicom whose `name` is not a DICOM keyword
(`pydicom.datadict.tag_for_keyword(name) is None`: a misspelled keyword in our code, verified on pydicom 3.0.2) also
propagates. Every other caught exception refuses with `REASON_MISSING_TAG` as today, after logging
`'convert: %s: %s at %s:%d'` with the reason, class name and `crash_site` at INFO (the CLI logs at INFO). The traceback
comes from `sys.exc_info()[2]`, as `convert.py`'s patient loop does. A propagated `KeyError` carries its key in its
message; that is safe only because the abort path formats class name and site and raises `ConvertAborted` unchained,
so no handler on that path may log the exception itself.

Accepted behaviour: an attribute access on a value pydicom returned as None (an empty tag) raises from our frame and
aborts the run. The plan audits the wrapped functions for such accesses and records the result.

### B6. Dose finiteness

`_check_dose_header`: any non-finite frame offset refuses with `REASON_DOSE_GRID`. B1's extraction: any non-finite
IOP/IPP/PixelSpacing value refuses with `REASON_DOSE_UNREADABLE`.

## Acceptance criteria

- **A1** A `TypeError` raised from `_read_dose`'s arithmetic step aborts the convert run: the patient's row is
  `REASON_INTERNAL`, `ConvertAborted` is raised, and no row says `REASON_DOSE_UNREADABLE`. Injection: patch
  `_check_dose_header` (via `monkeypatch.setattr` on the convert module) to return an offsets object whose `len()` and
  indexing work but whose subtraction raises `TypeError`. Patching numpy is wrong: `_header_call(_assemble)` would
  catch it first. On the pre-change code the test fails (the row is `REASON_DOSE_UNREADABLE`).
- **A2** An RTDOSE whose `ImageOrientationPatient` has 5 values, or whose `PixelSpacing` has 1, refuses the plan with
  `REASON_DOSE_UNREADABLE` (fixture built through `save_as` and read back with `dcmread`). The run continues. This
  pins behaviour (it also passes on the old code); it is the only test asserting `REASON_DOSE_UNREADABLE`.
- **A3** All existing tests pass unmodified, in particular every dose-geometry test.
- **A4** `quiet_call(fn)` where `fn` raises `KeyboardInterrupt('SYNTHETIC_x')` raises `KeyboardInterrupt` with empty
  `args`, no `__context__` and no `__cause__`. `SystemExit('SYNTHETIC_x')` → `SystemExit(1)`; `SystemExit(3)` →
  `SystemExit(3)`. All output channels are restored afterwards. A `BaseException` subclass whose constructor needs
  arguments becomes a bare `KeyboardInterrupt()`.
- **A5** With `os.dup` failing on its second call (a counting patch, removed before the fds are listed), `_fds_quiet`
  raises and the set of open fds (`/proc/self/fd`) is the same before and after, compared after leaving
  `pytest.raises` with the exception info dropped (its traceback keeps the generator frame alive). Skipped where
  `/proc/self/fd` does not exist. Fails on the pre-change code.
- **A6** A `warnings.warn` issued in a thread started and joined inside `fn` emits nothing (no warning recorded outside,
  nothing on fd 2).
- **A7** (#24) A separate test (not a new case of the existing A2.10 parametrize, whose `run_plan` half would re-read
  headers and refuse with `REASON_CT_SLICE_MISSING`): one planning-CT slice is deleted after the header read (a
  wrapper around `convert.read_folder_headers`), so ITK fails at `Execute` with "Could not open file: <path>"
  (verified; garbage or a truncated slice names no path and must not be used). The plan refuses with
  `REASON_CT_UNREADABLE`, and no path or `SYNTHETIC_` id appears in rows, log records or fds 1/2. Mutation check,
  run once and recorded in findings.md: with `convert.quiet_call` (the name convert imported, not `dicom_io`'s)
  patched to pass `str(exc)` through as the reason, the test fails.
- **A9** `_header_call`: (a) a step raising `AttributeError` from our own frame (an attribute missing on a plain
  object) propagates, and a convert run with it injected into `masks.classify` aborts with `REASON_INTERNAL`;
  (b) a step reading a missing tag (`ds.X` on a Dataset without it) still refuses with `REASON_MISSING_TAG`, and one
  INFO record carries the class name and a `rt_dataprep.*` module and line, with no path or `SYNTHETIC_` id;
  (c) `KeyError` from our own dict propagates; `ds['X']` on a missing tag refuses; (d) a misspelled keyword
  (`ds.SYNTHETICNotAKeyword`) propagates. (a), (c)-ours and (d) fail on the pre-change code.
- **A10** An RTDOSE with a NaN frame offset refuses with `REASON_DOSE_GRID`, and one with a NaN `ImagePositionPatient`
  value refuses with `REASON_DOSE_UNREADABLE`; fixtures through `save_as`/`dcmread`. Each fails on the pre-change code
  (if one does not, its guard is unreachable: drop the guard and record why in findings.md).
- **A8** Build, lint and the full suite pass, existing tests unmodified except the two whose injected `KeyError`
  stood for a refusal (`tests/test_convert.py` `refuse_after_the_plan_folder_exists`, `tests/test_series.py`
  `test_a_tag_error_while_building_the_series_rows_refuses_the_plan_and_removes_the_folder`): they raise `ValueError`
  instead (owner decision), and what they pin is unchanged.

## Guardrails

- Vault [[third-party-loader-errors-and-warnings-embed-phi-paths]]: raise after the except block; tests check message,
  `__context__`, `__cause__`.
- Vault [[pydicom-validates-lazily-on-first-tag-access]]: every pydicom value `_read_dose` uses is converted inside the
  quiet extraction (the outside step sees only floats and arrays); quiet tests assert that nothing at all is emitted.
- Commit gate: at most 5 staged files per commit; the private-API scan flags a dotted private reference, dunders included, so code reads
  the traceback via `sys.exc_info()[2]` and tests read `__context__`/`__cause__` via `getattr(...)  # noqa: B009` as
  the existing tests do; tests reach private helpers with
  `getattr(module, '_name')  # noqa: B009`.
- PHI: fixed `REASON_` text and counts only; test data `SYNTHETIC_`-prefixed; no `.dcm` committed.
- New guard tests run against the pre-change code first (A1, A5 must fail there; A7 by mutation).

## Issue disposition

Closes #40 (quiet_call and the same masking in `_header_call`), #20 (thread-safety part moved to #2 by comment), #24.
The NaN dose finiteness fix has no issue; the PR body names it.

## Deferred

- Thread safety / parallel quiet_call: #2's design (process-based parallelism or a serializing lock).
- Splitting `_place_struct`, `_run` and other closures.
- `_header_call`'s `TypeError`/`ValueError`/`IndexError` from our own logic still refuse; the INFO site log is the
  diagnostic. An extract-then-decide rewrite of the header steps waits for a real masked bug.
- A pre-existing worker thread's warnings under 3.14 context-aware warnings (A6 covers threads started inside `fn`).
- CI on Python 3.14 or free-threaded builds (A6 pins today's behaviour only).

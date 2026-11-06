# Findings: per-patient process pool (#171)

Spec: docs/specs/2026-11-05-patient-process-pool.md. Plan: docs/plans/patient-process-pool/plan.json.

Summary: `--workers N` = N patient processes (spawn start method, pinned) in organize, convert, preprocess; one
scheduler (src/rt_dataprep/pool.py) for every N (N=1 calls inline, no executor); bounded submission (<= N in flight);
parent owns rows (sorted order), claims, kept, hold-by-casefolded-name; failures become data inside the worker; the
parent never formats an exception from a future. read_pooled (threads) is retired; serial reads keep _pydicom_quiet.

Run tests with `uv run --group dev pytest -q --tb=short` inside the worktree. Never print names, MRNs or paths.

Append below under `## <task-id>` (append only).

## task-01

- read_pooled and the ThreadPoolExecutor import are gone from dicom_io. organize `_organize_patient` wraps its per-folder `map(_read_one, paths)` in `dicom_io._pydicom_quiet()`; convert `read_folder_headers(folder)` does the same around its list comprehension. Both reach the underscore helper from another module (pre-commit private-API flag expected).
- Signatures: `read_folder_headers(folder)`, `_organize_patient(folder, out_root, claimed, exclude_models, overwrite)`, `_convert_patient(folder, out_root, exclude_models, overwrite, claimed)` (workers dropped from the internal convert helper too: it was only passed through). `organize()`/`convert()` still accept and log `workers`; convert's docstring says unused until patients run in processes.
- Tests: the six read_pooled tests and the two "pooled read gets workers value" tests are deleted; one silence test per caller added (tests/test_organize.py, tests/test_convert.py; invalid-value RTDOSE, control read proves it warns unquieted, capfd empty, validation mode / pydicom logger / warnings.filters restored). Removing the quiet block makes both fail. test_series seam is `headers(folder)`.
- Workers=1 vs 4 parametrized organize/convert tests still pass untouched (workers is accepted and ignored).

## task-02

- API (src/rt_dataprep/pool.py): `pool.run(workers, tasks, handle)`. `tasks` is a list of `Task(fn, args=(), key=None)` (fn module-level and picklable; key = casefolded name or None). `handle(index, result)` is called by the parent; `result` is the fn's return value or `Failed(type_name)`. handle returns None, `pool.ABORT`, or a follow-up `Task` for the same index (queued ahead of unstarted work, shares the N slots, its result comes back through handle(index, result), may chain).
- Ordering: FIRST-stage results reach handle in input order. Follow-up results reach handle in completion order (not input order, to keep copies parallel): organize must store its row by index. After ABORT or a Failed nothing more is submitted; running tasks finish and are delivered (a follow-up returned then is dropped); never-submitted items get no call (no row).
- Exceptions: anything raised by fn (BaseException other than KeyboardInterrupt) becomes `Failed(type(exc).__name__)` and stops submission; the exception is never formatted/logged/chained. A KeyboardInterrupt (from a task, or from the parent's wait) stops submission, running tasks finish, returned results still reach handle, tasks that raise get no call, then a fresh `KeyboardInterrupt` is raised from run(). So callers keep their rows via handle and write the report in their existing `finally`.
- `run` returns None; the caller maps Failed to its REASON_INTERNAL row, ABORT, and raises `*Aborted` after run() returns. The worker wrapper (OSError -> IO row, other Exception -> crash_site log + internal marker) lives in the task fn, so Failed only fires for what escapes it (BrokenProcessPool, unpicklable result...).
- Test workers in tests/pool_workers.py: raisers `convert_patient`, `preprocess_patient`, `organize_read`, `organize_copy` take `(folder, *args, **kwargs)` and raise RuntimeError(SECRET) when `Path(folder).name == 'boom'` (RAISE_NAME), else return a minimal result in the real shape (convert `([], 0.0, 0, None)`, preprocess `([], None)`; organize shapes are placeholders: adjust them in task-03/04 when the real shapes are fixed, tests/pool_workers.py is shared).
- Test helper pattern: run pool.run in a daemon thread, join(60), assert not alive (tests/test_pool.py `run_guarded`). Spy executor subclass counting not-done futures at submit proves the N bound (done() is set before the parent wakes; done-callbacks are racy). `pool.wait` monkeypatch raising KeyboardInterrupt once simulates Ctrl-C in the parent.
- Gotcha: `os._exit` in a test worker trips the pre-commit private-API flag; use `os.kill(os.getpid(), signal.SIGKILL)`.
- 33 tests in tests/test_pool.py, 3 consecutive runs green (~17 s each: spawn startup).

## review-group-2
- B1 fixed (6b98f61): idle workers ignore SIGINT (initializer), `pool._call` enables KeyboardInterrupt only while a task runs and resets to ignore after. A real killpg(SIGINT) test is in test_pool.py.
- W2 fixed: after ABORT/Failed no new first-stage task starts, but follow-ups of already-started indexes still run (unless interrupted). Organize: a patient whose read finished still gets its copy and row.
- For task-03/04: a broken pool gives every running task its own Failed('BrokenProcessPool') → expect possibly several REASON_INTERNAL rows; handle each by index. Follow-up results arrive in completion order: store rows by index and write them in sorted order at the end.
- N6 (deferred): picking the next task scans the queue (quadratic in patients); negligible at these sizes.

## task-03

- convert() and preprocess() (new `workers=1` keyword) build one `pool.Task(_convert_worker | _preprocess_worker, (patient_fn, folder, ...), key=casefolded name)` per dispatched folder. The module-level worker wraps `_patient_fn` (default `_convert_patient` / `preprocess_patient`, resolved at call time so monkeypatches of the module attribute still work at N=1): OSError -> REASON_IO_ERROR row + warning, other Exception -> crash_site + the type/module/line log line, internal marker. Worker return: convert `(rows, reading, files, status, patient_seconds, internal)`, preprocess `(rows, status, internal)`; the inner patient functions keep the 4-tuple / 2-tuple shapes, so tests/pool_workers.py raisers needed no change.
- The `claimed` argument is gone from `_convert_patient` / `preprocess_patient`. The parent refuses a casefold repeat under overwrite (no task) and gives every task key=casefolded name. `convert.rows_in_order(count, refused, done)` (public, preprocess reuses it) writes rows in folder order; a never-dispatched repeat's row is written only if every dispatched folder before it has a result, so an abort leaves no row for patients that were never reached (as the serial loop did).
- Pool.Failed in handle: log `<stage>: REASON_INTERNAL: <type name>`, REASON_INTERNAL row, ABORT. Except `Failed('SystemExit')`: handle re-raises SystemExit (pool turns every non-KeyboardInterrupt BaseException into Failed; tests/test_series.py::test_ctrl_c_and_exit... expects SystemExit to propagate with no row, report still written by the finally). Not in the plan; kept to preserve behaviour.
- Timing line now 'convert: %.1f s total, %.1f s reading (summed over patients), %.1f s converting (summed over patients), %d files, workers %d'; converting = sum(patient seconds - reading), IO/internal patients add their time but no reading/files.
- Tests moved to workers=1 (they patch or use caplog): test_convert.py `test_one_timing_line_after_the_summary_...` (was workers=2, files/workers assertion now (12, 1)); `test_the_header_read_goes_through_the_dicom_io_module_attribute` (parametrization [1, 4] dropped). The 1-vs-4 equality at test_convert.py ~1323 and the organize one keep N=4. TIMING regex updated.
- New tests (tests/test_convert.py, tests/test_preprocess.py, end of file): N=1 vs N=2 on 3 patients (rows, file names, bytes), overwrite claim at N=1 and 2, held repeat without overwrite (rows equal N=1; "never concurrent" is covered by the hold tests in test_pool.py), abort via `_patient_fn=pool_workers.convert_patient/preprocess_patient` with a 120 s thread guard and capfd (no SECRET, 'RuntimeError' present = the worker's log line arrived). Each spawn run costs ~3-5 s; suite 1024 passed in ~90 s.
- examples/synthetic_workflow.py: preprocess now `workers=2` (organize/convert already had it); the script ran green end to end.

## task-04

- organize() builds one `pool.Task(_worker, (read_fn, folder, exclude_models))` per folder (no key). `_worker(fn, folder, *args)` is shared by task A and B and returns `(fn result, failure row or None)`: OSError -> REASON_IO_ERROR row + warning, other Exception -> crash_site (imported from convert) + type log line + REASON_INTERNAL row. Parent `handle` tells stages apart by `i in held`.
- Seam shapes: `_read_fn(folder, exclude_models)` -> `(name, refusal, cases, misc, unreadable, walked, reading_s)` with `refusal` None or `(PatientRow, may_be_kept)`; cases/misc already include blank-id and unreadable files (grouping runs in task A, before the parent knows whether the patient is skipped). `_copy_fn(folder, out_root, name, cases, misc, overwrite)` -> `(copy_s, status)`; `folder` is only for the failure row. tests/pool_workers.py `organize_read` changed to that 7-tuple (`None` refusal added); `organize_copy` already matched.
- Parent: refusal -> `kept` computed as before (only NO_SOP_UID / SPLIT_SERIES set may_be_kept), no claim; else claim check, resolve check (ValueError(REASON_OUTSIDE_ROOT) propagates from handle through pool.run, report still written by finally), recover (OSError -> IO row, claim stays taken), existence check, then follow-up Task B. Rows stored by index in `done`, written sorted. Reading/files of a patient whose copy fails add nothing (held until B succeeds).
- Failed('SystemExit') re-raised as SystemExit, as in convert/preprocess.
- Abort at N>1 is timing dependent for patients after the failing one: a freed slot starts the next folder before the failing result is delivered, so those rows exist. Deterministic "no further patient" check is at N=1; N=2 test asserts OrganizeAborted, exactly one REASON_INTERNAL row, no SECRET on stderr, returns within the 120 s guard.
- Tests moved to workers=1 (patch or caplog): test_ctrl_c_still_writes_the_report... (parametrization dropped), test_no_timing_line_on_ctrl_c (dropped), the preamble OSError test (renamed `..._refuses_that_patient_only`, workers=4 dropped), the timing-line test (workers=2 -> 1, expects ('7','1')). Unpatched parametrized tests and the 1-vs-4 equality keep N=4. TIMING regex updated.
- New tests (end of tests/test_organize.py): N=1 vs 2 under overwrite (same-PatientID pair, replaced, refused no-SOP-UID folder with old output kept, counts), abort via _read_fn at N=2 and N=1, abort via _copy_fn, OSError in copy/recover, ValueError outside root, resume (N=1 and 2, thread watches for any .partial). Suite 1030 passed, 4 skipped (~100 s).
- OrganizeAborted is not wired into the CLI (task-05).

## review-group-3
- No BLOCKING. W1 fixed (9097754): organize ignores a read result delivered after the abort (no claim/recover/copy/row); N=2 abort test tightened (fails without the fix).
- Deferred W2: an exception raised from handle at N>1 (ValueError REASON_OUTSIDE_ROOT, SystemExit) lets running copies publish but their rows are lost from the report.
- Deferred N1: `raise SystemExit` in handle drops the original exit code (old code kept it).
- N2/N3 fine as is (grouping before skip; recover for a read delivered after Ctrl-C).

## task-05

- `preprocess` takes `--workers` (positive_int, cap 64) and passes `workers=` to `preprocess()`. WORKERS_HELP: "number of patients processed in parallel, each in its own process (about 7 GB RAM each at 701x701x433); output is the same for any value".
- `OrganizeAborted` -> exit 1 with `rt_dataprep: {REASON_INTERNAL}` of organize. In `__main__` the name `organize` is the function (not the module), so the organize reason is imported as `ORGANIZE_REASON_INTERNAL` (as preprocess's already is); convert's keeps the bare `REASON_INTERNAL`.
- tests/test_cli.py: seam tests (`calls` fixture also records preprocess) parametrized over organize/convert/preprocess for default, `--workers 3`, cap 64 and invalid values; manifest/split still reject `--workers`; help test checks each stage's help; fixed-text abort test for organize and convert; subprocess `convert --workers 2` with one junk file in misc_files shows the worker's 'convert: 1 unreadable files in misc_files' INFO line and 'workers 2' on stderr, no patient id or path.
- README: preprocess row gets `--workers`; new "Parallel patients" section (processes, ~7 GB each, default 1, SimpleITK/BLAS threads uncapped, `kill -INT` to the parent alone waits for running patients, terminal Ctrl-C reaches workers too); header-reading-thread sentence removed.
- Full suite 1046 passed, 4 skipped; ruff and compileall clean.

## review-loop-pass-1
- Opus bug review and spec-axis review: no BLOCKING. README `kill -INT` wording fixed (41b1e60). Deferred: #177, #178, #179, #180.
- Codex: P2 out-of-order internal error kept dispatching → fixed with pool.Stop (b0c37e7); P2 SystemExit(n) exited 0 → fixed, Failed carries the sanitized code (b2b22f2); P2 handler raise at N>1 loses running rows → deferred: #178.

## review-loop-pass-1-verify
- b0c37e7 VERIFIED (Opus, fixes only): stop at completion, unwrapped delivery to every handle, organize task-B Stop still records its row; mutation fails at N=1 and N=2.

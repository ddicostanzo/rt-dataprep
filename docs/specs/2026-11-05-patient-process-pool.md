# Per-patient process pool (#171)

## Problem

On radonc-phgpu (20 patients, 41 plans) organize took 110 s, convert 859 s and preprocess 788 s, all serial over
patients. `--workers` only threads header reading inside one folder (`dicom_io.read_pooled`). Header parsing is
GIL-bound, so 8 threads made reading about 2x slower, and the per-patient work (about 28 s per plan in convert) stays
serial. Patients are independent, so the fix is one patient per worker process.

## Decisions (owner, 2026-11-05)

- `--workers N` now means N patient processes in organize, convert **and** preprocess (preprocess gains the flag).
  Same validation as today (`positive_int`, 1 to `MAX_WORKERS`). Help text says processes, about 7 GB RAM each at
  701x701x433, and that output is the same for any N.
- Threaded header reading is retired: `read_pooled` and its tests are deleted, header reads are serial within a
  patient. The serial read keeps the whole read under `dicom_io._pydicom_quiet()` (read_pooled held it; the callers
  rely on it, `read_header_core` expects it), so no pydicom warning or log line, which can quote tag values, reaches
  stderr.
- Default 1. One driver for every N: the same parent-side scheduler (claims, hold, kept, bounded submission, abort,
  row order) runs for N=1 and N>1. N=1 calls the worker function directly in the calling process (no executor), so the
  existing suite and its monkeypatched seams exercise every parent-side rule. N > 1 runs the same worker function on
  `concurrent.futures.ProcessPoolExecutor(N)`. Never an `if workers == 1: old loop` branch.
- RAM: documented only (help text and README). No probe, no clamp. Auto-sizing from a smoke benchmark is a separate
  issue.
- Timing log: "X s total" stays wall clock; reading/converting/copying seconds are labelled as summed over patients.
- /dev auto-merges when the review round is clean.
- Per-process thread counts (SimpleITK, BLAS) stay at their defaults: capping can change float bits. Documented; a cap
  is a follow-up if real runs show oversubscription.
- Organize handles an unexpected exception the way convert does: `REASON_INTERNAL` row, crash-site log line,
  `OrganizeAborted`, CLI exit 1 with a fixed text.

## Behaviour

### Common to all three stages

- The parent owns report rows, summary counts, replaced/kept counts and claims. Rows are written in sorted
  patient-folder order whatever the completion order, so the report is byte-identical for any N (excluding timings).
- Workers are fresh interpreters: the executor pins the `spawn` start method on every Python version (the venv is
  3.13, whose Linux default is still `fork`, which would leak parent state and monkeypatches into workers; forkserver
  workers keep the stderr fd of whichever test started the server, so capfd misses them later); the pool `initializer` configures logging the way `__main__`
  does (`%(message)s`, at the parent's root level) so worker INFO/WARNING lines reach stderr
  (vault: `spawned-worker-without-logging-config-only-warning-survives`).
- PHI boundary in the parent: the parent never formats, logs or chains an exception taken from a future
  (`future.result()` re-raises with a `_RemoteTraceback` holding the original message; `BrokenProcessPool` has its own
  text). Anything that escapes the worker wrapper becomes that patient's fixed `REASON_INTERNAL` row and a fixed log
  line naming only the exception type, and aborts the run.
- Entry points (`__main__.py`, `examples/synthetic_workflow.py`) keep their `if __name__ == '__main__'` guard: workers
  re-import the main module.
- Failures are turned into data **inside the worker**, because a traceback does not cross the process boundary:
  - OSError → the patient's fixed `REASON_IO_ERROR` row and warning, as today.
  - Other Exception (convert, preprocess) → the `REASON_INTERNAL` row; the worker computes `crash_site` and logs
    the type/module/line line itself; the parent sees an "internal" marker. Nothing of the exception message
    is returned or logged (PHI).
- Bounded submission: at most N patient tasks are submitted at a time; the next is submitted when one completes.
  (ProcessPoolExecutor moves N+1 futures into its call queue, where `cancel()` no longer works.) So nothing queued
  inside the executor needs cancelling.
- Abort on the first internal error (all three stages): nothing further is submitted; patients already running
  finish and their rows are kept; then the report is written and `*Aborted` is raised as
  today. Never `shutdown(cancel_futures=True)` inside a wait loop; iterate with `wait(..., FIRST_COMPLETED)` and drop
  cancelled futures (vault: `processpool-cancel-futures-hangs-as-completed`). Patients after the failing one in
  sorted order that had already finished keep their rows (this differs from N=1 only in which later patients ran).
- Ctrl-C: nothing further is submitted, running workers (which receive SIGINT from the terminal too) clean up their
  `.partial` as today, the report is written by the existing `finally`, KeyboardInterrupt propagates.

### convert and preprocess

- The claim key is the input folder name (casefold), known before dispatch. Under `--overwrite` the parent refuses a
  repeat with `REASON_NAME_CLAIMED` without dispatching it; the first in sorted order wins, as today.
- Without `--overwrite` the parent still tracks casefolded names: a folder whose casefolded name matches one still
  running is held until that one completes, then dispatched and run as today (it sees the first one's output and is
  skipped). Two folders of one casefolded name never build into the same `.partial` at once.
- The worker runs today's `_convert_patient` / `preprocess_patient` for one folder and returns its rows, status and
  (convert) reading seconds and files.

### organize

- Task A (pool): walk, read and group one source folder; return the sanitized name, the refusal row if any, the
  per-plan path lists and misc paths (paths only, no datasets), the unreadable count and reading seconds.
- Task A returns a refusal raw. The parent, in sorted source order, computes the `kept` status of a refused patient
  (no-SOP-UID and split-series refusals read `claimed` and `lexists` at that point, as `organize.py` does today) and
  does not claim its name.
- Parent, in sorted source order: claim check (`REASON_NAME_CLAIMED`), `recover`, existence check
  (`OUTCOME_SKIPPED`/`REASON_EXISTS`), then submits task B.
- Task B (pool): today's copy into `.<name>.partial` and `publish`; returns copy seconds and status.
- Headers are read once; nothing is copied for a patient that will be skipped or refused.
- A and B tasks share the N-slot bound; claims are decided in sorted order, so one slow read holds back later claims
  (correct, possibly slower).

### Testing seam

- The pool driver takes the per-patient worker callable as an internal argument (default: the real one). N>1 tests
  pass a module-level test function (picklable, importable in the worker) that raises for one chosen patient.
  Worker log output is checked with `capfd`, not `caplog`.

## Acceptance criteria

1. (N=1 reruns of the synthetic workflow are byte-identical: checked 2026-11-05, 36 NIfTI files.) For a synthetic
   tree of at least 3 patients, organize, convert and preprocess with `workers=2` produce the same
   output files (names and bytes of written data files) and the same report rows (in order) as `workers=1`.
2. Under `--overwrite`, two source folders whose names differ only in case: N=2 gives the same claim outcome as N=1
   (first sorted wins, second `REASON_NAME_CLAIMED`) in convert and preprocess; for organize, two folders with the
   same PatientID give the same claim outcome for N=1 and N=2.
3. An injected internal error in one patient with N=2 (organize, convert, preprocess), through the worker-callable
   seam: the `*Aborted` error is raised, the report holds the `REASON_INTERNAL` row, at most N patients started
   (patients after the running set have no row), and the call returns (no hang; test with a timeout). stderr (capfd)
   holds no exception message text.
3a. Without `--overwrite`, two convert/preprocess folders that differ only in case, N=2: never run at the same time
   (the second is dispatched after the first completes) and their rows equal N=1's.
3b. A header read that makes pydicom warn prints nothing to stderr (serial read keeps the quiet block).
4. `preprocess` CLI accepts `--workers` and passes `workers=` (asserted through the existing CLI seam);
   organize/convert pass it through; manifest/split still reject it.
   Help text says processes and the RAM figure.
5. `read_pooled` and its tests are gone; header reads are serial; the full suite and ruff pass.
6. README documents `--workers` as patient processes with about 7 GB per worker at 701x701x433.
7. Worker INFO log lines appear on stderr in a real N=2 CLI run (subprocess test).

## Deferred

- Auto-sizing `--workers` from a smoke benchmark of RAM and time per patient (separate issue, owner request).
- Real-data timing of N > 1 on radonc-phgpu (owner runs it after merge).
- A worker killed by the OOM killer (`BrokenProcessPool`): today it would surface as an unexpected error; a fixed
  reason and per-patient row are hardening.
- Second Ctrl-C during pool shutdown.
- `kill -INT` to the parent only (scheduler, nohup): workers get no SIGINT, so the parent waits for the running
  patients to finish. Documented, not fixed.
- Capping per-process SimpleITK/BLAS threads.

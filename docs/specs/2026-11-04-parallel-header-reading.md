# Parallel header reading, with run timings (#2)

## Problem

`organize` and `convert` read every DICOM header one file at a time. The real export lives on a network share (NFS), so
each read waits on the network: organize opens every file twice (`_has_preamble`, then `read_header`), and convert reads
every file of the organized tree again (`read_folder_headers`). Legacy read files with `Pool(6)`. #2 deferred a pool until
real export sizes show reading is the bottleneck; nothing has measured that yet.

## Decisions (owner, 2026-11-04 interview)

- **Measure and offer in one PR.** Both commands log per-run phase timings, and both get an opt-in `--workers N`. The
  owner runs the real export with `--workers 1` and `--workers 6` and compares. If the pool gives no gain, a later PR
  deletes it and closes #2 with the numbers.
- **Scope:** the two loops #2 names: organize's per-patient file read and convert's `read_folder_headers`. Patients still
  run one after another; copying, conversion and preprocess are unchanged.
- **Control:** `--workers N` on `organize` and `convert`, default 1. One code path for every N: `--workers 1` runs the
  same pooled helper with one thread, so the path the real export runs is the path every test runs. Its output is
  today's (R1).
- **Executor:** threads (`concurrent.futures.ThreadPoolExecutor`), not processes. `_pydicom_quiet`'s state (pydicom
  `reading_validation_mode`, the `pydicom` logger's `disabled`, the warning filters) is process-global, so it is entered
  **once** around the whole pooled read. `read_header` splits into an unquieted core and the existing quiet wrapper.
  The core is today's `read_header` minus the quiet block: dcmread, the access that converts every value, and the same
  conversion of any `Exception` to `OSError(REASON_UNREADABLE)`, raised after the except so nothing path-bearing is
  chained. Workers look the core up as a module attribute of `dicom_io` at call time, so a test can patch that one seam.
  The organize Ctrl-C test (`test_organize.py`, which patches `org.read_header`) is repointed to it. Convert's patch point
  `conv.read_folder_headers` stays, but its stub takes `(folder, *args, **kwargs)` once `workers` is passed down.
  A patient that is refused by an `OSError`, or that aborts the convert run, contributes 0 files and 0 reading seconds;
  its time counts toward "total" only. Workers run the core inside the outer block and never enter the
  quiet block or call `quiet_call` (it swaps fds 1/2 and is not thread-safe, per #20). The executor lives inside the
  quiet block, so no worker outlives it. This assumes a GIL build of CPython (not 3.13t), and nothing else in the process
  runs while the block is held: the pool is the only work in flight.
- **Preamble in the pool (organize):** one pooled step per path, which checks the preamble and then reads the header. A
  non-DICOM file yields nothing. The step catches only the header-read `OSError` and returns that file as "bad". An
  `OSError` from the preamble check propagates, as it does today when it escapes the sort. On NFS the preamble open is a
  full round-trip, so leaving it serial would cap the gain.
- **Timings:** one INFO line per run for each command, with seconds and counts only, e.g.
  `organize: 412.3 s total, 300.1 s reading, 98.7 s copying, 54321 files, workers 6`, and
  `convert: … s total, … s reading, … s converting, … files, workers N`. No patient id, path, UID or folder name.
  - **organize:** "reading" is everything before grouping (the `os.walk` listing, the preamble checks, the header reads).
    "copying" is the copy phase. "files" counts every file the walk listed.
  - **convert:** "reading" is the `read_folder_headers` calls. "converting" is total minus reading, with salvage reads
    included. "files" counts the headers read (readable files), so `read_folder_headers` keeps its return shape.
  - Each patient function returns its seconds and counts next to its row (no module globals). "total" is the
    `time.monotonic()` span of the patient loop.

## Requirements

1. **Identical output for any N.** For the same input, `--workers 1` and `--workers N` give byte-identical output trees
   and identical report CSV rows (the dated report copies differ only in their timestamp name). Headers are returned in
   the same order as today (sorted by path), whatever order the threads finish in.
2. **Errors classify as today.**
   - organize: a file with a preamble whose `read_header` fails is still a "bad" file (counted unreadable, copied to
     `misc_files`). An `OSError` from the preamble check (e.g. permission) still fails the whole patient as
     `REASON_IO_ERROR`. `os.walk` stays serial and its listing errors are unchanged (#68).
   - convert: an unreadable file still sets the folder's `unreadable` flag, and nothing else changes.
3. **Quiet holds under threads.** With N > 1, a header whose value conversion would warn or log (the PHI case in vault
   [[pydicom-validates-lazily-on-first-tag-access]]) emits no warning, no `pydicom` log record and nothing on stdout or
   stderr. Every tag value is still converted inside the quiet block. Afterwards, pydicom's validation mode, the
   `pydicom` logger and the warning filters are back as they were, also when a read raises and on Ctrl-C.
4. **Ctrl-C and early exit.** Any exception that leaves the pooled read (a `KeyboardInterrupt`, or the preamble
   `OSError` that refuses the patient) shuts the executor down with `cancel_futures=True`. The shape is
   `try: … finally: executor.shutdown(cancel_futures=True)`, never a `raise` inside an except block (a guard test forbids
   that). Queued reads don't run, only
   the reads already running are waited for, and the quiet state is restored after them. The existing
   report-in-`finally` and partial-folder cleanup behaviour is unchanged.
5. **CLI:** `--workers` accepts an integer of at least 1. Anything else is an argparse usage error (exit 2) whose text
   names no path from the run (argparse echoes only what was typed). `organize()` and `convert()` take `workers=1` as a
   keyword argument.
6. **Timings:** the INFO line is logged once per run, right after the existing summary line. Like the summary, it isn't
   logged on Ctrl-C. On a convert abort (`ConvertAborted`), it is logged before the raise, as the summary is.

## Out of scope

- Per-patient parallelism, process pools, and pooling preprocess or the pixel reads.
- Picking a default N above 1, which waits for the owner's measurement.

## Acceptance

- An equivalence test on a synthetic multi-patient export: organize then convert with `workers=1` and `workers=4` give
  identical `tree_hashes`, with the dated report copies (`<name>_<UTC time>.csv`) filtered out in the test, and identical
  report rows.
- An order test through the real pooled helper: a stubbed core whose delays make later paths finish first still gives
  headers sorted by path.
- A test that an unreadable file and a non-DICOM file are classified the same with `workers=4` as with 1, for both commands.
- A test that, with `workers=4` over enough files that reads overlap, a header with an invalid value leaves no warning,
  no `pydicom` record and no captured output, and that the pydicom/warning state afterwards equals the state before. A
  control in the same test shows the same header warns under default validation outside the quiet block.
- A Ctrl-C test: a stubbed read raises `KeyboardInterrupt` with `workers=4`. Queued reads after it don't run, the quiet
  state is restored, the report is written, and no `.partial` folder is left.
- A test that an `OSError` from the preamble check with `workers=4` still refuses that patient as `REASON_IO_ERROR`.
- A CLI test: `--workers 0` and `--workers x` exit 2, and `--workers 3` reaches `organize`/`convert` as `workers=3`.
- A test that the timing line is logged once per run and contains only numbers and fixed words (no `SYNTHETIC`, no `/`).

## Deferred

- Default N above 1, depending on the owner's real-export comparison.
- A process pool, if the timings show CPU-bound parsing dominates.
- Per-patient parallelism (covers copy and conversion too).
- A second Ctrl-C while in-flight reads are being joined could exit the quiet block with workers still reading, so a
  warning quoting a tag value could reach stderr. The window is at most N in-flight header reads. File it as a
  `hardening` issue; one fix is to make the join uninterruptible.

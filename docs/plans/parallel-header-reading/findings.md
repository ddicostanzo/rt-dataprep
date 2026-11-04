# Findings: parallel header reading (#2)

Spec: docs/specs/2026-11-04-parallel-header-reading.md. Plan: docs/plans/parallel-header-reading/plan.json.

Summary: organize and convert get an opt-in `--workers N` (default 1, same path for every N). Header reads run in a
thread pool through one dicom_io helper that holds _pydicom_quiet once around the executor; workers call
`dicom_io.read_header_core` (read_header minus its quiet block, same Exception -> OSError(REASON_UNREADABLE)).
organize pools preamble check + header read per path; convert pools read_folder_headers. Each command logs one INFO
timing line per run (seconds and counts only). Output is identical for any N.

Order: task-01 (dicom_io) -> task-02 (organize), task-03 (convert) -> task-04 (CLI + end-to-end equivalence).
Append under `## <task-id>`; append only.

## task-01

- Helper is `read_pooled(step, paths, workers)` in dicom_io (the plan left the name open; tasks 02 and 03 import it).
  `read_header` is now the quiet block around `read_header_core`; its tests are unchanged.
- Shape: quiet block, then `executor = ThreadPoolExecutor(workers)`, then try/finally with
  `executor.shutdown(cancel_futures=True)`. Results come from `future.result()` over futures in submit order, so input
  order holds. A worker's `KeyboardInterrupt` is stored on its future and re-raised in the caller by `result()`; no raise
  sits inside an except block.
- Mutation check: swapping the quiet block for a nullcontext makes the quiet test fail on the recorded warning, so the
  test does exercise the block (the control asserts the same file warns unquieted).
- Gotcha: `warnings.filters` inside a `catch_warnings` differs from outside it; the test compares the filters inside the
  recording block, and the mode and logger flag after it.
- `write_rtdose(..., invalid_values=True)` gives the invalid-value file (a PN over 64 chars warns on value access).

## review-group-1

No blocking findings; all task-01 criteria met. WARNING (deferred, already in spec Deferred): a second Ctrl-C during
the shutdown join restores the quiet state while workers still read. NOTEs for the next tasks:
- read_pooled raises the first step failure in input order, so the organize and convert steps must catch the core's
  OSError per path themselves (one bad file must never end the folder read).
- future.result() re-raises exactly what the step raised: any step exception that can escape must stay path-free in
  how callers report it (organize/convert report fixed REASON_ text only, never str(exc)).
- Criterion "workers resolve read_header_core as a module attribute" is met by how steps are built: tasks 02/03 steps
  must call dicom_io.read_header_core through the module.
- read_pooled does not validate workers; the CLI (task-04) enforces >= 1.

## task-02

- organize pools one private step per path (named read_one): preamble check, then `dicom_io.read_header_core` through the
  module; it returns None (non-DICOM), (path, None) (core OSError, i.e. a bad file) or (path, header). The walk is listed
  and sorted before `read_pooled`, so headers and bad paths keep sorted order for any workers.
- The preamble OSError is not caught in the step: `future.result()` re-raises it in input order, the patient function
  raises, and organize refuses the patient as REASON_IO_ERROR as before (test runs it with workers 4).
- `_organize_patient` returns (row, reading s, copying s, files walked) through a local `done` closure on every return
  after the read; a raise (walk, preamble, copy OSError) adds nothing. "copying" starts at the partial-folder rmtree.
- Timing line is logged right after the summary, so it is skipped on Ctrl-C like the summary. Test of the line refuses a
  patient by preamble OSError and checks the files count excludes it (6 of 12 files, workers 2).
- Ctrl-C with workers 4: a worker's KeyboardInterrupt is stored on its future and re-raised by `result()`; no .partial
  folder is left because the interrupt happens in the read, before any copy.

## task-03

- convert pools one folder at a time: `read_folder_headers(folder, workers=1)` lists and sorts the files, then
  `read_pooled` runs a private step named read_one (convert's own, same shape as organize's but with no preamble check):
  `dicom_io.read_header_core` through the module, core OSError caught per path, returns (path, header) or (path, None).
  The unreadable flag is `len(headers) < len(results)`. Plan folders and misc_files are read one after the other.
- `_convert_patient` returns (rows, reading s, header files) on all three returns (exists-skip: 0.0 and 0; no-plans skip and
  the end: the read time and count). A raise (OSError refusal, abort) adds nothing, so the numbers go into `convert`'s totals
  only after the call returns. No closure: the file count is computed as an int right after the reads, because
  misc_headers is rebound by the blank-id filter further down.
- "reading" is the wall time of the plan and misc folder reads only; the salvage reads (`_salvage_study`) stay serial and
  count under converting. Timing line follows the summary line, so it is logged before the ConvertAborted raise and skipped
  on Ctrl-C (tested both ways).
- Fixture gotcha: pydicom tolerates a DICOM cut off after the preamble (no error), so "unreadable" test files are an empty
  .dcm and a text file; both are the same OSError path in convert, there is no separate non-DICOM case.
- Count test: patient with 6 + 4 headers (plan_1 and misc_files, 2 bad files each), a no-plans patient with 2 headers, and
  a patient whose misc_files read raises OSError after its plan folder was read (adds 0): 12 headers of 16+ files.

## review-group-2

No blocking findings; all task-02/03 criteria met; workers=1 behaviour unchanged; full suite 564 passed, 1 skipped.
Correction to the task-02 section: after 36a3e45 the organize timing test expects 7 walked files (6 DICOM + 1 text).
Deferred (file as hardening at the review round):
- W1: no convert test patches dicom_io.read_header_core, so a from-import there would pass every test.
- W2: the organize timing test does not pin that the line follows the summary (convert's does).
- N3: no organize "no timing line on Ctrl-C" test (code is right: the interrupt leaves before the log line).
- N5: the organize Ctrl-C .partial assertion passes trivially (interrupt during reading, before any partial exists).

## task-04

- CLI: one argparse type, positive_int (int >= 1; a non-integer or below 1 raises ArgumentTypeError, so argparse exits 2), and one WORKERS_HELP
  string shared by the organize and convert subparsers. Default 1 is passed as workers=N in both dispatch branches (organize is the
  `else` branch of the command chain in main). preprocess, manifest and split are untouched; a test pins that `--workers` on them is a usage error.
- CLI tests monkeypatch `cli.organize` and `cli.convert` (the names __main__ imports) and compare the recorded call; 0, -1, x and 1.5 all exit 2 with no call.
- End-to-end test (test_organize_then_convert_is_the_same_for_one_worker_and_four): two synthetic patients, each with a dose and a box ROI, organized and
  converted with workers 1 and 4 into separate roots. The hashed trees (minus the dated report copies, matched by the same regex as `listing`) and
  the report rows are equal. Two fresh conversions are byte-identical (the NIfTI files, series.csv and the reports), so no comparison was loosened.
- The test can fail: it asserts 2 organize rows, 2 converted rows, 6 NIfTI files in the converted tree and .dcm files in the organized tree before comparing.
- The equivalence test passed before the CLI change, as expected: it calls the library kwarg, which tasks 02 and 03 added. Only the CLI tests were red first.

## review-loop-pass-1

One round: Opus bug review of the full diff (integration seams included), an Opus spec-axis review against plan.json
(group 3 given full attention, standing in for its wave review), and Codex (completed on 1d88762, no findings).
BLOCKING: none, so no fixes and no verification pass. Deferred and filed: #97 (second Ctrl-C during the shutdown join),
#98 (test gaps: workers forwarding, convert core seam, organize timing placement and Ctrl-C, trivial .partial assertion,
spec 'total' wording), #99 (--workers upper bound, positive_int looseness).

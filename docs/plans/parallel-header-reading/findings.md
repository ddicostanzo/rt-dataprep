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

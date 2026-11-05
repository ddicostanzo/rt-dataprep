# --overwrite for organize, convert and preprocess

- Date: 2026-11-05
- Branch: `feat/overwrite-flag`
- Status: REVIEWED

## Background
After a code fix, the owner reruns a stage on the same input into the same output root. Today every stage skips a patient
whose output folder already exists (`REASON_EXISTS`, outcome `skipped`), so refreshing output means hand-deleting folders or
using a new root. The owner asked for a flag that replaces existing output instead (2026-11-05). The main expected use is
"rerun after a code fix", not trying other preprocess flags in place (a separate root suits that better).

## Problem
An owner rerunning `organize`, `convert` or `preprocess` after an rt-dataprep fix cannot refresh existing patient folders
without deleting them by hand, which loses the old output before the new one is known to be good.

## Constraints
- Safety: no dose, plan or patient-matching computation changes. The feature replaces derived output folders, so it is a
  data-loss path and gets full rigor. Source trees are only read (profile `data-safety` rule); nothing outside the output
  root is touched. Only direct children of the output root are renamed or deleted; `shutil.rmtree` does not follow links.
- PHI: no new PHI surface. Log lines carry counts only; no path, patient name or ID in logs or exception text (each stage's
  per-patient `OSError` handler already logs a fixed reason only). Reports keep their current schema.
- FHIR: not applicable.
- Concurrency: one writer per output root, as the README already requires.
- Commit gate: at most 5 staged files per commit; private-API regex applies to prose too.

## Premises (confirmed by the owner)
- Only patients this run processes are touched. Output folders with no matching input patient are never deleted.
- A name written or claimed earlier in the same run is never replaced, renamed or restored by a later input folder of that
  run. Organize already refuses such a folder (`REASON_NAME_CLAIMED`). Under `--overwrite` only, convert and preprocess
  gain the same casefolded guard (owner decision, review issue 4) with their own `REASON_NAME_CLAIMED` constant, because
  on a case-insensitive output root `abc` and `ABC` share one folder. Without the flag the existing exists-check already
  prevents the clobber, so default behavior stays as today.
- Downstream outputs are not cascaded: overwriting converted output does not touch the preprocessed tree, `manifest.csv`
  or `splits.json`. The README says to rerun them; `split`'s refusal of changed content stays.
- `organize()`, `convert()` and `preprocess()` gain a keyword-only `overwrite=False`; the CLI flag passes it through.
  Default behavior is unchanged in effect: the recovery step below runs without the flag too, but acts only on hidden
  folders that only `--overwrite` creates.

## Approaches
- Chosen: one shared helper in `dicom_io.py` (already imported by all three stages) does the recovery and the swap.
  The data-loss logic lives in one tested place.
- Rejected: inline swap code in each stage (three copies of deletion logic can drift).
- Rejected: `--fresh` that wipes the whole output root (loses everything up front, no crash safety, deletes patients
  missing from the input).
- Rejected after review: "mirror the new run" (replace or delete even when the new run produced nothing). A transient read
  error turns plans into refusals, not a patient error (`convert.py` header read), so mirroring would swap good output for
  an empty folder. Owner chose to keep the old output whenever the new run has no successful case.

## Behavior
Hidden names per patient folder `<name>` under the output root: `.<name>.partial` (existing) and `.<name>.old` (new: the
previous output while a swap is in progress). All three stages and `manifest` already ignore `.`-prefixed folders.

1. Recovery, with or without `--overwrite`, at the point where each stage checks whether `<name>` exists today (in organize
   that is after `claimed.add`, so a folder refused earlier, including by `REASON_NO_SOP_UID`, `REASON_SPLIT_SERIES` or
   `REASON_NAME_CLAIMED`, never recovers, renames or deletes anything):
   - `.<name>.old` exists and `<name>` does not: rename `.<name>.old` back to `<name>` (an interrupted swap).
   - both exist: delete `.<name>.old` (the swap had finished).
   - a leftover `.<name>.partial` is deleted (a kill between the two swap renames leaves one that a restore-then-skip
     would otherwise never remove).
2. Without `--overwrite`: unchanged (skip an existing `<name>` with `REASON_EXISTS`).
3. With `--overwrite`, the patient's build runs as if `<name>` did not exist. Then:
   - The build produced at least one successful case (organize: `organized` with at least one plan folder, amended after
     review: an unreadable RTPLAN sends files to `misc_files` and still yields an `organized` row with no plan; convert: a `converted` plan or study series
     row; preprocess: a `preprocessed` row): if `<name>` exists, rename `<name>` to `.<name>.old`, rename the partial to
     `<name>`, delete `.<name>.old`. Otherwise rename as today.
   - The build produced no successful case (every case refused or skipped, or the build stopped before a partial): if
     `<name>` exists, it is kept unchanged and the partial (if any) is removed; the row keeps its refusal or skip reason;
     the patient is counted as "kept". If `<name>` does not exist, behavior is as today.
4. Failure during a swap: if renaming the partial into place fails after `<name>` was moved aside, rename `.<name>.old`
   back to `<name>` before the error propagates. If that rollback rename also fails, `.<name>.old` remains and step 1
   restores it on the next run. Either way the error follows the existing per-patient `OSError` path (`REASON_IO_ERROR`)
   and the existing cleanup removes the partial. A crash or Ctrl-C at any point leaves either `<name>` intact, or
   `.<name>.old` that step 1 restores on the next run.
5. Deleting `.<name>.old` after a completed swap uses `ignore_errors=True`, as partial cleanup does today: the row keeps its
   real outcome because the new output is complete, and a leftover is removed by step 1 on the next run.
5a. Root check (added after Codex review): with `--overwrite`, each stage refuses the whole run before writing anything
   when its input root is the output root or inside it (`REASON_INPUT_INSIDE_OUTPUT`), because a replaced patient folder
   could contain the input tree. Default runs are unchanged.
6. Reporting: rows keep their normal outcome. Each stage's summary log line adds two counts: patient folders replaced, and
   existing folders kept because the new run had no successful case. No report column changes.
7. CLI: `organize`, `convert` and `preprocess` gain `--overwrite` (help: replace a patient's existing output folder once its
   new output holds at least one successful case). `split` and `manifest` are unchanged.

## Acceptance criteria
1. Without `--overwrite`, each stage still skips an existing patient folder with `REASON_EXISTS` and leaves it
   byte-identical.
2. With `--overwrite`, rerunning each stage over a root holding a stale patient folder (a marker file added, an expected
   file removed) leaves that folder equal to a fresh run's output: the marker is gone, the expected files are present, and
   no hidden `.<name>.*` folder remains.
3. With `--overwrite`, if the new build raises (patched failure inside the build), the old patient folder is unchanged
   and no `.<name>.old` or `.<name>.partial` remains.
4. With `--overwrite`, if renaming the partial into place fails after the old folder was moved aside, the old folder is
   back at `<name>` with its contents unchanged, and the patient's row is `REASON_IO_ERROR`.
5. Recovery: with `.<name>.old` present and `<name>` missing, a run without `--overwrite` restores `.<name>.old` to
   `<name>` and then skips it with `REASON_EXISTS`. With both present, `.<name>.old` is deleted and `<name>` kept.
6. Keep-if-empty: with `--overwrite`, a patient whose new run has no successful case (convert: every plan refused via a
   patched unreadable file; convert: no plans; preprocess: no cases or every case refused; organize: refused after the name
   is known) keeps its existing folder byte-identical, leaves no hidden leftovers, keeps its refusal/skip row, and is
   counted as kept in the summary log.
7. Same-run guard: in organize, a second source folder mapping to a name written earlier in the run is refused with
   `REASON_NAME_CLAIMED` and the first folder's new output is untouched; a later same-name folder refused before claiming
   neither restores, renames nor deletes anything. In convert and preprocess with `--overwrite`, two input folders whose
   names differ only in case: the second is refused with `REASON_NAME_CLAIMED` and the first's output is untouched.
8. Output folders in the root with no matching input patient are untouched by an `--overwrite` run.
9. The source tree is unchanged after an `--overwrite` run of `organize` (file list and contents).
10. Each stage's summary log line reports the replaced and kept counts; no log line or exception text contains a path or
    the synthetic patient folder name (asserted on captured logs in an `--overwrite` run, including the failure of
    criterion 4).
11. The CLI accepts `--overwrite` on `organize`, `convert` and `preprocess` and passes it through; `split` rejects it.
    Existing test seams that replace `preprocess.preprocess` or the stage functions (`tests/test_cli.py` `PRE_RUN` lambda
    and the `lambda *args` stubs) accept the new keyword, and the suite stays green.
12. README documents `--overwrite`: when old output is replaced or kept, recovery after an interrupted run (including that
    an organize patient now refused before the claimed check never runs recovery, so its `.<name>.old` stays until a run
    that knows the name), the same-run guard, and that downstream trees, manifests and splits must be regenerated. The
    README's "Review outputs before deleting or replacing any prior result" line is updated to fit the flag.

## Failure handling
- Build failure (any exception, Ctrl-C): old output untouched; partial removed by the existing cleanup.
- Transient read errors that refuse every case: old output kept (Behavior 3), row shows the refusal, summary counts it kept.
- Rename failure mid-swap: rolled back to the old output (or left as `.<name>.old` for the next run), reported as
  `REASON_IO_ERROR`.
- Failure deleting `.<name>.old` after a completed swap: ignored; the new output is in place; next run removes it.

## Delivery and validation
Local suite and ruff (no CI). Owner confirms push and merge. Real-data check: rerun one patient with `--overwrite` on a
copy and compare with a fresh-root run.

## Deferred
- Windows: `rmtree` fails on read-only files (organize copies keep source read-only bits), so a `.<name>.old` can persist
  and later swaps fail as `REASON_IO_ERROR`. File as a `hardening` issue; fix with an `onexc` handler if it bites.
- A `--overwrite` for `manifest` or `split` (manifest already rewrites; split refusal is deliberate).
- Selecting a subset of patients to overwrite (use an input root holding only those patients).
- Detecting stale downstream trees automatically.

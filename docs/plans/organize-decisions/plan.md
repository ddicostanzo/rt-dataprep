# organize: owner decisions #10, #9, #13 (2026-11-04)

Worktree `/home/dominic/Repositories/rt-dataprep-organize`, branch `feat/organize-decisions`. Only touch `src/rt_dataprep/organize.py`, `tests/test_organize.py`, and `tests/synthetic_dicom.py` if a writer needs an option. Run tests with `uv run --group dev python -m pytest -q --tb=short` and lint with `uv run --group dev ruff check .`. The decisions are recorded on issues #10, #9 and #13.

## Guardrails
- **PHI:** log lines and exceptions carry fixed REASON_ constants and counts only. Test ids use the `SYNTHETIC` prefix.
- **Commit gate:**
  - At most 5 staged files.
  - No added line may hold a dotted name whose last part starts with `_`.
  - Don't edit `tests/test_organize.py:24`, which trips the MRN rule.
- **Red/green:** each new test must fail with the change reverted. Check this by restoring the file from a copy, not with `git checkout`/`stash`.
- **Stop counter:** after 3 failed attempts at one test, stop and report.

## T1 — #10: read files by DICM preamble
- [ ] organize reads every regular file under the patient folder whose bytes 128–131 are `DICM`, whatever its name or extension: `.dcm`, `.DCM`, or none. Other files are ignored and not counted as unreadable.
- [ ] A file with the preamble that `read_header` cannot read is counted in `unreadable_count`, as today.
- [ ] Tests:
  - A tree mixing `.DCM` and extension-less copies organizes like the all-`.dcm` tree.
  - A `notes.txt` and a 10-byte junk file are neither copied nor counted.
- [ ] Existing tests stay green.

## T2 — #9: date fallback for the fx order
- [ ] The sort key uses the planning CT's SeriesDate. If it is missing or blank, it falls back to StudyDate, then AcquisitionDate, then ContentDate.
- [ ] A plan whose CT has none of these sorts after every dated plan. Among undated plans, the rest of the existing key decides (SeriesTime, RTPlanDate, RTPlanTime, UID).
- [ ] Tests:
  - A CT without SeriesDate but with a later StudyDate sorts by that StudyDate.
  - A CT with no date at all becomes the last plan, not `plan_1`.

## T3 — #13: copy unreadable files to misc_files
- [ ] Files with the DICM preamble that can't be read are copied into the patient's `misc_files/` (they're never in a plan). `unreadable_count` still counts them, and `misc_count` counts them too.
- [ ] Test: one truncated DICM file ends up in `misc_files/`, and both counts are right.

## Deferred
Nothing in this plan; anything else found goes in your report.

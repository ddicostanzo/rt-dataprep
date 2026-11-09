# organize: read DICOM inside zip archives

Status: REVIEWED (OSC probe 2026-11-09: 3 zips, Deflate only, none encrypted, testzip ok, members DICOM)

## Problem
An imaging-only export keeps some accessions as a single zip in their source folder. On OSC (job 8440157, 20 accession
folders, v0.1.0) 17 organized and 3 were refused `source folder holds no readable DICOM file`: a counts-only check showed
each of the 3 holds exactly one zip (180-240 MiB, `PK` signature) and nothing else. A zip is untrusted input in a PHI
pipeline, so reading it must not open a path-traversal, zip-bomb, PHI-leak or data-loss hole.

## Decisions (owner, 2026-11-09)

### Detection and scope
- Every file `_walk` yields (recursive, hidden folders skipped, hidden files kept) is classified from one 132-byte read
  (the existing `_has_preamble` read): DICM at 128 → DICOM; else first 4 bytes `PK\x03\x04` → zip; else neither (dropped,
  as today). An empty zip (`PK\x05\x06`) is neither.
- Zips are numbered 1.. in sorted walk order, members 1.. in `infolist()` order. Directory members are skipped and count
  toward no cap. A member that is itself a zip is not opened (it is just a non-DICOM file once staged). Any other member,
  a symlink entry included, is written as a regular file holding its bytes.

### Staging
- Each run creates one staging parent `out_root/.unzip-<random>/` (`tempfile.mkdtemp`); task A extracts a folder's zips
  into a fresh sub-folder of it, each member written as `<zip #>/<member #>`. Member names are never used in a path.
- Members are read only through `ZipFile.open()`, streamed to the staged file; never `extract()`, never a seek on the raw
  archive.
- Task A extracts the folder's zips before reading any header; staged files then join the loose files and go through the
  existing read and grouping unchanged. Staged members count in "files walked".
- Task B hard-links each staged file into every destination folder (`os.link`, falling back to a copy, as
  `masks._link_or_copy`), because `_group` can put one study file into several plan folders; it copies loose files as
  today, then removes the staging sub-folder. Peak disk stays one copy of the staged data. Staged files keep their index
  names; the existing collision suffix keeps names distinct.
- Lifetime: task A owns its staging sub-folder until it returns normally with files to copy (`try/except BaseException:
  remove; raise`, like `_copy_patient`), and removes it itself when it returns without files to copy (refusal, skip).
  Task B removes what is left of it in a `finally`. In the parent, every `handle()` branch that receives a task-A result
  and does not submit task B removes the staging that result names. The run's staging parent is removed in
  `organize()`'s `finally`, after `pool.run_auto` returned or raised; if it cannot be removed, one fixed warning with a
  count is logged.
- Lock (POSIX only; on Windows no lock, documented): the run holds `fcntl.flock(LOCK_EX | LOCK_NB)` (imported inside a
  platform guard, never at module top level) on `out_root/.organize.lock` for its whole duration. convert and preprocess
  skip hidden names, so the lock file is never read as a patient. If another run holds it, organize stops with the fixed `REASON_LOCKED` (a CLI refusal, like the other fixed ValueErrors). While
  holding it, the run removes any leftover `.unzip-*` from a killed run before starting.
- A killed job (SIGTERM/scancel) leaves staging hidden in the output root, which has the outputs' protection, until the
  next run on that root removes it.

### Caps (zip bomb)
- Checked from the central directory before anything of the folder is extracted, over all zips of one source folder:
  sum of declared `file_size` ≤ `MAX_UNZIP_BYTES` = 16 GiB, and non-directory members ≤ `MAX_ZIP_MEMBERS` = 200,000.
- Every member must pass an allow-list before extraction: `flag_bits & 0x61 == 0` (not encrypted, not strong-encrypted,
  not patched) and `compress_type in {0, 8, 12, 14}` (stored, deflate, bzip2, lzma). Else the zip is unreadable.
- A stream that disagrees with its declared size or CRC is unreadable (zipfile stops at `file_size` and checks the CRC,
  so declared sizes bound what is written).

### Bad zips
- A zip is bad when it is over a cap, fails the allow-list, or raises while opened or read: `BadZipFile`, `EOFError`,
  `zlib.error`, `lzma.LZMAError`, `UnicodeDecodeError`, `NotImplementedError`, `RuntimeError` (all mapped inside the
  extraction function; `OSError` stays the existing I/O path, `REASON_IO_ERROR`).
- "Loose DICOM" means a loose file with the DICM preamble (the classification read), not a readable header.
- A bad zip in a folder that has at least one loose DICOM file is skipped: its staged members (if any) are removed, it is
  counted (`N zips skipped`), and the folder organizes from the rest as today. A bad zip in a folder with no loose DICOM
  file refuses the folder: `REASON_ZIP_TOO_LARGE` (a cap) or `REASON_ZIP_UNREADABLE` (anything else), nothing kept. The
  row uses the sanitized folder name with `reads_kept=False`; with `--overwrite` an existing output folder is untouched.
  Over-cap is decided per folder (all zips), so in a folder with loose DICOM an over-cap total skips every zip.
- No zipfile exception text, member name or path ever reaches a log line, an exception message, a chained exception or a
  `Failed`: the extraction function catches and re-raises fixed-reason refusals `from None`.

### Contract
- Task A gains an input, the run's staging parent (created in the parent process, passed in the task args to workers),
  and its result tuple gains `staging` (`Path | None`), `zips`, `zip_members`, `zips_skipped`; `_copy_patient` gains the
  staging path. Existing `_read_fn`/`_copy_fn` test fakes are updated to the new tuple. Lifecycle tests patch
  `zipfile`/`dicom_io.read_header_core`, not `_read_fn`, so the real extraction runs.

### Reporting
- The organize summary line gains `N zips, M zip members, K zips skipped`. `organize_report.csv` columns are unchanged.

## Acceptance
- A synthetic patient (CT + RTSTRUCT + RTPLAN chain plus one MR series) exported as one zip in its folder organizes and
  converts to the same plan and study outputs (same series, same voxels) as the same files loose.
- Loose files plus a zip holding the rest organize as one patient. One instance present both loose and zipped is
  deduplicated by SOP (the copy goes to `misc_files`, as for two loose copies).
- Members named `../x.dcm`, `/abs/x.dcm`, and two members with one name: every file stays inside the output root; both
  same-name members are copied.
- Zip-only folder: declared total over the cap → `REASON_ZIP_TOO_LARGE`, nothing extracted; member count over the cap →
  `REASON_ZIP_TOO_LARGE` (caps patched low).
- Zip-only folder: encrypted member, CRC flipped, truncated archive, `compress_type` 9, invalid-UTF-8 name →
  `REASON_ZIP_UNREADABLE`; the next patient of the same run still organizes.
- Loose DICOM plus a bad zip → organized from the loose files, `1 zips skipped` in the log.
- caplog for the traversal, CRC and encrypted cases (member names `SYNTHETIC_PATIENT_NAME...`) holds no member name or
  path.
- No `.unzip-*` left after: success, refusal, skip of an existing output, I/O error during header reading, REASON_INTERNAL
  abort, KeyboardInterrupt during extraction, and with `workers=2`. A leftover from a killed run is removed by the next
  run. A second run on a locked root refuses with `REASON_LOCKED`.
- A zip inside a zip is not opened.
- A zipped patient with two RTPLANs on one planning CT: both plan folders hold the full CT series.
- Existing tests pass unchanged except the updated seam fakes.

## Release check (manual)
- Release v0.1.1; rerun the 20-folder OSC subset (`/fs/ess/PDE0046/raw_subset`): 20 of 20 organized, and the 17 folders
  that organized in v0.1.0 give the same `organize_report.csv` rows.

## Deferred
- tar, 7z, gzip archives; nested archives; Deflate64 (method 9; needs a dependency).
- A `--max-unzip-gib` option; disk-free check before staging; disk use in worker sizing.
- Early-out on reruns: today a zipped patient whose output exists is fully extracted before task A can see the skip.
- Central-directory parse memory grows with member count (bounded only by the member cap after parsing).
- A loose DICOMDIR (preamble, no PatientID) beside a bad zip gives `REASON_NO_ID`, not `REASON_ZIP_UNREADABLE`.
- Keeping member base names; a per-folder zip column in organize_report.csv.

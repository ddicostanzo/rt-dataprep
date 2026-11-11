# F3: manifest and split

**Date:** 2026-11-04 · **Branch:** `feat/f3-manifest-split` · **Status:** REVIEWED (stress-test round 1 applied)
**Parent plan:** `docs/plans/2026-11-03-rt-dataprep/phase1.md` (F3, A3.1/A3.2) · **Builds on:** F2a convert, F2b preprocess

## Background

Convert and preprocess each write `<patient>/<plan folder>/` (`plan_<n>` or `fx_<n>`, `convert.PLAN_FOLDER`) with a
`series.csv` listing the plan's ct, dose, mask and applicator files. Nothing yet lists the cases in a tree or splits
them by patient.

rnt_transformer does not read an outside split: it splits with `GroupShuffleSplit(random_state=split_seed)` and
records `{split: {mrn: [sorted fraction ids]}}` in its run's `splits.json`, with a `fraction_id_hash` =
`sha256('\n'.join(sorted(ids)))`. rnt stays unchanged (phase1.md, out of scope). F3's split is for new projects.

## Decisions (user, 2026-11-04)

- D1 `split()` is a tool for new projects; rnt keeps its own split.
- D2 `splits.json` has rnt's shape, `{train|val|test: {patient: [sorted case ids]}}`, so rnt's splits.json readers
  (`rr_probe.split_counts` indexes `splits[name]`) parse it. The per-split hashes go under one extra top-level key,
  `case_id_hash: {train, val, test}`.
- D3 A case id is `<patient>/<plan folder>`, the case path relative to the root. The hash uses rnt's method, but its
  values never equal rnt's `fraction_id_hash` (different ids).
- D4 `manifest()` reads any tree whose plan folders hold `series.csv`: convert output or preprocess output.
- D5 `val_size` and `test_size` are shares of all patients (0.1/0.1 is about 80/10/10).
- D6 Counts round up: `ceil(round(n * size, 9))`, so any size > 0 holds out at least one patient and float noise
  (`0.07 * 100 = 7.000000000000001`) does not add one (user, stress-test F6).
- D7 Two CLI subcommands: `rt_dataprep manifest <root>` and
  `rt_dataprep split <root> --test-size --val-size --seed [--external <manifest.csv>]`. `--seed` is a required int
  (a missing seed would not be reproducible); any int, negative included (#75). The disjoint check runs
  inside `split` when `--external` is given, before anything is written.

## Design

### `manifest(root)` (`rt_dataprep/manifest.py`)

Walks `<root>/<patient>/<plan folder>/` (non-hidden patient folders; plan folders matching `PLAN_FOLDER` that hold
`series.csv`; `images/` and `.partial` folders are not cases). For each plan it reuses `preprocess.check_rows`, which
validates `series.csv` and reads the CT header only. A plan it refuses is left out and counted, and so is a patient folder whose name is not UTF-8 (one per folder,
#260: its surrogate escape would make `split` refuse the whole manifest); the count goes to
the log, nothing else.

Writes `<root>/manifest.csv` with `dicom_io.write_report` (dated copy plus latest), one row per case, sorted by
`case_dir`. Columns:

| column | value |
|---|---|
| `case_dir` | `f'{patient}/{plan folder}'`, always `/` (the case id, D3) |
| `patient` | the patient folder name, taken from the path component, never parsed from a string (vault `patient-group-key-parsed-from-relative-path-defeats-split-and-overlap-guards`) |
| `fx` | the plan folder name |
| `ct_size`, `ct_spacing` | the CT header's size and spacing, `x;y;z`; spacing as its float32 value (`0.8`, #74) |
| `dose` | `1` if `series.csv` lists a dose file, else `0` |
| `masks` | the mask rows' `roi_name`s, sorted, as a JSON list (`["bladder", "ptv"]`; a name may hold `;`, #74) |
| `rtplan_uid`, `ct_series_uid`, `study_uid` | from the ct row of `series.csv` |
| `retained` (plan lines, last key) | the lowest dose/mask `retained` of a preprocessed `series.json`, as a number; left out when no row has one (a converted tree, a non-geometric run); a cell that is not a finite number refuses the plan (#225) |

Returns the number of cases written and refused.

### `split(manifest_path, *, test_size, val_size, seed, external=None)` (`rt_dataprep/split.py`)

1. Reads `manifest.csv` with `csv.DictReader`, every value a string (pandas would turn `00123` into `123` and
   break patient matching); patients = sorted unique `patient` values. `external` is read the same way.
2. Refuses (fixed `REASON_`) when a size is outside `[0, 1)` or the test and val counts (D6) leave train empty. `seed` is required.
3. Patients ordered by `sha256(f'{seed}\n{patient}')`, the same on every numpy and Python version (#75): the first `ceil(n*test_size)` go to test, the next
   `ceil(n*val_size)` to val, the rest to train. No scikit-learn.
4. With `external`, runs `check_disjoint` first.
5. Writes `<manifest dir>/splits.json` (D2). If a `splits.json` with different content is already there, refuses
   instead of overwriting: a split already used for training is never replaced silently. Identical content is
   left as it is. The write is atomic and exclusive (temp file in the same folder, then `os.link`, which refuses an
   existing target, so of two concurrent runs only the first writes, #73; a plain rename where hard links are
   unsupported). A manifest without a `patient` or `case_dir` column is refused (#72). So is one that is not UTF-8 or holds a row without a
   `patient` or `case_dir` value (#78), or whose `patient` or `case_dir` holds a lone surrogate escape that cannot be encoded as UTF-8 (#249).

### `check_disjoint(splits, external_manifest)`

Refuses with a fixed `REASON_OVERLAP` if any patient of the external manifest is in any split. The reason names
no patient and no count.

Every new `REASON_` refusal is a `ValueError` whose text the CLI's fixed-refusal tuple (`__main__.py`) lists, so
it prints as a refusal, not as an unexpected error.

## Constraints

- **PHI.** `manifest.csv` and `splits.json` hold patient folder names (sanitized MRNs). They are written under the
  data root only, are in `.gitignore` (`manifest*.csv`, so the dated copies too), and the global `pre-commit.sh` refuses them by name
  even under `git add -f` (user, advisor finding; changed outside this repo, 2026-11-04), and are added to the `project-profile.yml` phi rule. Log lines,
  exceptions and CLI output carry counts and fixed `REASON_` constants only.
- **Data safety.** The data tree is only read; the only files written are `manifest.csv` (and its dated copy) and
  `splits.json`.
- **Dependencies.** None added.

## Acceptance tests

- **A3.1** The manifest has at least 10 patients and at least one patient with two or more plans; every plan of that
  patient lands in the same split, and no patient is in more than one split. The same seed and manifest give a
  byte-identical `splits.json`; two fixed seeds known to differ give a different patient-to-split assignment
  (vault `seed-everything-masks-missing-random-state-in-determinism-test`).
- **A3.2** An external manifest that shares a patient is refused and no `splits.json` is written. A distinctive
  synthetic patient name is absent from `str(exc)`, caplog, and `capfd` output.
- **A3.3** `manifest` on a converted synthetic tree and on its preprocessed tree lists the same `case_dir`s, one row
  per plan, with the CT size, dose flag, masks and UIDs from that tree; a plan with a malformed `series.csv` is
  left out and counted.
- **A3.4** Sizes: `n=10, test=0.1, val=0.1` gives 1/1/8 patients; `n=10, test=0.15` gives 2 test (round up); `n=100, test=0.07` gives 7; an
  empty train is refused.
- **A3.5** A different existing `splits.json` is refused and left untouched.
- **A3.6** The CLI subcommands run end to end; a refusal prints its fixed text (not "unexpected error"); no patient
  name appears in `capfd` output.

## Deferred

- Stratified splits (by site, plan type or fraction count).
- K-fold.
- Reading `case_id_hash` back to verify a split in a later run.

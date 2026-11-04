# Hardening B10: split reservation leftovers (#105, #106)

Branch `fix/hardening-b10-split-reservation`, worktree `/home/dominic/Repositories/rt-dataprep-b10` off `main` at `604ca3d`. Run tests with `uv run --group dev pytest -q --tb=short` and lint with `uv run --group dev ruff check .`. Baseline: 630 passed, 4 skipped.

Scope: `src/rt_dataprep/split.py`, `tests/test_split.py`, this plan. Nothing else.

## Guardrails
- **PHI:** no new message, log or exception may carry a path, ID or tag value. A `FileNotFoundError` names the path, so it must never propagate out of `split()`. Test data is `SYNTHETIC_`-prefixed.
- **Commit gate:** no added line with a dotted name whose last part starts with `_` (use `getattr(module, '_name')  # noqa: B009`); at most 5 staged files per commit.
- **Handlers:** no `raise <expr>` inside an except block (a guard test forbids it).
- **Stop counter:** after 3 failed attempts at one test, drop it and report.
- Don't push, open a PR or close issues: the owner confirms those.

## Tasks and acceptance

### T1 — split: a reservation released during the wait is taken over, not a raw FileNotFoundError (#105)
In `split()`, the sequence `if not path.exists(): _write_new(path, text)` → `found = path.read_text(...)` → empty-wait loop has two unguarded reads (the first read and the re-read in the loop). Without hard links, another run whose `os.replace` failed unlinks its reservation; if that lands between our `exists()` and a read, `read_text` raises `FileNotFoundError` naming the path.

Fold the sequence into the one loop so every read goes through the same guard, for example:

```python
for attempt in range(21):  # comment: empty only while another run's O_EXCL reservation is not yet renamed over; missing once that run released it after a failed rename
    if not path.exists():
        _write_new(path, text)
    try:
        found = path.read_text(encoding='utf-8')
    except FileNotFoundError:  # released between exists() and the read: the next pass writes it
        found = ''
    if found or attempt == 20:
        break
    time.sleep(0.05)  # keep the existing shortcut: comment
if found != text:
    raise ValueError(REASON_EXISTS)
```

- Total wait stays at most 20 sleeps (about 1 s); no extra trailing sleep before the refusal.
- Keep the existing `# shortcut:` comment (killed run holding the reservation) on the sleep line; update the loop comment as above.
- A file that keeps vanishing until the loop ends falls through to `REASON_EXISTS` (fixed text). Acceptable.
- [ ] Test (red first), no patching of `os.link`: the handler under test is the read in `split()`, not the write path. Pre-create an empty `splits.json`, start `threading.Timer(0.2, (tmp_path / 'splits.json').unlink)`, call `split()`. It returns the dict, and `splits.json` then reads exactly `json.dumps(obj, indent=1, sort_keys=True) + '\n'`. Cancel/join the timer in a `finally`. On current `main` it fails with `FileNotFoundError`; state that in the report.
- [ ] The existing empty-reservation tests, `test_without_hard_links_*` tests and the hard-link race test pass unchanged (apart from T2's edits).

### T2 — the empty-reservation race test fills the file atomically, and the refusal bound has headroom (#106)
Test-only. **No red step:** it removes an intermittent flake in the test itself, which cannot be made to fail deterministically.
- [ ] In `test_an_identical_split_is_not_refused_by_another_runs_empty_reservation`, the Timer target becomes a local `fill()` that writes `text` to a temp file in `tmp_path` and `os.replace`s it over `splits.json`, as the real code does. Don't use `_write_new`: with the file present, its `os.link` branch writes nothing.
- [ ] In `test_an_empty_splits_json_that_is_never_filled_is_still_refused_after_about_a_second`, raise the bound in place from `1.5` to `3`. It still catches an unbounded wait. Don't lift the `20`/`0.05` into constants.

## Done
Full suite and ruff green; test count 631 passed, 4 skipped. Report: T1 red-before/green-after, files changed, test count. Commit on the branch (plan, `split.py`, `test_split.py`). The PR body, once the owner confirms the push, says "Closes #105. Closes #106."; check after the merge and close them by hand if they stay open.

## Deferred
- A `splits.json` reservation left empty by a killed run (SIGKILL between reservation and rename) still refuses every later split until it is removed by hand. Only on file systems without hard links. (Carried from B9.)

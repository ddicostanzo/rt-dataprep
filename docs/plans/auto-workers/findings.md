# Findings: --workers auto (#175)

Spec: docs/specs/2026-11-08-auto-workers.md. Plan: docs/plans/auto-workers/plan.json (5 tasks; groups 01 → 02 → 03+04 → 05).
Append under `## <task-id>`; never rewrite earlier sections.

## orchestrator
- task-01 and task-02 run in one implementer on feat/auto-workers (both touch pool.py, strictly sequential); one commit per task.
- Vault guardrails: [[measure-a-phase-peak-rss-by-resetting-vmhwm]] (ru_maxrss is lifetime HWM; overestimate by design), [[dataloader-cpu-bound-on-hyperthreaded-cores]] (ITK thread oversubscription: deferred).
- OSC enforces cgroup v2 memory.max (job 8403365).

## task-01
- available_ram(root='/') walks the cgroup path and all ancestors (incl. the cgroup root); a v1 memory line wins over a `0::` line (hybrid hosts). An unreadable level (missing memory.max/current/stat) is no limit for that level; a missing inactive key counts as 0 (conservative).
- size_workers uses integer math (4*available // (5*peak)) so the 0.8 floor has no float rounding at exact boundaries.
- Private helpers (_read, _cgroup_free) are exercised only through the public readers in tests, so the commit hook's private-access override was not needed.
- tests/test_pool.py alone takes ~2.5 min (spawned pools), full suite ~3m45s.

## task-02
- run_auto returns 1 when the probe ends (ABORT/Failed/raise) or no patient qualified, and the sized N otherwise (1 when remaining is 0; the INFO line is still logged then).
- Probe handle is one closure with a mutable current index (ruff B023 rejects a closure defined per loop pass); the probe pool.run(1, [task]) always passes index 0, so the wrapper ignores it and passes k.
- Unknown peak (no `resource`) is passed to size_workers as 0, which it maps to 1; None would raise on `peak <= 0`.
- Tests spy on pool.run by patching the module attribute, so run_auto must keep calling the module-global `run`/`size_workers`/`available_ram`/`usable_cpus`. The N=2 cases start a real spawned pool (about 0.1 s each).

## review-group-1-2
- Inline wave review by the orchestrator (pool.py diff 7f10c19, 25f79af): matches task-01/02 criteria; no BLOCKING. The fresh-context Opus pass is the final /code-review round.

## task-05
- `positive_int` became `workers_arg` (accepts the exact string 'auto', otherwise the same digit-only 1..MAX_WORKERS rule); the usage error now reads "must be 'auto' or an integer from 1 to 64". MAX_WORKERS is imported from pool; `Auto` and `auto ` are rejected (tests added).
- WORKERS_HELP is now a parenthesised multi-line string (it contains an apostrophe); it keeps the old phrases the help test pins and avoids '%' (argparse formats help) and the word 'thread'.
- README command table uses "`auto` or `1..64`" because a pipe inside a table cell breaks the row.

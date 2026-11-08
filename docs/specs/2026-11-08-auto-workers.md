# `--workers auto`: size the patient pool from a measured probe (#175)

## Problem
`--workers` defaults to 1, and picking N by hand means guessing peak RAM per patient (the help text says ~7 GB at
701x701x433). Too many workers OOM-kill the job; too few waste a node. On OSC (job 8388216, 20 patients, 4 workers,
48 GB) the run used 16 min wall time with sacct MaxRSS 11.5 GB, so the hand-picked N left most of the node idle.

## Decisions (owner, 2026-11-08)
- Measurement: probe in-run. With `workers='auto'` the stage runs patients in the calling process, one at a time, until
  one is actually processed; its peak RSS sizes the pool for the remaining patients. No subcommand, no cache.
- Probe target: a patient counts as processed when at least one of its rows has the stage's success outcome
  (`organized`, `converted`, `preprocessed`). Skipped (existing output) and wholly refused patients keep the probe going.
  If no patient qualifies, every patient runs in-process (N=1), as today. Handles return None/ABORT/Task, never rows, so
  each stage passes its own `processed(k)` predicate over its own state (organize `done[i]`; convert/preprocess
  `results[...]`, keyed by folder, not task index).
- Probe contract: the probe calls `pool.run(1, [task_k], wrapped)` once per patient; the remaining patients go to one
  `pool.run(N, rest, handle offset by k)`. The probe phase ends the stage (no further `pool.run`) when the wrapped handle
  returns ABORT, is given a `Failed`, or raises; KeyboardInterrupt propagates. pool.run is unchanged. (A `Stop` result is
  unwrapped by pool.run before handle sees it; every stage turns it into ABORT today, so ABORT covers it.)
- N = max(1, min(floor(0.8 × available RAM ÷ probe peak), usable CPUs, MAX_WORKERS (64), patients left)).
  - Available RAM: `MemAvailable` from `/proc/meminfo`, capped by the tightest cgroup limit over the process's cgroup and
    all its ancestors, so a Slurm `--mem` set on `job_X`/`step_Y` is found from the leaf `task_0`. The path comes from
    `/proc/self/cgroup` (v2 `0::/path`, v1 the `memory` controller line). Per level, free = limit − (usage −
    reclaimable cache): v2 `memory.max`, `memory.current`, `inactive_file` from `memory.stat`; v1
    `memory.limit_in_bytes`, `memory.usage_in_bytes`, `total_inactive_file`. A limit of `max` (or v1's huge sentinel)
    is no limit. Page cache from an earlier stage in the same job must not read as used. Unreadable (non-Linux) → N=1.
  - Probe peak: `resource.getrusage(RUSAGE_SELF).ru_maxrss` of the calling process after the probe patient (Linux KiB).
    It is the process's lifetime high-water mark, so it also counts the caller's own memory; that overestimates and
    errs toward fewer workers (vault: [[measure-a-phase-peak-rss-by-resetting-vmhwm]]).
  - Usable CPUs: `len(os.sched_getaffinity(0))` (respects Slurm `--cpus-per-task`), else `os.cpu_count()`.
- Default: the CLI's `--workers` defaults to `auto` and still accepts a positive int. The Python functions keep
  `workers=1` and also accept `'auto'`.
- One log line after sizing, counts only: probe patients, peak GB, available GB, CPUs, chosen N. Organize's and convert's
  existing `workers %d` timing lines log the chosen N (preprocess has none). Peak = `ru_maxrss` × 1024 bytes; `resource`
  is imported only when auto runs, so Windows still imports the package (auto → 1 there).
- Output is the same for any N, as today (report row order, files).
- /dev may auto-merge after a clean review round.

## Acceptance
1. `organize/convert/preprocess(..., workers='auto')` produce the same report rows and files as `workers=1`, on a
   fixture of at least 3 patients with the sizer patched to return 2, so the probe → pool → offset-handle path runs
   (with the real sizer a small fixture never starts a pool).
2. With a probe that sees skipped patients first, sizing happens only after the first processed patient; all-skipped
   runs never start a process pool.
3. The sizer is a pure function of (peak, available, cpus, remaining); the RAM and CPU readers are separate and take
   a root path. It returns the min of the four limits and at least 1; RAM unreadable → 1. Remaining 0 → no second
   `pool.run` call at all; N = 1 → the second call is inline.
4. The CLI parses `--workers auto` and `--workers 3`, defaults to auto, and still rejects `0`, `abc` and `65` (the
   type function keeps the MAX_WORKERS cap).
5. An ABORT, Failed, Stop or exception during the probe phase ends the stage exactly as it does at N=1 (no pool started
   for the remaining patients).
6. README and `--workers` help describe auto, the 0.8 headroom and the OOM-abort consequence.
7. The RAM reader, on fixture `/proc` and cgroup trees: finds a limit set on an ancestor, not the leaf (v1 and v2);
   subtracts `inactive_file`; treats `max` as no limit; returns None when nothing is readable.

## Deferred
- SimpleITK/ITK threads per worker oversubscribe the CPUs at N>1 (vault:
  [[dataloader-cpu-bound-on-hyperthreaded-cores]]); set ITK threads per worker = CPUs ÷ N if throughput is flat.
- A later patient much larger than the probe (a 512x512x150 probe vs 701x701x433, ~3-4x) can exceed the 0.8 headroom:
  a worker is OOM-killed, the pool breaks and the run aborts (report kept, no corruption; a rerun skips finished patients;
  a fixed `--workers N` overrides). Owner: defer, fix if seen on real data. Documented in the README.
- Slurm without cgroup memory enforcement (OSC enforces cgroup v2 `memory.max`: 2147483648 seen for `--mem=2G`, job 8403365, 2026-11-08) (`SLURM_MEM_PER_NODE` only) and a cgroup v2 `cpu.max` quota are not read.
- Per-patient wall time is not measured or used.
- macOS/Windows: auto falls back to 1.

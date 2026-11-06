# Triage: #200 test_a_dying_worker_is_a_failed_and_does_not_hang flakes

## Feedback loop
Standalone stdlib script (no rt_dataprep code): 2-worker spawn `ProcessPoolExecutor`, submit `die` (SIGKILL self)
and `nap(3)`, time until the `die` future is done.

    .venv/bin/python race.py 10 die-first    -> slow detections: 3 of 10   (Python 3.13.16)
    .venv/bin/python race.py 10 die-second   -> slow detections: 8 of 10
    python3          race.py 10 die-second   -> slow detections: 5 of 10   (Python 3.14.4)

"Slow" = the death is noticed only when the sibling's 3 s nap returns.

## Root cause (HIGH)
CPython `ProcessPoolExecutor.submit` wakes the manager thread *before* `_adjust_process_count` spawns the new
worker. The manager wakes, snapshots `self.processes` for its sentinel list
(`wait_result_broken_or_wakeup`), and blocks in `mp.connection.wait` without the new worker's sentinel. If
that worker then dies, nothing wakes the manager until some other event (a result, a submit, shutdown).

In the test: task 0 (`die`) lands on the second-spawned worker in a fraction of runs (whichever worker boots
first takes it). The death goes unseen; task 1 (`after_file`) waits its full 30 s cap for `released`, which
the parent only creates after task 0's Failed is delivered -> the 30.5 s run. Task 1's result then wakes the
manager; meanwhile the scheduler, with `stop` still False, submits task 2 and may deliver it -> the
`2 not in delivered` assertion fails (the full-suite failure).

The same race, same code, exists on 3.14.

## Correction to an earlier note
Vault note `broken-processpool-keeps-running-workers-test-race` (from #183) says a broken pool keeps its
running workers. In fact `terminate_broken` terminates them; the sibling "ran to completion" because the
death was not yet detected. The #183 file-release fix narrowed the window but cannot close it.

## Product impact
Not just a test issue: an OOM-killed worker (#47 territory) is detected late, at the next pool event. Until
then the run keeps going; when detection comes, every pending future fails and running workers are
terminated, so one more patient can be started and then killed as Failed('BrokenProcessPool'). Publish is
atomic (.partial/.old + recover), so no data corruption; outcome is a delay and an extra Failed row.

Why one slow run passed and the full-suite run failed: same race. If pool.run's thread is not scheduled between
task 1's set_result and task 0's set_exception, wait returns both, stop is set before the next pick and task 2 is
never submitted (slow pass). If it is, task 2 is submitted and can be delivered (fail).

## advisor
- Fix: kick the manager (`executor._executor_manager_thread_wakeup.wakeup()`) in `pool._submit` after each
  process-pool submit; the kick lands after the spawn, so the next snapshot includes the new worker. Owner chose
  this over test-only tolerance. Attribute and order verified on 3.12.15, 3.13.16, 3.14.4; no getattr guard,
  so a removal turns tests red instead of bringing the flake back.
- Deterministic RED: patch `ProcessPoolExecutor._spawn_process` to sleep 1 s on its second call; tasks
  `[after_file(0, marker, 2.0), die_after_marker(1, marker)]`; both must be Failed('BrokenProcessPool').
- Use the worktree's own venv (`uv sync`), and post the analysis on #200 rather than opening a new issue.

## Result
RED before the fix (task 0 returned 0). After: test_pool 53 passed; the two target tests 20/20 at ~1.2 s and 8/8
under full CPU load; full suite 1083 passed, 4 skipped.

"""The scheduler: one code path for N=1 (inline) and N>1 (spawned processes). Every consumer runs in a daemon thread under a timeout."""
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pool_workers as w
import pytest

from rt_dataprep import pool
from rt_dataprep.pool import ABORT, Failed, Task

TIMEOUT = 60
BOTH = pytest.mark.parametrize('workers', [1, 2])


def run_guarded(workers, tasks, handle=None):
    """Run pool.run in a daemon thread; return (delivered list of (index, result), raised exception or None)."""
    delivered, outcome = [], {}

    def record(index, result):
        delivered.append((index, result))
        return handle(index, result) if handle else None

    def target():
        try:
            pool.run(workers, tasks, record)
        except BaseException as exc:  # KeyboardInterrupt too
            outcome['error'] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(TIMEOUT)
    assert not thread.is_alive(), 'the scheduler hung'
    return delivered, outcome.get('error')


def times(directory, name):
    return float(Path(directory, f'{name}.start').read_text()), float(Path(directory, f'{name}.end').read_text())


def started(directory):
    return {int(p.stem) for p in Path(directory).glob('*.start')}


@BOTH
def test_results_in_input_order_whatever_the_completion_order(workers):
    tasks = [Task(w.sleep_then, (0, 0.8))] + [Task(w.echo, (i,)) for i in (1, 2, 3)]
    delivered, error = run_guarded(workers, tasks)
    assert error is None
    assert delivered == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_one_runs_inline_and_more_run_in_other_processes():
    assert run_guarded(1, [Task(w.pid)])[0] == [(0, os.getpid())]
    (_, child), = run_guarded(2, [Task(w.pid)])[0]
    assert child != os.getpid()


def test_spawned_worker_does_not_see_a_parent_monkeypatch(monkeypatch):
    monkeypatch.setattr(w, 'MARKER', 'patched')
    assert run_guarded(1, [Task(w.marker)])[0] == [(0, 'patched')]  # inline sees it, so the next line proves spawn
    assert run_guarded(2, [Task(w.marker)])[0] == [(0, 'original')]


class Spy(ProcessPoolExecutor):
    """Records how many submitted futures were unfinished at each submit (done() is set before the parent wakes)."""
    live = []
    futures = []

    def submit(self, *args, **kwargs):
        Spy.live.append(sum(not f.done() for f in Spy.futures) + 1)
        Spy.futures.append(super().submit(*args, **kwargs))
        return Spy.futures[-1]


def test_at_most_n_are_submitted_at_a_time(monkeypatch, tmp_path):
    Spy.live, Spy.futures = [], []
    monkeypatch.setattr(pool, 'ProcessPoolExecutor', Spy)
    tasks = [Task(w.record, (str(i), str(tmp_path), 0.3)) for i in range(6)]
    delivered, error = run_guarded(2, tasks)
    assert error is None and len(delivered) == 6
    assert max(Spy.live) == 2 and len(Spy.live) == 6


@BOTH
def test_running_never_exceeds_n(workers, tmp_path):
    tasks = [Task(w.record, (str(i), str(tmp_path), 0.4)) for i in range(4)]
    run_guarded(workers, tasks)
    spans = [times(tmp_path, str(i)) for i in range(4)]
    overlap = max(sum(s <= s0 < e for s, e in spans) for s0, _ in spans)  # running at the moment each one starts
    assert overlap <= workers


@BOTH
def test_same_key_never_runs_at_the_same_time(workers, tmp_path):
    keys = ['a', 'a', 'b', 'a']
    tasks = [Task(w.record, (str(i), str(tmp_path), 0.4), k) for i, k in enumerate(keys)]
    delivered, error = run_guarded(workers, tasks)
    assert error is None and [i for i, _ in delivered] == [0, 1, 2, 3]
    spans = {i: times(tmp_path, str(i)) for i in (0, 1, 3)}
    assert spans[0][1] <= spans[1][0] and spans[1][1] <= spans[3][0]


def test_a_held_task_does_not_block_unrelated_ones(tmp_path):
    keys = ['a', 'a', 'b']
    tasks = [Task(w.record, (str(i), str(tmp_path), 0.8), k) for i, k in enumerate(keys)]
    delivered, error = run_guarded(2, tasks)
    assert error is None and [i for i, _ in delivered] == [0, 1, 2]  # input order regardless
    assert times(tmp_path, '2')[0] < times(tmp_path, '1')[0]  # 2 went ahead of the held 1
    assert times(tmp_path, '0')[1] <= times(tmp_path, '1')[0]


@BOTH
def test_abort_from_handle_submits_nothing_further(workers, tmp_path):
    tasks = [Task(w.record, (str(i), str(tmp_path), 0.3)) for i in range(6)]
    delivered, error = run_guarded(workers, tasks, lambda i, r: ABORT if i == 1 else None)
    assert error is None
    ran = started(tmp_path)
    assert {i for i, _ in delivered} == ran and len(ran) < 6  # everything that started was delivered
    assert [i for i, _ in delivered] == sorted(ran)
    if workers == 1:
        assert ran == {0, 1}


@BOTH
def test_escaping_exception_is_a_failed_with_only_its_type_name(workers, tmp_path, capfd):
    tasks = [Task(w.fail, (0,)), Task(w.record, ('1', str(tmp_path), 0.8)), Task(w.record, ('2', str(tmp_path), 0.0))]
    delivered, error = run_guarded(workers, tasks)
    assert error is None
    assert delivered[0] == (0, Failed('RuntimeError'))
    assert started(tmp_path) == ({1} if workers == 2 else set())  # the running one finished, none was added
    assert delivered[1:] == ([(1, '1')] if workers == 2 else [])
    seen = capfd.readouterr()
    assert w.SECRET not in seen.out + seen.err + repr(delivered)


def test_a_dying_worker_is_a_failed_and_does_not_hang(tmp_path, capfd):
    released = tmp_path / 'released'  # created once handle has task 0's Failed, so stop is already set when task 1 returns (#183)
    tasks = [Task(w.die, (0,)), Task(w.after_file, (1, str(released), 0.0)), Task(w.echo, (2,))]
    delivered, error = run_guarded(2, tasks, lambda i, r: released.touch() if i == 0 else None)
    assert error is None
    assert delivered[0] == (0, Failed('BrokenProcessPool'))
    assert 2 not in {i for i, _ in delivered}  # task 1 may or may not have finished before the pool broke


def test_a_worker_spawned_after_the_manager_waits_is_still_watched(tmp_path, monkeypatch, capfd):
    """CPython's submit wakes its manager before spawning the worker: a late spawn is missing from the sentinels it waits on (#200)."""
    real, calls = ProcessPoolExecutor._spawn_process, []

    def late_second(self):
        calls.append(1)
        if len(calls) == 2:
            time.sleep(1)  # the manager has already re-snapshotted its workers without this one
        return real(self)

    monkeypatch.setattr(ProcessPoolExecutor, '_spawn_process', late_second)
    marker = tmp_path / 'dying'
    tasks = [Task(w.after_file, (0, str(marker), 2.0)), Task(w.die_after_marker, (1, str(marker)))]
    delivered, error = run_guarded(2, tasks)
    assert error is None
    assert delivered == [(0, Failed('BrokenProcessPool')), (1, Failed('BrokenProcessPool'))]  # task 0 ends when the pool breaks, not after its sleep


def test_a_raising_manager_wakeup_keeps_the_submitted_task(monkeypatch):
    """pool's extra wakeup (#200) is best effort: if it raises, the task already submitted still runs and is delivered (#201)."""
    from concurrent.futures import process
    real = process._ThreadWakeup.wakeup

    def wakeup(self):
        if sys._getframe(1).f_globals.get('__name__') == pool.__name__:  # only pool's own kick; CPython's calls stay real
            raise RuntimeError(w.SECRET)
        return real(self)

    monkeypatch.setattr(process._ThreadWakeup, 'wakeup', wakeup)
    delivered, error = run_guarded(2, [Task(w.echo, (0,)), Task(w.echo, (1,))])
    assert error is None and delivered == [(0, 0), (1, 1)]


def test_a_submit_that_raises_is_a_failed(monkeypatch):
    class Broken(ProcessPoolExecutor):
        def submit(self, *args, **kwargs):
            raise RuntimeError(w.SECRET)

    monkeypatch.setattr(pool, 'ProcessPoolExecutor', Broken)
    delivered, error = run_guarded(2, [Task(w.echo, (0,)), Task(w.echo, (1,))])
    assert error is None and delivered[0] == (0, Failed('RuntimeError'))


@BOTH
def test_keyboard_interrupt_from_a_task_stops_submission_and_propagates(workers, tmp_path, capfd):
    tasks = [Task(w.record, ('0', str(tmp_path), 0.5)), Task(w.interrupt, (1,)), Task(w.record, ('2', str(tmp_path), 0.0))]
    delivered, error = run_guarded(workers, tasks)
    assert isinstance(error, KeyboardInterrupt)
    assert delivered == [(0, '0')]  # the task that returned keeps its result; the interrupted one has none
    assert 2 not in started(tmp_path)


@BOTH
def test_keyboard_interrupt_in_the_parent_wait(workers, tmp_path, monkeypatch, capfd):
    real, calls = pool.wait, []

    def interrupted_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(pool, 'wait', interrupted_once)
    tasks = [Task(w.fail, (0, 0.3)), Task(w.record, ('1', str(tmp_path), 0.5)), Task(w.record, ('2', str(tmp_path), 0.0))]
    delivered, error = run_guarded(workers, tasks)
    assert isinstance(error, KeyboardInterrupt)
    assert 0 not in {i for i, _ in delivered}  # ended with an exception after the interrupt: no row
    assert 2 not in started(tmp_path)
    assert ({1} if workers == 2 else set()) == {i for i, _ in delivered}
    seen = capfd.readouterr()
    assert w.SECRET not in seen.out + seen.err


@pytest.mark.skipif(not (hasattr(signal, 'pthread_sigmask') and hasattr(signal, 'sigtimedwait')), reason='no signal masks to poll')
def test_workers_spawn_with_ctrl_c_blocked_but_a_running_task_gets_it_and_the_parent_mask_is_restored():
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    delivered, error = run_guarded(2, [Task(w.sigint_state)])
    assert error is None and delivered == [(0, ([], True))]  # _init_worker unblocked it (#177)
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before


@pytest.mark.skipif(not (hasattr(signal, 'pthread_sigmask') and hasattr(signal, 'sigtimedwait')), reason='no signal masks to poll')
def test_a_caller_that_blocked_ctrl_c_gets_workers_with_it_still_blocked():
    before = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})  # run_guarded's thread inherits this mask
    try:
        delivered, error = run_guarded(2, [Task(w.sigint_state)])
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, before)
    assert error is None and delivered == [(0, ([int(signal.SIGINT)], True))]  # #192


@pytest.mark.skipif(not (hasattr(signal, 'pthread_sigmask') and hasattr(signal, 'sigtimedwait')), reason='no signal masks to poll')
def test_a_caller_that_blocked_ctrl_c_keeps_its_pending_one():
    before = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})  # on the main thread: the mask is per thread
    try:
        signal.pthread_kill(threading.get_ident(), signal.SIGINT)  # pending, as the caller meant it
        delivered = []
        try:
            pool.run(2, [Task(w.echo, (0,))], lambda i, r: delivered.append(i))
        except KeyboardInterrupt:  # a bare one would abort the whole pytest run
            pytest.fail('run took the caller\'s pending Ctrl-C (#194)')
        assert delivered == [0]
        assert signal.SIGINT in signal.sigpending()
    finally:
        signal.sigtimedwait({signal.SIGINT}, 0)  # consumed here, or the restore below raises it
        signal.pthread_sigmask(signal.SIG_SETMASK, before)


@pytest.mark.skipif(not (hasattr(signal, 'pthread_sigmask') and hasattr(signal, 'sigtimedwait')), reason='no signal masks to poll')
def test_a_ctrl_c_to_the_parent_during_handle_starts_no_new_patient(tmp_path):
    def handle(index, result):
        if index == 0:
            signal.pthread_kill(threading.get_ident(), signal.SIGINT)  # this (main) thread alone: pending until the next pick

    delivered = []
    tasks = [Task(w.echo, (0,)), Task(w.sleep_then, (1, 0.5)), Task(w.record, ('2', str(tmp_path), 0.0))]
    with pytest.raises(KeyboardInterrupt):  # on the main thread, as the CLI runs it: the mask is per thread
        pool.run(2, tasks, lambda i, r: (delivered.append(i), handle(i, r))[1])
    assert delivered == [0, 1]  # the running task still ends and is delivered
    assert 2 not in started(tmp_path)


@pytest.mark.skipif(not (hasattr(signal, 'pthread_sigmask') and hasattr(signal, 'sigtimedwait')), reason='no signal masks to poll')
def test_a_task_failing_after_a_parent_ctrl_c_with_every_slot_busy_is_skipped_not_failed():
    main = threading.get_ident()
    timer = threading.Timer(0.2, signal.pthread_kill, (main, signal.SIGINT))  # parent only, while both slots run
    delivered = []
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):  # on the main thread: the mask is per thread
            pool.run(2, [Task(w.sleep_then, (0, 1.0)), Task(w.fail, (1, 0.5))], lambda i, r: delivered.append((i, r)))
    finally:
        timer.cancel()
    assert delivered == [(0, 0)]  # not (1, Failed('RuntimeError')): an interrupted run writes no error row (#177)


def test_a_worker_info_line_reaches_stderr(capfd, caplog):
    caplog.set_level(logging.INFO)  # the parent's root level is what the pool initializer copies
    run_guarded(2, [Task(w.log_info, (7,))])
    assert 'SYNTHETIC worker line 7' in capfd.readouterr().err.splitlines()  # bare message: format is '%(message)s'


@BOTH
def test_follow_up_tasks_share_the_slots_and_results_carry_the_index(workers, monkeypatch, tmp_path):
    Spy.live, Spy.futures = [], []
    if workers == 2:
        monkeypatch.setattr(pool, 'ProcessPoolExecutor', Spy)
    tasks = [Task(w.sleep_then, (f'read{i}', 0.2)) for i in range(4)]

    def handle(index, result):
        if result.startswith('read') and index % 2 == 0:  # inspect the read, then copy
            return Task(w.sleep_then, (f'copy{index}', 0.2))

    delivered, error = run_guarded(workers, tasks, handle)
    assert error is None
    assert [r for i, r in delivered if r.startswith('read')] == ['read0', 'read1', 'read2', 'read3']  # first stage in order
    assert sorted(r for _, r in delivered if r.startswith('copy')) == ['copy0', 'copy2']
    assert all(i % 2 == 0 for i, r in delivered if r.startswith('copy'))
    if workers == 2:
        assert max(Spy.live) == 2 and len(Spy.live) == 6


@BOTH
def test_abort_on_a_read_result_starts_no_new_patient_but_finishes_started_ones(workers):
    tasks = [Task(w.echo, (i,)) for i in range(3)]
    handle = lambda i, r: ABORT if r == 0 else Task(w.echo, (f'copy {i}',)) if isinstance(r, int) else None  # noqa: E731
    delivered, error = run_guarded(workers, tasks, handle)
    assert error is None and delivered[0] == (0, 0)
    assert all(i < workers for i, _ in delivered)  # only the first N were started; 2 never is
    assert ('copy 0' not in [r for _, r in delivered]) and (workers == 1 or (1, 'copy 1') in delivered)


@BOTH
@pytest.mark.parametrize('fn', [w.convert_patient, w.preprocess_patient, w.organize_read, w.organize_copy])
def test_per_stage_raisers_fail_on_the_chosen_folder_only(fn, workers, tmp_path):
    delivered, error = run_guarded(workers, [Task(fn, (tmp_path / name,)) for name in ('ok', w.RAISE_NAME)])
    assert error is None
    assert [isinstance(r, Failed) for _, r in delivered] == [False, True]


def test_a_real_ctrl_c_keeps_the_row_of_a_running_task_while_another_worker_is_idle(tmp_path):
    # A real SIGINT to the process group, as a terminal sends it: an idle worker must not die and break the pool.
    import signal
    import subprocess
    import sys
    import time
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path(w.__file__).parent), str(Path(pool.__file__).parents[1])])}
    proc = subprocess.Popen([sys.executable, '-c', w.INTERRUPT_SCRIPT, str(tmp_path)], env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + TIMEOUT
        while not (tmp_path / 'a.start').exists():
            assert time.monotonic() < deadline and proc.poll() is None, 'the task never started'
            time.sleep(0.05)
        time.sleep(1.0)  # 'b' has finished: its worker is idle
        os.killpg(proc.pid, signal.SIGINT)
        out, err = proc.communicate(timeout=TIMEOUT)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert out.strip() == "interrupted [(0, 'cleaned a'), (1, 'b')]"
    assert 'Traceback' not in err  # no worker died on the signal


@BOTH
def test_a_failed_later_index_still_runs_the_follow_up_of_an_earlier_one(workers, tmp_path):
    # index 0's read finishes before index 1 fails (or in the same pass): 0's copy still runs, nothing new starts
    tasks = [Task(w.echo, ('read 0',)), Task(w.fail, ('read 1', 0.5)), Task(w.echo, ('read 2',))]
    follow = lambda i, r: Task(w.sleep_then, (f'copy {i}', 1.0)) if r == 'read 0' else None  # noqa: E731
    delivered, error = run_guarded(workers, tasks, follow)
    assert error is None
    assert (0, 'copy 0') in delivered and (1, Failed('RuntimeError')) in delivered
    assert all(i != 2 for i, _ in delivered) or workers == 1  # N=2 never starts 2 after the failure; N=1 ran it before


@BOTH
def test_a_stop_result_halts_submission_when_it_completes_not_when_its_turn_comes(workers):
    # Codex P2 on #176: a slow patient 0 and an immediately failing patient 1 must not let the rest of the cohort start.
    tasks = [Task(w.sleep_then, ('slow 0', 1.5)), Task(w.stop_now, ('failed 1',))] + [Task(w.echo, (f'later {i}',)) for i in range(2, 6)]
    delivered, error = run_guarded(workers, tasks)
    assert error is None
    assert delivered == [(0, 'slow 0'), (1, 'failed 1')]  # unwrapped, in input order; nothing after 1 ever started


@BOTH
@pytest.mark.parametrize('code, expected', [(3, 3), (None, None), ('SYNTHETIC_secret_message', 1)])
def test_a_system_exit_keeps_its_sanitized_code_and_never_its_message(workers, code, expected):
    delivered, error = run_guarded(workers, [Task(w.exit_with, (code,))])
    assert error is None and delivered == [(0, Failed('SystemExit', expected))]


@BOTH
def test_a_raising_handle_still_delivers_the_running_tasks_then_raises(workers):
    # #178: a stage handle that raised lost the rows of patients still running (they had published, so a rerun skipped them)
    def handle(index, result):
        if index == 0:
            raise ValueError('SYNTHETIC_refusal')
        return Task(w.echo, ('follow-up',)) if index == 1 else None  # returned after the error: dropped, nothing new starts

    tasks = [Task(w.echo, ('fast 0',)), Task(w.sleep_then, ('slow 1', 1.0))] + [Task(w.echo, (f'later {i}',)) for i in range(2, 5)]
    delivered, error = run_guarded(workers, tasks, handle)
    assert isinstance(error, ValueError) and str(error) == 'SYNTHETIC_refusal'
    assert delivered == [(0, 'fast 0'), (1, 'slow 1')][:workers]  # N=1: patient 1 never started


def test_when_handle_raises_twice_the_first_exception_is_raised():
    # #182: two running patients both make handle raise (e.g. two SystemExit codes): input order decides, the second is dropped
    def handle(index, result):
        raise SystemExit(index + 3)

    delivered, error = run_guarded(2, [Task(w.sleep_then, ('a', 0.5)), Task(w.echo, ('b',))], handle)
    assert delivered == [(0, 'a'), (1, 'b')] and isinstance(error, SystemExit) and error.code == 3


# --- sizing (#175): fixture /proc and /sys/fs/cgroup trees under tmp_path, never the real ones ---

GB = 10**9


def put(root, rel, text):
    path = Path(root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def fake_host(root, cgroup_lines, available_kb=8_000_000):
    put(root, 'proc/meminfo', f'MemTotal: 16000000 kB\nMemAvailable: {available_kb} kB\n')
    put(root, 'proc/self/cgroup', cgroup_lines)


def v2_level(root, path, limit, current=0, inactive=0):
    base = f'sys/fs/cgroup{path}'
    put(root, f'{base}/memory.max', f'{limit}\n')
    put(root, f'{base}/memory.current', f'{current}\n')
    put(root, f'{base}/memory.stat', f'anon 1\ninactive_file {inactive}\nactive_file 5\n')


def v1_level(root, path, limit, usage=0, inactive=0):
    base = f'sys/fs/cgroup/memory{path}'
    put(root, f'{base}/memory.limit_in_bytes', f'{limit}\n')
    put(root, f'{base}/memory.usage_in_bytes', f'{usage}\n')
    put(root, f'{base}/memory.stat', f'cache 1\ntotal_inactive_file {inactive}\n')


def test_available_ram_is_memavailable_without_any_cgroup_limit(tmp_path):
    fake_host(tmp_path, '0::/job_1/step_2/user/task_0\n')
    v2_level(tmp_path, '', 'max')
    v2_level(tmp_path, '/job_1', 'max')
    assert pool.available_ram(tmp_path) == 8_000_000 * 1024


def test_cgroup_v2_ancestor_limit_caps_the_result_minus_non_reclaimable_use(tmp_path):
    fake_host(tmp_path, '0::/job_1/step_2/user/task_0\n')
    v2_level(tmp_path, '/job_1', 4 * GB, current=3 * GB, inactive=GB)  # free = 4 - (3 - 1) = 2 GB
    v2_level(tmp_path, '/job_1/step_2/user/task_0', 'max', current=GB)  # 'max' on the leaf is no limit
    assert pool.available_ram(tmp_path) == 2 * GB


def test_cgroup_v2_tightest_level_wins(tmp_path):
    fake_host(tmp_path, '0::/job_1/step_2\n')
    v2_level(tmp_path, '/job_1', 4 * GB)
    v2_level(tmp_path, '/job_1/step_2', 3 * GB)
    assert pool.available_ram(tmp_path) == 3 * GB
    v2_level(tmp_path, '/job_1/step_2', 6 * GB)
    assert pool.available_ram(tmp_path) == 4 * GB


def test_a_cgroup_limit_above_memavailable_does_not_raise_the_result(tmp_path):
    fake_host(tmp_path, '0::/job_1\n')
    v2_level(tmp_path, '/job_1', 100 * GB)
    assert pool.available_ram(tmp_path) == 8_000_000 * 1024


def test_cgroup_v1_uses_the_memory_controller_line_and_ignores_the_unlimited_sentinel(tmp_path):
    fake_host(tmp_path, '12:cpu,cpuacct:/other\n4:memory:/job_1/step_2\n1:name=systemd:/x\n')
    v1_level(tmp_path, '/job_1', 5 * GB, usage=3 * GB, inactive=GB)  # free = 5 - (3 - 1) = 3 GB
    v1_level(tmp_path, '/job_1/step_2', 2**63 - 4096, usage=GB)  # unlimited sentinel
    v1_level(tmp_path, '', 2**63 - 4096)
    assert pool.available_ram(tmp_path) == 3 * GB


def test_unreadable_meminfo_is_none(tmp_path):
    put(tmp_path, 'proc/self/cgroup', '0::/job_1\n')
    assert pool.available_ram(tmp_path) is None


def test_an_unreadable_cgroup_file_is_no_limit_at_that_level(tmp_path):
    fake_host(tmp_path, '0::/job_1/step_2\n')
    v2_level(tmp_path, '/job_1', 3 * GB)
    put(tmp_path, 'sys/fs/cgroup/job_1/step_2/memory.max', '2000000000\n')  # no memory.current or memory.stat beside it
    assert pool.available_ram(tmp_path) == 3 * GB
    put(tmp_path, 'proc/self/cgroup', '')  # no cgroup line at all
    assert pool.available_ram(tmp_path) == 8_000_000 * 1024


def test_a_negative_free_value_clamps_to_zero(tmp_path):
    fake_host(tmp_path, '0::/job_1\n')
    v2_level(tmp_path, '/job_1', GB, current=3 * GB)
    assert pool.available_ram(tmp_path) == 0


def test_usable_cpus_follows_the_affinity_mask_else_cpu_count(monkeypatch):
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 3, 5}, raising=False)
    assert pool.usable_cpus() == 3
    monkeypatch.delattr(os, 'sched_getaffinity')
    monkeypatch.setattr(os, 'cpu_count', lambda: 6)
    assert pool.usable_cpus() == 6
    monkeypatch.setattr(os, 'cpu_count', lambda: None)
    assert pool.usable_cpus() == 1


@pytest.mark.parametrize('peak, available, cpus, remaining, expected', [
    (GB, 10 * GB, 32, 100, 8),  # RAM binds: floor(0.8 * 10)
    (3 * GB, 10 * GB, 32, 100, 2),  # floor(2.67)
    (GB, 100 * GB, 4, 100, 4),  # CPUs bind
    (GB, 1000 * GB, 128, 500, 64),  # MAX_WORKERS binds
    (GB, 100 * GB, 32, 3, 3),  # patients left bind
    (10 * GB, GB, 32, 100, 1),  # RAM for less than one worker: still 1
    (GB, 0, 32, 100, 1),
    (GB, 10 * GB, 32, 0, 1),
    (GB, None, 32, 100, 1),  # RAM unknown
    (0, 10 * GB, 32, 100, 1),  # peak unknown
    (-1, 10 * GB, 32, 100, 1),
])
def test_size_workers_table(peak, available, cpus, remaining, expected):
    assert pool.size_workers(peak, available, cpus, remaining) == expected


def test_max_workers_is_64_and_resource_is_not_imported_at_module_level():
    assert pool.MAX_WORKERS == 64
    assert not hasattr(pool, 'resource')


# --- run_auto (#175): probe in-process, then one sized pool.run ---

class Probe:
    """Patches the sizer's inputs and spies on pool.run; `chosen` is what the sizer returns."""

    def __init__(self, monkeypatch, peak_kib=2_000_000, available=40 * GB, cpus=8, chosen=1):
        self.calls, self.sized = [], []
        real_run = pool.run

        def spy(workers, tasks, handle):
            self.calls.append((workers, list(tasks), handle))
            return real_run(workers, tasks, handle)

        def sizer(*args):
            self.sized.append(args)
            return chosen

        class Usage:
            ru_maxrss = peak_kib

        import resource
        monkeypatch.setattr(pool, 'run', spy)
        monkeypatch.setattr(pool, 'size_workers', sizer)
        monkeypatch.setattr(pool, 'available_ram', lambda root: available)
        monkeypatch.setattr(pool, 'usable_cpus', lambda: cpus)
        monkeypatch.setattr(resource, 'getrusage', lambda who: Usage)


def echo_tasks(n):
    return [Task(w.echo, (i,)) for i in range(n)]


def test_run_auto_with_an_int_is_one_unchanged_pool_run(monkeypatch):
    probe = Probe(monkeypatch)
    tasks, seen = echo_tasks(3), []

    def handle(index, result):
        seen.append((index, result))

    assert pool.run_auto(1, tasks, handle, lambda k: pytest.fail('processed is for auto only')) == 1
    assert [(n, t) for n, t, _ in probe.calls] == [(1, tasks)] and probe.calls[0][2] is handle
    assert seen == [(0, 0), (1, 1), (2, 2)] and probe.sized == []


def test_auto_probes_until_a_processed_patient_then_sizes_one_pool_for_the_rest(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    probe = Probe(monkeypatch, peak_kib=3_000_000, available=40 * GB, cpus=8, chosen=2)
    tasks, seen, done = echo_tasks(5), [], set()

    def handle(index, result):
        seen.append((index, result))
        if index == 1 and result == 1:  # patient 1 is read first, its follow-up is what processes it
            return Task(w.echo, ('copy',))
        if result == 'copy' or index >= 2:
            done.add(index)

    assert pool.run_auto('auto', tasks, handle, lambda k: k in done) == 2
    assert [(n, [t.args for t in ts]) for n, ts, _ in probe.calls] == [
        (1, [(0,)]), (1, [(1,)]), (2, [(2,), (3,), (4,)])]  # the follow-up ran inside the probe call for patient 1
    assert probe.sized == [(3_000_000 * 1024, 40 * GB, 8, 3)]
    assert sorted(seen, key=str) == sorted([(0, 0), (1, 1), (1, 'copy'), (2, 2), (3, 3), (4, 4)], key=str)  # original indexes
    assert any(r.levelno == logging.INFO and '2' in r.getMessage() for r in caplog.records)


def test_the_pool_call_after_the_probe_hands_the_handle_original_indexes(monkeypatch):
    Probe(monkeypatch, chosen=2)
    seen = []
    tasks = echo_tasks(4)
    pool.run_auto('auto', tasks, lambda i, r: seen.append((i, r)), lambda k: True)  # a real N=2 pool for tasks 1..3
    assert sorted(seen) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_no_second_pool_run_when_the_probe_patient_was_the_last(monkeypatch):
    probe = Probe(monkeypatch)
    assert pool.run_auto('auto', echo_tasks(1), lambda i, r: None, lambda k: True) == 1
    assert [n for n, _, _ in probe.calls] == [1]


def test_all_unprocessed_means_every_patient_in_its_own_single_pool_run(monkeypatch):
    probe = Probe(monkeypatch, chosen=4)
    assert pool.run_auto('auto', echo_tasks(3), lambda i, r: None, lambda k: False) == 1
    assert [(n, len(ts)) for n, ts, _ in probe.calls] == [(1, 1)] * 3 and probe.sized == []


def test_no_tasks_means_no_pool_run(monkeypatch):
    probe = Probe(monkeypatch)
    assert pool.run_auto('auto', [], lambda i, r: None, lambda k: True) == 1
    assert probe.calls == []


def test_probe_ends_on_abort_failed_or_a_raising_handle(monkeypatch):
    probe = Probe(monkeypatch, chosen=2)
    assert pool.run_auto('auto', echo_tasks(3), lambda i, r: ABORT, lambda k: False) == 1
    assert len(probe.calls) == 1

    probe.calls.clear()
    seen = []
    tasks = [Task(w.fail, (0,)), *echo_tasks(2)]
    pool.run_auto('auto', tasks, lambda i, r: seen.append(r), lambda k: False)
    assert len(probe.calls) == 1 and seen == [Failed('RuntimeError')]

    probe.calls.clear()

    def raising(index, result):
        raise ValueError('SYNTHETIC')

    with pytest.raises(ValueError):
        pool.run_auto('auto', echo_tasks(3), raising, lambda k: True)
    assert len(probe.calls) == 1 and probe.sized == []


def test_keyboard_interrupt_in_the_probe_propagates(monkeypatch):
    probe = Probe(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        pool.run_auto('auto', [Task(w.interrupt, (0,)), *echo_tasks(2)], lambda i, r: None, lambda k: False)
    assert len(probe.calls) == 1


def test_without_the_resource_module_the_remaining_patients_run_one_at_a_time(monkeypatch):
    probe = Probe(monkeypatch, chosen=9)
    monkeypatch.setitem(sys.modules, 'resource', None)  # import resource -> ImportError, as on Windows
    sizes = []
    monkeypatch.setattr(pool, 'size_workers', lambda *args: sizes.append(args) or 1)
    assert pool.run_auto('auto', echo_tasks(3), lambda i, r: None, lambda k: True) == 1
    assert [n for n, _, _ in probe.calls] == [1, 1] and sizes[0][0] <= 0

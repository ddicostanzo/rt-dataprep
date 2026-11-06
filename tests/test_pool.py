"""The scheduler: one code path for N=1 (inline) and N>1 (spawned processes). Every consumer runs in a daemon thread under a timeout."""
import logging
import os
import signal
import threading
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

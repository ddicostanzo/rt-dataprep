"""The scheduler: one code path for N=1 (inline) and N>1 (spawned processes). Every consumer runs in a daemon thread under a timeout."""
import logging
import os
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


def test_a_dying_worker_is_a_failed_and_does_not_hang(capfd):
    tasks = [Task(w.die, (0,)), Task(w.sleep_then, (1, 0.5)), Task(w.echo, (2,))]
    delivered, error = run_guarded(2, tasks)
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
def test_abort_on_a_read_result_drops_its_copy(workers):
    tasks = [Task(w.echo, (i,)) for i in range(3)]
    delivered, _ = run_guarded(workers, tasks, lambda i, r: ABORT if i == 0 else Task(w.echo, ('copy',)))
    assert ('copy' not in [r for _, r in delivered]) and delivered[0] == (0, 0)


@BOTH
@pytest.mark.parametrize('fn', [w.convert_patient, w.preprocess_patient, w.organize_read, w.organize_copy])
def test_per_stage_raisers_fail_on_the_chosen_folder_only(fn, workers, tmp_path):
    delivered, error = run_guarded(workers, [Task(fn, (tmp_path / name,)) for name in ('ok', w.RAISE_NAME)])
    assert error is None
    assert [isinstance(r, Failed) for _, r in delivered] == [False, True]

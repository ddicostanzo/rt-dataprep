"""
One scheduler for every --workers N: tasks run on N patient processes, or inline when N is 1.

The parent owns all bookkeeping. Workers turn their own failures into data, because a traceback does not cross the process
boundary; whatever still escapes a worker reaches the caller as Failed(type name), never as an exception from a future (its
message and remote traceback can quote a patient). Nothing here formats, logs or chains an exception.
"""
import contextlib
import logging
import multiprocessing
import signal
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Callable, NamedTuple


class Task(NamedTuple):
    fn: Callable  # module-level and picklable for N > 1
    args: tuple = ()
    key: str | None = None  # hold key, e.g. a casefolded name: tasks of one key never run at the same time


class Failed(NamedTuple):
    """The result of a task whose function raised: only the exception type's name (and a SystemExit's sanitized code)."""
    type_name: str
    code: int | None = None  # SystemExit only: an int code is kept, None stays None, anything else (a message) becomes 1


class Stop(NamedTuple):
    """A task result that stops new first-stage submissions as soon as it completes; handle gets `value` in its turn."""
    value: object


ABORT = object()  # returned by handle: submit nothing further
_SKIP = object()  # a first-stage task that ended with KeyboardInterrupt (or any exception once interrupted): no result
_MASKABLE = hasattr(signal, 'pthread_sigmask')  # POSIX; on Windows Ctrl-C reaches the parent as before


class _Inline:
    """The N=1 executor: runs the task in the calling process, so the same scheduling code serves every N."""

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:  # KeyboardInterrupt too: the scheduler decides what it means
            future.set_exception(exc)
        return future


def _init_worker(level):
    # An idle worker ignores Ctrl-C: one killed while waiting for work breaks the pool, which kills the busy ones mid-cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if _MASKABLE:  # spawned with SIGINT blocked (the parent's mask): a Ctrl-C during startup is dropped here, not fatal (#177)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    logging.basicConfig(level=level, format='%(message)s')  # a spawned worker starts with no handler: INFO would vanish


def _call(fn, *args):
    """Run fn in a worker with Ctrl-C raising KeyboardInterrupt, as in the parent, so its own cleanup runs."""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return fn(*args)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _take_ctrl_c():
    """Consume a pending, blocked Ctrl-C (N > 1 only): True if there was one."""
    if signal.SIGINT not in signal.sigpending():
        return False
    signal.sigwait({signal.SIGINT})
    return True


def _submit(executor, task):
    call = (task.fn,) if isinstance(executor, _Inline) else (_call, task.fn)  # inline: the parent's handler stays as it is
    try:
        return executor.submit(*call, *task.args)
    except Exception as exc:  # e.g. a pool that broke between two checks: the same Failed as any other escape
        future = Future()
        future.set_exception(exc)
        return future


def run(workers, tasks, handle):
    """
    Run tasks on `workers` slots and give each result to handle(index, result), index being the task's position in `tasks`.

    At most `workers` tasks are in flight, a task whose key matches a running task waits (later tasks may start ahead of it).
    First results reach handle in input order whatever the completion order; a result is a task's return value or Failed.
    handle returns None, ABORT, or a Task: a follow-up for the same index, queued ahead of unstarted work and sharing the
    slots; its result reaches handle(index, result) when it completes (in completion order) and may chain again.
    After ABORT, a Failed, or a Stop result (from the moment it completes), no further first-stage task is submitted; running tasks finish and are delivered, and a
    follow-up of an index already started still runs (that patient was under way). KeyboardInterrupt, from a task or the waiting parent, stops submission the same way; a task that returns still
    reaches handle, one that raises gets no call, no follow-up runs, and the interrupt is raised once the running tasks end.
    Idle workers ignore Ctrl-C; a running task gets it as KeyboardInterrupt.
    An exception from handle stops submission like ABORT and drops the follow-ups handle returns after it; running tasks
    still finish and reach handle (their output is published, so their rows must be kept), then the first one is raised;
    a later exception from handle is dropped. A caller that must not start a delivered patient after its own raise (organize's
    claim) refuses it in handle itself (#182).
    With N > 1 on POSIX the parent keeps SIGINT blocked and polls for it before every pick and every 0.2 s of waiting
    (#177): workers start with it blocked, and a Ctrl-C never lands inside wait's lock handling or in handle. A task already
    running when the Ctrl-C reached only the parent (or a starting worker) runs to its end and is delivered. The mask is the
    calling thread's: call run from the main thread (the CLI does), or Ctrl-C reaches another thread unmasked.
    """
    queue = deque((i, task, True) for i, task in enumerate(tasks))  # (index, task, is the first stage)
    count = len(queue)
    running = {}  # future -> (index, key, first stage)
    started, ready = set(), {}  # first-stage indexes submitted; first-stage results waiting for their turn
    next_index = 0
    stop = interrupted = False
    raised = None  # the first exception from handle (SystemExit too): raised once the running tasks are delivered (#178)

    def deliver(index, result):
        nonlocal stop, raised
        try:
            follow = handle(index, result)
        except (Exception, SystemExit) as exc:
            stop, raised = True, exc if raised is None else raised
            return
        if follow is ABORT:
            stop = True
        elif follow is not None and not interrupted and raised is None:
            queue.appendleft((index, follow, False))

    masked = workers > 1 and _MASKABLE  # N=1 runs tasks in this process: they get Ctrl-C as before
    saved = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT}) if masked else None  # before any worker or thread exists
    try:
        context = (contextlib.nullcontext(_Inline()) if workers == 1 else
                   ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),  # fork would leak parent state
                                       initializer=_init_worker, initargs=(logging.getLogger().level,)))
        with context as executor:
            while True:
                while len(running) < workers:
                    if masked and _take_ctrl_c():
                        stop = interrupted = True
                    busy = {key for _, key, _ in running.values()}
                    pick = next((e for e in queue if not (e[2] and stop) and not interrupted
                                 and (e[1].key is None or e[1].key not in busy)), None)
                    if pick is None:
                        break
                    queue.remove(pick)
                    index, task, first = pick
                    running[_submit(executor, task)] = (index, task.key, first)
                    started.add(index)
                if not running:
                    break
                try:
                    done, _ = wait(running, timeout=0.2 if masked else None, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:  # the workers got the same signal; let them end, then re-raise
                    stop = interrupted = True
                    continue
                for future in done:
                    index, _, first = running.pop(future)
                    error = future.exception()
                    if error is None:
                        result = future.result()
                        if isinstance(result, Stop):  # at completion, not at its in-order delivery: a slow earlier patient must not let the rest start
                            stop, result = True, result.value
                    elif isinstance(error, KeyboardInterrupt):
                        stop = interrupted = True
                        result = _SKIP
                    else:
                        stop = True
                        code = error.code if isinstance(error, SystemExit) and isinstance(error.code, int | None) else 1
                        result = _SKIP if interrupted else Failed(type(error).__name__, code if isinstance(error, SystemExit) else None)
                    if first:
                        ready[index] = result
                    elif result is not _SKIP:
                        deliver(index, result)
                while next_index < count:
                    if next_index in ready:
                        result = ready.pop(next_index)
                        if result is not _SKIP:
                            deliver(next_index, result)
                    elif not (stop and next_index not in started):  # never submitted after a stop: nobody waits for it
                        break
                    next_index += 1
        if masked and _take_ctrl_c():  # one during shutdown: raised here, not by the mask restore below
            interrupted = True
        if interrupted:
            raise KeyboardInterrupt
        if raised is not None:
            raise raised
    finally:
        if masked:
            signal.pthread_sigmask(signal.SIG_SETMASK, saved)

"""Module-level worker functions for the scheduler tests: picklable and importable in a spawned child (tests/ is on pythonpath)."""
import logging
import os
import signal
import time
from pathlib import Path

MARKER = 'original'  # a test patches this in the parent; a spawned worker re-imports the module and sees the original
RAISE_NAME = 'boom'  # the patient folder name the per-stage raisers below fail on
SECRET = 'SYNTHETIC_secret_message'  # what a raising worker puts in its exception: it must never appear anywhere


def echo(value):
    return value


def pid():
    return os.getpid()


def marker():
    return MARKER


def sleep_then(value, seconds):
    time.sleep(seconds)
    return value


def record(name, directory, seconds):
    """Write <name>.start and <name>.end (monotonic clock, shared by all processes) around a sleep."""
    Path(directory, f'{name}.start').write_text(repr(time.monotonic()))
    time.sleep(seconds)
    Path(directory, f'{name}.end').write_text(repr(time.monotonic()))
    return name


def fail(value, seconds=0.0):
    time.sleep(seconds)
    raise RuntimeError(SECRET)


def interrupt(value):
    raise KeyboardInterrupt


def die(value):
    os.kill(os.getpid(), signal.SIGKILL)  # kills the worker process: BrokenProcessPool in the parent (never call it inline)


def after_file(value, path, seconds):
    """Wait (at most 30 s) for the parent to create path, then sleep."""
    deadline = time.monotonic() + 30
    while not Path(path).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(seconds)
    return value


def log_info(value):
    logging.getLogger('pool_workers').info('SYNTHETIC worker line %s', value)
    return value


def _raise_for(folder):
    if Path(folder).name == RAISE_NAME:
        raise RuntimeError(SECRET)


# The per-stage raisers stand in for the real per-patient call (internal seam): they raise for the patient folder named
# RAISE_NAME and otherwise return a minimal result of the stage's shape.
def convert_patient(folder, *args, **kwargs):
    _raise_for(folder)
    return [], 0.0, 0, None  # rows, reading seconds, files, status


def preprocess_patient(folder, *args, **kwargs):
    _raise_for(folder)
    return [], None  # rows, status


def organize_read(folder, *args, **kwargs):
    _raise_for(folder)
    return Path(folder).name, None, [], [], 0, 0, 0.0  # name, refusal, per-plan paths, misc paths, unreadable, walked, reading seconds


def organize_read_escaping(folder, *args, **kwargs):
    """Folder 'a' reads as the hostile name '..'; any other folder reads normally (in-order delivery puts it after a's raise)."""
    if Path(folder).name == 'a':
        return '..', None, [], [], 0, 0, 0.0
    return organize_read(folder)


def organize_read_a_exits(folder, *args, **kwargs):
    """Folder 'a' exits with code 3; any other folder reads normally (in-order delivery puts it after a's raise)."""
    if Path(folder).name == 'a':
        raise SystemExit(3)
    return organize_read(folder)


def organize_copy(folder, *args, **kwargs):
    _raise_for(folder)
    return 0.0, None  # copy seconds, status


def cleanup_on_interrupt(value, directory):
    """Sleep until Ctrl-C, then clean up and return: the result must still reach the parent."""
    Path(directory, f'{value}.start').write_text('')
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        time.sleep(0.5)
        return f'cleaned {value}'
    return f'slept {value}'


INTERRUPT_SCRIPT = '''
import sys
import pool_workers as w
from rt_dataprep import pool
got = []
try:
    pool.run(2, [pool.Task(w.cleanup_on_interrupt, ('a', sys.argv[1])), pool.Task(w.echo, ('b',))],
             lambda i, r: got.append((i, r)))
except KeyboardInterrupt:
    print('interrupted', sorted(got))
'''


def stop_now(value):
    from rt_dataprep import pool
    return pool.Stop(value)


def exit_with(code):
    raise SystemExit(code)


def patient_exits_3(folder, *args, **kwargs):
    raise SystemExit(3)

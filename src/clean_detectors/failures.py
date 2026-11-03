"""Narrow infrastructure classification; semantic/programming errors never loop."""
import errno
from pathlib import Path
import sys
import time
import traceback
from .core import atomic_json


def retryable_exception(exc):
    if isinstance(exc,OSError) and exc.errno in {errno.EIO,errno.ESTALE,errno.ETIMEDOUT,errno.ECONNRESET}:
        return True
    return isinstance(exc,RuntimeError) and any(value in str(exc) for value in (
        'CUDA error: unknown error','driver shutting down','CUDA-capable devices are busy or unavailable'))


def worker_entry(main):
    try:return main() or 0
    except Exception as exc:
        retry=retryable_exception(exc)
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            atomic_json(output/'errors'/f'{time.time_ns()}.json',{'at':time.time(),'error':str(exc),
              'exception_type':type(exc).__name__,'retryable_infrastructure':retry,'traceback':traceback.format_exc()})
        traceback.print_exc()
        return 76 if retry else 1

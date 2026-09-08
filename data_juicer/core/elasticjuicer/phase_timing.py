"""Diagnostic wall spans, separate from the durable execution/metrics contract.

Driver spans use ONE process clock. Nested/overlapping spans are retained, not
summed; consumers must partition the time axis and report the residual. Worker
counters are actor-seconds, never additive components of driver wall time.
"""

from contextlib import contextmanager, nullcontext
from functools import wraps
from threading import Lock
import time


class PhaseTiming:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.origin = clock()
        self.spans = []
        self.lock = Lock()

    @contextmanager
    def span(self, name, **attributes):
        start = self.clock() - self.origin
        try:
            yield
        finally:
            end = self.clock() - self.origin
            with self.lock:
                self.spans.append(dict(phase=name, start=start, end=end, **attributes))

    def snapshot(self):
        return {
            "schema_version": 1,
            "clock": "driver_perf_counter",
            "wall_seconds": self.clock() - self.origin,
            "spans": list(self.spans),
        }


def phase(owner, name, **attributes):
    timing = getattr(owner, "_ej_timing", None)
    return timing.span(name, **attributes) if timing is not None else nullcontext()


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            with phase(self, name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorate

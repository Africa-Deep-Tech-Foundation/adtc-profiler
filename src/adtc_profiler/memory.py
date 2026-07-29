"""Sample RSS during a subprocess run; return peak + steady-state.

Used by wrapping a llama-bench (or llama-cli) invocation in `sample_during()`:
  with sample_during() as sampler:
      run_some_subprocess(...)
  report = sampler.report()

Implementation: background daemon thread polls psutil.Process(self_pid).memory_info()
plus any child processes spawned during the window. Peak is max observed RSS;
steady-state is the mean of the last 60s of samples (or the last half if run <120s).
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import psutil


@dataclass
class _Sample:
    t: float
    rss_mb: float
    vms_mb: float


@dataclass
class MemorySampler:
    interval_s: float = 0.1
    _samples: list[_Sample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _root_pid: int = 0

    def _poll(self) -> None:
        root = psutil.Process(self._root_pid)
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                family = [root] + root.children(recursive=True)
                rss = sum(p.memory_info().rss for p in family if p.is_running())
                vms = sum(p.memory_info().vms for p in family if p.is_running())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._stop.wait(self.interval_s)  # don't busy-spin on errors
                continue
            self._samples.append(
                _Sample(time.monotonic() - t0, rss / (1024**2), vms / (1024**2))
            )
            self._stop.wait(self.interval_s)

    def start(self, pid: int) -> None:
        self._root_pid = pid
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def report(self) -> dict:
        if not self._samples:
            return {"peak_rss_mb": 0.0, "steady_state_rss_mb": 0.0, "peak_vms_mb": 0.0}
        rss = [s.rss_mb for s in self._samples]
        vms = [s.vms_mb for s in self._samples]
        duration = self._samples[-1].t - self._samples[0].t
        # Steady-state = mean over the last 60s, or last half of run if shorter
        cutoff = self._samples[-1].t - min(60.0, duration / 2)
        tail = [s.rss_mb for s in self._samples if s.t >= cutoff]
        steady = sum(tail) / len(tail) if tail else (sum(rss) / len(rss))
        return {
            "peak_rss_mb": round(max(rss), 2),
            "steady_state_rss_mb": round(steady, 2),
            "peak_vms_mb": round(max(vms), 2),
        }


@contextlib.contextmanager
def sample_during(pid: int | None = None) -> Iterator[MemorySampler]:
    """Context manager — sample memory of `pid` (default: current process)."""
    sampler = MemorySampler()
    sampler.start(pid if pid is not None else psutil.Process().pid)
    try:
        yield sampler
    finally:
        sampler.stop()

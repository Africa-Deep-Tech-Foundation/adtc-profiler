"""CPU utilization (p99) + best-effort core temperature.

Cloud VMs typically do not expose host thermal sensors — schema allows
core_temp_c_peak to be null. CPU percent is always available via psutil.
"""
from __future__ import annotations

import contextlib
import json
import platform
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import psutil


@dataclass
class ThermalSampler:
    interval_s: float = 0.5
    _cpu_samples: list[float] = field(default_factory=list)
    _temp_samples: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    # Sensor label substrings that indicate a CPU core/die temperature.
    # Matched case-insensitively before falling back to the first available reading.
    _CORE_HINTS: tuple[str, ...] = ("core", "cpu", "tdie", "tccd", "package")

    def _read_temp(self) -> float | None:
        # 1. macOS (Darwin) fallback: use ismc CLI if present
        if platform.system() == "Darwin" and shutil.which("ismc"):
            try:
                out = subprocess.check_output(["ismc", "temp", "-o", "json"], text=True, timeout=2)
                data = json.loads(out)
                if isinstance(data, dict):
                    candidates = []
                    for name, info in data.items():
                        if not isinstance(info, dict):
                            continue
                        quantity = info.get("quantity")
                        if not isinstance(quantity, (int, float)) or quantity <= 0:
                            continue
                        key = str(info.get("key", "")).lower()
                        name_lower = name.lower()
                        is_cpu = any(h in name_lower for h in self._CORE_HINTS) or \
                                 any(h in name_lower for h in ("heatsink", "die", "soc")) or \
                                 key.startswith("td") or key.startswith("th") or key.startswith("tc")
                        if is_cpu:
                            candidates.append(float(quantity))
                    if candidates:
                        return max(candidates)
            except Exception:
                pass

        # 2. psutil's sensors_temperatures works on Linux when /sys/class/thermal is exposed
        if hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures() or {}
                fallback: float | None = None
                for entries in temps.values():
                    for entry in entries:
                        if not entry.current or entry.current <= 0:
                            continue
                        label = (entry.label or "").lower()
                        if any(h in label for h in self._CORE_HINTS):
                            return float(entry.current)  # prefer CPU core/die sensor
                        if fallback is None:
                            fallback = float(entry.current)  # remember first positive reading
                if fallback is not None:
                    return fallback
            except (AttributeError, OSError):
                pass
        # 3. Fallback: lm-sensors CLI if present
        if shutil.which("sensors"):
            try:
                out = subprocess.check_output(["sensors", "-u"], text=True, timeout=2)
                # Two-pass parse: prefer lines whose preceding chip/label block
                # contains a core hint; fall back to the first positive _input: value.
                lines = out.splitlines()
                fallback_value: float | None = None
                current_block_is_core = False
                for line in lines:
                    stripped = line.strip()
                    # A chip header has no leading whitespace and no colon-value pattern
                    if stripped and not stripped.startswith("+") and ":" not in stripped:
                        current_block_is_core = any(h in stripped.lower() for h in self._CORE_HINTS)
                    if "_input:" in line:
                        try:
                            value = float(line.split(":", 1)[1].strip())
                            if value > 0:
                                if current_block_is_core:
                                    return value
                                if fallback_value is None:
                                    fallback_value = value
                        except ValueError:
                            continue
                if fallback_value is not None:
                    return fallback_value
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                pass
        return None

    def _poll(self) -> None:
        # psutil.cpu_percent(interval=…) blocks; the first call returns a
        # meaningless 0.0 — prime it.
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            pct = psutil.cpu_percent(interval=self.interval_s)
            self._cpu_samples.append(pct)
            temp = self._read_temp()
            if temp is not None:
                self._temp_samples.append(temp)

    def start(self) -> None:
        self._stop.clear()
        self._cpu_samples.clear()
        self._temp_samples.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def report(self) -> dict:
        if not self._cpu_samples:
            return {"cpu_percent_p99": 0.0, "core_temp_c_peak": None, "throttled": False}
        sorted_cpu = sorted(self._cpu_samples)
        # p99 — last percentile; for small N this is effectively the max
        idx = max(0, int(len(sorted_cpu) * 0.99) - 1)
        p99 = sorted_cpu[idx]
        peak_temp = max(self._temp_samples) if self._temp_samples else None
        # "throttled" detection requires kernel events; best-effort flag based
        # on observed temp crossing the published penalty threshold (85°C —
        # must match the scoring rule, which applies P_thermal above 85°C).
        # Real throttle-event detection is deferred to Phase 2.
        throttled = bool(peak_temp and peak_temp >= 85.0)
        return {
            "cpu_percent_p99": round(min(100.0, p99), 1),
            "core_temp_c_peak": round(peak_temp, 1) if peak_temp is not None else None,
            "throttled": throttled,
        }


@contextlib.contextmanager
def sample_thermal() -> Iterator[ThermalSampler]:
    sampler = ThermalSampler()
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()

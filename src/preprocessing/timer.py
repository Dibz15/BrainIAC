"""
Author: Austin Dibble
Date: 17-02-2026


Description: Simple and lightweight utility class for profiling code with wall time. 


Example: 
```
t = Timer(max_samples=10_000)

for _ in range(100):
    with t.track("parse"):
        pass

for _ in range(40):
    with t.track("db"):
        pass

print(t.report(sort_by="total"))
```

Output: 

name                      lt_cnt   lt_total    lt_mean |   w_cnt     w_mean      w_p95      w_max (window: last 3)
---------------------------------------------------------------------------------------------------------------
db                           10     42.8ms     4.28ms |       3     5.12ms     6.40ms     6.40ms
parse                        84    118.3ms     1.41ms |       3     1.02ms     1.30ms     1.30ms
cache_hit                  1200     95.6ms    79.7µs |       3    70.2µs    88.0µs    88.0µs
external_api                 6      2.41s    401.7ms |       3    520.3ms    790.0ms    790.0ms

"""

from __future__ import annotations

from time import perf_counter
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, Optional
import math
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Union
import pickle

def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile; p in [0, 100]."""
    if not sorted_vals:
        return math.nan
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = math.ceil((p / 100) * len(sorted_vals)) - 1
    return sorted_vals[k]


@dataclass(frozen=True)
class Stat:
    # Lifetime
    lifetime_count: int
    lifetime_total: float
    lifetime_mean: float

    # Window (last N samples if max_samples is set, else lifetime window)
    window_count: int
    window_total: float
    window_mean: float
    window_min: float
    window_max: float
    window_median: float
    window_p95: float


class Timer:
    """
    Named timer with optional rolling window samples and lifetime aggregates.

    max_samples:
        If set, keep only the last N samples per name for distribution stats.
        Lifetime totals/counts are still tracked accurately.
    """
    def __init__(self, *, max_samples: Optional[int] = None, dump_path:Path = None):
        self._max_samples = max_samples
        self._samples: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples) if max_samples else deque()
        )
        self._lifetime_total = defaultdict(float)
        self._lifetime_count = defaultdict(int)
        self._lock = Lock()
        self._dump_path = dump_path

    @contextmanager
    def track(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - start)

    def add(self, name: str, seconds: float) -> None:
        seconds = float(seconds)
        with self._lock:
            self._samples[name].append(seconds)
            self._lifetime_total[name] += seconds
            self._lifetime_count[name] += 1

    def reset(self, name: Optional[str] = None) -> None:
        """Clear one timer or all timers (both window samples and lifetime aggregates)."""
        with self._lock:
            if name is None:
                self._samples.clear()
                self._lifetime_total.clear()
                self._lifetime_count.clear()
            else:
                self._samples.pop(name, None)
                self._lifetime_total.pop(name, None)
                self._lifetime_count.pop(name, None)

    def stats(self) -> dict[str, Stat]:
        with self._lock:
            samples = {k: list(v) for k, v in self._samples.items()}
            lt_total = dict(self._lifetime_total)
            lt_count = dict(self._lifetime_count)

        out: dict[str, Stat] = {}
        for name in set(samples) | set(lt_total) | set(lt_count):
            vals = samples.get(name, [])
            lc = lt_count.get(name, 0)
            lt = lt_total.get(name, 0.0)

            # Window stats
            if vals:
                w_total = sum(vals)
                w_count = len(vals)
                w_mean = w_total / w_count
                sorted_vals = sorted(vals)
                w_min = sorted_vals[0]
                w_max = sorted_vals[-1]
                w_median = _percentile(sorted_vals, 50)
                w_p95 = _percentile(sorted_vals, 95)
            else:
                w_total = 0.0
                w_count = 0
                w_mean = math.nan
                w_min = math.nan
                w_max = math.nan
                w_median = math.nan
                w_p95 = math.nan

            # Lifetime stats
            l_mean = (lt / lc) if lc else math.nan

            out[name] = Stat(
                lifetime_count=lc,
                lifetime_total=lt,
                lifetime_mean=l_mean,
                window_count=w_count,
                window_total=w_total,
                window_mean=w_mean,
                window_min=w_min,
                window_max=w_max,
                window_median=w_median,
                window_p95=w_p95,
            )
        return out

    def report(self, *, sort_by: str = "lifetime_total", limit: Optional[int] = None) -> str:
        """
        Return a readable multi-line report.
        sort_by: one of {"lifetime_total","lifetime_mean","lifetime_count","window_total","window_mean","window_p95","window_max"}.
        """
        stats = self.stats()
        key_map = {
            "lifetime_total": lambda s: s.lifetime_total,
            "lifetime_mean": lambda s: s.lifetime_mean,
            "lifetime_count": lambda s: s.lifetime_count,
            "window_total": lambda s: s.window_total,
            "window_mean": lambda s: s.window_mean,
            "window_p95": lambda s: s.window_p95,
            "window_max": lambda s: s.window_max,
        }
        if sort_by not in key_map:
            raise ValueError(f"sort_by must be one of {set(key_map)}")

        rows = sorted(stats.items(), key=lambda kv: key_map[sort_by](kv[1]), reverse=True)
        if limit is not None:
            rows = rows[:limit]

        def fmt(sec: float) -> str:
            if math.isnan(sec):
                return "n/a"
            if sec < 1e-6:
                return f"{sec*1e9:.0f}ns"
            if sec < 1e-3:
                return f"{sec*1e6:.1f}µs"
            if sec < 1:
                return f"{sec*1e3:.2f}ms"
            return f"{sec:.3f}s"

        window_note = f" (window: last {self._max_samples})" if self._max_samples else ""
        lines = []
        lines.append(
            f"{'name':24} "
            f"{'lt_cnt':>7} {'lt_total':>10} {'lt_mean':>10} | "
            f"{'w_cnt':>7} {'w_mean':>10} {'w_p95':>10} {'w_max':>10}{window_note}"
        )
        lines.append("-" * (24 + 1 + 7 + 1 + 10 + 1 + 10 + 3 + 7 + 1 + 10 + 1 + 10 + 1 + 10) )
        for name, s in rows:
            lines.append(
                f"{name:24} "
                f"{s.lifetime_count:7d} {fmt(s.lifetime_total):>10} {fmt(s.lifetime_mean):>10} | "
                f"{s.window_count:7d} {fmt(s.window_mean):>10} {fmt(s.window_p95):>10} {fmt(s.window_max):>10}"
            )
        return "\n".join(lines)

    def to_dict(
        self,
        *,
        include_samples: bool = False,
        include_empty: bool = False,
        sort_by: str = "lifetime_total",
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Return a JSON-serializable snapshot of the timer.

        include_samples:
            If True, include the raw rolling-window samples per name.
        include_empty:
            If False, exclude names with lifetime_count == 0 (usually none).
        sort_by / limit:
            Apply sorting/limiting consistent with report().
        """
        stats = self.stats()

        key_map = {
            "lifetime_total": lambda s: s.lifetime_total,
            "lifetime_mean": lambda s: s.lifetime_mean,
            "lifetime_count": lambda s: s.lifetime_count,
            "window_total": lambda s: s.window_total,
            "window_mean": lambda s: s.window_mean,
            "window_p95": lambda s: s.window_p95,
            "window_max": lambda s: s.window_max,
        }
        if sort_by not in key_map:
            raise ValueError(f"sort_by must be one of {set(key_map)}")

        items = list(stats.items())
        if not include_empty:
            items = [(n, s) for (n, s) in items if s.lifetime_count > 0]

        items.sort(key=lambda kv: key_map[sort_by](kv[1]), reverse=True)
        if limit is not None:
            items = items[:limit]

        def num(x: float) -> Optional[float]:
            # JSON does not officially support NaN/Infinity.
            # Convert them to None so dumps(..., allow_nan=False) works.
            return None if (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x

        out_items: list[dict[str, Any]] = []
        for name, s in items:
            entry: dict[str, Any] = {
                "name": name,
                "lifetime": {
                    "count": int(s.lifetime_count),
                    "total_seconds": num(s.lifetime_total),
                    "mean_seconds": num(s.lifetime_mean),
                },
                "window": {
                    "count": int(s.window_count),
                    "total_seconds": num(s.window_total),
                    "mean_seconds": num(s.window_mean),
                    "min_seconds": num(s.window_min),
                    "max_seconds": num(s.window_max),
                    "median_seconds": num(s.window_median),
                    "p95_seconds": num(s.window_p95),
                },
            }

            if include_samples:
                with self._lock:
                    entry["samples_seconds"] = list(self._samples.get(name, ()))

            out_items.append(entry)

        return {
            "meta": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "max_samples": self._max_samples,
                "sort_by": sort_by,
                "limit": limit,
                "include_samples": include_samples,
            },
            "items": out_items,
        }

    def dumps_json(
        self,
        *,
        include_samples: bool = False,
        include_empty: bool = False,
        sort_by: str = "lifetime_total",
        limit: Optional[int] = None,
        indent: Optional[int] = 2,
    ) -> str:
        """Return a JSON string snapshot."""
        payload = self.to_dict(
            include_samples=include_samples,
            include_empty=include_empty,
            sort_by=sort_by,
            limit=limit,
        )
        return json.dumps(payload, indent=indent, allow_nan=False)

    def dump_json(
        self,
        path: Union[str, "Path"],
        *,
        include_samples: bool = False,
        include_empty: bool = False,
        sort_by: str = "lifetime_total",
        limit: Optional[int] = None,
        indent: Optional[int] = 2,
        encoding: str = "utf-8",
    ) -> None:
        """Write a JSON snapshot to `path`."""
        if path is None:
            raise ValueError("Invalid path: path is None")
        
        p = Path(path)
        payload = self.to_dict(
            include_samples=include_samples,
            include_empty=include_empty,
            sort_by=sort_by,
            limit=limit,
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=indent, allow_nan=False), encoding=encoding)

    def dump_json_default(
        self,
        *,
        include_samples: bool = False,
        include_empty: bool = False,
        sort_by: str = "lifetime_total",
        limit: Optional[int] = None,
        indent: Optional[int] = 2,
        encoding: str = "utf-8",
    ) -> None:
        """Write a JSON snapshot to `path`."""
        self.dump_json(self._dump_path, 
                       include_samples=include_samples,
                       include_empty=include_empty,
                       sort_by=sort_by,
                       limit=limit,
                       indent=indent,
                       encoding=encoding
                       )
        return

    # --- Pickle support -------------------------------------------------

    def __getstate__(self):
        """Prepare state for pickling (exclude the lock)."""
        state = self.__dict__.copy()
        state["_lock"] = None  # Locks are not picklable
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._lock = Lock()  # Recreate the lock


    def dump_pickle(self, path: Union[str, "Path"]) -> None:
        """Persist the Timer instance to a pickle file."""
        if path is None:
            raise ValueError("Invalid path: path is None")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            with p.open("wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load_pickle(cls, path: Union[str, "Path"]) -> "Timer":
        """Load a Timer instance from a pickle file."""
        if path is None:
            raise ValueError("Invalid path: path is None")

        p = Path(path)
        with p.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"Pickle at {path} is not a Timer instance")

        return obj

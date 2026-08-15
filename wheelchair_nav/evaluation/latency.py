"""Per-inference latency measurement, shared by every evaluation script so
depth models, the obstacle detector and the full navigation pipeline are
all timed the same way.

WHY LATENCY AND NOT JUST FPS
----------------------------
FPS as reported elsewhere in this repo is a throughput number: total cycles
divided by total elapsed time. It answers "how many frames per second can
this model sustain?". For an autonomous wheelchair that is the wrong
question on its own, because throughput hides the tail: a model averaging
8 ms but occasionally spiking to 90 ms has the same FPS as one that is
flat at 9 ms, yet only the second is safe to drive behind.

What matters for safety is the DISTRIBUTION of the per-frame delay between
a photon reaching the camera and a drive command being issued, and
specifically its upper percentiles. At the controller's 0.6 m/s cruise
speed, every 100 ms of latency is 6 cm travelled on stale information --
so p95/p99 latency, not the mean, is what bounds the worst-case reaction
distance.

MEASUREMENT CHOICES
-------------------
- Wall clock via time.perf_counter() (monotonic, ns resolution), not CUDA
  events. CUDA events would time the GPU kernels alone and silently
  exclude the cv2 resize, the H2D copy and the D2H copy that .infer()
  really performs -- costs the wheelchair actually pays.
- torch.cuda.synchronize() around every sample. CUDA launches are
  asynchronous, so without it the timer would record queue-submission
  time, not completion time, and the numbers would be meaninglessly small.
- Percentiles by linear interpolation (numpy default). With the default
  200 cycles, p99 is interpolated from the top two samples -- honest, but
  raise --latency_cycles if you intend to quote p99 in a paper.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np

# Columns every latency-reporting script writes, in display order.
LATENCY_FIELDS = [
    "lat_mean_ms", "lat_std_ms", "lat_min_ms",
    "lat_p50_ms", "lat_p90_ms", "lat_p95_ms", "lat_p99_ms", "lat_max_ms",
]


def _synchronize(device) -> None:
    """No-op unless we are timing CUDA work."""
    if device is None:
        return
    try:
        import torch

        if getattr(device, "type", str(device)) == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def latency_stats(samples_ms) -> Dict[str, float]:
    """Summarises a list of per-call latencies (milliseconds).

    Returns mean/std/min/max plus the p50, p90, p95 and p99 percentiles,
    and `fps` derived as 1000 / mean so throughput stays consistent with
    the latency it is computed from.
    """
    a = np.asarray(list(samples_ms), dtype=np.float64)
    if a.size == 0:
        return {k: float("nan") for k in LATENCY_FIELDS + ["fps"]}

    p50, p90, p95, p99 = np.percentile(a, [50, 90, 95, 99])
    mean = float(a.mean())
    return {
        "lat_mean_ms": mean,
        # ddof=1: these are a sample of runs, not the whole population.
        # Guard n=1, where the unbiased estimator is undefined.
        "lat_std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "lat_min_ms": float(a.min()),
        "lat_p50_ms": float(p50),
        "lat_p90_ms": float(p90),
        "lat_p95_ms": float(p95),
        "lat_p99_ms": float(p99),
        "lat_max_ms": float(a.max()),
        "fps": 1000.0 / mean if mean > 0 else float("inf"),
    }


def measure_latency(
    fn: Callable[[], object],
    warmup: int = 20,
    cycles: int = 200,
    device=None,
    return_samples: bool = False,
):
    """Times `fn()` `cycles` times after `warmup` untimed calls.

    The warmup matters more than it looks: the first CUDA call pays lazy
    context creation, cuDNN autotuning picks an algorithm on first sight of
    each tensor shape, and clocks are still ramping. Including those in the
    sample would inflate the mean and put a fictional outlier in the tail.

    Returns the latency_stats() dict, plus a "samples_ms" list when
    `return_samples` is set (useful for plotting a latency histogram).
    """
    for _ in range(warmup):
        fn()
    _synchronize(device)

    samples_ms: List[float] = []
    for _ in range(cycles):
        t0 = time.perf_counter()
        fn()
        _synchronize(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = latency_stats(samples_ms)
    if return_samples:
        stats["samples_ms"] = samples_ms
    return stats


def reaction_distance_m(latency_ms: float, speed_mps: float) -> float:
    """Distance travelled while a decision is still in flight.

    The wheelchair keeps executing its PREVIOUS command for the whole time
    the current frame is being processed, so this is the blind-travel
    distance that a given latency buys -- the number to compare against
    config.SAFE_DISTANCE_M and EMERGENCY_DISTANCE_M.
    """
    return speed_mps * latency_ms / 1000.0


def format_latency_table(rows: List[Dict], name_key: str = "model", title: Optional[str] = None) -> str:
    """Renders one latency row per model as a fixed-width table."""
    headers = ["Model", "mean(ms)", "std(ms)", "min(ms)", "p50(ms)",
               "p90(ms)", "p95(ms)", "p99(ms)", "max(ms)", "FPS"]
    keys = [name_key] + LATENCY_FIELDS + ["fps"]

    widths = [max(len(h), 10) for h in headers]
    widths[0] = max(widths[0], max((len(str(r.get(name_key, ""))) for r in rows), default=10) + 2)

    def fmt(values):
        return "".join(f"{str(v):>{w}}" for v, w in zip(values, widths))

    lines = []
    if title:
        lines.append(title)
    lines.append(fmt(headers))
    for r in rows:
        values = [r.get(name_key, "")]
        for k in LATENCY_FIELDS + ["fps"]:
            v = r.get(k)
            values.append("-" if v is None else f"{float(v):.2f}")
        lines.append(fmt(values))
    return "\n".join(lines)

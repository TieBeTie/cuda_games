import csv
import statistics
import time

import torch

# Isolation is not code: plug in the charger, close the browser, then set this.
POWER = "ac"          # "ac" or "battery"
SIZES = [32, 128, 1024, 8192]
WARMUP = 50           # discarded iterations, Reddi MLSysBook p. 713
NUMBER = 5            # calls inside one measurement
REPEAT = 10           # measurements per configuration
CSV_PATH = "results.csv"


def _warm(f, args, warmup):
    for _ in range(warmup):
        f(*args)


def bench_events(f, *args, warmup=WARMUP, number=NUMBER, repeat=REPEAT):
    """Device-side timestamps. Measures the work, host is never stalled inside."""
    _warm(f, args, warmup)
    tms, peaks = [], []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()

        start.record()
        for _ in range(number):
            f(*args)
        end.record()

        torch.cuda.synchronize()
        tms.append(start.elapsed_time(end) / number)
        peaks.append(torch.cuda.max_memory_allocated())
    return tms, peaks


def bench_host_sync(f, *args, warmup=WARMUP, number=NUMBER, repeat=REPEAT):
    """Host clock, but the host waits for the device before each reading."""
    _warm(f, args, warmup)
    tms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(number):
            f(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tms.append((t1 - t0) * 1e3 / number)
    return tms


def bench_host_nosync(f, *args, warmup=WARMUP, number=NUMBER, repeat=REPEAT):
    """Host clock with no waiting. Measures the enqueue, not the work."""
    _warm(f, args, warmup)
    tms = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            f(*args)
        t1 = time.perf_counter()
        tms.append((t1 - t0) * 1e3 / number)
        torch.cuda.synchronize()   # outside the interval: keeps the queue bounded
    return tms


def summarize(tms):
    p50 = statistics.median(tms)
    p90 = statistics.quantiles(tms, n=100)[89]
    cv = statistics.stdev(tms) / statistics.mean(tms)
    return p50, p90, cv


METHODS = [
    ("events", bench_events),
    ("host+sync", bench_host_sync),
    ("host-nosync", bench_host_nosync),
]

device = torch.cuda.get_device_name(0)
print(f"{device} | torch {torch.__version__} | cuda {torch.version.cuda} | power {POWER}")
print(f"warmup={WARMUP} number={NUMBER} repeat={REPEAT}\n")

rows = []
table = {name: [] for name, _ in METHODS}

for n in SIZES:
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)

    print(f"n = {n}")
    print(f"  {'method':<13} {'p50 ms':>10} {'p90 ms':>10} {'CV %':>7} {'peak MiB':>10}")

    for name, fn in METHODS:
        result = fn(torch.matmul, a, b)
        tms, peaks = result if isinstance(result, tuple) else (result, None)
        p50, p90, cv = summarize(tms)
        peak_mib = max(peaks) / 1024 ** 2 if peaks else float("nan")
        table[name].append(p50)

        flag = "" if cv < 0.05 else "  <- noisy"
        print(f"  {name:<13} {p50:>10.5f} {p90:>10.5f} {cv * 100:>7.1f} {peak_mib:>10.1f}{flag}")

        rows.append({
            "label": "matmul",
            "method": name,
            "n": n,
            "dtype": "float32",
            "median_ms": round(p50, 6),
            "p90_ms": round(p90, 6),
            "cv": round(cv, 4),
            "peak_mib": round(peak_mib, 1),
            "device": device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "power": POWER,
            "warmup": WARMUP,
            "number": NUMBER,
            "repeat": REPEAT,
        })
    print()

print(f"Growth relative to n = {SIZES[0]} (work grows as n^3)")
header = "  ".join(f"{n:>10}" for n in SIZES)
print(f"  {'method':<13} {header}")
for name, _ in METHODS:
    base = table[name][0]
    ratios = "  ".join(f"{t / base:>10.1f}" for t in table[name])
    print(f"  {name:<13} {ratios}")

work = "  ".join(f"{(n / SIZES[0]) ** 3:>10.1f}" for n in SIZES)
print(f"  {'work':<13} {work}")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{len(rows)} rows written to {CSV_PATH}")

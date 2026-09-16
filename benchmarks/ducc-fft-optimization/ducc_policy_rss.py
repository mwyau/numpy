#!/usr/bin/env python3
"""Measure one-thread FFT timing and peak RSS for staged candidates.

Each candidate/case is executed in a fresh process.  The benchmark is
intentionally limited to the existing one-dimensional public FFT wrapper.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


SEED = 20260915


def worker(args: argparse.Namespace) -> None:
    import resource
    import numpy as np  # type: ignore

    rng = np.random.default_rng(SEED + args.n + args.batch * 100003)
    shape = (args.n,) if args.batch == 1 else (args.batch, args.n)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        args.dtype)
    fn = np.fft.fft if args.operation == "fft" else np.fft.ifft

    fn(x)
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        for _ in range(args.repeats):
            result = fn(x)
        samples.append((time.perf_counter() - start) * 1000 / args.repeats)
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        "median_ms": statistics.median(samples),
        "best_ms": min(samples),
        "rss_kib": rss_kib,
        "samples": args.samples,
        "repeats": args.repeats,
        "status": "ok",
    }))


def stage_env(stage: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stage)
    env.update({name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS")})
    return env


def driver(args: argparse.Namespace) -> None:
    fields = ["candidate", "operation", "dtype", "n", "batch", "median_ms",
              "best_ms", "rss_kib", "samples", "repeats", "status", "notes"]
    rows = []
    for spec in args.candidate:
        label, raw_stage = spec.split("=", 1)
        for n in args.sizes:
            for batch in args.batches:
                command = [sys.executable, "-S", str(Path(__file__).resolve()),
                           "--worker", "--operation", args.operation,
                           "--dtype", args.dtype, "--n", str(n), "--batch",
                           str(batch), "--samples", str(args.samples),
                           "--repeats", str(args.repeats)]
                proc = subprocess.run(
                    command, cwd="/tmp", env=stage_env(Path(raw_stage)),
                    capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    rows.append({
                        "candidate": label, "operation": args.operation,
                        "dtype": args.dtype, "n": n, "batch": batch,
                        "status": "error", "notes": proc.stderr[-2000:],
                    })
                    continue
                result = json.loads(proc.stdout)
                rows.append({
                    "candidate": label, "operation": args.operation,
                    "dtype": args.dtype, "n": n, "batch": batch,
                    **result,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--operation", choices=("fft", "ifft"))
    parser.add_argument("--dtype", choices=("complex64", "complex128"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    required = (args.candidate, args.output, args.operation, args.dtype,
                args.sizes, args.batches)
    if any(value is None for value in required):
        parser.error("driver mode requires candidates, output, operation, dtype, sizes, and batches")
    driver(args)


if __name__ == "__main__":
    main()
